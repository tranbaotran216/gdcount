from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from groundingdino.util.misc import NestedTensor
from groundingdino.util.inference import load_model  # dùng để load từ config + ckpt


@dataclass
class GDCountConfig:
    """
    Cấu hình cơ bản cho CountGD-style:
    - threshold: ngưỡng hard count trên logit từng query
    - soa_level: index feature level dùng SOA (0 là feature coarse nhất, -1 là level cuối)
    - feature_dim: hidden dim của GroundingDINO (thường 256)
    - freeze_keywords: các từ khoá trong tên param để freeze (backbone, BERT)
    """

    threshold: float = 0.0
    soa_level: int = -1
    feature_dim: int = 256
    freeze_keywords: Sequence[str] = ("backbone.0", "bert")


class SmallObjectAdapter(nn.Module):
    """
    SOA: tăng nhạy với object nhỏ trên feature map phân giải cao.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

        # Squeeze-Excitation
        hidden = max(in_channels // 4, 16)
        self.se_fc1 = nn.Linear(in_channels, hidden)
        self.se_fc2 = nn.Linear(hidden, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # SE channel attention
        b, c, h, w = out.shape
        pooled = out.mean(dim=(2, 3))  # (B,C)
        w_se = self.act(self.se_fc1(pooled))
        w_se = torch.sigmoid(self.se_fc2(w_se)).view(b, c, 1, 1)
        out = out * w_se

        out = self.act(out + residual)
        return out


class SOABackboneWrapper(nn.Module):
    """
    Bọc backbone Joiner của GroundingDINO để chèn SOA vào một feature level.
    """

    def __init__(
        self,
        backbone: nn.Module,
        level_index: int,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone  # Joiner
        self.level_index = level_index
        # sẽ auto-adapt lại kênh nếu cần trong forward
        self.soa = SmallObjectAdapter(in_channels)

    def __getitem__(self, idx):
        return self.backbone[idx]
    
    def __len__(self):
        try:
            return len(self.backbone)
        except Exception:
            return 0
        
        
    def forward(
        self, samples: NestedTensor
    ) -> Tuple[List[NestedTensor], List[torch.Tensor]]:
        features, pos = self.backbone(samples)  # list[NestedTensor], list[pos]
        if not isinstance(features, list) or len(features) == 0:
            return features, pos

        idx = self.level_index
        if idx < 0:
            idx = len(features) + idx
        if idx < 0 or idx >= len(features):
            # không chèn được, trả về như cũ
            return features, pos

        feat_nt: NestedTensor = features[idx]
        src, mask = feat_nt.decompose()  # src: (B,C,H,W)

        # nếu số kênh không khớp với SOA hiện tại, khởi tạo lại
        C = src.shape[1]
        if self.soa.conv1.in_channels != C:
            self.soa = SmallObjectAdapter(C).to(src.device)

        src = self.soa(src)
        features[idx] = NestedTensor(src, mask)
        return features, pos


class TransformerWrapper(nn.Module):
    """
    Bọc transformer của GroundingDINO để lấy hs (decoder queries) mỗi lần forward.
    """

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.last_hs = None

    def forward(self, *args, **kwargs):
        outputs = self.transformer(*args, **kwargs)
        # Theo code GroundingDINO: hs là output[0]
        if isinstance(outputs, tuple) and len(outputs) > 0:
            self.last_hs = outputs[0]
        else:
            self.last_hs = None
        return outputs


class CountHead(nn.Module):
    """
    Count head: đọc hs (decoder queries) -> logit per query -> soft/hard count.
    """

    def __init__(self, d_model: int, threshold: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.threshold = threshold


    def forward(self, hs):
        """
        hs :
          - Tensor [num_layers, B, Q, D]
          - Tensor [B, Q, D]
          - List[Tensor[B, Q, D]] (một số bản GroundingDINO trả kiểu list)
        """
        # Nếu là list/tuple các tensor -> stack lại
        if isinstance(hs, (list, tuple)):
            if len(hs) == 0 or not isinstance(hs[0], torch.Tensor):
                raise ValueError(f"Unexpected hs list: {type(hs)}, first={type(hs[0]) if hs else None}")
            hs = torch.stack(hs, dim=0)  # [L,B,Q,D]

        if hs.dim() == 4:
            hs_last = hs[-1]  # (B,Q,D)
        elif hs.dim() == 3:
            hs_last = hs
        else:
            raise ValueError(f"Unexpected hs shape: {hs.shape}")

        hs_last = self.norm(hs_last)           # (B,Q,D)
        logits = self.fc(hs_last).squeeze(-1)  # (B,Q)
        soft = torch.relu(logits).sum(dim=1)   # (B,)
        hard = (logits > self.threshold).sum(dim=1).to(torch.int64)  # (B,)
        return logits, soft, hard


class GroundingDINOCounter(nn.Module):
    """
    Wrapper chính: GroundingDINO + SOA + CountHead.
    """

    def __init__(self, base_model: nn.Module, cfg: GDCountConfig) -> None:
        super().__init__()
        self.base_model = base_model

        # 1) Lấy đúng số kênh của feature level dùng SOA
        try:
            num_channels_list = self.base_model.backbone.num_channels
            lvl = cfg.soa_level
            if lvl < 0:
                lvl = len(num_channels_list) + lvl
            lvl = max(0, min(lvl, len(num_channels_list) - 1))
            in_channels = num_channels_list[lvl]
        except Exception:
            in_channels = cfg.feature_dim

        # 2) Bọc backbone bằng SOABackboneWrapper
        self.base_model.backbone = SOABackboneWrapper(
            backbone=self.base_model.backbone,
            level_index=cfg.soa_level,
            in_channels=in_channels,
        )

        # 3) Bọc transformer để lấy hs
        self.base_model.transformer = TransformerWrapper(self.base_model.transformer)

        # 4) Tạo count head
        hidden_dim = getattr(self.base_model, "hidden_dim", cfg.feature_dim)
        self.count_head = CountHead(hidden_dim, threshold=cfg.threshold)

        # 5) Freeze image encoder + text encoder theo keyword
        self._apply_freeze(cfg)

    def _apply_freeze(self, cfg: GDCountConfig) -> None:
        # Mặc định cho phép gradient
        for _, p in self.base_model.named_parameters():
            p.requires_grad = True

        # Freeze theo keyword (ví dụ "backbone.0", "bert")
        for name, p in self.base_model.named_parameters():
            if any(kw in name for kw in cfg.freeze_keywords):
                p.requires_grad = False

        # Đảm bảo SOA + CountHead luôn trainable
        for p in self.base_model.backbone.soa.parameters():
            p.requires_grad = True
        for p in self.count_head.parameters():
            p.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        captions: List[str],
        targets: Any = None,
        exemplars: Any = None,
        labels: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        images: (B,3,H,W)
        captions: list[str] length B
        exemplars: thường là list[Tensor(K,4)] xyxy pixel (roi_align style) hoặc tùy base_model
        labels: list/tensor label phrase index per image (vd [tensor([0]), ...])
        """

        # 1) gọi base_model có/không exemplar
        try:
            if exemplars is not None and labels is not None:
                outputs: Dict[str, Any] = self.base_model(
                    images,
                    captions=captions,
                    targets=targets,
                    exemplars=exemplars,
                    labels=labels,
                    **kwargs,
                )
            else:
                outputs: Dict[str, Any] = self.base_model(
                    images,
                    captions=captions,
                    targets=targets,
                    **kwargs,
                )
        except TypeError:
            # fallback cho trường hợp base_model.forward không có tham số exemplars/labels
            outputs: Dict[str, Any] = self.base_model(
                images,
                captions=captions,
                targets=targets,
                **kwargs,
            )

        # 2) lấy hs từ TransformerWrapper
        hs = self.base_model.transformer.last_hs
        if hs is None:
            raise RuntimeError("TransformerWrapper không capture được hs (decoder output).")

        # 3) count head
        query_logits, soft_counts, hard_counts = self.count_head(hs)
        outputs["query_logits"] = query_logits
        outputs["soft_counts"] = soft_counts
        outputs["hard_counts"] = hard_counts

        # 4) meta text
        outputs["caption"] = captions
        outputs["text"] = [[p.strip() for p in cap.split(".") if p.strip()] for cap in captions]

        return outputs



def build_gdcount_model(
    config_path: str,
    checkpoint_path: str,
    device: str,
    gdcount_cfg: GDCountConfig,
) -> GroundingDINOCounter:
    """
    Load GroundingDINO + wrap thành GroundingDINOCounter.
    """
    base = load_model(
        model_config_path=config_path,
        model_checkpoint_path=checkpoint_path,
        device=device,
    )
    # load_model đặt model.eval(), ta chuyển lại sang train()
    base.train()

    model = GroundingDINOCounter(base_model=base, cfg=gdcount_cfg)
    model.to(device)
    # print("Has tokenizer:", hasattr(model.base_model, "tokenizer"))
    # print("Base model type:", type(model.base_model))
    # print("Attrs:", [k for k in dir(model.base_model) if "token" in k.lower()])

    return model
