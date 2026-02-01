from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn

from groundingdino.util.misc import NestedTensor
from groundingdino.util.inference import load_model  # load từ config + ckpt


@dataclass
class GDCountConfig:
    """
    CountGD-style config
    - threshold: ngưỡng hard count trên logit từng query
    - soa_level: index feature level dùng SOA (0..n-1 hoặc -1..), None = tắt SOA
    - feature_dim: hidden dim của GroundingDINO (thường 256)
    - freeze_keywords: các từ khoá trong tên param để freeze (backbone, BERT)
    """
    threshold: float = 0.0
    soa_level: Optional[int] = -1  # None => disable SOA
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
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        b, c, _, _ = out.shape
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
        self.level_index = int(level_index)
        self.soa = SmallObjectAdapter(in_channels)

    def __getitem__(self, idx):
        return self.backbone[idx]

    def __len__(self):
        try:
            return len(self.backbone)
        except Exception:
            return 0

    def __getattr__(self, name: str):
        # proxy mọi attribute không có ở wrapper sang backbone gốc
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)

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
            return features, pos

        feat_nt: NestedTensor = features[idx]
        src, mask = feat_nt.decompose()  # src: (B,C,H,W)

        # nếu số kênh không khớp, khởi tạo lại SOA đúng kênh
        C = src.shape[1]
        if getattr(self.soa.conv1, "in_channels", None) != C:
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

    def __getattr__(self, name: str):
        # proxy attribute sang transformer gốc (tránh thiếu field)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(self, *args, **kwargs):
        outputs = self.transformer(*args, **kwargs)
        # GroundingDINO thường trả tuple, hs ở outputs[0]
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
        self.threshold = float(threshold)

    def forward(self, hs):
        """
        hs :
          - Tensor [num_layers, B, Q, D]
          - Tensor [B, Q, D]
          - List[Tensor[B, Q, D]]
        """
        if isinstance(hs, (list, tuple)):
            if len(hs) == 0 or not isinstance(hs[0], torch.Tensor):
                raise ValueError(f"Unexpected hs list: {type(hs)}, first={type(hs[0]) if hs else None}")
            hs = torch.stack(hs, dim=0)  # [L,B,Q,D]

        if hs.dim() == 4:
            hs_last = hs[-1]  # (B,Q,D)
        elif hs.dim() == 3:
            hs_last = hs
        else:
            raise ValueError(f"Unexpected hs shape: {tuple(hs.shape)}")

        hs_last = self.norm(hs_last)
        logits = self.fc(hs_last).squeeze(-1)     # (B,Q)

        soft = torch.relu(logits).sum(dim=1)      # (B,)
        hard = (logits > self.threshold).sum(dim=1).to(torch.int64)
        return logits, soft, hard


class GroundingDINOCounter(nn.Module):
    """
    Wrapper chính: GroundingDINO (+ optional SOA) + CountHead.
    """

    def __init__(self, base_model: nn.Module, cfg: GDCountConfig) -> None:
        super().__init__()
        self.base_model = base_model

        # 1) Optional SOA: chỉ wrap backbone nếu cfg.soa_level != None
        if cfg.soa_level is not None:
            # lấy số kênh đúng level nếu có
            in_channels = cfg.feature_dim
            try:
                num_channels_list = self.base_model.backbone.num_channels
                lvl = int(cfg.soa_level)
                if lvl < 0:
                    lvl = len(num_channels_list) + lvl
                lvl = max(0, min(lvl, len(num_channels_list) - 1))
                in_channels = int(num_channels_list[lvl])
            except Exception:
                in_channels = cfg.feature_dim

            self.base_model.backbone = SOABackboneWrapper(
                backbone=self.base_model.backbone,
                level_index=int(cfg.soa_level),
                in_channels=in_channels,
            )

        # 2) Bọc transformer để lấy hs
        self.base_model.transformer = TransformerWrapper(self.base_model.transformer)

        # 3) Count head
        hidden_dim = getattr(self.base_model, "hidden_dim", cfg.feature_dim)
        self.count_head = CountHead(int(hidden_dim), threshold=cfg.threshold)

        # 4) Freeze theo keyword
        self._apply_freeze(cfg)

    def _apply_freeze(self, cfg: GDCountConfig) -> None:
        # reset: cho phép gradient
        for _, p in self.base_model.named_parameters():
            p.requires_grad = True

        # freeze theo keyword
        for name, p in self.base_model.named_parameters():
            if any(kw in name for kw in cfg.freeze_keywords):
                p.requires_grad = False

        # đảm bảo CountHead luôn trainable
        for p in self.count_head.parameters():
            p.requires_grad = True

        # nếu có SOA thì luôn trainable
        bb = getattr(self.base_model, "backbone", None)
        if bb is not None and hasattr(bb, "soa") and (bb.soa is not None):
            for p in bb.soa.parameters():
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
        # 1) forward base_model (có/không exemplar)
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

        # 3) CountHead
        query_logits, soft_counts, hard_counts = self.count_head(hs)
        outputs["query_logits"] = query_logits
        outputs["soft_counts"] = soft_counts
        outputs["hard_counts"] = hard_counts

        # 4) meta text (giữ tương thích code cũ)
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
    base.train()

    model = GroundingDINOCounter(base_model=base, cfg=gdcount_cfg)
    model.to(device)
    return model
