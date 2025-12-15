# gdcount_model.py
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from groundingdino.util.misc import NestedTensor
from groundingdino.util.inference import load_model  # dùng để load từ config + ckpt


@dataclass
class GDCountConfig:
    """
    Cấu hình cơ bản cho CountGD-style:
    - threshold: ngưỡng hard count trên logit từng query (CountHead)
    - soa_level: index feature level dùng SOA (0 là coarse nhất, -1 là level cuối)
    - feature_dim: hidden dim của GroundingDINO (thường 256)
    - freeze_keywords: các từ khoá trong tên param để freeze (backbone, BERT)
    - unfreeze_bert_last_n: mở lại n layer cuối của BERT (0 = giữ freeze)
    - use_exemplar_mod: bật exemplar-conditioned modulation cho calibrated score
    """
    threshold: float = 0.0
    soa_level: int = -1
    feature_dim: int = 256
    freeze_keywords: Sequence[str] = ("backbone.0", "bert")

    unfreeze_bert_last_n: int = 0       # 0/1/2...
    use_exemplar_mod: bool = True       # modulation chỉ ảnh hưởng calibrated score


class SmallObjectAdapter(nn.Module):
    """SOA: tăng nhạy với object nhỏ trên feature map phân giải cao."""
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

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
    """Bọc backbone Joiner của GroundingDINO để chèn SOA vào một feature level."""
    def __init__(self, backbone: nn.Module, level_index: int, in_channels: int) -> None:
        super().__init__()
        self.backbone = backbone  # Joiner
        self.level_index = level_index
        self.soa = SmallObjectAdapter(in_channels)

    def __getitem__(self, idx):
        return self.backbone[idx]

    def __len__(self):
        try:
            return len(self.backbone)
        except Exception:
            return 0

    def forward(self, samples: NestedTensor) -> Tuple[List[NestedTensor], List[torch.Tensor]]:
        features, pos = self.backbone(samples)
        if not isinstance(features, list) or len(features) == 0:
            return features, pos

        idx = self.level_index
        if idx < 0:
            idx = len(features) + idx
        if idx < 0 or idx >= len(features):
            return features, pos

        feat_nt: NestedTensor = features[idx]
        src, mask = feat_nt.decompose()  # src: (B,C,H,W)

        C = src.shape[1]
        if self.soa.conv1.in_channels != C:
            self.soa = SmallObjectAdapter(C).to(src.device)

        src = self.soa(src)
        features[idx] = NestedTensor(src, mask)
        return features, pos


class TransformerWrapper(nn.Module):
    """Bọc transformer của GroundingDINO để lấy hs (decoder queries) mỗi lần forward."""
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.last_hs = None

    def forward(self, *args, **kwargs):
        outputs = self.transformer(*args, **kwargs)
        if isinstance(outputs, tuple) and len(outputs) > 0:
            self.last_hs = outputs[0]  # GroundingDINO: hs = outputs[0]
        else:
            self.last_hs = None
        return outputs


class CountHead(nn.Module):
    """Count head: đọc hs (decoder queries) -> logit per query -> soft/hard count."""
    def __init__(self, d_model: int, threshold: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.threshold = threshold

    def forward(self, hs):
        # hs:
        # - Tensor [num_layers, B, Q, D]
        # - Tensor [B, Q, D]
        # - List[Tensor[B, Q, D]]
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

        hs_last = self.norm(hs_last)
        logits = self.fc(hs_last).squeeze(-1)          # (B,Q)
        soft = torch.relu(logits).sum(dim=1)           # (B,)
        hard = (logits > self.threshold).sum(dim=1).to(torch.int64)  # (B,)
        return logits, soft, hard


class ScoreCalibrator(nn.Module):
    """Calibrate per-query score từ decoder hs (không đụng backbone)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, hs):
        if isinstance(hs, (list, tuple)):
            hs = torch.stack(hs, dim=0)
        if hs.dim() == 4:
            hs = hs[-1]  # (B,Q,D)
        calib_logits = self.mlp(hs).squeeze(-1)     # (B,Q)
        calib_scores = torch.sigmoid(calib_logits)  # (B,Q)
        return calib_logits, calib_scores


class ExemplarModulator(nn.Module):
    """
    Gating nhẹ theo exemplars: tạo 1 bias/scale theo thống kê box exemplar.
    Input: e = [mean_w, mean_h, mean_area, std_area, mean_ar, std_ar] (6,)
    Output: gate scalar per image.
    """
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        # e: (B,6)
        return self.net(e)  # (B,1)


def _hs_last(hs: Any) -> torch.Tensor:
    """Chuẩn hóa hs về (B,Q,D)."""
    if isinstance(hs, (list, tuple)):
        hs = torch.stack(hs, dim=0)
    if hs.dim() == 4:
        return hs[-1]
    if hs.dim() == 3:
        return hs
    raise ValueError(f"Unexpected hs shape: {getattr(hs, 'shape', None)}")


def _exemplar_stats(ex: torch.Tensor) -> torch.Tensor:
    """
    ex: (K,4) xyxy pixel
    return: (6,) stats; nếu K==0 -> zeros
    """
    if ex is None or ex.numel() == 0:
        return torch.zeros((6,), device=ex.device if ex is not None else "cpu", dtype=torch.float32)

    x1, y1, x2, y2 = ex.unbind(-1)
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    area = w * h
    ar = (w / h).clamp(min=1e-3, max=1e3)

    # std dùng unbiased=False để ổn định khi K nhỏ
    feat = torch.stack([
        w.mean(),
        h.mean(),
        area.mean(),
        area.std(unbiased=False),
        ar.mean(),
        ar.std(unbiased=False),
    ], dim=0).to(torch.float32)
    return feat


class GroundingDINOCounter(nn.Module):
    """
    Wrapper chính: GroundingDINO + SOA + CountHead + (ScoreCalibrator + ExemplarModulator).
    """
    def __init__(self, base_model: nn.Module, cfg: GDCountConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg

        # 1) Lấy đúng số kênh feature level dùng SOA
        try:
            num_channels_list = self.base_model.backbone.num_channels
            lvl = cfg.soa_level
            if lvl < 0:
                lvl = len(num_channels_list) + lvl
            lvl = max(0, min(lvl, len(num_channels_list) - 1))
            in_channels = num_channels_list[lvl]
        except Exception:
            in_channels = cfg.feature_dim

        # 2) Bọc backbone bằng SOA
        self.base_model.backbone = SOABackboneWrapper(
            backbone=self.base_model.backbone,
            level_index=cfg.soa_level,
            in_channels=in_channels,
        )

        # 3) Bọc transformer để lấy hs
        self.base_model.transformer = TransformerWrapper(self.base_model.transformer)

        # 4) Heads
        hidden_dim = getattr(self.base_model, "hidden_dim", cfg.feature_dim)
        self.count_head = CountHead(hidden_dim, threshold=cfg.threshold)

        # (1) Score calibrator
        self.calibrator = ScoreCalibrator(hidden_dim)

        # (3) Exemplar modulation (gating theo e)
        self.ex_mod = ExemplarModulator(in_dim=6) if cfg.use_exemplar_mod else None

        # 5) Freeze theo keyword + optional unfreeze BERT last layers
        self._apply_freeze(cfg)

    def _apply_freeze(self, cfg: GDCountConfig) -> None:
        # allow grad by default
        for _, p in self.base_model.named_parameters():
            p.requires_grad = True

        # freeze by keyword
        for name, p in self.base_model.named_parameters():
            if any(kw in name for kw in cfg.freeze_keywords):
                p.requires_grad = False

        # optional: unfreeze last N layers of BERT encoder (bert-base: 0..11)
        n = int(getattr(cfg, "unfreeze_bert_last_n", 0) or 0)
        if n > 0:
            last_ids = list(range(12 - n, 12))
            for name, p in self.base_model.named_parameters():
                for lid in last_ids:
                    if f"bert.encoder.layer.{lid}." in name:
                        p.requires_grad = True

        # ensure SOA + heads trainable
        for p in self.base_model.backbone.soa.parameters():
            p.requires_grad = True
        for p in self.count_head.parameters():
            p.requires_grad = True
        for p in self.calibrator.parameters():
            p.requires_grad = True
        if self.ex_mod is not None:
            for p in self.ex_mod.parameters():
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
        exemplars: list[Tensor(K,4)] xyxy pixel (tối đa 3)
        labels: list/tensor label phrase index per image (vd [tensor([0,0,0]), ...])
        """

        # 1) gọi base_model (có/không exemplar)
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

        # 3) Count head
        query_logits, soft_counts, hard_counts = self.count_head(hs)
        outputs["query_logits"] = query_logits
        outputs["soft_counts"] = soft_counts
        outputs["hard_counts"] = hard_counts

        # 4) Calibrator score (per-query)
        calib_logits, calib_scores = self.calibrator(hs)

        # 5) Exemplar-conditioned modulation (tác động lên calibrated score)
        if self.ex_mod is not None:
            # exemplars: list per-image; lấy stats per-image -> (B,6)
            if exemplars is None:
                # zero-shot: gate = 1 (không đổi)
                gate = None
            else:
                e_list = []
                for ex in exemplars:
                    e_list.append(_exemplar_stats(ex.to(calib_logits.device)))
                e = torch.stack(e_list, dim=0)  # (B,6)
                gate = torch.sigmoid(self.ex_mod(e)).clamp(0.05, 1.0)  # (B,1) ổn định

            if gate is not None:
                # scale scores (không phá logits quá nhiều)
                calib_scores = calib_scores * gate
                # (tuỳ bạn) cũng có thể shift logits:
                # calib_logits = calib_logits + torch.log(gate / (1 - gate + 1e-6) + 1e-6)

                outputs["ex_gate"] = gate.squeeze(-1)

        outputs["calib_logits"] = calib_logits
        outputs["calib_scores"] = calib_scores

        # 6) meta text
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
    return model
