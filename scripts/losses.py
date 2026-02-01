# scripts/losses.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from torchvision.ops import nms as tv_nms
import torch
import torch.nn as nn
import torch.nn.functional as F

def _cxcywh_to_xyxy_px(boxes_cxcywh: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # boxes_cxcywh: (...,4) in normalized cx,cy,w,h
    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
    cx = cx * W
    cy = cy * H
    bw = bw * W
    bh = bh * H
    x1 = (cx - bw / 2.0).clamp(0, W - 1)
    y1 = (cy - bh / 2.0).clamp(0, H - 1)
    x2 = (cx + bw / 2.0).clamp(0, W - 1)
    y2 = (cy + bh / 2.0).clamp(0, H - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        criterion: nn.Module,
        weight_dict: Dict[str, float],
        lambda_query: float = 0.1,
        use_query_loss: bool = True,
        log_count_mae: bool = True,   
        threshold: float = 0.23,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.weight_dict = weight_dict
        self.lambda_query = float(lambda_query)
        self.use_query_loss = bool(use_query_loss)
        self.log_count_mae = bool(log_count_mae)
        self.matcher = getattr(criterion, "matcher", None)
        self.threshold = float(threshold)

    def _get_pred_count(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
   
        self.threshold = 0.23

        if "pred_logits" in outputs:
            logits = outputs["pred_logits"].float()

            # (B,Q) binary logits per query
            if logits.dim() == 2:
                logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
                scores = torch.sigmoid(logits)
                keep = scores > self.threshold
                # No boxes -> can't NMS
                return keep.sum(dim=1).to(torch.float32).view(-1)

            if logits.dim() == 3:
                B, Q, T = logits.shape

                # token mask
                token_mask = outputs.get("text_mask", None)
                if token_mask is None:
                    token_mask = torch.ones((B, T), device=logits.device, dtype=torch.bool)
                else:
                    token_mask = token_mask.to(device=logits.device, dtype=torch.bool)
                    if token_mask.shape[-1] < T:
                        pad = torch.zeros((B, T - token_mask.shape[-1]),
                                        device=logits.device, dtype=torch.bool)
                        token_mask = torch.cat([token_mask, pad], dim=-1)
                    token_mask = token_mask[:, :T]

                # remove specials
                input_ids = outputs.get("input_ids", None)
                if input_ids is not None:
                    ids = input_ids.to(device=logits.device)
                    if ids.shape[-1] < T:
                        pad = torch.zeros((B, T - ids.shape[-1]),
                                        device=logits.device, dtype=ids.dtype)
                        ids = torch.cat([ids, pad], dim=-1)
                    ids = ids[:, :T]
                    specials = (ids == 0) | (ids == 101) | (ids == 102)  # PAD/CLS/SEP
                    token_mask = token_mask & (~specials)

                # finite + mask invalid tokens
                logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
                logits = logits.masked_fill(~token_mask[:, None, :], -1e4)

                per_query_logit = logits.max(dim=-1).values  # (B,Q)
                scores = torch.sigmoid(per_query_logit)      # (B,Q)
                keep = scores > self.threshold               # (B,Q)

                # NMS requires boxes + torchvision nms
                if ("pred_boxes" not in outputs) or (tv_nms is None):
                    return keep.sum(dim=1).to(torch.float32).view(-1)

                boxes = outputs["pred_boxes"].float()  # (B,Q,4) cxcywh norm (assumed)

                # get image size for px conversion
                # Prefer outputs-provided size if you have it; otherwise infer from pred_boxes normalization fallback.
                # Best: store in outputs["img_hw"] = (H,W) during forward.
                img_hw = outputs.get("img_hw", None)
                if img_hw is None:
                    # fallback: cannot know true H,W -> NMS still works in normalized coords
                    # but torchvision nms expects xyxy in same coord system; normalized is OK.
                    # We'll convert as if W=H=1 to keep consistency.
                    H = 1
                    W = 1
                else:
                    H, W = int(img_hw[0]), int(img_hw[1])

                pred_counts = []
                iou_thr = float(getattr(self, "nms_iou", 0.5))  # set self.nms_iou if you want

                for b in range(B):
                    idx_keep = keep[b].nonzero(as_tuple=False).flatten()
                    if idx_keep.numel() == 0:
                        pred_counts.append(torch.tensor(0.0, device=logits.device))
                        continue

                    b_scores = scores[b, idx_keep]
                    b_boxes = boxes[b, idx_keep]  # cxcywh norm

                    # convert to xyxy (same coord system); if H,W=1 => normalized xyxy
                    if img_hw is None:
                        # normalized xyxy
                        cx, cy, bw, bh = b_boxes.unbind(-1)
                        x1 = (cx - bw / 2.0).clamp(0, 1)
                        y1 = (cy - bh / 2.0).clamp(0, 1)
                        x2 = (cx + bw / 2.0).clamp(0, 1)
                        y2 = (cy + bh / 2.0).clamp(0, 1)
                        b_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                    else:
                        b_xyxy = _cxcywh_to_xyxy_px(b_boxes, H=H, W=W)

                    keep_nms = tv_nms(b_xyxy, b_scores, iou_thr)
                    pred_counts.append(keep_nms.numel() * torch.ones((), device=logits.device, dtype=torch.float32))

                return torch.stack(pred_counts, dim=0).view(-1)

            return None

        # fallback only if no pred_logits
        if "hard_counts" in outputs:
            return outputs["hard_counts"].to(torch.float32).view(-1)

        return None
    

    def _get_gt_count(self, targets: List[Dict[str, Any]], dtype, device) -> torch.Tensor:
        return torch.stack([t["count"].to(device=device, dtype=dtype).view(()) for t in targets]).view(-1)

    def forward(self, outputs, targets, caption, cat_list) -> Dict[str, torch.Tensor]:
        device = next(iter(outputs.values())).device

        # normalize caption / cat_list như code của bạn ...
        caption_list = [caption] if isinstance(caption, str) else caption
        if len(caption_list) == 1 and isinstance(cat_list, list) and len(cat_list) > 0 and isinstance(cat_list[0], str):
            cat_list_batch = [cat_list]
        else:
            cat_list_batch = cat_list

        # det_targets
        det_targets: List[Dict[str, Any]] = []
        for t in targets:
            dt = {}
            if "boxes" in t: dt["boxes"] = t["boxes"]
            if "labels" in t: dt["labels"] = t["labels"]
            det_targets.append(dt)

        out: Dict[str, torch.Tensor] = {}

        # 1) Detect losses (có backprop)
        det_loss_dict = self.criterion(outputs, det_targets, cat_list_batch, caption_list)

        loss_det_total = torch.zeros((), device=device)
        for k, v in det_loss_dict.items():
            out[k] = v  # log term gốc
            if k in self.weight_dict:
                loss_det_total = loss_det_total + v * float(self.weight_dict[k])

        # 2) Query loss (tùy chọn, có backprop nếu bật)
        loss_q = torch.zeros((), device=device)
        if self.use_query_loss and ("query_logits" in outputs) and (self.matcher is not None):
            qlogits = outputs["query_logits"]
            if qlogits.dim() == 3 and qlogits.size(-1) == 1:
                qlogits = qlogits.squeeze(-1)

            indices = self.matcher(outputs, det_targets)
            B, Q = qlogits.shape[:2]
            tgt = torch.zeros((B, Q), device=qlogits.device, dtype=qlogits.dtype)
            for b, (src_idx, _) in enumerate(indices):
                if src_idx.numel() > 0:
                    tgt[b, src_idx] = 1.0

            loss_q = F.binary_cross_entropy_with_logits(qlogits.float(), tgt.float(), reduction="mean")
            loss_q = loss_q * self.lambda_query

        # 3) Count MAE chỉ để log/eval (KHÔNG backprop)
        if self.log_count_mae:
            with torch.no_grad():
                pred = self._get_pred_count(outputs)
                if pred is None:
                    out["count_mae"] = torch.zeros((), device=device)
                else:
                    gt = self._get_gt_count(targets, dtype=pred.dtype, device=pred.device)
                    out["count_mae"] = (pred - gt).abs().mean()

        out["loss_det"] = loss_det_total
        out["loss_query"] = loss_q
        out["loss_total"] = loss_det_total + loss_q  
        return out
