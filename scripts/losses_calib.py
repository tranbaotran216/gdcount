# scripts/losses.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms as tv_nms


def _cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)

def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (N,4), b: (M,4) -> iou: (N,M)
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)

    union = area_a + area_b.unsqueeze(0) - inter
    return inter / (union + 1e-6)

@torch.no_grad()
def _simple_iou_match(outputs: Dict[str, Any], det_targets: List[Dict[str, Any]], iou_thr: float = 0.05):
    """
    Trả về indices giống HungarianMatcher: list[(src_idx, tgt_idx)] per batch.
    Dùng greedy: mỗi GT chọn pred có IoU cao nhất, lấy unique preds.
    """
    B = outputs["pred_boxes"].shape[0]
    Q = outputs["pred_boxes"].shape[1]
    pred_xyxy = _cxcywh_to_xyxy_norm(outputs["pred_boxes"].float())  # (B,Q,4)

    all_indices = []
    for b in range(B):
        gt = det_targets[b].get("boxes", None)
        if gt is None or gt.numel() == 0:
            all_indices.append((torch.empty((0,), dtype=torch.long),
                                torch.empty((0,), dtype=torch.long)))
            continue

        gt_xyxy = _cxcywh_to_xyxy_norm(gt.float())  # (M,4)
        iou = _box_iou_xyxy(pred_xyxy[b], gt_xyxy)  # (Q,M)

        # mỗi GT chọn pred tốt nhất
        best_pred = iou.argmax(dim=0)              # (M,)
        best_iou  = iou[best_pred, torch.arange(gt_xyxy.shape[0], device=iou.device)]

        # lọc theo ngưỡng
        keep = best_iou > float(iou_thr)
        tgt_idx = torch.arange(gt_xyxy.shape[0], device=iou.device)[keep]
        src_idx = best_pred[keep]

        # unique pred (tránh 2 GT cùng 1 pred)
        if src_idx.numel() > 0:
            src_unique, inv = torch.unique(src_idx, return_inverse=True)
            # chọn tgt đầu tiên cho mỗi pred
            chosen_tgt = []
            for k in range(src_unique.numel()):
                chosen_tgt.append(tgt_idx[(inv == k).nonzero(as_tuple=False)[0,0]])
            src_idx = src_unique
            tgt_idx = torch.stack(chosen_tgt, dim=0)

        all_indices.append((src_idx.to(torch.long), tgt_idx.to(torch.long)))

    return all_indices


class MultiTaskLoss(nn.Module):
    """
    Tổng loss:
      - det losses (criterion): backprop
      - query loss (optional): backprop
      - calib loss (NEW): backprop (dựa trên Hungarian matcher)
      - count_mae: chỉ log (no_grad)

    NOTE:
      - pred_count dùng outputs["calib_scores"] nếu có (NEW)
      - giữ nguyên logic NMS/threshold cho counting
    """

    def __init__(
        self,
        criterion: nn.Module,
        weight_dict: Dict[str, float],
        lambda_query: float = 0.1,
        use_query_loss: bool = True,
        log_count_mae: bool = True,
        threshold: float = 0.23,
        nms_iou: float = 0.5,
        lambda_calib: float = 1.0,
        use_calib_loss: bool = True,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.weight_dict = weight_dict
        self.lambda_query = float(lambda_query)
        self.use_query_loss = bool(use_query_loss)
        self.log_count_mae = bool(log_count_mae)
        self.matcher = getattr(criterion, "matcher", None)

        self.threshold = float(threshold)
        self.nms_iou = float(nms_iou)

        self.lambda_calib = float(lambda_calib)
        self.use_calib_loss = bool(use_calib_loss)

    # -------------------------
    # Counting (for logging/eval)
    # -------------------------
    def _get_pred_count(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Return predicted count (B,) float32.
        Ưu tiên dùng calibrated score (outputs["calib_scores"]) nếu có.
        Fallback: pred_logits → sigmoid(max_token_logit) như cũ.
        """
        # 0) ưu tiên calibrated score
        scores: Optional[torch.Tensor] = None
        if "calib_scores" in outputs and isinstance(outputs["calib_scores"], torch.Tensor):
            scores = outputs["calib_scores"].float()  # (B,Q)

        # 1) fallback pred_logits -> scores (B,Q)
        if scores is None:
            if "pred_logits" not in outputs:
                # fallback cuối
                if "hard_counts" in outputs:
                    return outputs["hard_counts"].to(torch.float32).view(-1)
                return None

            logits = outputs["pred_logits"].float()

            # (B,Q)
            if logits.dim() == 2:
                logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
                scores = torch.sigmoid(logits)

            # (B,Q,T)
            elif logits.dim() == 3:
                B, Q, T = logits.shape

                token_mask = outputs.get("text_mask", None)
                if token_mask is None:
                    token_mask = torch.ones((B, T), device=logits.device, dtype=torch.bool)
                else:
                    token_mask = token_mask.to(device=logits.device, dtype=torch.bool)
                    if token_mask.shape[-1] < T:
                        pad = torch.zeros((B, T - token_mask.shape[-1]), device=logits.device, dtype=torch.bool)
                        token_mask = torch.cat([token_mask, pad], dim=-1)
                    token_mask = token_mask[:, :T]

                input_ids = outputs.get("input_ids", None)
                if input_ids is not None:
                    ids = input_ids.to(device=logits.device)
                    if ids.shape[-1] < T:
                        pad = torch.zeros((B, T - ids.shape[-1]), device=logits.device, dtype=ids.dtype)
                        ids = torch.cat([ids, pad], dim=-1)
                    ids = ids[:, :T]
                    specials = (ids == 0) | (ids == 101) | (ids == 102)  # PAD/CLS/SEP
                    token_mask = token_mask & (~specials)

                logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
                logits = logits.masked_fill(~token_mask[:, None, :], -1e4)

                per_query_logit = logits.max(dim=-1).values  # (B,Q)
                scores = torch.sigmoid(per_query_logit)      # (B,Q)
            else:
                return None

        # 2) threshold
        keep = scores > self.threshold  # (B,Q)

        # 3) nếu không có boxes hoặc không có nms -> chỉ đếm threshold
        if ("pred_boxes" not in outputs) or (tv_nms is None):
            return keep.sum(dim=1).to(torch.float32).view(-1)

        boxes = outputs["pred_boxes"].float()  # (B,Q,4) cxcywh norm

        # Option: outputs["img_hw"] = (H,W) để NMS theo pixel; nếu không có thì NMS theo norm

        pred_counts = []
        iou_thr = self.nms_iou

        B = scores.shape[0]
        for b in range(B):
            idx_keep = keep[b].nonzero(as_tuple=False).flatten()
            if idx_keep.numel() == 0:
                pred_counts.append(torch.tensor(0.0, device=scores.device))
                continue

            b_scores = scores[b, idx_keep]
            b_boxes = boxes[b, idx_keep]  # cxcywh norm
            b_xyxy = _cxcywh_to_xyxy_norm(b_boxes)

            kept = tv_nms(b_xyxy, b_scores, iou_thr)
            pred_counts.append(torch.tensor(float(kept.numel()), device=scores.device))

        return torch.stack(pred_counts, dim=0).view(-1)

    def _get_gt_count(self, targets: List[Dict[str, Any]], dtype, device) -> torch.Tensor:
        return torch.stack([t["count"].to(device=device, dtype=dtype).view(()) for t in targets]).view(-1)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, outputs, targets, caption, cat_list) -> Dict[str, torch.Tensor]:
        device = outputs["pred_boxes"].device if "pred_boxes" in outputs else outputs["pred_logits"].device

        # normalize caption/cat_list (giữ y như code của bạn)
        caption_list = [caption] if isinstance(caption, str) else caption
        if len(caption_list) == 1 and isinstance(cat_list, list) and len(cat_list) > 0 and isinstance(cat_list[0], str):
            cat_list_batch = [cat_list]
        else:
            cat_list_batch = cat_list

        # det_targets
        det_targets: List[Dict[str, Any]] = []
        for t in targets:
            dt: Dict[str, Any] = {}
            if "boxes" in t:
                dt["boxes"] = t["boxes"]
            if "labels" in t:
                dt["labels"] = t["labels"]
            det_targets.append(dt)

        out: Dict[str, torch.Tensor] = {}

        # 1) Detect losses (backprop)
        det_loss_dict = self.criterion(outputs, det_targets, cat_list_batch, caption_list)

        loss_det_total = torch.zeros((), device=device)
        for k, v in det_loss_dict.items():
            out[k] = v
            if k in self.weight_dict:
                loss_det_total = loss_det_total + v * float(self.weight_dict[k])

        # 2) Query loss (optional, backprop)
        loss_q = torch.zeros((), device=device)
        indices = None
        if ("pred_boxes" in outputs):
            # luôn dùng IoU-match để tránh phụ thuộc matcher nội bộ (đang lỗi shape idx_map)
            indices = _simple_iou_match(outputs, det_targets, iou_thr=0.05)


        if self.use_query_loss and ("query_logits" in outputs) and (indices is not None):
            qlogits = outputs["query_logits"]
            if qlogits.dim() == 3 and qlogits.size(-1) == 1:
                qlogits = qlogits.squeeze(-1)

            B, Q = qlogits.shape[:2]
            tgt = torch.zeros((B, Q), device=qlogits.device, dtype=torch.float32)
            for b, (src_idx, _) in enumerate(indices):
                if src_idx.numel() > 0:
                    tgt[b, src_idx] = 1.0

            loss_q = F.binary_cross_entropy_with_logits(qlogits.float(), tgt, reduction="mean")
            loss_q = loss_q * self.lambda_query

        # 3) Calibrator loss (NEW, backprop)
        loss_calib = torch.zeros((), device=device)
        if self.use_calib_loss and ("calib_logits" in outputs) and (indices is not None):
            calib_logits = outputs["calib_logits"]  # (B,Q)
            if calib_logits.dim() != 2:
                # an toàn: ép về (B,Q)
                calib_logits = calib_logits.view(calib_logits.shape[0], -1)

            B, Q = calib_logits.shape
            tgt = torch.zeros((B, Q), device=calib_logits.device, dtype=torch.float32)
            for b, (src_idx, _) in enumerate(indices):
                if src_idx.numel() > 0:
                    tgt[b, src_idx] = 1.0

            loss_calib = F.binary_cross_entropy_with_logits(calib_logits.float(), tgt, reduction="mean")
            loss_calib = loss_calib * self.lambda_calib

        # 4) Count MAE chỉ để log/eval (NO backprop)
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
        out["loss_calib"] = loss_calib

        # only backprop through det + query + calib
        out["loss_total"] = loss_det_total + loss_q + loss_calib
        return out
