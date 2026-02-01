import argparse
import os
import csv
from typing import Any, Dict, List, Tuple
import re
import math
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torchvision.ops import nms as tv_nms

def _get_query_scores(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "pred_logits" not in outputs:
        return None
    logits = outputs["pred_logits"]  # (B,Q) hoặc (B,Q,D)
    if logits.dim() == 3:
        logits = logits.max(dim=-1).values
    return logits.sigmoid()  # (B,Q)

def _match_points_to_queries(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # cost: (N, K)  (N=gt points, K=selected queries)
    try:
        ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.as_tensor(ri, device=cost.device), torch.as_tensor(ci, device=cost.device)
    except Exception:
        # greedy fallback
        N, K = cost.shape
        used = torch.zeros(K, dtype=torch.bool, device=cost.device)
        ri_list, ci_list = [], []
        for r in range(N):
            c = cost[r].clone()
            c[used] = 1e9
            j = torch.argmin(c).item()
            if c[j].item() >= 1e8:
                break
            used[j] = True
            ri_list.append(r)
            ci_list.append(j)
        if len(ri_list) == 0:
            return torch.empty(0, dtype=torch.long, device=cost.device), torch.empty(0, dtype=torch.long, device=cost.device)
        return torch.tensor(ri_list, device=cost.device), torch.tensor(ci_list, device=cost.device)

def center_supervision_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    device: str,
    lambda_center: float = 5.0,
    topk_min: int = 300,
) -> Dict[str, torch.Tensor]:
    pred_boxes = outputs["pred_boxes"]  # (B,Q,4) in cxcywh (normalized)
    B, Q, _ = pred_boxes.shape
    centers = pred_boxes[..., :2]  # (B,Q,2)

    scores = _get_query_scores(outputs)  # (B,Q) or None

    loss_center_sum = centers.new_tensor(0.0)
    matched_points = 0

    for b in range(B):
        H = int(batch["images"][b].shape[-2])
        W = int(batch["images"][b].shape[-1])

        if "points" in batch:
            pts_pad = batch["points"].to(device)          # (B,maxN,2)
            mask = batch.get("points_mask", None)         # (B,maxN)
            if mask is None:
                pts = pts_pad[b]
            else:
                pts = pts_pad[b][mask[b]]
        else:
            pts = batch["meta"]["points"][b].to(device)   # fallback (List)
        N = int(pts.shape[0])
        if N == 0:
            continue

        gt = pts.clone()
        gt[:, 0] = gt[:, 0] / float(W)
        gt[:, 1] = gt[:, 1] / float(H)
        gt = gt.clamp(0, 1)  # (N,2)

        if scores is None:
            K = min(Q, max(topk_min, 3 * N))
            idx = torch.arange(K, device=device)
        else:
            K = min(Q, max(topk_min, 3 * N))
            idx = torch.topk(scores[b], k=K, largest=True).indices  # (K,)

        pred_c = centers[b, idx]  # (K,2)

        cost = torch.cdist(gt.float(), pred_c.float(), p=2)  # (N,K)
        ri, ci = _match_points_to_queries(cost)
        if ri.numel() == 0:
            continue

        loss_center_sum = loss_center_sum + F.smooth_l1_loss(pred_c[ci], gt[ri], reduction="sum")
        matched_points += int(ri.numel())

    if matched_points > 0:
        loss_center = (loss_center_sum / matched_points) * float(lambda_center)
    else:
        loss_center = loss_center_sum * 0.0

    gt_counts = batch["gt_counts"].to(device)
    with torch.no_grad():
        pred_counts = count_by_det_nms(outputs, threshold=0.23, nms_iou=getattr(batch, "nms_iou", 0.5)).to(gt_counts.dtype)
        count_mae = (pred_counts - gt_counts).abs().mean()

    return {
        "loss_total": loss_center,
        "loss_center": loss_center.detach(),
        "count_mae": count_mae.detach(),
        "loss_ce": loss_center.new_tensor(0.0),
        "loss_bbox": loss_center.new_tensor(0.0),
        "loss_giou": loss_center.new_tensor(0.0),
        "loss_query": loss_center.new_tensor(0.0),
    }

def build_exemplar_inputs(batch: Dict[str, Any], device: str):
    ex_list = [t.to(device) for t in batch["meta"]["exemplar_xyxy"]]
    labels_list = [torch.zeros((ex.shape[0],), dtype=torch.long, device=device) for ex in ex_list]
    return ex_list, labels_list


def sanitize_caption(p: str) -> str:
    p = "" if p is None else str(p)
    p = p.strip()
    p = re.sub(r"\s+", " ", p)
    p = re.sub(r"\s+\.", ".", p)
    if p == "" or p == ".":
        p = "object."
    if not p.endswith("."):
        p = p + "."
    return p


def _scores_from_pred_logits(outputs: Dict[str, Any]) -> torch.Tensor:
    """scores (B,Q) = sigmoid(max_token_logit over valid tokens)"""
    logits = outputs["pred_logits"].float()  # (B,Q,T) or (B,Q)

    if logits.dim() == 2:
        logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
        return torch.sigmoid(logits)

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

    per_q = logits.max(dim=-1).values  # (B,Q)
    return torch.sigmoid(per_q)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def count_by_det_nms(outputs: Dict[str, Any], threshold: float, nms_iou: float) -> torch.Tensor:
    scores = _scores_from_pred_logits(outputs)  # (B,Q)
    keep = scores > threshold

    if ("pred_boxes" not in outputs) or (tv_nms is None):
        return keep.sum(dim=1).to(torch.float32)

    boxes = outputs["pred_boxes"].float()  # (B,Q,4) cxcywh norm
    boxes_xyxy = _cxcywh_to_xyxy(boxes).clamp(0, 1)

    B, Q = scores.shape
    out_counts = []
    for b in range(B):
        idx = keep[b].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            out_counts.append(torch.tensor(0.0, device=scores.device))
            continue
        b_boxes = boxes_xyxy[b, idx]
        b_scores = scores[b, idx]
        kept = tv_nms(b_boxes, b_scores, nms_iou)
        out_counts.append(torch.tensor(float(kept.numel()), device=scores.device))
    return torch.stack(out_counts, dim=0)

@torch.no_grad()
def estimate_side0_from_points(
    pts_xy: torch.Tensor,
    H: int,
    W: int,
    device: torch.device,
    k: float = 0.8,
    max_samples: int = 256,
) -> float:
    """
    Ước lượng kích thước object theo từng ảnh từ phân bố points:
      side0 = k * median(nearest_neighbor_distance(points))
    """
    if pts_xy is None or pts_xy.numel() == 0:
        return float(0.6 * math.sqrt(H * W))

    N = int(pts_xy.shape[0])
    if N <= 1:
        return float(0.6 * math.sqrt(H * W))

    pts = pts_xy.to(device=device, dtype=torch.float32)

    # subsample để giảm O(N^2)
    if N > int(max_samples):
        idx = torch.randperm(N, device=device)[: int(max_samples)]
        pts = pts[idx]

    # cdist trên subset
    D = torch.cdist(pts, pts)  # (M,M)
    D.fill_diagonal_(1e9)
    nn = D.min(dim=1).values
    d = float(nn.median().item())

    if not math.isfinite(d) or d <= 0:
        return float(0.6 * math.sqrt(H * W))

    side0 = float(k) * d
    return float(side0)


def sample_log_uniform(low: float, high: float, device: torch.device) -> float:
    """
    Sample u ~ LogUniform(low, high)
    """
    low = float(low)
    high = float(high)
    assert low > 0 and high > 0 and high >= low
    u = torch.empty(1, device=device).uniform_(math.log(low), math.log(high)).exp().item()
    return float(u)

# lấy raw logit per query chưa sigmoid
def _logits_per_query(outputs: Dict[str, Any]) -> torch.Tensor:
    """
    return raw logits per query (B,Q) = max_token_logit over valid tokens (NO sigmoid)
    """
    logits = outputs["pred_logits"].float()  # (B,Q,T) or (B,Q)

    if logits.dim() == 2:
        logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
        return logits

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
    return logits.max(dim=-1).values  # (B,Q)