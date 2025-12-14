# debugs/debug_gdcount.py
import os
import argparse
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import random
from PIL import Image, ImageDraw
import numpy as np
from datasets.fsc147_dataset import FSC147Dataset, fsc147_collate
from models.gdcount_model import GDCountConfig, build_gdcount_model
from scripts.criterion_detect import build_criterion_detect
from scripts.losses import MultiTaskLoss
from torchvision.ops import nms

def parse_args():
    p = argparse.ArgumentParser("Debug GDCount (CountGD-style criterion)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ann", type=str, required=True)
    p.add_argument("--img-root", type=str, required=True)
    p.add_argument("--split-file", type=str, required=True)
    p.add_argument("--class-map", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--soa-level", type=int, default=-1)
    p.add_argument("--outdir", type=str, default="debug_out")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--img-id", type=str, default=None,
                   help="Debug đúng 1 ảnh theo tên file, ví dụ: 7.jpg")
    p.add_argument("--index", type=int, default=None,
                   help="Debug theo index trong dataset (ưu tiên thấp hơn img-id)")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed cho random chọn ảnh (khi không truyền img-id/index)")
    p.add_argument("--max-iter", type=int, default=100000,
                   help="Giới hạn số batch scan khi tìm img-id")

    return p.parse_args()

def query_scores_from_pred_logits(outputs: Dict[str, Any]) -> torch.Tensor:
    """
    Return scores (B,Q) from pred_logits using CountGD rule:
    score = sigmoid(max_token_logit over valid tokens)
    """
    logits = outputs["pred_logits"].float()  # (B,Q,T)
    B, Q, T = logits.shape

    tm = outputs.get("text_mask", None)
    if tm is None:
        tm = torch.ones((B, T), device=logits.device, dtype=torch.bool)
    else:
        tm = tm.to(device=logits.device, dtype=torch.bool)
        if tm.shape[-1] < T:
            pad = torch.zeros((B, T - tm.shape[-1]), device=logits.device, dtype=torch.bool)
            tm = torch.cat([tm, pad], dim=-1)
        tm = tm[:, :T]

    ids = outputs.get("input_ids", None)
    if ids is not None:
        ids = ids.to(device=logits.device)
        if ids.shape[-1] < T:
            pad = torch.zeros((B, T - ids.shape[-1]), device=logits.device, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=-1)
        ids = ids[:, :T]
        specials = (ids == 0) | (ids == 101) | (ids == 102)
        tm = tm & (~specials)

    logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
    logits = logits.masked_fill(~tm[:, None, :], -1e4)

    per_q = logits.max(dim=-1).values  # (B,Q)
    return torch.sigmoid(per_q)

def set_deterministic(seed: int = 0):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # TF32 cũng làm lệch nhẹ
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ép thuật toán deterministic (có thể làm chậm)
    torch.use_deterministic_algorithms(True, warn_only=True)

def points_to_boxes_xyxy(points_xy: torch.Tensor, H: int, W: int, box_size: int = 16) -> torch.Tensor:
    half = box_size / 2.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x1 = (x - half).clamp(0, W - 1)
    y1 = (y - half).clamp(0, H - 1)
    x2 = (x + half).clamp(0, W - 1)
    y2 = (y + half).clamp(0, H - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    bw = (x2 - x1).clamp(min=1.0) / W
    bh = (y2 - y1).clamp(min=1.0) / H
    return torch.stack([cx, cy, bw, bh], dim=-1)

def cxcywh_norm_to_xyxy_px(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    cx, cy, bw, bh = boxes.unbind(-1)
    cx = cx * W
    cy = cy * H
    bw = bw * W
    bh = bh * H
    x1 = (cx - bw / 2).clamp(0, W - 1)
    y1 = (cy - bh / 2).clamp(0, H - 1)
    x2 = (cx + bw / 2).clamp(0, W - 1)
    y2 = (cy + bh / 2).clamp(0, H - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)

def scale_points_xy(points_xy: torch.Tensor, orig_h: int, orig_w: int, new_h: int, new_w: int) -> torch.Tensor:
    if points_xy.numel() == 0:
        return points_xy
    sx = new_w / float(orig_w)
    sy = new_h / float(orig_h)
    out = points_xy.clone().float()
    out[:, 0] = out[:, 0] * sx
    out[:, 1] = out[:, 1] * sy
    return out


def build_targets(batch: Dict[str, Any], device: str, points_list_scaled=None):
    images = batch["images"]
    B, _, H, W = images.shape
    gt_counts = batch["gt_counts"].to(device)

    points_list = points_list_scaled if points_list_scaled is not None else batch["meta"]["points"]

    targets = []
    for i in range(B):
        pts = points_list[i].to(device)
        boxes_xyxy = points_to_boxes_xyxy(pts, H=H, W=W, box_size=16)
        boxes = xyxy_to_cxcywh_norm(boxes_xyxy, H=H, W=W)
        labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
        targets.append({"count": gt_counts[i], "boxes": boxes, "labels": labels})
    return targets


def points_to_exemplar_boxes_xyxy(points_xy: torch.Tensor, H: int, W: int,
                                  box_size: int = 32, max_ex: int = 3):
    # points_xy: (N,2) pixel (x,y)
    if points_xy.numel() == 0:
        return torch.zeros((0, 4), device=points_xy.device, dtype=torch.float32)

    pts = points_xy[:max_ex].float()
    r = box_size / 2.0
    x1 = (pts[:, 0] - r).clamp(0, W - 1)
    y1 = (pts[:, 1] - r).clamp(0, H - 1)
    x2 = (pts[:, 0] + r).clamp(0, W - 1)
    y2 = (pts[:, 1] + r).clamp(0, H - 1)
    return torch.stack([x1, y1, x2, y2], dim=1)

def visualize(image_tensor: torch.Tensor, img_id: str, gt_points: torch.Tensor,
              pred_boxes_xyxy: torch.Tensor, pred_scores: torch.Tensor, outpath: str):
    # image_tensor: (3,H,W), normalized; dataset dùng mean/std ImageNet -> convert thô bằng clamp
    x = image_tensor.detach().cpu()
    x = x.clamp(-3, 3)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    x = (x * 255).byte()
    img = Image.fromarray(x.permute(1, 2, 0).numpy())

    draw = ImageDraw.Draw(img)

    # GT points (red)
    for p in gt_points.cpu().tolist():
        px, py = p[0], p[1]
        r = 2
        draw.ellipse((px - r, py - r, px + r, py + r), outline="red", width=2)

    # Pred boxes (green)
    for (b, s) in zip(pred_boxes_xyxy.cpu().tolist(), pred_scores.cpu().tolist()):
        x1, y1, x2, y2 = b
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1, y1), f"{s:.2f}", fill="lime")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    img.save(outpath)

def debug_countgd_token_maps(outputs, targets, indices, special_tokens=(101, 102, 1012, 1029), topk=20):
    """
    outputs: dict after criterion (must contain pred_logits, one_hot, text_mask, input_ids)
    targets: list[dict] length B
    indices: list[(src_idx, tgt_idx)] length B
    """
    assert "pred_logits" in outputs and "one_hot" in outputs
    assert "text_mask" in outputs and "input_ids" in outputs

    pl = outputs["pred_logits"]     # [B,Q,T]
    oh = outputs["one_hot"]         # [B,Q,T] (int/bool)
    ids = outputs["input_ids"]      # [B,T]
    tm = outputs.get("text_mask", None)      # [B,T]   (0/1)
    if tm is None:
        tm = torch.ones_like(ids, dtype=torch.bool)
    else:
        tm = tm.bool()
    print(outputs.keys())
    for k,v in outputs.items():
        if torch.is_tensor(v):
            print(k, v.shape, v.dtype)
        elif isinstance(v, list):
            print(k, "list", len(v))


    B, Q, T = pl.shape

    # valid token mask = attention_mask & ~special
    special = torch.zeros_like(ids, dtype=torch.bool)
    for t in special_tokens:
        special |= (ids == t)
    valid = tm & (~special) # [B,T]

    print("\n==== CountGD token/map debug ====")
    print(f"B={B}, Q={Q}, T={T}")
    print("pred_logits finite:", torch.isfinite(pl).all().item(),
          "| has_inf:", torch.isinf(pl).any().item(),
          "| min/max:", pl.min().item(), pl.max().item())

    # 1) text_mask sums
    print("text_mask.sum (incl specials):", tm.sum(dim=1).tolist())
    print("valid_tokens.sum (excl specials):", valid.sum(dim=1).tolist())

    # 2) one_hot sums
    # total positives over all queries/tokens
    oh_sum_total = oh.sum().item() if oh.dtype != torch.bool else oh.int().sum().item()
    print("one_hot.sum total:", int(oh_sum_total))

    # positives per sample
    for b in range(B):
        oh_b = oh[b].int() if oh[b].dtype != torch.bool else oh[b].int()
        pos_tokens_b = int(oh_b.sum().item())
        print(f"[b={b}] one_hot.sum:", pos_tokens_b)

    # 3) ratio matched queries with any positive token in one_hot
    for b in range(B):
        src_idx, tgt_idx = indices[b]
        if src_idx.numel() == 0:
            print(f"[b={b}] matched=0 (skip)")
            continue

        oh_b = oh[b].int() if oh[b].dtype != torch.bool else oh[b].int()  # [Q,T]
        matched_oh = oh_b[src_idx]  # [M,T]
        matched_has_pos = (matched_oh.sum(dim=1) > 0)  # [M]
        ratio = matched_has_pos.float().mean().item() if matched_has_pos.numel() else 0.0

        print(f"[b={b}] matched queries: {src_idx.numel()}")
        print(f"[b={b}] matched queries with one_hot!=0: {matched_has_pos.sum().item()} "
              f"({ratio*100:.1f}%)")

        # Optional: show a few matched queries and how many positive tokens they got
        m = min(topk, src_idx.numel())
        pos_counts = matched_oh.sum(dim=1)[:m].tolist()
        print(f"[b={b}] first {m} matched pos-token-counts:", [int(x) for x in pos_counts])

    # Extra: verify that one_hot only uses valid tokens (should be true ideally)
    for b in range(B):
        oh_b = oh[b].int() if oh[b].dtype != torch.bool else oh[b].int()
        invalid_cols = (~valid[b]).nonzero(as_tuple=False).flatten()
        if invalid_cols.numel() > 0:
            leak = int(oh_b[:, invalid_cols].sum().item())
            print(f"[b={b}] one_hot positives on invalid tokens:", leak)

def get_batch_by_img_id(dl: DataLoader, target_img_id: str, max_iter: int = 100000):
    for it, batch in enumerate(dl):
        img_id = batch["meta"]["image_ids"][0]
        if img_id == target_img_id:
            return batch
        if it + 1 >= max_iter:
            break
    raise ValueError(f"Không tìm thấy img_id='{target_img_id}' trong dataloader (scan {max_iter} batch).")

def get_batch_by_index(ds, idx: int):
    # Lấy sample trực tiếp từ dataset rồi đóng gói lại bằng collate_fn
    sample = ds[idx]
    return fsc147_collate([sample])

def get_random_batch(dl: DataLoader, seed: int = 0):
    rnd = random.Random(seed)
    # reservoir sampling để chọn 1 batch random mà không list(dl)
    chosen = None
    for i, batch in enumerate(dl, start=1):
        if rnd.randrange(i) == 0:
            chosen = batch
    if chosen is None:
        raise RuntimeError("Dataloader rỗng.")
    return chosen


@torch.no_grad()
def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    ds = FSC147Dataset(
        ann_path=args.ann,
        img_root=args.img_root,
        split=args.split,
        split_file=args.split_file,
        class_map_file=args.class_map,
        img_size=None,
        normalize=True,
        density_root=None,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=fsc147_collate)

    cfg = GDCountConfig(
        threshold=args.threshold,
        soa_level=args.soa_level,
        feature_dim=256,
        freeze_keywords=["backbone.0", "bert"],
    )
    model = build_gdcount_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        gdcount_cfg=cfg,
    ).to(device)
    model.eval()
    

    criterion_detect = build_criterion_detect(
        tokenizer=model.base_model.tokenizer,
        num_classes=1,
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        lambda_cls=1.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
    )
    criterion = MultiTaskLoss(
        criterion=criterion_detect,
        weight_dict=criterion_detect.weight_dict,
        lambda_query=0.0,
        use_query_loss=False,
        log_count_mae=True, 
    ).to(device)

    if args.img_id is not None:
        batch = get_batch_by_img_id(dl, args.img_id, max_iter=args.max_iter)
    elif args.index is not None:
        batch = get_batch_by_index(ds, args.index)
    else:
        batch = get_random_batch(dl, seed=args.seed)


    images = batch["images"].to(device)     # (1,3,H,W)
    prompts = batch["prompts"]              # list[str]
    
    caption = prompts
    img_id = batch["meta"]["image_ids"][0]
    gt_points = batch["meta"]["points"][0]  # (N,2) pixel
    img_path = os.path.join(args.img_root, img_id)
    with Image.open(img_path) as im:
        orig_w, orig_h = im.size

        #----------

    H, W = images.shape[-2], images.shape[-1]
    print("tensor H,W:", H, W)

    img_path = os.path.join(args.img_root, img_id)
    with Image.open(img_path) as im:
        orig_w, orig_h = im.size
    print("file orig_w,orig_h:", orig_w, orig_h)

    pts = batch["meta"]["points"][0]
    print("pts min/max x:", pts[:,0].min().item(), pts[:,0].max().item())
    print("pts min/max y:", pts[:,1].min().item(), pts[:,1].max().item())

    gt_points = batch["meta"]["points"][0]          # đã scaled đúng
    points_list_scaled = batch["meta"]["points"]    # list[tensor]

    gt_count = batch["gt_counts"].item()
    cat_list = [[batch["meta"]["class_names"][0]]]  # B=1

    print("==== Batch ====")
    print("image_id:", img_id)
    print("prompt:", prompts[0])
    print("GT count:", gt_count)
    print("num_points:", gt_points.shape)

    with autocast(enabled=False):
        # --- tạo exemplars từ gt_points để ép chạy nhánh add_exemplar_tokens trong groundingdino.py ---
        B, _, H, W = images.shape
        points_list = batch["meta"]["points"]  # list[tensor(N,2)] pixel

        exemplars = [
            points_to_exemplar_boxes_xyxy(points_list_scaled[i].to(device), H, W, box_size=32, max_ex=3)
            for i in range(B)
        ]
        labels_ex = torch.zeros((B, 1), dtype=torch.long, device=device)  # phrase 0

        
        outputs = model(images, captions=prompts, exemplars=exemplars, labels=labels_ex)
        print("5 first output boxes")
        b = outputs["pred_boxes"][0, :5]  # (5,4)
        print(b.min().item(), b.max().item(), b)
        scores_all = query_scores_from_pred_logits(outputs)[0]
        pred_cnt = (scores_all > args.threshold).sum().item()
        print("DEBUG pred_count(thresholded det):", pred_cnt)

        # thêm count head (để criterion multitask vẫn dùng được như trước)
        hs = model.base_model.transformer.last_hs
        qlogits, softc, hardc = model.count_head(hs)
        outputs["query_logits"] = qlogits
        outputs["soft_counts"] = softc
        outputs["hard_counts"] = hardc

        print("MODEL outputs has text_mask?", "text_mask" in outputs)
        pl = outputs["pred_logits"]
        print("pred_logits finite:", torch.isfinite(pl).all().item())
        print("pred_logits nan:", torch.isnan(pl).any().item(), "inf:", torch.isinf(pl).any().item())
        print("pred_logits range:", pl.min().item(), pl.max().item())

        targets = build_targets(batch, device=device, points_list_scaled=points_list_scaled)

        loss_dict = criterion(outputs, targets, caption=caption, cat_list=cat_list)
        print("MODEL outputs has text_mask after loss_dict?", "text_mask" in outputs)

        scores_all = query_scores_from_pred_logits(outputs)[0]  # lúc này outputs đã có text_mask đúng
        print("DEBUG pred_count(thresholded det) AFTER criterion:", pred_cnt)
        scores_all = query_scores_from_pred_logits(outputs)[0]
        print("threshold:", args.threshold)
        print("scores min/max/mean:", scores_all.min().item(), scores_all.max().item(), scores_all.mean().item())
        print("ratio > thr:", (scores_all > args.threshold).float().mean().item())

        keep = scores_all > args.threshold
        idx_keep = keep.nonzero(as_tuple=False).flatten()

        boxes = outputs["pred_boxes"][0][idx_keep]
        boxes_xyxy = cxcywh_norm_to_xyxy_px(boxes, H=H, W=W)
        scores_keep = scores_all[idx_keep]

        keep2 = nms(boxes_xyxy, scores_keep, iou_threshold=0.5)
        pred_count = int(keep2.numel())
        print("pred_count(after thr+nms):", pred_count)



    print("\n==== Outputs keys ====")
    print(sorted(list(outputs.keys())))

    print("\n==== Shapes ====")
    if "pred_logits" in outputs:
        print("pred_logits:", outputs["pred_logits"].shape)
    if "pred_boxes" in outputs:
        print("pred_boxes:", outputs["pred_boxes"].shape)
    if "hs" in outputs:
        try:
            print("hs[-1]:", outputs["hs"][-1].shape)
        except Exception:
            print("hs is:", type(outputs["hs"]))

    print("\n==== Losses ====")
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            print(f"{k}: {v.item():.6f}")
        else:
            print(f"{k}: {v}")

    # ---- Matcher debug: lấy indices từ SetCriterion bằng return_indices=True ----
    losses_det, indices = criterion_detect(
        outputs,
        [{"boxes": targets[0]["boxes"], "labels": targets[0]["labels"]}],
        cat_list=cat_list,
        caption=caption,
        return_indices=True,
    )

    src_idx, tgt_idx = indices[0]
    print("\n==== Hungarian match ====")
    print("matched:", src_idx.numel())
    print("matched queries:", src_idx.numel())
    print("first 10 src_idx:", src_idx[:10].tolist() if src_idx.numel() else [])
    print("first 10 tgt_idx:", tgt_idx[:10].tolist() if tgt_idx.numel() else [])
    debug_countgd_token_maps(outputs, targets, indices)
    if "query_logits" in outputs and "pred_boxes" in outputs:
        scores_all = query_scores_from_pred_logits(outputs)[0]  # (Q,)
        topk = min(args.topk, scores_all.numel())
        scores, idx = torch.topk(scores_all, k=topk)
        keep = scores_all > args.threshold
        idx_keep = keep.nonzero(as_tuple=False).flatten()
        scores_keep = scores_all[idx_keep]

        topk = min(args.topk, scores_keep.numel())
        scores, order = torch.topk(scores_keep, k=topk)
        idx = idx_keep[order]

        boxes = outputs["pred_boxes"][0][idx]  # cxcywh norm
        boxes_xyxy = cxcywh_norm_to_xyxy_px(boxes, H=H, W=W)

        outpath = os.path.join(args.outdir, f"{os.path.splitext(img_id)[0]}_viz.png")
        visualize(images[0], img_id, gt_points, boxes_xyxy, scores, outpath)

        print("\nSaved visualization:", outpath)
    else:
        print("\nSkip viz (missing query_logits/pred_boxes).")


if __name__ == "__main__":
    main()
