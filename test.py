# test.py
import argparse
import os
import csv
from typing import Any, Dict, Tuple
import re

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

from datasets.fsc147_dataset import FSC147Dataset, fsc147_collate
from models.gdcount_model import GDCountConfig, build_gdcount_model
from scripts.losses import MultiTaskLoss
from scripts.criterion_detect import build_criterion_detect


# ========================
#  UTILS (giữ giống train)
# ========================

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


def _scores_from_outputs(outputs: Dict[str, Any]) -> torch.Tensor:
    """
    Return scores (B,Q).
    Ưu tiên: outputs["calib_scores"] (nếu có)
    Fallback: sigmoid(max_token_logit) từ pred_logits.
    """
    if "calib_scores" in outputs and isinstance(outputs["calib_scores"], torch.Tensor):
        s = outputs["calib_scores"]
        if s.dim() == 2:
            return s.float()
        # an toàn: reshape về (B,Q)
        return s.float().view(s.shape[0], -1)

    # fallback pred_logits
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
    """
    Return pred_count (B,) from scores/pred_boxes with threshold + NMS.
    Uses normalized xyxy (0..1).
    """
    scores = _scores_from_outputs(outputs)  # (B,Q)
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


# ========================
#  ARGS
# ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Test GDCount on FSC147 (split=val & test)")

    # ---- GroundingDINO ----
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    # ---- Data ----
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--img-root", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--class-map", type=str, required=True)

    # ---- Runtime ----
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)

    # ---- Inference params ----
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--nms-iou", type=float, default=0.5)

    # ---- GDCount config ----
    parser.add_argument("--soa-level", type=int, default=-1)
    parser.add_argument(
        "--freeze-keywords",
        type=str,
        nargs="+",
        default=["backbone.0", "bert"],
    )

    # ---- Model checkpoint to test ----
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Checkpoint GDCount đã train (vd: .../gdcount_epoch_013_best.pth)",
    )

    # ---- Which splits ----
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val", "test"],
        help='Danh sách split cần chạy. Mặc định: val test. Ví dụ: --splits val test',
    )

    # ---- Logging ----
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--log-file", type=str, default="test_log.csv")
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--no-soa", action="store_true", help="Tắt SOA (không dùng SmallObjectAdapter)")

    return parser.parse_args()


def create_dataloader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = FSC147Dataset(
        ann_path=args.ann,
        img_root=args.img_root,
        split=split,
        split_file=args.split_file,
        class_map_file=args.class_map,
        img_size=None,
        normalize=True,
        density_root=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=fsc147_collate,
    )
    return loader


def create_model(args: argparse.Namespace) -> torch.nn.Module:
    cfg = GDCountConfig(
        threshold=args.threshold,
        soa_level=None if args.no_soa else args.soa_level,
        feature_dim=256,
        freeze_keywords=args.freeze_keywords,
    )
    model = build_gdcount_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        gdcount_cfg=cfg,
    )
    return model


def build_targets(batch: Dict[str, Any], device: str):
    images = batch["images"]
    prompts = [sanitize_caption(p) for p in batch["prompts"]]

    gt_counts = batch["gt_counts"].to(device)
    points_list = batch["meta"]["points"]

    B = images.shape[0]
    targets = []
    cat_list = []
    captions = prompts

    for i in range(B):
        H = int(images[i].shape[-2])
        W = int(images[i].shape[-1])

        pts = points_list[i].to(device)
        Ni = int(pts.shape[0])

        cnt = float(gt_counts[i].item()) if Ni > 0 else 1.0
        base = (H * W / max(cnt, 1.0)) ** 0.5
        side = max(6.0, min(160.0, 0.5 * base))

        if Ni > 0:
            cx = pts[:, 0].clamp(0, W - 1)
            cy = pts[:, 1].clamp(0, H - 1)
            w = torch.full((Ni,), float(side), device=device)
            h = torch.full((Ni,), float(side), device=device)

            x0 = (cx - w / 2).clamp(0, W - 1)
            y0 = (cy - h / 2).clamp(0, H - 1)
            x1 = (cx + w / 2).clamp(0, W - 1)
            y1 = (cy + h / 2).clamp(0, H - 1)

            cxn = ((x0 + x1) / 2) / W
            cyn = ((y0 + y1) / 2) / H
            wn = (x1 - x0) / W
            hn = (y1 - y0) / H
            boxes = torch.stack([cxn, cyn, wn, hn], dim=-1).clamp(0, 1)
        else:
            boxes = torch.zeros((0, 4), device=device)

        labels = torch.zeros((Ni,), dtype=torch.long, device=device)

        targets.append({"boxes": boxes, "labels": labels, "count": gt_counts[i].view(())})

        phrase = prompts[i].strip()
        if phrase.endswith("."):
            phrase = phrase[:-1].strip()
        if len(phrase) == 0:
            phrase = "object"
        cat_list.append([phrase])

    return targets, captions, cat_list


LOSS_KEYS = ["loss_total", "count_mae", "loss_ce", "loss_bbox", "loss_giou", "loss_query"]


def _init_loss_sums() -> Dict[str, float]:
    return {k: 0.0 for k in LOSS_KEYS}


def _avg_losses(loss_sums: Dict[str, float], num_samples: int) -> Dict[str, float]:
    if num_samples == 0:
        return {k: 0.0 for k in LOSS_KEYS}
    return {k: v / num_samples for k, v in loss_sums.items()}


def evaluate_one_split(
    split: str,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()

    loss_sums = _init_loss_sums()
    num_samples = 0

    mae_sum = 0.0
    mse_sum = 0.0
    n_count_samples = 0

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Eval [{split}]",
        ncols=120,
    )

    with torch.no_grad():
        for _, batch in pbar:
            images = batch["images"].to(device)
            prompts = [sanitize_caption(x) for x in batch["prompts"]]
            gt_counts = batch["gt_counts"].to(device)

            batch_size = images.shape[0]
            num_samples += batch_size

            with autocast(enabled=(args.amp and device.startswith("cuda"))):
                outputs: Dict[str, Any] = model(images, captions=prompts)
                outputs.pop("hard_counts", None)

                batch = dict(batch)
                batch["prompts"] = prompts
                targets, captions, cat_list = build_targets(batch, device)
                loss_dict = criterion(outputs, targets, caption=captions, cat_list=cat_list)

            for k in LOSS_KEYS:
                v = loss_dict.get(k, torch.tensor(0.0, device=device)).item()
                loss_sums[k] += v * batch_size

            pred_counts = count_by_det_nms(outputs, threshold=args.threshold, nms_iou=args.nms_iou).to(gt_counts.dtype)

            diff = (pred_counts - gt_counts).abs()
            mae_sum += diff.sum().item()
            mse_sum += (diff ** 2).sum().item()
            n_count_samples += gt_counts.numel()

            avg_now = _avg_losses(loss_sums, num_samples)
            pbar.set_postfix(
                {
                    "L_total": f"{avg_now['loss_total']:.3f}",
                    "MAE_cnt": f"{avg_now.get('count_mae', 0.0):.3f}",
                    "L_ce": f"{avg_now['loss_ce']:.3f}",
                    "L_box": f"{avg_now['loss_bbox']:.3f}",
                    "L_giou": f"{avg_now['loss_giou']:.3f}",
                    "L_q": f"{avg_now['loss_query']:.3f}",
                }
            )

    avg_losses = _avg_losses(loss_sums, num_samples)
    if n_count_samples > 0:
        mae = mae_sum / n_count_samples
        rmse = (mse_sum / n_count_samples) ** 0.5
    else:
        mae, rmse = 0.0, 0.0

    metrics = {"mae": mae, "rmse": rmse}
    return avg_losses, metrics


def load_model_checkpoint(model_ckpt_path: str, model: torch.nn.Module, device: str) -> Dict[str, Any]:
    ckpt = torch.load(model_ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format: {model_ckpt_path}")

    model.load_state_dict(state, strict=True)
    meta = {}
    if isinstance(ckpt, dict):
        meta["epoch"] = ckpt.get("epoch", None)
    return meta


def init_test_csv(log_dir: str, filename: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "experiment_name",
                    "model_ckpt",
                    "ckpt_epoch",
                    "split",
                    "batch_size",
                    "threshold",
                    "nms_iou",
                    "loss_total",
                    "count_mae",
                    "loss_ce",
                    "loss_bbox",
                    "loss_giou",
                    "loss_query",
                    "mae",
                    "rmse",
                ]
            )
    return path


def append_test_csv(
    csv_path: str,
    args: argparse.Namespace,
    ckpt_meta: Dict[str, Any],
    split: str,
    losses: Dict[str, float],
    metrics: Dict[str, float],
) -> None:
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                args.exp,
                args.model_ckpt,
                ckpt_meta.get("epoch", ""),
                split,
                args.batch_size,
                args.threshold,
                args.nms_iou,
                losses.get("loss_total", 0.0),
                losses.get("count_mae", 0.0),
                losses.get("loss_ce", 0.0),
                losses.get("loss_bbox", 0.0),
                losses.get("loss_giou", 0.0),
                losses.get("loss_query", 0.0),
                metrics.get("mae", 0.0),
                metrics.get("rmse", 0.0),
            ]
        )


def main():
    args = parse_args()

    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    print(f"Using device: {device}")

    # model + criterion
    model = create_model(args).to(device)

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
        threshold=args.threshold,
    )
    criterion.nms_iou = args.nms_iou

    # load trained ckpt
    if not os.path.isfile(args.model_ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_ckpt}")
    ckpt_meta = load_model_checkpoint(args.model_ckpt, model, device)
    print(f"Loaded model checkpoint: {args.model_ckpt} (epoch={ckpt_meta.get('epoch', None)})")

    # CSV
    csv_path = init_test_csv(args.log_dir, args.log_file)

    # run for each split
    for split in args.splits:
        loader = create_dataloader(args, split=split)
        losses, metrics = evaluate_one_split(
            split=split,
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            args=args,
        )

        print(
            f"[{split.upper()}] loss_total={losses.get('loss_total',0.0):.4f} "
            f"count_mae={losses.get('count_mae',0.0):.4f} "
            f"MAE={metrics.get('mae',0.0):.4f} "
            f"RMSE={metrics.get('rmse',0.0):.4f}"
        )

        append_test_csv(csv_path, args, ckpt_meta, split, losses, metrics)

    print(f"Saved test log to: {csv_path}")


if __name__ == "__main__":
    main()
