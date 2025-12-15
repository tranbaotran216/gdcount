# finetune_bert.py
import argparse
import os
import csv
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

# Dataset
from datasets.fsc147_dataset import FSC147Dataset, fsc147_collate

# Model
from models.gdcount_model import GDCountConfig, build_gdcount_model

# Loss / Criterion
from scripts.losses import MultiTaskLoss
from scripts.criterion_detect import build_criterion_detect


# ========================
#  EXEMPLAR INPUTS
# ========================
def build_exemplar_inputs(batch: Dict[str, Any], device: str):
    # meta["exemplar_xyxy"] : list[Tensor(K,4)] (pixel xyxy)
    ex_list = [t.to(device) for t in batch["meta"]["exemplar_xyxy"]]
    # labels per exemplar: all zeros (single phrase)
    labels_list = [torch.zeros((ex.shape[0],), dtype=torch.long, device=device) for ex in ex_list]
    return ex_list, labels_list


# ========================
#  TEXT HELPERS
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


# ========================
#  COUNTING BY DET + NMS
# ========================
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
    """
    Return pred_count (B,) from scores + pred_boxes with threshold + NMS.
    Ưu tiên dùng outputs["calib_scores"] (nếu có), fallback pred_logits.
    boxes dùng normalized xyxy (0..1).
    """
    if "calib_scores" in outputs and isinstance(outputs["calib_scores"], torch.Tensor):
        scores = outputs["calib_scores"].float()  # (B,Q)
    else:
        scores = _scores_from_pred_logits(outputs)  # (B,Q)

    keep = scores > float(threshold)

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
        kept = tv_nms(b_boxes, b_scores, float(nms_iou))
        out_counts.append(torch.tensor(float(kept.numel()), device=scores.device))
    return torch.stack(out_counts, dim=0)


# ========================
#  ARGUMENTS
# ========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Fine-tune BERT last layers for GDCount on FSC147")

    # ---- GroundingDINO ----
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    # ---- Data ----
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--img-root", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--class-map", type=str, required=True)

    # ---- Train ----
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5, help="LR cho head params (không phải BERT)")
    parser.add_argument("--bert-lr", type=float, default=2e-6, help="LR cho BERT last layers")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="checkpoints_gdcount")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", default=False, help="Bật AMP")

    parser.add_argument("--use-exemplar", action="store_true", help="Train text+exemplar")
    parser.add_argument("--exp-name", type=str, default="", help="Tên experiment để tách log/ckpt")

    # ---- GDCount config ----
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--soa-level", type=int, default=-1)
    parser.add_argument("--freeze-keywords", type=str, nargs="+", default=["backbone.0", "bert"])

    # Fine-tune BERT
    parser.add_argument("--unfreeze-bert-last-n", type=int, default=2, help="Mở lại n layer cuối BERT (bert-base: tối đa 12)")

    # Resume
    parser.add_argument("--resume", type=str, default="", help="Path checkpoint để resume")
    parser.add_argument("--start-epoch", type=int, default=-1, help="Override start epoch (>=1)")
    parser.add_argument("--accum-steps", type=int, default=1)

    # LR schedule (optional)
    parser.add_argument("--lr-step", type=int, default=10, help="StepLR step_size")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="StepLR gamma")

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
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        collate_fn=fsc147_collate,
    )
    return loader


# ========================
#  MODEL + OPTIMIZER
# ========================
def create_model(args: argparse.Namespace) -> torch.nn.Module:
    cfg = GDCountConfig(
        threshold=args.threshold,
        soa_level=args.soa_level,
        feature_dim=256,
        freeze_keywords=args.freeze_keywords,
        unfreeze_bert_last_n=int(args.unfreeze_bert_last_n),
        use_exemplar_mod=True,
    )
    model = build_gdcount_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        gdcount_cfg=cfg,
    )
    return model


def create_optimizer(model: torch.nn.Module, head_lr: float, bert_lr: float, weight_decay: float):
    """
    Param groups:
      - BERT last layers: bert_lr
      - Others trainable: head_lr
    """
    head_params = []
    bert_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # match bert-base encoder layers
        if ".bert.encoder.layer." in name:
            # chỉ muốn last layers -> đã được requires_grad=True trong gdcount_model._apply_freeze
            bert_params.append(p)
        else:
            head_params.append(p)

    # Nếu vì config mà bert_params rỗng thì vẫn OK
    param_groups = []
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": float(head_lr)})
    if len(bert_params) > 0:
        param_groups.append({"params": bert_params, "lr": float(bert_lr)})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(weight_decay))
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, step_size: int, gamma: float):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))


# ========================
#  TARGETS
# ========================
def build_targets(batch: Dict[str, Any], device: str):
    images = batch["images"]
    prompts = [sanitize_caption(p) for p in batch["prompts"]]

    gt_counts = batch["gt_counts"].to(device)      # (B,)
    points_list = batch["meta"]["points"]          # list[(Ni,2)]

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
        side = max(6.0, min(40.0, 0.5 * base))

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


# ========================
#  TRAIN / EVAL
# ========================
LOSS_KEYS = [
    "loss_total",
    "count_mae",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "loss_query",
    "loss_calib",
]


def _init_loss_sums() -> Dict[str, float]:
    return {k: 0.0 for k in LOSS_KEYS}


def _avg_losses(loss_sums: Dict[str, float], num_samples: int) -> Dict[str, float]:
    if num_samples == 0:
        return {k: 0.0 for k in LOSS_KEYS}
    return {k: v / num_samples for k, v in loss_sums.items()}


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: MultiTaskLoss,
    device: str,
    log_interval: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()

    loss_sums = _init_loss_sums()
    num_samples = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [train]", ncols=120)
    accum_steps = max(1, int(args.accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for step, batch in pbar:
        images = batch["images"].to(device)
        prompts = [sanitize_caption(x) for x in batch["prompts"]]

        with autocast(enabled=(args.amp and str(device).startswith("cuda"))):
            if args.use_exemplar:
                exemplars, labels = build_exemplar_inputs(batch, device)
                has_any = any(ex.shape[0] > 0 for ex in exemplars)
                if has_any:
                    outputs: Dict[str, Any] = model(images, captions=prompts, exemplars=exemplars, labels=labels)
                else:
                    outputs = model(images, captions=prompts)
            else:
                outputs = model(images, captions=prompts)

            # tránh nhầm với MultiTaskLoss fallback hard_counts
            outputs.pop("hard_counts", None)
            outputs["img_hw"] = (images.shape[-2], images.shape[-1])

            batch2 = dict(batch)
            batch2["prompts"] = prompts
            targets, captions, cat_list = build_targets(batch2, device=device)

            loss_dict = criterion(outputs, targets, caption=captions, cat_list=cat_list)
            loss = loss_dict["loss_total"]
            loss_to_backprop = loss / accum_steps

        scaler.scale(loss_to_backprop).backward()

        if step == 0:
            ok = False
            for _, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    ok = True
                    break
            print("Has gradients on trainable params:", ok)

        do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        num_samples += batch_size
        for k in LOSS_KEYS:
            v = loss_dict.get(k, torch.tensor(0.0, device=device)).item()
            loss_sums[k] += v * batch_size

        avg_now = _avg_losses(loss_sums, num_samples)
        pbar.set_postfix(
            {
                "L_total": f"{avg_now['loss_total']:.3f}",
                "MAE_cnt": f"{avg_now.get('count_mae', 0.0):.3f}",
                "L_ce": f"{avg_now['loss_ce']:.3f}",
                "L_box": f"{avg_now['loss_bbox']:.3f}",
                "L_giou": f"{avg_now['loss_giou']:.3f}",
                "L_cal": f"{avg_now.get('loss_calib', 0.0):.3f}",
            }
        )

        if (step + 1) % log_interval == 0:
            print(
                f"[Train] Epoch {epoch} Step {step+1}/{len(loader)} "
                f"L_total={avg_now['loss_total']:.4f} "
                f"MAE_cnt={avg_now.get('count_mae', 0.0):.4f} "
                f"L_ce={avg_now['loss_ce']:.4f} "
                f"L_box={avg_now['loss_bbox']:.4f} "
                f"L_giou={avg_now['loss_giou']:.4f} "
                f"L_cal={avg_now.get('loss_calib', 0.0):.4f}"
            )

    return _avg_losses(loss_sums, num_samples)


def evaluate(
    epoch: int,
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

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [val]  ", ncols=120)

    with torch.no_grad():
        for step, batch in pbar:
            images = batch["images"].to(device)
            prompts = [sanitize_caption(x) for x in batch["prompts"]]
            gt_counts = batch["gt_counts"].to(device)

            batch_size = images.shape[0]
            num_samples += batch_size

            with autocast(enabled=(args.amp and str(device).startswith("cuda"))):
                if args.use_exemplar:
                    exemplars, labels = build_exemplar_inputs(batch, device)
                    has_any = any(ex.shape[0] > 0 for ex in exemplars)
                    if has_any:
                        outputs: Dict[str, Any] = model(images, captions=prompts, exemplars=exemplars, labels=labels)
                    else:
                        outputs = model(images, captions=prompts)
                else:
                    outputs = model(images, captions=prompts)

                outputs.pop("hard_counts", None)
                outputs["img_hw"] = (images.shape[-2], images.shape[-1])

                batch2 = dict(batch)
                batch2["prompts"] = prompts
                targets, captions, cat_list = build_targets(batch2, device)
                loss_dict = criterion(outputs, targets, caption=captions, cat_list=cat_list)

            for k in LOSS_KEYS:
                v = loss_dict.get(k, torch.tensor(0.0, device=device)).item()
                loss_sums[k] += v * batch_size

            pred_counts = count_by_det_nms(outputs, threshold=criterion.threshold, nms_iou=args.nms_iou).to(gt_counts.dtype)
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
                    "L_cal": f"{avg_now.get('loss_calib', 0.0):.3f}",
                }
            )

    avg_losses = _avg_losses(loss_sums, num_samples)
    if n_count_samples > 0:
        mae = mae_sum / n_count_samples
        rmse = (mse_sum / n_count_samples) ** 0.5
    else:
        mae = 0.0
        rmse = 0.0

    return avg_losses, {"mae": mae, "rmse": rmse}


# ========================
#  CHECKPOINT + LOG CSV
# ========================
def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    save_dir: str,
    suffix: str = "",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    if suffix:
        ckpt_path = os.path.join(save_dir, f"gdcount_epoch_{epoch:03d}_{suffix}.pth")
    else:
        ckpt_path = os.path.join(save_dir, f"gdcount_epoch_{epoch:03d}.pth")

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: str,
    load_optim: bool = False,   # NEW
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[CKPT] Missing keys: {len(missing)}")
    if len(missing) > 0:
        print("  - missing (first 20):", missing[:20])
    print(f"[CKPT] Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print("  - unexpected (first 20):", unexpected[:20])

    if load_optim:
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])

    return int(ckpt.get("epoch", 0)) + 1



def init_csv_logger(log_dir: str = "logs", filename: str = "train_log.csv") -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss_total", "train_count_mae", "train_loss_ce", "train_loss_bbox", "train_loss_giou", "train_loss_query", "train_loss_calib",
                    "val_loss_total", "val_count_mae", "val_loss_ce", "val_loss_bbox", "val_loss_giou", "val_loss_query", "val_loss_calib",
                    "val_mae", "val_rmse",
                ]
            )
    return path


def append_csv_log(
    csv_path: str,
    epoch: int,
    train_losses: Dict[str, float],
    val_losses: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                train_losses.get("loss_total", 0.0),
                train_losses.get("count_mae", 0.0),
                train_losses.get("loss_ce", 0.0),
                train_losses.get("loss_bbox", 0.0),
                train_losses.get("loss_giou", 0.0),
                train_losses.get("loss_query", 0.0),
                train_losses.get("loss_calib", 0.0),

                val_losses.get("loss_total", 0.0),
                val_losses.get("count_mae", 0.0),
                val_losses.get("loss_ce", 0.0),
                val_losses.get("loss_bbox", 0.0),
                val_losses.get("loss_giou", 0.0),
                val_losses.get("loss_query", 0.0),
                val_losses.get("loss_calib", 0.0),

                val_metrics.get("mae", 0.0),
                val_metrics.get("rmse", 0.0),
            ]
        )


# ========================
#  MAIN
# ========================
def main():
    args = parse_args()

    exp = args.exp_name.strip()
    if not exp:
        exp = "finetune_bert_exemplar" if args.use_exemplar else "finetune_bert_text"

    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    print(f"Using device: {device}")

    # Data
    train_loader = create_dataloader(args, split="train")
    val_loader = create_dataloader(args, split="val")

    # Model
    model = create_model(args)
    model.to(device)

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    print(f"Trainable params: {trainable}/{total}")

    # Criterion detect
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

    # MultiTaskLoss: bật calib loss để train calibrator
    criterion = MultiTaskLoss(
        criterion=criterion_detect,
        weight_dict=criterion_detect.weight_dict,
        lambda_query=0.0,
        use_query_loss=False,
        log_count_mae=True,
        threshold=args.threshold,
        nms_iou=args.nms_iou,
        lambda_calib=1.0,
        use_calib_loss=True,
    )

    # Optimizer + Scheduler + Scaler
    optimizer = create_optimizer(model, args.lr, args.bert_lr, args.weight_decay)
    scheduler = create_scheduler(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scaler = GradScaler(enabled=(args.amp and device.startswith("cuda")))

    # CSV logger
    csv_path = init_csv_logger(log_dir="logs", filename=f"train_log_{exp}.csv")

    best_val = float("inf")
    start_epoch = 1

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_epoch = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            load_optim=False,
        )

    if args.start_epoch is not None and args.start_epoch > 0:
        start_epoch = args.start_epoch

    print(f"Start epoch: {start_epoch}")

    save_dir = os.path.join(args.save_dir, exp)

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            log_interval=args.log_interval,
            args=args,
        )

        val_losses, val_metrics = evaluate(
            epoch=epoch,
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
        )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| Train: total={train_losses.get('loss_total',0.0):.4f}, "
            f"count_mae={train_losses.get('count_mae',0.0):.4f}, "
            f"ce={train_losses.get('loss_ce',0.0):.4f}, "
            f"bbox={train_losses.get('loss_bbox',0.0):.4f}, "
            f"giou={train_losses.get('loss_giou',0.0):.4f}, "
            f"calib={train_losses.get('loss_calib',0.0):.4f} "
            f"| Val: total={val_losses.get('loss_total',0.0):.4f}, "
            f"count_mae={val_losses.get('count_mae',0.0):.4f}, "
            f"MAE={val_metrics.get('mae',0.0):.4f}, "
            f"RMSE={val_metrics.get('rmse',0.0):.4f}"
        )

        append_csv_log(csv_path, epoch, train_losses, val_losses, val_metrics)

        scheduler.step()

        save_checkpoint(epoch, model, optimizer, scheduler, scaler, save_dir)

        if val_losses.get("loss_total", 1e9) < best_val:
            best_val = val_losses["loss_total"]
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, os.path.join(save_dir, "best"), suffix="best")


if __name__ == "__main__":
    main()
