import argparse
import os, sys
import csv
from typing import Any, Dict, List, Tuple
import re
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "groundingdino"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torchvision.ops import nms as tv_nms

from datasets.fsc147_dataset import FSC147Dataset, collate_batch
from models.gdcount_model import GDCountConfig, build_gdcount_model
from scripts.losses import MultiTaskLoss
from scripts.helpers import build_exemplar_inputs, sanitize_caption, _scores_from_pred_logits, _cxcywh_to_xyxy, count_by_det_nms, estimate_side0_from_points,_logits_per_query, sample_log_uniform, _get_query_scores,_match_points_to_queries,center_supervision_loss



# ========================
#  ARGUMENTS
# ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train GDCount on FSC147 (CountGD-style)")

    # ---- GroundingDINO ----
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument(
            "--data-root",
            type=str,
            default=None,
            help="Root folder that contains FSC147 and image folder. "
                "If not set, will default to <project_root>/data",
        )
    # ---- Data ----
    parser.add_argument("--ann", type=str, default="FSC147/annotation_FSC147_384.json")
    parser.add_argument("--img-root", type=str, default="images_384_VarV2")
    parser.add_argument("--split-file", type=str, default="FSC147/Train_Test_Val_FSC_147.json")
    parser.add_argument("--class-map", type=str, default="FSC147/ImageClasses_FSC147.txt")

    # ---- Train ----
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="checkpoints_gdcount")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--use-exemplar", action="store_true")
    parser.add_argument("--exp-name", type=str, default="")

    # ---- GDCount config ----
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--soa-level", type=int, default=-1)
    parser.add_argument("--freeze-keywords", type=str, nargs="+", default=["backbone.0", "bert"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start-epoch", type=int, default=-1)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--no-soa", action="store_true")

    # ---- PSEUDO BOX SIZE: per-image NN + jitter + fallback ----
    parser.add_argument("--side-mode", type=str, default="nn",
                        choices=["nn", "sqrt"],
                        help="nn: side từ nearest-neighbor; sqrt: side từ sqrt(H*W/cnt)")
    parser.add_argument("--nn-k", type=float, default=0.8,
                        help="side0 = nn_k * median(nearest_neighbor_dist)")
    parser.add_argument("--nn-max-samples", type=int, default=256,
                        help="subsample points cho cdist")
    parser.add_argument("--side-jitter-low", type=float, default=0.7,
                        help="log-uniform jitter low (per-image)")
    parser.add_argument("--side-jitter-high", type=float, default=1.4,
                        help="log-uniform jitter high (per-image)")

    # fallback sqrt(HW/cnt)
    parser.add_argument("--side-mult", type=float, default=0.5,
                        help="sqrt-mode: side = side_mult * sqrt(H*W/cnt)")

    # clamps/boost
    parser.add_argument("--side-min", type=float, default=6.0)
    parser.add_argument("--side-max-ratio", type=float, default=0.8,
                        help="Max side = side_max_ratio * min(H,W)")
    parser.add_argument("--boost-cnt2-ratio", type=float, default=0.50,
                        help="If gt_count<=2: side >= boost_cnt2_ratio*min(H,W)")
    parser.add_argument("--boost-cnt5-ratio", type=float, default=0.35,
                        help="If gt_count<=5: side >= boost_cnt5_ratio*min(H,W)")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root if args.data_root else os.path.join(project_root, "data")
    data_root = os.path.normpath(data_root)

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(data_root, p))

    args.data_root = data_root
    args.ann = _resolve(args.ann)
    args.img_root = _resolve(args.img_root)
    args.split_file = _resolve(args.split_file)
    args.class_map = _resolve(args.class_map)

    return args



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
        collate_fn=collate_batch,
    )
    return loader


# ========================
#  MODEL + OPTIMIZER
# ========================

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


def create_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    head_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "count_head" in name or "soa" in name:
            head_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": head_params, "lr": lr},
        {"params": base_params, "lr": lr * 0.5},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    return optimizer, scheduler


# ========================
#  TARGETS (PER-IMAGE SIDE)
# ========================

# def build_targets(batch: Dict[str, Any], device: str, args: argparse.Namespace):
#     images = batch["images"]
#     prompts = [sanitize_caption(p) for p in batch["prompts"]]

#     gt_counts = batch["gt_counts"].to(device)
#     points_list = batch["meta"]["points"]

#     B = images.shape[0]
#     targets, cat_list = [], []
#     captions = prompts

#     dev = torch.device(device)

#     for i in range(B):
#         H = int(images[i].shape[-2])
#         W = int(images[i].shape[-1])
#         min_hw = float(min(H, W))

#         pts = points_list[i].to(device)
#         Ni = int(pts.shape[0])

#         cnt = float(gt_counts[i].item()) if Ni > 0 else 1.0
#         cnt = max(cnt, 1.0)

#         # ---- base side0 ----
#         side0 = None

#         if args.side_mode == "nn":
#             # per-image estimate from points
#             try:
#                 side0 = estimate_side0_from_points(
#                     pts_xy=pts,
#                     H=H,
#                     W=W,
#                     device=dev,
#                     k=float(args.nn_k),
#                     max_samples=int(args.nn_max_samples),
#                 )
#             except Exception:
#                 side0 = None

#         if side0 is None:
#             # fallback sqrt-mode
#             base = math.sqrt((H * W) / cnt)
#             side0 = float(args.side_mult) * float(base)

#         # ---- jitter per-image (log-uniform) ----
#         j = sample_log_uniform(float(args.side_jitter_low), float(args.side_jitter_high), device=dev)
#         side = float(side0) * float(j)

#         # ---- boosts for low-count images (tend to have large objects) ----
#         if cnt <= 2.0:
#             side = max(side, float(args.boost_cnt2_ratio) * min_hw)
#         elif cnt <= 5.0:
#             side = max(side, float(args.boost_cnt5_ratio) * min_hw)

#         # ---- clamp ----
#         side_min = float(args.side_min)
#         side_max = float(args.side_max_ratio) * min_hw
#         side = max(side_min, min(side, side_max))

#         # ---- create pseudo boxes from points ----
#         if Ni > 0:
#             cx = pts[:, 0].clamp(0, W - 1)
#             cy = pts[:, 1].clamp(0, H - 1)
#             w = torch.full((Ni,), float(side), device=device)
#             h = torch.full((Ni,), float(side), device=device)

#             x0 = (cx - w / 2).clamp(0, W - 1)
#             y0 = (cy - h / 2).clamp(0, H - 1)
#             x1 = (cx + w / 2).clamp(0, W - 1)
#             y1 = (cy + h / 2).clamp(0, H - 1)

#             cxn = ((x0 + x1) / 2) / W
#             cyn = ((y0 + y1) / 2) / H
#             wn = (x1 - x0) / W
#             hn = (y1 - y0) / H
#             boxes = torch.stack([cxn, cyn, wn, hn], dim=-1).clamp(0, 1)
#         else:
#             boxes = torch.zeros((0, 4), device=device)

#         labels = torch.zeros((Ni,), dtype=torch.long, device=device)

#         targets.append({"boxes": boxes, "labels": labels, "count": gt_counts[i].view(())})

#         phrase = prompts[i].strip()
#         if phrase.endswith("."):
#             phrase = phrase[:-1].strip()
#         if len(phrase) == 0:
#             phrase = "object"
#         cat_list.append([phrase])

#     return targets, captions, cat_list


# ========================
#  TRAIN / EVAL
# ========================

LOSS_KEYS = ["loss_total", "count_mae", "loss_ce", "loss_bbox", "loss_giou", "loss_query", "loss_center"]

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

        with autocast(enabled=(args.amp and device.startswith("cuda"))):
            if args.use_exemplar:
                exemplars, labels = build_exemplar_inputs(batch, device)
                has_any = any(ex.shape[0] > 0 for ex in exemplars)
                if has_any:
                    outputs: Dict[str, Any] = model(images, captions=prompts, exemplars=exemplars, labels=labels)
                else:
                    outputs: Dict[str, Any] = model(images, captions=prompts)
            else:
                outputs: Dict[str, Any] = model(images, captions=prompts)

            outputs.pop("hard_counts", None)

            batch = dict(batch)
            batch["prompts"] = prompts

            loss_dict = center_supervision_loss(
                outputs=outputs,
                batch=batch,
                device=device,
                lambda_center=5.0,
                topk_min=300,
            )
            loss = loss_dict["loss_total"]
            loss_to_backprop = loss / accum_steps

        scaler.scale(loss_to_backprop).backward()

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
        pbar.set_postfix({
            "L_total": f"{avg_now['loss_total']:.3f}",
            "MAE_cnt": f"{avg_now.get('count_mae', 0.0):.3f}",
            "L_ce": f"{avg_now['loss_ce']:.3f}",
            "L_box": f"{avg_now['loss_bbox']:.3f}",
            "L_giou": f"{avg_now['loss_giou']:.3f}",
            "L_q": f"{avg_now['loss_query']:.3f}",
        })

        if (step + 1) % log_interval == 0:
            print(
                f"[Train] Epoch {epoch} Step {step+1}/{len(loader)} "
                f"L_total={avg_now['loss_total']:.4f} "
                f"MAE_cnt={avg_now.get('count_mae', 0.0):.4f} "
                f"L_ce={avg_now['loss_ce']:.4f} "
                f"L_box={avg_now['loss_bbox']:.4f} "
                f"L_giou={avg_now['loss_giou']:.4f} "
                f"L_q={avg_now['loss_query']:.4f}"
            )

    return _avg_losses(loss_sums, num_samples)


def evaluate(
    epoch: int,
    model: torch.nn.Module,
    loader: DataLoader,
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
        for _, batch in pbar:
            images = batch["images"].to(device)
            prompts = [sanitize_caption(x) for x in batch["prompts"]]
            gt_counts = batch["gt_counts"].to(device)

            batch_size = images.shape[0]
            num_samples += batch_size

            with autocast(enabled=(args.amp and device.startswith("cuda"))):
                if args.use_exemplar:
                    exemplars, labels = build_exemplar_inputs(batch, device)
                    has_any = any(ex.shape[0] > 0 for ex in exemplars)
                    if has_any:
                        outputs: Dict[str, Any] = model(images, captions=prompts, exemplars=exemplars, labels=labels)
                    else:
                        outputs: Dict[str, Any] = model(images, captions=prompts)
                else:
                    outputs: Dict[str, Any] = model(images, captions=prompts)

                outputs.pop("hard_counts", None)

                batch = dict(batch)
                batch["prompts"] = prompts
            
                loss_dict = center_supervision_loss(
                    outputs=outputs,
                    batch=batch,
                    device=device,
                    lambda_center=5.0,
                    topk_min=300,
                )

            for k in LOSS_KEYS:
                v = loss_dict.get(k, torch.tensor(0.0, device=device)).item()
                loss_sums[k] += v * batch_size

            pred_counts = count_by_det_nms(outputs, threshold=args.threshold, nms_iou=args.nms_iou).to(gt_counts.dtype)
            diff = (pred_counts - gt_counts).abs()
            mae_sum += diff.sum().item()
            mse_sum += (diff ** 2).sum().item()
            n_count_samples += gt_counts.numel()

            avg_now = _avg_losses(loss_sums, num_samples)
            pbar.set_postfix({
                "L_total": f"{avg_now['loss_total']:.3f}",
                "MAE_cnt": f"{avg_now.get('count_mae', 0.0):.3f}",
                "L_ce": f"{avg_now['loss_ce']:.3f}",
                "L_box": f"{avg_now['loss_bbox']:.3f}",
                "L_giou": f"{avg_now['loss_giou']:.3f}",
                "L_q": f"{avg_now['loss_query']:.3f}",
            })

    avg_losses = _avg_losses(loss_sums, num_samples)
    if n_count_samples > 0:
        mae = mae_sum / n_count_samples
        rmse = math.sqrt(mse_sum / n_count_samples)
    else:
        mae, rmse = 0.0, 0.0
    return avg_losses, {"mae": mae, "rmse": rmse}


# ========================
#  CHECKPOINT + CSV
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
    ckpt_path = os.path.join(save_dir, f"gdcount_epoch_{epoch:03d}{'_' + suffix if suffix else ''}.pth")
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
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
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1


def init_csv_logger(log_dir: str = "logs", filename: str = "train_log.csv") -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "train_loss_total","train_count_mae","train_loss_ce","train_loss_bbox","train_loss_giou","train_loss_query",
                "val_loss_total","val_count_mae","val_loss_ce","val_loss_bbox","val_loss_giou","val_loss_query",
                "val_mae","val_rmse",
            ])
    return path


def append_csv_log(
    csv_path: str,
    epoch: int,
    train_losses: Dict[str, float],
    val_losses: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            train_losses.get("loss_total",0.0),
            train_losses.get("count_mae", 0.0),
            train_losses.get("loss_ce",0.0),
            train_losses.get("loss_bbox",0.0),
            train_losses.get("loss_giou",0.0),
            train_losses.get("loss_query",0.0),

            val_losses.get("loss_total",0.0),
            val_losses.get("count_mae", 0.0),
            val_losses.get("loss_ce",0.0),
            val_losses.get("loss_bbox",0.0),
            val_losses.get("loss_giou",0.0),
            val_losses.get("loss_query",0.0),

            val_metrics.get("mae",0.0),
            val_metrics.get("rmse",0.0),
        ])


# ========================
#  MAIN
# ========================

def main():
    args = parse_args()
    exp = args.exp_name.strip() or ("text_exemplar" if args.use_exemplar else "text_only")

    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    print(f"Using device: {device}")

    train_loader = create_dataloader(args, split="train")
    val_loader = create_dataloader(args, split="val")

    model = create_model(args).to(device)

    optimizer, scheduler = create_optimizer(model, args.lr, args.weight_decay)
    scaler = GradScaler(enabled=(args.amp and device.startswith("cuda")))

    best_val = float("inf")
    start_epoch = 1

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)

    if args.start_epoch is not None and args.start_epoch > 0:
        start_epoch = args.start_epoch

    print(f"Start epoch: {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            log_interval=args.log_interval,
            args=args,
        )

        val_losses, val_metrics = evaluate(
            epoch=epoch,
            model=model,
            loader=val_loader,
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
            f"q={train_losses.get('loss_query',0.0):.4f} "
            f"| Val: total={val_losses.get('loss_total',0.0):.4f}, "
            f"count_mae={val_losses.get('count_mae',0.0):.4f}, "
            f"MAE={val_metrics.get('mae',0.0):.4f}, "
            f"RMSE={val_metrics.get('rmse',0.0):.4f}"
        )

        save_dir = os.path.join(args.save_dir, exp)
        os.makedirs(save_dir, exist_ok=True)

        csv_path = init_csv_logger(log_dir="logs", filename=f"train_log_{exp}.csv")
        append_csv_log(csv_path, epoch, train_losses, val_losses, val_metrics)

        scheduler.step()
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, save_dir)

        if val_losses.get("loss_total", 1e9) < best_val:
            best_val = val_losses["loss_total"]
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, os.path.join(save_dir, "best"), suffix="best")


if __name__ == "__main__":
    main()
