# test_plt_imgs.py
import argparse
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import tqdm as tqdm_mod

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

from datasets.fsc147_dataset import FSC147Dataset, fsc147_collate
from models.gdcount_model import GDCountConfig, build_gdcount_model


# =========================
# Helpers
# =========================

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
        return s.float().view(s.shape[0], -1)

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


def pick_boxes_after_thresh_nms_one(
    outputs: Dict[str, Any],
    b: int,
    threshold: float,
    nms_iou: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (boxes_xyxy_norm(K,4), scores(K,))
    """
    scores = _scores_from_outputs(outputs)[b]  # (Q,)
    keep = scores > float(threshold)

    if "pred_boxes" not in outputs:
        idx = keep.nonzero(as_tuple=False).flatten()
        return torch.zeros((0, 4), device=scores.device), scores[idx]

    boxes = outputs["pred_boxes"].float()[b]  # (Q,4) cxcywh norm
    boxes_xyxy = _cxcywh_to_xyxy(boxes).clamp(0, 1)

    idx = keep.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.zeros((0, 4), device=scores.device), torch.zeros((0,), device=scores.device)

    bxy = boxes_xyxy[idx]
    s = scores[idx]

    if tv_nms is None or idx.numel() == 1:
        return bxy, s

    kept = tv_nms(bxy, s, float(nms_iou))  # boxes normalized vẫn OK cho IoU
    return bxy[kept], s[kept]


def _boxes_to_centers_norm(boxes_xyxy_norm: torch.Tensor) -> torch.Tensor:
    if boxes_xyxy_norm is None or boxes_xyxy_norm.numel() == 0:
        device = boxes_xyxy_norm.device if boxes_xyxy_norm is not None else "cpu"
        return torch.zeros((0, 2), device=device)
    x1, y1, x2, y2 = boxes_xyxy_norm.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return torch.stack([cx, cy], dim=-1).clamp(0, 1)


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = kernel_size
    ax = torch.arange(k, device=device) - (k // 2)
    # indexing="ij" để tránh warning meshgrid
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, k, k)


def centers_to_density_map(
    centers_norm: torch.Tensor,
    out_h: int,
    out_w: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    density = torch.zeros((1, 1, out_h, out_w), device=device, dtype=torch.float32)
    if centers_norm is None or centers_norm.numel() == 0:
        return density[0, 0]

    xs = (centers_norm[:, 0] * (out_w - 1)).round().long().clamp(0, out_w - 1)
    ys = (centers_norm[:, 1] * (out_h - 1)).round().long().clamp(0, out_h - 1)
    density[0, 0, ys, xs] += 1.0

    if sigma is not None and float(sigma) > 0:
        # kernel nhỏ hơn để đỡ "bệt"
        k = int(max(3, round(float(sigma) * 4)))
        kernel = _gaussian_kernel2d(k, float(sigma), device=device)
        pad = kernel.shape[-1] // 2
        density = torch.nn.functional.conv2d(density, kernel, padding=pad)

    return density[0, 0]


def overlay_density_on_pil(
    image: Image.Image,
    density: np.ndarray,
    alpha: float,
    pctl: float = 99.0,
    gamma: float = 0.45,
) -> Image.Image:
    """
    Overlay heatmap rõ hơn:
    - scale theo percentile (pctl)
    - gamma < 1 để làm sáng vùng đỉnh
    """
    img = image.convert("RGB")
    dens = density.astype(np.float32)

    if dens.size == 0:
        return img

    scale = float(np.percentile(dens, pctl)) if np.any(dens > 0) else 0.0
    if scale <= 1e-8:
        return img

    d = np.clip(dens / scale, 0.0, 1.0)
    d = np.power(d, float(gamma))  # gamma < 1 => đỉnh sáng hơn

    heat = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (d * 255).astype(np.uint8)
    heat[..., 1] = (d * 40).astype(np.uint8)
    heat[..., 2] = (d * 10).astype(np.uint8)

    heat_img = Image.fromarray(heat, mode="RGB")
    return Image.blend(img, heat_img, alpha=float(alpha))


def draw_centers_on_pil(img: Image.Image, centers_norm: np.ndarray, r: int = 2) -> Image.Image:
    out = img.copy().convert("RGB")
    W, H = out.size
    dr = ImageDraw.Draw(out)
    for cx, cy in centers_norm:
        x = int(np.clip(cx * (W - 1), 0, W - 1))
        y = int(np.clip(cy * (H - 1), 0, H - 1))
        dr.ellipse((x - r, y - r, x + r, y + r), outline="yellow", width=2)
        dr.ellipse((x - 1, y - 1, x + 1, y + 1), fill="yellow")
    return out


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


def _safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        if os.path.isfile(path):
            return Image.open(path).convert("RGB")
    except Exception:
        return None
    return None


def _fallback_tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    x = image_tensor.detach().cpu()
    x = x.clamp(-3, 3)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x).convert("RGB")


# =========================
# Args / Data / Model
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Save GDCount density overlay images (FSC147)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model-ckpt", type=str, required=True)

    p.add_argument("--ann", type=str, required=True)
    p.add_argument("--img-root", type=str, required=True)
    p.add_argument("--split-file", type=str, required=True)
    p.add_argument("--class-map", type=str, required=True)

    p.add_argument("--splits", type=str, nargs="+", default=["test"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)

    p.add_argument("--threshold", type=float, default=0.23)
    p.add_argument("--nms-iou", type=float, default=0.5)

    p.add_argument("--soa-level", type=int, default=-1)
    p.add_argument("--freeze-keywords", type=str, nargs="+", default=["backbone.0", "bert"])
    p.add_argument("--no-soa", action="store_true")

    p.add_argument("--out-dir", type=str, default="test_prediction")

    # density rendering
    p.add_argument("--sigma", type=float, default=1.5)   # giảm mặc định để rõ tâm
    p.add_argument("--alpha", type=float, default=0.75)  # tăng blend
    p.add_argument("--pctl", type=float, default=99.0)   # percentile normalize
    p.add_argument("--gamma", type=float, default=0.45)  # gamma < 1 làm sáng

    p.add_argument("--max-images", type=int, default=-1, help="-1 = lưu tất cả")
    p.add_argument("--no-write-text", action="store_true", help="Không vẽ gt/pred lên ảnh")
    p.add_argument("--draw-centers", action="store_true", default=True, help="Vẽ chấm tâm (default: bật)")
    p.add_argument("--center-r", type=int, default=2, help="Bán kính chấm tâm")
    p.add_argument("--per-split-subdir", action="store_true", help="Lưu theo out-dir/<split>/...")

    return p.parse_args()


def create_dataloader(args: argparse.Namespace, split: str) -> DataLoader:
    ds = FSC147Dataset(
        ann_path=args.ann,
        img_root=args.img_root,
        split=split,
        split_file=args.split_file,
        class_map_file=args.class_map,
        img_size=None,
        normalize=True,
        density_root=None,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=fsc147_collate,
    )


def create_model(args: argparse.Namespace, device: str) -> torch.nn.Module:
    cfg = GDCountConfig(
        threshold=args.threshold,
        soa_level=None if args.no_soa else args.soa_level,
        feature_dim=256,
        freeze_keywords=args.freeze_keywords,
    )
    model = build_gdcount_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        gdcount_cfg=cfg,
    )
    return model


# =========================
# Main save loop
# =========================

@torch.no_grad()
def run_one_split(args: argparse.Namespace, split: str, model: torch.nn.Module, device: str) -> int:
    loader = create_dataloader(args, split)
    saved = 0

    out_dir = args.out_dir
    if args.per_split_subdir:
        out_dir = os.path.join(out_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    for batch in tqdm_mod.tqdm(loader, desc=f"Save [{split}]", ncols=120):
        images = batch["images"].to(device)
        prompts = [sanitize_caption(x) for x in batch["prompts"]]
        gt_counts = batch["gt_counts"].to(device)
        image_ids: List[str] = batch["meta"]["image_ids"]

        with autocast(enabled=(args.amp and str(device).startswith("cuda"))):
            outputs: Dict[str, Any] = model(images, captions=prompts)
            outputs.pop("hard_counts", None)

        B = images.shape[0]
        for bi in range(B):
            if args.max_images is not None and int(args.max_images) >= 0 and saved >= int(args.max_images):
                return saved

            img_id = image_ids[bi]
            gt = int(gt_counts[bi].item())

            boxes_xyxy_norm, _ = pick_boxes_after_thresh_nms_one(
                outputs, b=bi, threshold=float(args.threshold), nms_iou=float(args.nms_iou)
            )
            pred = int(boxes_xyxy_norm.shape[0])

            img_path = os.path.join(args.img_root, img_id)
            pil = _safe_open_image(img_path)
            if pil is None:
                pil = _fallback_tensor_to_pil(images[bi])

            W, H = pil.size

            centers_norm = _boxes_to_centers_norm(boxes_xyxy_norm)
            density_t = centers_to_density_map(
                centers_norm=centers_norm,
                out_h=H,
                out_w=W,
                sigma=float(args.sigma),
                device=torch.device(device),
            )
            density = density_t.detach().cpu().numpy()

            vis = overlay_density_on_pil(
                pil, density,
                alpha=float(args.alpha),
                pctl=float(args.pctl),
                gamma=float(args.gamma),
            )

            if args.draw_centers and centers_norm.numel() > 0:
                vis = draw_centers_on_pil(vis, centers_norm.detach().cpu().numpy(), r=int(args.center_r))

            if not args.no_write_text:
                dr = ImageDraw.Draw(vis)
                dr.text((8, 8), f"gt={gt} pred={pred}", fill=(255, 255, 255))

            stem = os.path.splitext(os.path.basename(img_id))[0]
            out_name = f"gt-{gt}_pred-{pred}_{stem}.jpg"
            out_path = os.path.join(out_dir, out_name)
            vis.save(out_path, quality=95)

            saved += 1

    return saved


def main():
    args = parse_args()

    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    print(f"Using device: {device}")

    if not os.path.isfile(args.model_ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_ckpt}")

    model = create_model(args, device=device).to(device)
    meta = load_model_checkpoint(args.model_ckpt, model, device=device)
    model.eval()
    print(f"Loaded trained ckpt: {args.model_ckpt} (epoch={meta.get('epoch', None)})")

    total_saved = 0
    for split in args.splits:
        n = run_one_split(args, split=split, model=model, device=device)
        total_saved += n
        print(f"[{split}] saved: {n} images -> {args.out_dir}")

    print(f"Done. Total saved: {total_saved}")


if __name__ == "__main__":
    main()
