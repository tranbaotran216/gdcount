# app.py
# Streamlit demo cho GDCount (FSC147-style): upload ảnh + prompt -> dự đoán count
# Chạy: streamlit run app.py

import os
import re
import time
from typing import Any, Dict, Tuple, Optional

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

# ===== Import từ project của bạn (đảm bảo đúng path/module) =====
# Các module này đang được dùng trong train/test của bạn.
from models.gdcount_model import GDCountConfig, build_gdcount_model


# =========================
# Helpers (giống train/test)
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


def _scores_from_pred_logits(outputs: Dict[str, Any]) -> torch.Tensor:
    """
    scores (B,Q) = sigmoid(max_token_logit over valid tokens) hoặc sigmoid(logit) nếu 2D.
    """
    logits = outputs["pred_logits"].float()  # (B,Q,T) hoặc (B,Q)

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
        specials = (ids == 0) | (ids == 101) | (ids == 102)
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


def _pick_boxes_after_thresh_nms(
    outputs: Dict[str, Any],
    threshold: float,
    nms_iou: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về (boxes_xyxy_norm (K,4), scores (K,))
    """
    scores = _scores_from_pred_logits(outputs)[0]  # (Q,)
    keep = scores > threshold

    if "pred_boxes" not in outputs:
        idx = keep.nonzero(as_tuple=False).flatten()
        return torch.zeros((0, 4), device=scores.device), scores[idx]

    boxes = outputs["pred_boxes"].float()[0]  # (Q,4) cxcywh norm
    boxes_xyxy = _cxcywh_to_xyxy(boxes).clamp(0, 1)

    idx = keep.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.zeros((0, 4), device=scores.device), torch.zeros((0,), device=scores.device)

    b = boxes_xyxy[idx]
    s = scores[idx]

    if tv_nms is None or idx.numel() == 1:
        return b, s

    kept = tv_nms(b, s, nms_iou)
    return b[kept], s[kept]

def _boxes_to_centers_norm(boxes_xyxy_norm: torch.Tensor) -> torch.Tensor:
    """
    boxes_xyxy_norm: (K,4) in [0,1]
    return centers_norm: (K,2) in [0,1]
    """
    if boxes_xyxy_norm is None or boxes_xyxy_norm.numel() == 0:
        return torch.zeros((0, 2), device=boxes_xyxy_norm.device if boxes_xyxy_norm is not None else "cpu")
    x1, y1, x2, y2 = boxes_xyxy_norm.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return torch.stack([cx, cy], dim=-1).clamp(0, 1)


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: str) -> torch.Tensor:
    """
    return kernel: (1,1,K,K) normalized sum=1
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = kernel_size
    ax = torch.arange(k, device=device) - (k // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, k, k)


def centers_to_density_map(
    centers_norm: torch.Tensor,
    out_h: int = 384,
    out_w: int = 384,
    sigma: float = 3.0,
    kernel_size: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    centers_norm: (K,2) normalized
    return density: (H,W) float32
    """
    density = torch.zeros((1, 1, out_h, out_w), device=device, dtype=torch.float32)

    if centers_norm is None or centers_norm.numel() == 0:
        return density[0, 0]

    xs = (centers_norm[:, 0] * (out_w - 1)).round().long().clamp(0, out_w - 1)
    ys = (centers_norm[:, 1] * (out_h - 1)).round().long().clamp(0, out_h - 1)

    # đặt 1 tại tâm (impulse)
    density[0, 0, ys, xs] += 1.0

    # gaussian blur
    if sigma > 0:
        if kernel_size <= 0:
            kernel_size = int(max(3, round(sigma * 6)))  # ~6*sigma
        kernel = _gaussian_kernel2d(kernel_size, sigma, device=device)
        pad = kernel.shape[-1] // 2
        density = torch.nn.functional.conv2d(density, kernel, padding=pad)

    return density[0, 0]


def draw_boxes_on_pil(
    image: Image.Image,
    boxes_xyxy_norm: np.ndarray,
    scores: Optional[np.ndarray] = None,
    score_threshold_to_show: float = 0.0
) -> Image.Image:
    """
    Vẽ box lên ảnh (PIL). boxes là normalized xyxy.
    """
    img = image.copy().convert("RGB")
    W, H = img.size
    dr = ImageDraw.Draw(img)

    for i, box in enumerate(boxes_xyxy_norm):
        x1 = int(max(0, min(W - 1, box[0] * W)))
        y1 = int(max(0, min(H - 1, box[1] * H)))
        x2 = int(max(0, min(W - 1, box[2] * W)))
        y2 = int(max(0, min(H - 1, box[3] * H)))

        dr.rectangle([x1, y1, x2, y2], width=2)

        if scores is not None:
            sc = float(scores[i])
            if sc >= score_threshold_to_show:
                txt = f"{sc:.2f}"
                dr.text((x1 + 3, y1 + 3), txt)

    return img

def overlay_density_on_image(
    image_384: Image.Image,
    density: np.ndarray,
    alpha: float = 0.55
) -> Image.Image:
    """
    image_384: PIL RGB 384x384
    density: (384,384) float
    """
    img = image_384.convert("RGB")
    dens = density.astype(np.float32)

    # normalize 0..1 để hiển thị
    dmax = float(dens.max()) if dens.size else 0.0
    if dmax > 1e-8:
        dens = dens / dmax
    dens = np.clip(dens, 0.0, 1.0)

    # pseudo-color đơn giản: đỏ = dens, xanh/lam thấp
    heat = np.zeros((dens.shape[0], dens.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (dens * 255).astype(np.uint8)          # R
    heat[..., 1] = (dens * 120).astype(np.uint8)          # G
    heat[..., 2] = (dens * 30).astype(np.uint8)           # B

    heat_img = Image.fromarray(heat, mode="RGB")
    return Image.blend(img, heat_img, alpha=alpha)


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


def preprocess_image_for_model(img: Image.Image) -> torch.Tensor:
    """
    Chuẩn hóa kiểu FSC147 pipeline thường dùng:
    - Input của dataset bạn đang dùng đã normalize (mean/std ImageNet) và resize 384.
    Ở đây, mình làm theo cách an toàn:
    - Resize về 384 (nếu ảnh không phải 384)
    - ToTensor
    - Normalize ImageNet
    """
    img = img.convert("RGB")
    img = img.resize((384, 384), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    arr = arr.transpose(2, 0, 1)  # (3,H,W)
    x = torch.from_numpy(arr)

    # ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std

    return x  # (3,384,384)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="GDCount Streamlit", layout="wide")
st.title("GDCount – Demo đếm theo prompt (FSC147)")

with st.sidebar:
    st.header("Cấu hình")

    device_choice = st.selectbox("Device", ["cuda", "cpu"], index=0)
    device = device_choice if (device_choice == "cpu" or torch.cuda.is_available()) else "cpu"

    # Default path theo đúng bạn đưa
    default_config = "./weights/GroundingDINO_SwinT_OGC.cfg.py"
    default_gdino_ckpt = "./weights/groundingdino_swint_ogc.pth"
    default_model_ckpt = "./checkpoints/gdcount_epoch_011_best_fp16.pth"

    config_path = st.text_input("GroundingDINO config", value=default_config)
    gdino_ckpt_path = st.text_input("GroundingDINO checkpoint", value=default_gdino_ckpt)
    model_ckpt_path = st.text_input("GDCount trained checkpoint", value=default_model_ckpt)

    st.divider()

    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.23, step=0.01)
    nms_iou = st.slider("NMS IoU", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    show_density = st.checkbox("Hiển thị density map (từ tâm bbox)", value=True)

    sigma = 3.0
    alpha = 0.55
    if show_density:
        sigma = st.slider("Sigma (Gaussian blur)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
        alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

    show_scores = st.checkbox("Hiển thị score trên box", value=False)  # chỉ dùng khi show_density=False

    st.divider()

    prompt = st.text_input("Prompt", value="object")
    run_btn = st.button("Chạy đếm", type="primary")


@st.cache_resource(show_spinner=True)
def load_model_cached(
    config_path: str,
    gdino_ckpt_path: str,
    model_ckpt_path: str,
    device: str,
    threshold: float
):
    gd_cfg = GDCountConfig(
        threshold=threshold,
        soa_level=-1,
        feature_dim=256,
        freeze_keywords=["backbone.0", "bert"],
    )
    model = build_gdcount_model(
        config_path=config_path,
        checkpoint_path=gdino_ckpt_path,
        device=device,
        gdcount_cfg=gd_cfg,
    )
    meta = load_model_checkpoint(model_ckpt_path, model, device)
    model.eval()
    return model, meta


col_left, col_right = st.columns([1, 1])

with col_left:
    up = st.file_uploader("Upload ảnh (jpg/png)", type=["jpg", "jpeg", "png"])

    if up is not None:
        img = Image.open(up)
        st.image(img, caption="Ảnh gốc",width='stretch')
    else:
        img = None

with col_right:
    st.subheader("Kết quả")
    if img is None:
        st.info("Upload ảnh để bắt đầu.")
    else:
        if run_btn:
            if not os.path.isfile(config_path):
                st.error(f"Không tìm thấy config: {config_path}")
                st.stop()
            if not os.path.isfile(gdino_ckpt_path):
                st.error(f"Không tìm thấy GroundingDINO ckpt: {gdino_ckpt_path}")
                st.stop()
            if not os.path.isfile(model_ckpt_path):
                st.error(f"Không tìm thấy GDCount ckpt: {model_ckpt_path}")
                st.stop()

            with st.spinner("Đang load model (lần đầu có thể lâu)..."):
                model, meta = load_model_cached(
                    config_path=config_path,
                    gdino_ckpt_path=gdino_ckpt_path,
                    model_ckpt_path=model_ckpt_path,
                    device=device,
                    threshold=threshold,
                )

            cap = sanitize_caption(prompt)

            x = preprocess_image_for_model(img).unsqueeze(0)  # (1,3,384,384)
            x = x.to(device)

            t0 = time.time()
            with torch.no_grad():
                outputs: Dict[str, Any] = model(x, captions=[cap])
            dt = (time.time() - t0) * 1000.0

            boxes_t, scores_t = _pick_boxes_after_thresh_nms(outputs, threshold=threshold, nms_iou=nms_iou)
            pred_count = int(boxes_t.shape[0]) if boxes_t is not None else 0

            st.metric("Predicted count", pred_count)
            st.write(
                f"- Prompt: `{cap}`\n"
                f"- Device: `{device}`\n"
                f"- Checkpoint epoch: `{meta.get('epoch', '')}`\n"
                f"- Inference time: `{dt:.1f} ms`"
            )

            if show_density:
                # centers -> density
                centers_norm = _boxes_to_centers_norm(boxes_t)  # (K,2)
                density_t = centers_to_density_map(
                    centers_norm=centers_norm.to(device),
                    out_h=384, out_w=384,
                    sigma=float(sigma),
                    device=device,
                )
                density = density_t.detach().cpu().numpy()

                img_384 = img.resize((384, 384), Image.BILINEAR)
                vis = overlay_density_on_image(img_384, density, alpha=float(alpha))
                st.image(vis, caption="Ảnh 384×384 + density map (từ tâm bbox)", width='stretch')

                with st.expander("Density map (raw)"):
                    st.image(density / (density.max() + 1e-8), caption="density (normalized)", width='stretch')

            else:
                # hiển thị bbox như cũ
                boxes_np = boxes_t.detach().cpu().numpy() if boxes_t is not None else np.zeros((0, 4), dtype=np.float32)
                scores_np = scores_t.detach().cpu().numpy() if scores_t is not None else None

                vis = draw_boxes_on_pil(
                    image=img.resize((384, 384), Image.BILINEAR),
                    boxes_xyxy_norm=boxes_np,
                    scores=scores_np if show_scores else None,
                    score_threshold_to_show=0.0,
                )
                st.image(vis, caption="Ảnh 384×384 + boxes (sau threshold + NMS)", width='stretch')
