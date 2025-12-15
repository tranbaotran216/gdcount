# app.py
# Streamlit demo cho GDCount (FSC147-style): upload ảnh + prompt -> dự đoán count
# KHÔNG cho phép vẽ exemplar
# Chạy: streamlit run app.py

import os
import re
import time
from typing import Any, Dict, Tuple, Optional

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

import torch

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

from models.gdcount_model_calib import GDCountConfig, build_gdcount_model


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


def _get_scores(outputs: Dict[str, Any]) -> torch.Tensor:
    """
    Priority: calib_scores nếu có (calibrator)
    Fallback: sigmoid(max_token_logit) như bản cũ
    """
    if "calib_scores" in outputs and isinstance(outputs["calib_scores"], torch.Tensor):
        return outputs["calib_scores"].float()

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


def _pick_boxes_after_thresh_nms(
    outputs: Dict[str, Any],
    threshold: float,
    nms_iou: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về (boxes_xyxy_norm (K,4), scores (K,))
    """
    scores = _get_scores(outputs)[0]  # (Q,)
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


def draw_boxes_on_pil(
    image: Image.Image,
    boxes_xyxy_norm: np.ndarray,
    scores: Optional[np.ndarray] = None,
    score_threshold_to_show: float = 0.0
) -> Image.Image:
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
                dr.text((x1 + 3, y1 + 3), f"{sc:.2f}")

    return img


def load_model_checkpoint(model_ckpt_path: str, model: torch.nn.Module, device: str) -> Dict[str, Any]:
    ckpt = torch.load(model_ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format: {model_ckpt_path}")

    # support resume từ ckpt cũ chưa có calibrator
    missing, unexpected = model.load_state_dict(state, strict=False)

    meta: Dict[str, Any] = {}
    if isinstance(ckpt, dict):
        meta["epoch"] = ckpt.get("epoch", None)
    meta["missing_keys"] = missing
    meta["unexpected_keys"] = unexpected
    return meta


def preprocess_image_for_model(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((384, 384), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    arr = arr.transpose(2, 0, 1)  # (3,H,W)
    x = torch.from_numpy(arr)

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

    default_config = r"C:\Users\PC\Documents\college\CV\gdcount\groundingdino\groundingdino\config\GroundingDINO_SwinT_OGC.py"
    default_gdino_ckpt = r"C:\Users\PC\Documents\college\CV\gdcount\weights\groundingdino_swint_ogc.pth"
    default_model_ckpt = r"C:\Users\PC\Documents\college\CV\gdcount\checkpoints_gdcount\text_exemplar_calib\best\gdcount_epoch_013_best.pth"

    config_path = st.text_input("GroundingDINO config", value=default_config)
    gdino_ckpt_path = st.text_input("GroundingDINO checkpoint", value=default_gdino_ckpt)
    model_ckpt_path = st.text_input("GDCount trained checkpoint", value=default_model_ckpt)

    st.divider()

    threshold = st.slider("Threshold", 0.0, 1.0, 0.23, 0.01)
    nms_iou = st.slider("NMS IoU", 0.0, 1.0, 0.50, 0.01)

    show_boxes = st.checkbox("Hiển thị bounding boxes (sau threshold+NMS)", value=True)
    show_scores = st.checkbox("Hiển thị score trên box", value=False)

    st.divider()

    prompt = st.text_input("Prompt", value="object")
    run_btn = st.button("Chạy đếm", type="primary")


@st.cache_resource(show_spinner=True)
def load_model_cached(config_path: str, gdino_ckpt_path: str, model_ckpt_path: str, device: str, threshold: float):
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
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Ảnh gốc", use_container_width=True)
    else:
        img = None

with col_right:
    st.subheader("Kết quả")
    if img is None:
        st.info("Upload ảnh để bắt đầu.")
    elif run_btn:
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
            model, meta = load_model_cached(config_path, gdino_ckpt_path, model_ckpt_path, device, threshold)

        cap = sanitize_caption(prompt)
        x = preprocess_image_for_model(img).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs: Dict[str, Any] = model(x, captions=[cap])
        dt = (time.time() - t0) * 1000.0

        boxes_t, scores_t = _pick_boxes_after_thresh_nms(outputs, threshold=threshold, nms_iou=nms_iou)
        pred_count = int(boxes_t.shape[0])

        st.metric("Predicted count", pred_count)
        st.write(
            f"- Prompt: `{cap}`\n"
            f"- Device: `{device}`\n"
            f"- Checkpoint epoch: `{meta.get('epoch', '')}`\n"
            f"- Inference time: `{dt:.1f} ms`"
        )

        if meta.get("missing_keys"):
            with st.expander("Checkpoint warning (missing/unexpected keys)"):
                st.write("missing_keys:", meta.get("missing_keys"))
                st.write("unexpected_keys:", meta.get("unexpected_keys"))

        if show_boxes:
            boxes_np = boxes_t.detach().cpu().numpy()
            scores_np = scores_t.detach().cpu().numpy() if scores_t is not None else None

            vis = draw_boxes_on_pil(
                image=img.resize((384, 384), Image.BILINEAR),
                boxes_xyxy_norm=boxes_np,
                scores=scores_np if show_scores else None,
                score_threshold_to_show=0.0,
            )
            st.image(vis, caption="Ảnh 384×384 + boxes (threshold + NMS)", use_container_width=True)

        with st.expander("Debug (tensors)"):
            st.write("outputs keys:", list(outputs.keys()))
            if "pred_boxes" in outputs:
                st.write("pred_boxes shape:", tuple(outputs["pred_boxes"].shape))
            if "pred_logits" in outputs:
                st.write("pred_logits shape:", tuple(outputs["pred_logits"].shape))
            if "calib_scores" in outputs:
                st.write("calib_scores shape:", tuple(outputs["calib_scores"].shape))
            st.write("kept boxes:", int(boxes_t.shape[0]))
