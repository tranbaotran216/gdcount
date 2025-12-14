import os
import re
import time
from typing import Any, Dict, Tuple, Optional, List

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

import torch

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

from streamlit_drawable_canvas import st_canvas
from models.gdcount_model import GDCountConfig, build_gdcount_model



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

    model.load_state_dict(state, strict=True)
    meta = {}
    if isinstance(ckpt, dict):
        meta["epoch"] = ckpt.get("epoch", None)
    return meta


def preprocess_image_for_model(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((384, 384), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    arr = arr.transpose(2, 0, 1)  # (3,H,W)
    x = torch.from_numpy(arr)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std

    return x  # (3,384,384)


def canvas_rects_to_exemplars(canvas_json: Optional[Dict[str, Any]], max_k: int = 3) -> np.ndarray:
    """
    Convert objects from st_canvas to xyxy pixel boxes in 384x384 space.
    Return (K,4) float32, K<=max_k.
    """
    if not canvas_json:
        return np.zeros((0, 4), dtype=np.float32)

    objs = canvas_json.get("objects", []) or []
    rects: List[np.ndarray] = []

    for o in objs:
        # chỉ nhận rect
        if o.get("type") != "rect":
            continue

        left = float(o.get("left", 0.0))
        top = float(o.get("top", 0.0))
        w = float(o.get("width", 0.0))
        h = float(o.get("height", 0.0))

        # fabric.js có thể có scaleX/scaleY
        sx = float(o.get("scaleX", 1.0))
        sy = float(o.get("scaleY", 1.0))

        x1 = left
        y1 = top
        x2 = left + w * sx
        y2 = top + h * sy

        # clamp
        x1 = max(0.0, min(383.0, x1))
        y1 = max(0.0, min(383.0, y1))
        x2 = max(0.0, min(383.0, x2))
        y2 = max(0.0, min(383.0, y2))

        # đảm bảo x1<x2, y1<y2
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        rects.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    if len(rects) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    rects = rects[:max_k]  # lấy tối đa 3 theo thứ tự vẽ
    return np.stack(rects, axis=0)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="GDCount Streamlit", layout="wide")
st.title("GDCount – Demo đếm theo prompt (FSC147)")

if "exemplar_boxes" not in st.session_state:
    st.session_state.exemplar_boxes = np.zeros((0, 4), dtype=np.float32)

with st.sidebar:
    st.header("Cấu hình")

    device_choice = st.selectbox("Device", ["cuda", "cpu"], index=0)
    device = device_choice if (device_choice == "cpu" or torch.cuda.is_available()) else "cpu"

    default_config = r"C:\Users\PC\Documents\college\CV\gdcount\groundingdino\groundingdino\config\GroundingDINO_SwinT_OGC.py"
    default_gdino_ckpt = r"C:\Users\PC\Documents\college\CV\gdcount\weights\groundingdino_swint_ogc.pth"
    default_model_ckpt = r"C:\Users\PC\Documents\college\CV\gdcount\checkpoints_gdcount\text_exemplar\best\gdcount_epoch_026_best.pth"

    config_path = st.text_input("GroundingDINO config", value=default_config)
    gdino_ckpt_path = st.text_input("GroundingDINO checkpoint", value=default_gdino_ckpt)
    model_ckpt_path = st.text_input("GDCount trained checkpoint", value=default_model_ckpt)

    st.divider()

    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.23, step=0.01)
    nms_iou = st.slider("NMS IoU", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    show_boxes = st.checkbox("Hiển thị bbox output (sau threshold+NMS)", value=True)
    show_scores = st.checkbox("Hiển thị score trên bbox output", value=False)

    st.divider()

    prompt = st.text_input("Prompt", value="object")
    use_user_exemplars = st.checkbox("Dùng exemplars do người dùng vẽ (tối đa 3)", value=True)

    colb1, colb2 = st.columns(2)
    with colb1:
        clear_ex = st.button("Xoá exemplars")
    with colb2:
        run_btn = st.button("Chạy đếm", type="primary")

    if clear_ex:
        st.session_state.exemplar_boxes = np.zeros((0, 4), dtype=np.float32)


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
        img = Image.open(up).convert("RGB")
        img_384 = img.resize((384, 384), Image.BILINEAR)

        st.subheader("Vẽ exemplars (tối đa 3 box)")
        st.caption("Kéo chuột để vẽ hình chữ nhật. Chỉ lấy tối đa 3 rect theo thứ tự bạn vẽ. Không bắt buộc đủ 3.")

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=img_384,
            update_streamlit=True,
            height=384,
            width=384,
            drawing_mode="rect",
            key="canvas_exemplars",
        )

        ex_boxes = canvas_rects_to_exemplars(canvas.json_data, max_k=3)
        st.session_state.exemplar_boxes = ex_boxes

        st.write(f"Exemplars hiện tại: **{ex_boxes.shape[0]}** / 3")
        if ex_boxes.shape[0] > 0:
            st.code(ex_boxes.astype(int))

        st.image(img, caption="Ảnh gốc", use_container_width=True)
    else:
        img = None
        img_384 = None

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

            x = preprocess_image_for_model(img).unsqueeze(0).to(device)  # (1,3,384,384)

            exemplars_tensor = None
            labels_tensor = None
            if use_user_exemplars:
                ex = st.session_state.exemplar_boxes
                if ex is not None and ex.shape[0] > 0:
                    exemplars_tensor = torch.from_numpy(ex).to(device)  # (K,4) xyxy pixel (384-space)
                    labels_tensor = torch.zeros((ex.shape[0],), dtype=torch.long, device=device)  # phrase index 0

            t0 = time.time()
            with torch.no_grad():
                if exemplars_tensor is not None:
                    outputs: Dict[str, Any] = model(
                        x,
                        captions=[cap],
                        exemplars=[exemplars_tensor],     # list per-image
                        labels=[labels_tensor],           # list per-image
                    )
                else:
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

            if exemplars_tensor is not None:
                st.write(f"- User exemplars: **{int(exemplars_tensor.shape[0])}** box(es)")

            if show_boxes:
                boxes_np = boxes_t.detach().cpu().numpy() if boxes_t is not None else np.zeros((0, 4), dtype=np.float32)
                scores_np = scores_t.detach().cpu().numpy() if scores_t is not None else None

                vis = draw_boxes_on_pil(
                    image=img.resize((384, 384), Image.BILINEAR),
                    boxes_xyxy_norm=boxes_np,
                    scores=scores_np if show_scores else None,
                    score_threshold_to_show=0.0,
                )
                st.image(vis, caption="Ảnh 384×384 + bbox output (threshold + NMS)", use_container_width=True)

            with st.expander("Debug (tensors)"):
                st.write("outputs keys:", list(outputs.keys()))
                if "pred_boxes" in outputs:
                    st.write("pred_boxes shape:", tuple(outputs["pred_boxes"].shape))
                if "pred_logits" in outputs:
                    st.write("pred_logits shape:", tuple(outputs["pred_logits"].shape))
                st.write("kept boxes:", int(boxes_t.shape[0]))
