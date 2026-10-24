# app_exp3.py
# Streamlit Demo for GDCount + Exp3 Image-Adaptive Prototype Refinement
# Features: Anti-Hallucination (Absent-Query Gate) & Multi-Object Competitor Filtering
# Checkpoint default: gdcount_epoch_013_best.pth (Calibrated GDCount)
# Run: streamlit run app_exp3.py

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import torch
import torch.nn.functional as F

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

from models.gdcount_model_calib import GDCountConfig, build_gdcount_model
from models.exp3_postprocessor import Exp3Config, Exp3PostProcessor, PrototypeBankLite
from models.feature_extractor import CandidateFeatureExtractor


st.set_page_config(
    page_title="GDCount + Exp3 (Anti-Hallucination & Adaptive ZSC)",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .badge-present {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-absent {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    """Extract prediction scores from calibrated head or token logits."""
    if "calib_scores" in outputs and isinstance(outputs["calib_scores"], torch.Tensor):
        return outputs["calib_scores"].float()

    logits = outputs["pred_logits"].float()
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
    per_q = logits.max(dim=-1).values
    return torch.sigmoid(per_q)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def pick_raw_candidates(
    outputs: Dict[str, Any],
    threshold: float,
    nms_iou: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract raw candidate bounding boxes and scores after initial threshold + NMS."""
    scores = _get_scores(outputs)[0]  # (Q,)
    keep = scores > threshold

    if "pred_boxes" not in outputs or tv_nms is None:
        idx = keep.nonzero(as_tuple=False).flatten()
        return np.zeros((len(idx), 4), dtype=np.float32), scores[idx].detach().cpu().numpy()

    boxes_cxcywh = outputs["pred_boxes"][0].float()
    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh).clamp(0.0, 1.0)

    idx = keep.nonzero(as_tuple=False).flatten()
    if len(idx) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    b = boxes_xyxy[idx]
    s = scores[idx]

    nms_idx = tv_nms(b, s, iou_threshold=nms_iou)
    return b[nms_idx].detach().cpu().numpy(), s[nms_idx].detach().cpu().numpy()


def preprocess_image_for_model(img_pil: Image.Image) -> torch.Tensor:
    img = img_pil.convert("RGB").resize((384, 384), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - mean) / std


def draw_colored_boxes_on_pil(
    image: Image.Image,
    diagnostics: List[Dict[str, Any]],
    show_labels: bool = True,
) -> Image.Image:
    """Draw bounding boxes color-coded by their Exp3 verification status."""
    img = image.copy().convert("RGB")
    W, H = img.size
    dr = ImageDraw.Draw(img)

    # Color scheme
    color_map = {
        "kept": "#10B981",                   # Emerald Green: Valid target
        "rejected_competitor_class": "#EF4444", # Red: Co-present competitor class
        "rejected_low_adapt_sim": "#F59E0B",  # Amber: Visual similarity below adapt threshold
        "rejected_low_conf": "#9CA3AF",      # Gray: Low detector score
        "rejected_target_absent": "#DC2626", # Deep Red: Target absent
    }

    for d in diagnostics:
        box = d["box"]
        status = d.get("status", "kept")
        color = color_map.get(status, "#3B82F6")

        x1 = int(np.clip(box[0] * W if max(box) <= 1.0 else box[0], 0, W - 1))
        y1 = int(np.clip(box[1] * H if max(box) <= 1.0 else box[1], 0, H - 1))
        x2 = int(np.clip(box[2] * W if max(box) <= 1.0 else box[2], 0, W - 1))
        y2 = int(np.clip(box[3] * H if max(box) <= 1.0 else box[3], 0, H - 1))

        width = 3 if status == "kept" else 2
        dr.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if show_labels:
            if status == "kept":
                label = f"ID:{d['candidate_id']} ({d.get('s_adapt', d.get('score', 0)):.2f})"
            elif status == "rejected_competitor_class":
                label = f"X: {d.get('competitor', 'other')}"
            else:
                label = f"X: {d.get('score', 0):.2f}"
            dr.text((x1 + 3, max(0, y1 - 12)), label, fill=color)

    return img


def centers_to_density_map(
    centers_norm: torch.Tensor,
    out_h: int = 384,
    out_w: int = 384,
    sigma: float = 3.0,
    device: str = "cpu",
) -> torch.Tensor:
    if centers_norm is None or centers_norm.shape[0] == 0:
        return torch.zeros((out_h, out_w), device=device, dtype=torch.float32)

    cx = (centers_norm[:, 0] * (out_w - 1)).clamp(0, out_w - 1)
    cy = (centers_norm[:, 1] * (out_h - 1)).clamp(0, out_h - 1)

    yy, xx = torch.meshgrid(
        torch.arange(out_h, device=device, dtype=torch.float32),
        torch.arange(out_w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    density = torch.zeros((out_h, out_w), device=device, dtype=torch.float32)
    s2 = 2.0 * (sigma ** 2)
    inv_norm = 1.0 / (2.0 * np.pi * (sigma ** 2))

    for x_i, y_i in zip(cx, cy):
        dist_sq = (xx - x_i) ** 2 + (yy - y_i) ** 2
        density += torch.exp(-dist_sq / s2) * inv_norm

    return density


def overlay_density_on_image(img_pil: Image.Image, density_np: np.ndarray, alpha: float = 0.55) -> Image.Image:
    import matplotlib.cm as cm
    d_norm = density_np / (np.max(density_np) + 1e-8)
    cmap = cm.get_cmap("jet")
    colored = cmap(d_norm)[:, :, :3]
    colored = (colored * 255.0).astype(np.uint8)

    base = img_pil.convert("RGB")
    overlay = Image.fromarray(colored)
    return Image.blend(base, overlay, alpha=float(alpha))


# ==========================================
# CACHED RESOURCE LOADERS
# ==========================================

@st.cache_resource(show_spinner="Đang khởi tạo GDCount Model (Checkpoint 013 Best)...")
def load_gdcount_model(
    config_path: str,
    gdino_ckpt_path: str,
    model_ckpt_path: str,
    device: str,
):
    gd_cfg = GDCountConfig(
        threshold=0.0,
        soa_level=-1,
        feature_dim=256,
        freeze_keywords=["backbone.0", "bert"],
        unfreeze_bert_last_n=0,
        use_exemplar_mod=True,
    )
    model = build_gdcount_model(
        config_path=config_path,
        checkpoint_path=gdino_ckpt_path,
        device=device,
        gdcount_cfg=gd_cfg,
    )

    if os.path.isfile(model_ckpt_path):
        ckpt = torch.load(model_ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Filter matching keys
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        epoch = ckpt.get("epoch", "13") if isinstance(ckpt, dict) else "unknown"
    else:
        epoch = "Not found"

    model.eval()
    return model, epoch


@st.cache_resource(show_spinner="Đang nạp Prototype Bank & Feature Extractor...")
def load_prototype_bank_and_extractor(
    prototype_cache_path: str,
    dinov3_repo_path: str,
    dinov3_weights_path: str,
    device: str,
):
    bank = None
    if os.path.isdir(prototype_cache_path):
        try:
            bank = PrototypeBankLite.from_cache_dir(prototype_cache_path)
            st.toast(f"✅ Đã nạp Prototype Bank ({bank.num_classes} classes, dim {bank.embedding_dim})")
        except Exception as e:
            st.toast(f"⚠️ Không tải được Prototype Bank: {e}")

    extractor = CandidateFeatureExtractor(
        dinov3_repo_path=dinov3_repo_path if os.path.isdir(dinov3_repo_path) else None,
        dinov3_weights_path=dinov3_weights_path if os.path.isfile(dinov3_weights_path) else None,
        device=device,
    )
    return bank, extractor


# ==========================================
# MAIN APPLICATION UI
# ==========================================

st.title("🎯 GDCount + Exp3: Image-Adaptive Anti-Hallucination Counting")
st.caption(
    "Zero-Shot Object Counting nâng cấp với hậu xử lý **Exp3 (Image-Adaptive Prototype Refinement)**. "
    "Triệt tiêu hoàn toàn **Object-Presence Hallucination (OPH)** khi đối tượng vắng mặt và tự động phân tách khi ảnh chứa nhiều loại đối tượng khác nhau."
)

with st.sidebar:
    st.header("⚙️ 1. Cấu hình Mô hình")

    device_choice = st.selectbox("Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
    device = device_choice if (device_choice == "cpu" or torch.cuda.is_available()) else "cpu"

    # Default Checkpoint Paths
    default_config = "./weights/GroundingDINO_SwinT_OGC.cfg.py"
    default_gdino_ckpt = "./weights/groundingdino_swint_ogc.pth"
    default_model_ckpt = "./checkpoints/gdcount_epoch_013_best.pth"
    if not os.path.isfile(default_model_ckpt):
        # Fallback to fp16 checkpoint if 013 path is relative
        fallback_011 = "./checkpoints/gdcount_epoch_011_best_fp16.pth"
        if os.path.isfile(fallback_011):
            default_model_ckpt = fallback_011

    config_path = st.text_input("GroundingDINO Config", value=default_config)
    gdino_ckpt_path = st.text_input("GroundingDINO Weights", value=default_gdino_ckpt)
    model_ckpt_path = st.text_input("GDCount Checkpoint (Epoch 13)", value=default_model_ckpt)

    st.header("🧬 2. Prototype Bank & DINOv3")
    default_cache = r"D:\projects\zsc\data\syn_FSC147_classes\.dinov3_cache"
    default_dinov3_repo = r"D:\projects\zsc\dinov3"
    default_dinov3_weights = r"D:\projects\zsc\dinov3\dinov3\weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    cache_path = st.text_input("Prototype Cache (.dinov3_cache)", value=default_cache)
    dinov3_repo = st.text_input("DINOv3 Repo", value=default_dinov3_repo)
    dinov3_weights = st.text_input("DINOv3 Weights", value=default_dinov3_weights)

    st.divider()
    st.header("🎛️ 3. Siêu tham số Hậu xử lý (Exp3)")

    enable_exp3 = st.toggle("Kích hoạt Hậu xử lý Exp3", value=True)
    presence_gate = st.checkbox("Presence Gate (Triệt tiêu Ảo giác khi Target vắng mặt)", value=True)
    use_competitor = st.checkbox("Competitor Filter (Lọc đối tượng lớp khác trong cảnh đa vật thể)", value=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        base_threshold = st.slider("Detector Conf (τ_gd)", 0.0, 1.0, 0.23, 0.01)
        adapt_threshold = st.slider("Exp3 Adapt Sim (τ_adapt)", 0.0, 1.0, 0.60, 0.01)
    with col_t2:
        nms_iou = st.slider("NMS IoU", 0.0, 1.0, 0.50, 0.01)
        synthetic_blend = st.slider("Synthetic Blend (λ)", 0.0, 1.0, 0.20, 0.05)

    consistency_threshold = st.slider("Seed Consistency (τ_c)", 0.0, 1.0, 0.70, 0.05)


# Main Interaction Area
prompt_col, run_col = st.columns([5, 1])
with prompt_col:
    prompt = st.text_input("📝 Truy vấn Đếm (Category Query Prompt):", value="strawberries")
with run_col:
    st.write("")
    run_btn = st.button("🚀 Chạy Đếm", type="primary", use_container_width=True)

col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("🖼️ Ảnh Đầu vào")
    uploaded_file = st.file_uploader("Upload ảnh (JPG / PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption=f"Ảnh gốc ({input_image.width} × {input_image.height})", use_container_width=True)
    else:
        input_image = None
        st.info("💡 Vui lòng upload một ảnh để bắt đầu thử nghiệm.")

with col_output:
    st.subheader("📊 Kết quả Đếm & Phân tích")
    if input_image is None:
        st.write("Đang chờ tải ảnh...")
    elif run_btn:
        # 1. Validation checks
        if not os.path.isfile(config_path):
            st.error(f"Không tìm thấy config: {config_path}")
            st.stop()
        if not os.path.isfile(gdino_ckpt_path):
            st.error(f"Không tìm thấy GroundingDINO weights: {gdino_ckpt_path}")
            st.stop()

        # 2. Load model & bank
        model, epoch = load_gdcount_model(config_path, gdino_ckpt_path, model_ckpt_path, device)
        bank, extractor = load_prototype_bank_and_extractor(cache_path, dinov3_repo, dinov3_weights, device)

        clean_cap = sanitize_caption(prompt)
        query_key = prompt.strip().lower()

        # 3. Base Detector Forward
        t0 = time.time()
        tensor_img = preprocess_image_for_model(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor_img, captions=[clean_cap])
        detector_time = (time.time() - t0) * 1000.0

        # Extract Raw Candidate boxes and scores
        raw_boxes_norm, raw_scores = pick_raw_candidates(outputs, threshold=base_threshold, nms_iou=nms_iou)
        raw_count = len(raw_boxes_norm)

        # Convert normalized boxes to pixel coordinates on original image
        W_orig, H_orig = input_image.size
        raw_boxes_px = np.zeros_like(raw_boxes_norm)
        if raw_count > 0:
            raw_boxes_px[:, 0] = raw_boxes_norm[:, 0] * W_orig
            raw_boxes_px[:, 1] = raw_boxes_norm[:, 1] * H_orig
            raw_boxes_px[:, 2] = raw_boxes_norm[:, 2] * W_orig
            raw_boxes_px[:, 3] = raw_boxes_norm[:, 3] * H_orig

        # 4. Exp3 Post-Processing
        exp3_cfg = Exp3Config(
            countgd_threshold=base_threshold,
            adapt_threshold=adapt_threshold,
            synthetic_blend=synthetic_blend,
            consistency_threshold=consistency_threshold,
            presence_threshold=base_threshold,
            use_competitor_filter=use_competitor,
        )
        postprocessor = Exp3PostProcessor(config=exp3_cfg, prototype_bank=bank)

        # Extract ROI features if Exp3 is enabled
        t_exp3_start = time.time()
        if enable_exp3 and raw_count > 0:
            roi_features = extractor.extract_box_features(
                image=input_image,
                boxes_xyxy=raw_boxes_px,
                expected_dim=bank.embedding_dim if bank else 1024,
            )
        else:
            roi_features = None

        exp3_result = postprocessor.verify_and_refine(
            boxes_xyxy=raw_boxes_norm,
            scores=raw_scores,
            features=roi_features,
            query=query_key,
            presence_gate=presence_gate,
        )
        exp3_time = (time.time() - t_exp3_start) * 1000.0

        final_count = exp3_result["final_count"] if enable_exp3 else raw_count
        presence = exp3_result["presence"] if enable_exp3 else (raw_count > 0)
        eliminated = exp3_result["eliminated_hallucinations"] if enable_exp3 else 0

        # Metrics display
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Exp3 Final Count", final_count, delta=f"{final_count - raw_count} so với raw" if enable_exp3 else None)
        with m2:
            st.metric("Raw Detector Count", raw_count)
        with m3:
            if presence:
                st.markdown("<p style='margin-bottom:4px; font-weight:600;'>Trạng thái Hiện diện:</p><span class='badge-present'>✅ TARGET PRESENT</span>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='margin-bottom:4px; font-weight:600;'>Trạng thái Hiện diện:</p><span class='badge-absent'>❌ ABSENT (Count = 0)</span>", unsafe_allow_html=True)

        if enable_exp3 and eliminated > 0:
            st.success(f"🛡️ **Exp3 Anti-Hallucination:** Đã triệt tiêu thành công **{eliminated} bounding boxes ảo giác / đối tượng lớp khác!**")

        st.caption(f"⚡ Thời gian xử lý: Base Detector: `{detector_time:.1f}ms` | Exp3 Post-processing: `{exp3_time:.1f}ms` | Epoch: `{epoch}`")

        # Tabs for visualization
        tab_verified, tab_analysis, tab_density, tab_diagnostics = st.tabs([
            "🎯 Bounding Boxes Hợp lệ",
            "🔍 Phân tích Ảo giác & Lớp Đối thủ",
            "🔥 Density Map",
            "📋 Bảng Chi tiết Candidates",
        ])

        with tab_verified:
            # Draw only kept boxes
            kept_diagnostics = [d for d in exp3_result["diagnostics"] if d["status"] == "kept"] if enable_exp3 else [
                {"candidate_id": i+1, "box": raw_boxes_norm[i].tolist(), "score": float(raw_scores[i]), "status": "kept"}
                for i in range(raw_count)
            ]
            vis_kept = draw_colored_boxes_on_pil(input_image, kept_diagnostics, show_labels=True)
            st.image(vis_kept, caption=f"Các đối tượng được Exp3 xác thực (Số lượng: {final_count})", use_container_width=True)

        with tab_analysis:
            # Draw all boxes with full color coding
            vis_analysis = draw_colored_boxes_on_pil(input_image, exp3_result["diagnostics"], show_labels=True)
            st.image(
                vis_analysis,
                caption="Phân loại BBoxes: 🟢 Xanh: Đúng Target | 🔴 Đỏ: Lớp đối thủ (Co-present) | 🟠 Cam: Low Adapt Sim | ⚪ Xám: Low Conf",
                use_container_width=True,
            )

        with tab_density:
            # Generate density map from kept box centers
            kept_boxes = exp3_result["final_boxes"] if enable_exp3 else raw_boxes_norm
            if len(kept_boxes) > 0:
                centers = torch.from_numpy(np.stack([
                    (kept_boxes[:, 0] + kept_boxes[:, 2]) / 2.0,
                    (kept_boxes[:, 1] + kept_boxes[:, 3]) / 2.0,
                ], axis=1)).float()
                d_map = centers_to_density_map(centers, out_h=input_image.height, out_w=input_image.width, sigma=4.0)
                d_vis = overlay_density_on_image(input_image, d_map.numpy(), alpha=0.55)
                st.image(d_vis, caption="Density Map phủ trên ảnh gốc", use_container_width=True)
            else:
                st.info("Không có đối tượng nào hiện diện để sinh Density Map.")

        with tab_diagnostics:
            if exp3_result["diagnostics"]:
                df_diag = pd.DataFrame(exp3_result["diagnostics"])
                st.dataframe(df_diag, use_container_width=True)
            else:
                st.write("Không có candidate nào được sinh ra.")
