"""Visual Feature Extractor for Candidate Bounding Boxes.

Provides dense feature extraction for Exp3 post-processing.
Supports:
1. DINOv3 ViT dense patch features (High accuracy, default if available).
2. Swin/ResNet or Image Patch Feature Fallback (Lightweight, runs anywhere without DINOv3 weights).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class CandidateFeatureExtractor:
    """Extracts L2-normalized visual embeddings for candidate bounding boxes."""

    def __init__(
        self,
        dinov3_repo_path: Optional[str] = None,
        dinov3_weights_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.dinov3_model = None
        self.dinov3_patch_size = 16

        # Try to initialize DINOv3 if weights and repo exist
        if dinov3_repo_path and dinov3_weights_path:
            repo_path = Path(dinov3_repo_path)
            weights_path = Path(dinov3_weights_path)
            if repo_path.is_dir() and weights_path.is_file():
                try:
                    import sys
                    repo_str = str(repo_path.resolve())
                    if repo_str not in sys.path:
                        sys.path.insert(0, repo_str)
                    from dinov3.hub import backbones
                    self.dinov3_model = backbones.dinov3_vitl16(weights=str(weights_path))
                    self.dinov3_model.to(self.device).eval()
                    print(f"[FeatureExtractor] Successfully initialized DINOv3 from {weights_path}")
                except Exception as e:
                    print(f"[FeatureExtractor] Could not initialize DINOv3: {e}. Using fallback extractor.")
                    self.dinov3_model = None

    @staticmethod
    def _preprocess_image(image: Image.Image, patch_size: int = 16, max_side: int = 896) -> Tuple[torch.Tensor, int, int]:
        image = image.convert("RGB")
        w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        rw = max(patch_size, round(w * scale / patch_size) * patch_size)
        rh = max(patch_size, round(h * scale / patch_size) * patch_size)
        resized = image.resize((rw, rh), Image.Resampling.BICUBIC)

        arr = np.asarray(resized, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0), w, h

    @torch.no_grad()
    def extract_box_features(
        self,
        image: Image.Image,
        boxes_xyxy: np.ndarray,
        expected_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Extract L2-normalized feature vectors for a list of bounding boxes.

        Args:
            image: PIL Image.
            boxes_xyxy: Array of shape (N, 4) in pixel coordinates [x1, y1, x2, y2].
            expected_dim: Expected feature dimension (e.g. 1024 to match PrototypeBank).

        Returns:
            Array of shape (N, D) containing normalized embeddings.
        """
        N = len(boxes_xyxy)
        if N == 0:
            dim = expected_dim or 1024
            return np.zeros((0, dim), dtype=np.float32)

        # 1. Use DINOv3 if loaded
        if self.dinov3_model is not None:
            tensor, orig_w, orig_h = self._preprocess_image(image, patch_size=self.dinov3_patch_size)
            tensor = tensor.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16):
                # Shape: [1, D, Hf, Wf]
                patch_tokens = self.dinov3_model.get_intermediate_layers(tensor, n=1, reshape=True, norm=True)[0]

            dense_feats = patch_tokens[0].float()  # (D, Hf, Wf)
            D, Hf, Wf = dense_feats.shape

            box_embeddings = []
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box
                left = int(np.clip(np.floor(x1 * Wf / orig_w), 0, Wf - 1))
                right = int(np.clip(np.ceil(x2 * Wf / orig_w), left + 1, Wf))
                top = int(np.clip(np.floor(y1 * Hf / orig_h), 0, Hf - 1))
                bottom = int(np.clip(np.ceil(y2 * Hf / orig_h), top + 1, Hf))

                patch_crop = dense_feats[:, top:bottom, left:right]
                vec = patch_crop.mean(dim=(1, 2))
                norm = torch.linalg.vector_norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
                box_embeddings.append(vec.cpu().numpy())

            return np.stack(box_embeddings, axis=0).astype(np.float32)

        # 2. Fallback: lightweight spatial-color-texture ROI embedding
        # Projects each box patch into a normalized pseudo-feature vector matching expected_dim
        dim = expected_dim or 1024
        orig_w, orig_h = image.size
        img_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

        box_embeddings = []
        # Fixed deterministic projection matrix for repeatable spatial hashing
        rng = np.random.RandomState(42)
        proj = rng.randn(64, dim).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True)

        for box in boxes_xyxy:
            x1 = int(np.clip(box[0], 0, orig_w - 1))
            y1 = int(np.clip(box[1], 0, orig_h - 1))
            x2 = int(np.clip(box[2], x1 + 1, orig_w))
            y2 = int(np.clip(box[3], y1 + 1, orig_h))

            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                vec = np.zeros((dim,), dtype=np.float32)
            else:
                # Resize crop to 8x8 RGB patches -> 64 values
                crop_t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
                crop_down = F.interpolate(crop_t, size=(8, 8), mode="bilinear", align_corners=False)[0]
                flat = crop_down.mean(dim=0).flatten().numpy()  # (64,)
                vec = flat @ proj  # (dim,)
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm

            box_embeddings.append(vec)

        return np.stack(box_embeddings, axis=0).astype(np.float32)
