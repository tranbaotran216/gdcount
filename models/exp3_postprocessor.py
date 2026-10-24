"""Exp3 Image-Adaptive Prototype Refinement and Anti-Hallucination Post-Processor.

This module implements the Exp3 post-processing pipeline for Zero-Shot Counting (ZSC).
It addresses two fundamental challenges in open-world counting:
1. Target-Absence Hallucination (Object-Presence Hallucination - OPH):
   When the queried object does not exist in the image (e.g. querying 'elephants'
   in an image of strawberries), baseline detectors often force boxes onto background
   textures or other objects. Exp3 verifies target presence and suppresses phantom counts (Count = 0).
2. Multi-Object Co-Presence Disambiguation:
   When an image contains multiple different classes (e.g. apples and oranges),
   Exp3 uses competitor class prototypes and image-adaptive refinement to filter out
   candidates that belong to other co-present classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


EPSILON = 1e-8


@dataclass
class Exp3Config:
    """Hyperparameters for Exp3 Post-Processing."""

    # Base detection threshold
    countgd_threshold: float = 0.23

    # Seed score weighting: alpha * score_det + (1 - alpha) * score_syn
    alpha_gd: float = 0.70

    # Number of candidates in candidate pool to inspect for seeds
    seed_pool_size: int = 8

    # Sequential consistency threshold (cosine similarity) among trusted seeds
    consistency_threshold: float = 0.70

    # Maximum number of trusted seeds to average into p_real
    max_trusted_seeds: int = 5

    # Weighting method for p_real: 'weighted' or 'mean'
    weighting: str = "weighted"

    # Synthetic prototype blend ratio: p_adapt = (1 - lambda) * p_real + lambda * p_syn
    synthetic_blend: float = 0.20

    # Final verification threshold on adapted cosine similarity: s_adapt >= adapt_threshold
    adapt_threshold: float = 0.60

    # Presence gating threshold: if max candidate score is below this, target is flagged absent
    presence_threshold: float = 0.23

    # Reject candidate if another non-target class has higher similarity
    use_competitor_filter: bool = True

    # Margin required over competitor classes (0.0 = strict argmax)
    competitor_margin: float = 0.0


class PrototypeBankLite:
    """Lightweight in-memory Prototype Bank for class prototype lookup and competitor ranking."""

    def __init__(
        self,
        class_keys: List[str],
        prototype_matrix: torch.Tensor,
        class_display_names: Optional[List[str]] = None,
        prototype_class_ids: Optional[torch.Tensor] = None,
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.class_keys = [str(k).strip().lower() for k in class_keys]
        self.class_display_names = class_display_names or self.class_keys
        self.prototype_matrix = F.normalize(prototype_matrix.detach().float().cpu(), p=2, dim=1)
        
        if prototype_class_ids is not None:
            self.prototype_class_ids = prototype_class_ids.detach().long().cpu()
        else:
            self.prototype_class_ids = torch.arange(len(self.class_keys), dtype=torch.long)
            
        self.records = records or []
        self._key_to_id = {key: i for i, key in enumerate(self.class_keys)}

    @property
    def num_classes(self) -> int:
        return len(self.class_keys)

    @property
    def embedding_dim(self) -> int:
        return self.prototype_matrix.shape[1] if self.prototype_matrix.ndim == 2 else 0

    def has_class(self, class_name: str) -> bool:
        normalized = self._normalize_name(class_name)
        return normalized in self._key_to_id

    def get_class_prototype(self, class_name: str) -> Optional[torch.Tensor]:
        normalized = self._normalize_name(class_name)
        if normalized not in self._key_to_id:
            # Fallback: check partial substring match
            for k in self.class_keys:
                if normalized in k or k in normalized:
                    idx = self._key_to_id[k]
                    return self.prototype_matrix[idx]
            return None
        idx = self._key_to_id[normalized]
        return self.prototype_matrix[idx]

    @staticmethod
    def _normalize_name(name: str) -> str:
        s = str(name).strip().lower()
        if s.endswith("."):
            s = s[:-1].strip()
        return s

    @classmethod
    def from_cache_dir(cls, cache_dir: Union[str, Path]) -> PrototypeBankLite:
        """Load from a saved .dinov3_cache directory."""
        cache_dir = Path(cache_dir)
        tensors_path = cache_dir / "prototypes.pt"
        records_path = cache_dir / "records.json"

        if not tensors_path.is_file():
            raise FileNotFoundError(f"Prototypes tensor file not found at: {tensors_path}")

        try:
            tensors = torch.load(tensors_path, map_location="cpu", weights_only=True)
        except TypeError:
            tensors = torch.load(tensors_path, map_location="cpu")

        records = []
        if records_path.is_file():
            try:
                records = json.loads(records_path.read_text(encoding="utf-8"))
            except Exception:
                records = []

        return cls(
            class_keys=tensors["class_keys"],
            prototype_matrix=tensors["prototype_matrix"],
            class_display_names=tensors.get("class_display_names", tensors["class_keys"]),
            prototype_class_ids=tensors.get("prototype_class_ids", None),
            records=records,
        )


class Exp3PostProcessor:
    """Exp3 Post-Processing Engine for Adaptive Prototype Verification."""

    def __init__(
        self,
        config: Optional[Exp3Config] = None,
        prototype_bank: Optional[PrototypeBankLite] = None,
    ) -> None:
        self.config = config or Exp3Config()
        self.bank = prototype_bank

    def set_prototype_bank(self, bank: PrototypeBankLite) -> None:
        self.bank = bank

    @staticmethod
    def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
        v = v.detach().float().cpu().reshape(-1)
        norm = torch.linalg.vector_norm(v)
        if norm <= EPSILON or not torch.isfinite(norm):
            return torch.zeros_like(v)
        return v / norm

    def verify_and_refine(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
        query: str = "object",
        presence_gate: bool = True,
    ) -> Dict[str, Any]:
        """Apply Exp3 verification to raw candidate boxes.

        Args:
            boxes_xyxy: Array of shape (N, 4) in XYXY format.
            scores: Array of shape (N,) detection / calibrated scores.
            features: Optional array of shape (N, D) normalized visual embeddings.
                      If None, standard threshold filtering is performed.
            query: Query text category string.
            presence_gate: Whether to enforce the target presence check.

        Returns:
            Dictionary containing:
                - 'final_boxes': Filtered bounding boxes.
                - 'final_scores': Filtered prediction scores.
                - 'final_count': Final predicted count.
                - 'presence': Boolean indicating whether the object is present.
                - 'p_real': Real-image adaptive prototype (if computed).
                - 'p_adapt': Blended adapted prototype (if computed).
                - 'diagnostics': List of per-candidate status metadata.
                - 'eliminated_hallucinations': Number of false positives eliminated.
        """
        N = len(boxes_xyxy)
        if N == 0:
            return {
                "final_boxes": np.zeros((0, 4), dtype=np.float32),
                "final_scores": np.zeros((0,), dtype=np.float32),
                "final_count": 0,
                "presence": False,
                "p_real": None,
                "p_adapt": None,
                "diagnostics": [],
                "eliminated_hallucinations": 0,
            }

        # 1. Target-Absence / Presence Gate
        max_score = float(np.max(scores)) if N > 0 else 0.0
        if presence_gate and max_score < self.config.presence_threshold:
            # Query target is completely absent in the image -> Suppress all boxes (Count = 0)
            diagnostics = [
                {
                    "candidate_id": i + 1,
                    "box": boxes_xyxy[i].tolist(),
                    "score": float(scores[i]),
                    "status": "rejected_target_absent",
                    "reason": f"Max confidence ({max_score:.3f}) < presence threshold ({self.config.presence_threshold:.3f})",
                }
                for i in range(N)
            ]
            return {
                "final_boxes": np.zeros((0, 4), dtype=np.float32),
                "final_scores": np.zeros((0,), dtype=np.float32),
                "final_count": 0,
                "presence": False,
                "p_real": None,
                "p_adapt": None,
                "diagnostics": diagnostics,
                "eliminated_hallucinations": N,
            }

        # If visual features are not provided, fallback to standard score thresholding
        if features is None or self.bank is None or not self.bank.has_class(query):
            keep_mask = scores >= self.config.countgd_threshold
            kept_idx = np.where(keep_mask)[0]
            diagnostics = [
                {
                    "candidate_id": i + 1,
                    "box": boxes_xyxy[i].tolist(),
                    "score": float(scores[i]),
                    "status": "kept" if i in kept_idx else "rejected_low_confidence",
                }
                for i in range(N)
            ]
            return {
                "final_boxes": boxes_xyxy[kept_idx],
                "final_scores": scores[kept_idx],
                "final_count": int(len(kept_idx)),
                "presence": len(kept_idx) > 0,
                "p_real": None,
                "p_adapt": None,
                "diagnostics": diagnostics,
                "eliminated_hallucinations": int(N - len(kept_idx)),
            }

        # Visual feature verification with Prototype Bank
        feats_t = torch.from_numpy(features).float()
        feats_t = F.normalize(feats_t, p=2, dim=1)  # Ensure L2 normalized: (N, D)

        p_syn = self.bank.get_class_prototype(query)
        if p_syn is None:
            p_syn = torch.zeros(feats_t.shape[1])

        # Step 1: Compute synthetic scores: s_syn = feats_t @ p_syn
        s_syn = (feats_t @ p_syn.unsqueeze(1)).squeeze(1).numpy()
        dino01 = np.clip((s_syn + 1.0) / 2.0, 0.0, 1.0)

        # Candidate Seed Score: r_i = alpha * g_i + (1 - alpha) * dino01_i
        seed_scores = self.config.alpha_gd * scores + (1.0 - self.config.alpha_gd) * dino01

        # Step 2: Form Seed Pool (Top-M candidates satisfying threshold)
        eligible = np.where(scores >= self.config.countgd_threshold)[0]
        if len(eligible) == 0:
            return {
                "final_boxes": np.zeros((0, 4), dtype=np.float32),
                "final_scores": np.zeros((0,), dtype=np.float32),
                "final_count": 0,
                "presence": False,
                "p_real": None,
                "p_adapt": None,
                "diagnostics": [{"candidate_id": i+1, "status": "rejected_low_conf"} for i in range(N)],
                "eliminated_hallucinations": N,
            }

        # Sort eligible seeds by seed score descending
        eligible = eligible[np.argsort(-seed_scores[eligible])]
        pool_indices = eligible[: self.config.seed_pool_size]

        # Step 3: Greedy Sequential Consistency to pick trusted seeds
        trusted_indices = [pool_indices[0]]
        for idx in pool_indices[1:]:
            ref_feats = feats_t[trusted_indices]  # (K_cur, D)
            cand_feat = feats_t[idx].unsqueeze(0)  # (1, D)
            consist = float((cand_feat @ ref_feats.T).mean().item())
            if consist >= self.config.consistency_threshold and len(trusted_indices) < self.config.max_trusted_seeds:
                trusted_indices.append(idx)

        # Step 4: Build Real-Image Prototype (p_real)
        trusted_vectors = feats_t[trusted_indices]
        if self.config.weighting == "weighted":
            w = torch.from_numpy(seed_scores[trusted_indices]).clamp_min(0.0).float()
            if w.sum() > EPSILON:
                p_real_raw = (w[:, None] * trusted_vectors).sum(dim=0) / w.sum()
            else:
                p_real_raw = trusted_vectors.mean(dim=0)
        else:
            p_real_raw = trusted_vectors.mean(dim=0)

        p_real = self._normalize_vector(p_real_raw)

        # Step 5: Adapt with Canonical Prototype (p_adapt)
        blend = (1.0 - self.config.synthetic_blend) * p_real + self.config.synthetic_blend * p_syn
        p_adapt = self._normalize_vector(blend)

        # Step 6: Adapted Similarity per candidate
        s_adapt = (feats_t @ p_adapt.unsqueeze(1)).squeeze(1).numpy()

        # Step 7: Competitor Matrix check across all classes in bank
        # all_sims: (N, num_classes)
        all_sims = (feats_t @ self.bank.prototype_matrix.T).numpy()
        query_class_id = self.bank._key_to_id.get(self.bank._normalize_name(query), -1)

        kept_indices = []
        diagnostics = []

        for i in range(N):
            g_i = float(scores[i])
            s_a = float(s_adapt[i])
            diag_entry: Dict[str, Any] = {
                "candidate_id": i + 1,
                "box": boxes_xyxy[i].tolist(),
                "score": g_i,
                "s_adapt": s_a,
                "is_seed": bool(i in trusted_indices),
            }

            # Check 1: Base threshold
            if g_i < self.config.countgd_threshold:
                diag_entry["status"] = "rejected_low_conf"
                diagnostics.append(diag_entry)
                continue

            # Check 2: Adapted similarity threshold
            if s_a < self.config.adapt_threshold:
                diag_entry["status"] = "rejected_low_adapt_sim"
                diagnostics.append(diag_entry)
                continue

            # Check 3: Competitor class rejection (Disambiguate multi-object scenes)
            if self.config.use_competitor_filter and query_class_id >= 0:
                class_sims = all_sims[i]
                best_class_idx = int(np.argmax(class_sims))
                best_class_sim = float(class_sims[best_class_idx])
                query_sim = float(class_sims[query_class_id]) if query_class_id < len(class_sims) else s_syn[i]

                if best_class_idx != query_class_id and (best_class_sim - query_sim) > self.config.competitor_margin:
                    competitor_name = self.bank.class_display_names[best_class_idx]
                    diag_entry["status"] = "rejected_competitor_class"
                    diag_entry["competitor"] = competitor_name
                    diag_entry["competitor_sim"] = best_class_sim
                    diag_entry["query_sim"] = query_sim
                    diagnostics.append(diag_entry)
                    continue

            # Passed all checks
            diag_entry["status"] = "kept"
            kept_indices.append(i)
            diagnostics.append(diag_entry)

        final_boxes = boxes_xyxy[kept_indices] if len(kept_indices) > 0 else np.zeros((0, 4), dtype=np.float32)
        final_scores = scores[kept_indices] if len(kept_indices) > 0 else np.zeros((0,), dtype=np.float32)
        presence = len(kept_indices) > 0

        return {
            "final_boxes": final_boxes,
            "final_scores": final_scores,
            "final_count": len(kept_indices),
            "presence": presence,
            "p_real": p_real,
            "p_adapt": p_adapt,
            "diagnostics": diagnostics,
            "eliminated_hallucinations": N - len(kept_indices),
        }
