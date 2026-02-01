# scripts/set_criterion_countgd.py
from __future__ import annotations

from typing import Dict, List, Any
import torch
import torch.nn.functional as F
from torch import nn
from scripts import box_ops


def insert_visual_mask(old_mask: torch.Tensor | None,
                       batch_size: int,
                       insert_pos: int,
                       num_visual: int,
                       device=None,
                       dtype=None) -> torch.Tensor:
    """
    old_mask: (B, L_old) với 0/1 hoặc bool; có thể None.
    insert_pos: vị trí chèn (0..L_old)
    num_visual: số visual tokens chèn thêm (V)
    return: new_mask (B, L_old + V) kiểu bool
    """
    if device is None:
        device = old_mask.device if old_mask is not None else "cpu"
    if dtype is None:
        dtype = old_mask.dtype if old_mask is not None else torch.bool

    # Nếu old_mask=None: coi như toàn valid (text tokens đều hợp lệ)
    if old_mask is None:
        # Không biết L_old thì không làm được; caller phải truyền old_mask hoặc tự tạo theo input_ids length
        raise ValueError("old_mask is None: please provide old_mask or build it from input_ids length.")

    if old_mask.dtype != torch.bool:
        old_mask = old_mask.bool()

    B, L_old = old_mask.shape
    assert B == batch_size, f"batch mismatch: old_mask B={B}, batch_size={batch_size}"
    assert 0 <= insert_pos <= L_old, f"insert_pos must be in [0, {L_old}]"

    head = old_mask[:, :insert_pos]                  # (B, insert_pos)
    tail = old_mask[:, insert_pos:]                  # (B, L_old - insert_pos)

    vis = torch.ones((B, num_visual), device=device, dtype=torch.bool)  # (B, V)

    new_mask = torch.cat([head, vis, tail], dim=1)   # (B, L_old + V)
    return new_mask

def sigmoid_focal_loss_logits(inputs, targets, num_pos: float, alpha: float = 0.25, gamma: float = 2.0):
    """
    CountGD / GroundingDINO style focal over tokens.
    inputs: logits  (N, ...)  ; targets: {0,1} same shape
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # normalize by number of matched positives (CountGD uses num_pos_avg_per_gpu)
    return loss.sum() / max(float(num_pos), 1.0)


def create_positive_map_exemplar_from_input_ids(
    input_ids: torch.Tensor,
    label: int,
    special_tokens: List[int],
    max_len: int | None = None,
) -> torch.Tensor:
    """
    Same idea as groundingdino.py:create_positive_map_exemplar :contentReference[oaicite:1]{index=1}
    Build a 1D mask over token positions corresponding to the `label`-th phrase segment.

    input_ids: (L,)
    return: (T,) float, where T = max_len or L
    """
    if max_len is None:
        max_len = int(input_ids.numel())

    tokens_positive = torch.zeros((max_len,), dtype=torch.float32, device=input_ids.device)

    count = -1
    L = int(input_ids.numel())
    for token_ind in range(L):
        tid = int(input_ids[token_ind].item())
        prev_tid = int(input_ids[token_ind - 1].item()) if token_ind > 0 else None

        # start of a new "phrase" chunk
        if (tid not in special_tokens) and (token_ind == 0 or (prev_tid in special_tokens)):
            count += 1

        if count == int(label):
            k = token_ind
            while k < L and int(input_ids[k].item()) not in special_tokens and k < max_len:
                tokens_positive[k] = 1.0
                k += 1
            break

    return tokens_positive


class SetCriterion(nn.Module):
    """
    CountGD-style SetCriterion:
      forward(outputs, targets, cat_list, caption, return_indices=False)
    """

    def __init__(
        self,
        num_classes: int,
        matcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tokenizer=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tokenizer = tokenizer

        # CountGD/GroundingDINO typical special tokens for BERT:
        # [CLS]=101, [SEP]=102, '.'=1012, '?'=1029 (repo bạn dùng cũng thế) :contentReference[oaicite:2]{index=2}
        self._special_tokens = [101, 102, 1012, 1029]

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "centers": self.loss_centers,
            "cardinality": self.loss_cardinality,
        }
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # -------- CountGD labels loss (token focal on one_hot) --------
    def loss_labels(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        pred_logits: [B, Q, T]
        one_hot:     [B, Q, T] (created in forward() based on matcher indices + label_map)
        text_mask:   [B, T] optional
        """
        assert "pred_logits" in outputs
        assert "one_hot" in outputs

        pred_logits = outputs["pred_logits"]      # [B,Q,T]
        new_targets = outputs["one_hot"].to(pred_logits.device).float()  # [B,Q,T]

        # token info phải được lưu lại trong outputs ở forward() (xem mục 3)
        text_mask = outputs.get("text_mask", None)     # [B,T] attention_mask (1=valid token)
        input_ids  = outputs.get("input_ids", None)    # [B,T]

        if text_mask is not None:
            valid = text_mask.bool()  # [B,T]

            # loại special tokens nếu có input_ids
            if input_ids is not None:
                special = torch.zeros_like(input_ids, dtype=torch.bool)
                for t in self._special_tokens:
                    special |= (input_ids == t)
                valid = valid & (~special)

            # expand sang [B,Q,T]
            valid3 = valid[:, None, :].expand(pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2])

            pred_logits = pred_logits[valid3]
            new_targets = new_targets[valid3]

        # SANITIZE để chắc chắn không còn +/-inf (cực quan trọng với log của bạn)
        pred_logits = torch.nan_to_num(pred_logits, neginf=-20.0, posinf=20.0)

        new_targets = new_targets.float()

        total_num_pos = 0
        for (src_idx, _) in indices:
            total_num_pos += int(src_idx.numel())
        loss = sigmoid_focal_loss_logits(
            pred_logits,
            new_targets,
            num_pos=max(total_num_pos, 1.0),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        return {"loss_ce": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]  # (sumN,4) cxcywh norm
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / max(num_boxes, 1)

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou.sum() / max(num_boxes, 1)

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_centers(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Supervise ONLY center (cx,cy) of boxes using gt_dots (points).
        - pred_boxes: [B,Q,4] (cxcywh, normalized)
        - targets[j]["points"]: [Nj,2] normalized (x,y) in [0,1]
        (Bạn cần đảm bảo build targets có key "points")
        indices: matcher output (src_idx, tgt_idx) where tgt_idx indexes targets[j]["boxes"].
        Ở đây ta KHÔNG dùng target box, chỉ dùng target point tương ứng.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_centers = outputs["pred_boxes"][idx][:, :2]  # (sumN,2)

        # lấy points theo tgt_idx
        tgt_points = torch.cat(
            [t["points"][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )  # (sumN,2)

        loss_center = F.smooth_l1_loss(src_centers, tgt_points, reduction="none")
        loss_center = loss_center.sum() / max(num_boxes, 1)

        return {"loss_center": loss_center}


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        # logging only
        device = next(iter(outputs.values())).device
        return {"cardinality_error": torch.tensor(0.0, device=device)}

    # -------- CountGD forward: build label_map -> matcher -> one_hot --------
    def forward(self, outputs: Dict[str, Any], targets, cat_list, caption, return_indices: bool = False):
        """
        outputs must include:
          - pred_logits: [B,Q,T]
          - pred_boxes:  [B,Q,4]
        Prefer if outputs has:
          - token: dict with input_ids [B,T], attention_mask [B,T]
        else we tokenize caption here.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = outputs_without_aux["pred_logits"].device
        B, Q, T = outputs_without_aux["pred_logits"].shape

        # 1) Get tokenized input_ids / attention_mask
        token = outputs_without_aux.get("token", None)
        if token is None:
            assert self.tokenizer is not None, "Need tokenizer or outputs['token'] like CountGD."
            token = self.tokenizer(
                caption,
                padding="longest",
                return_tensors="pt",
                truncation=True,
            ).to(device)

        input_ids = token["input_ids"]  # [B,T_token]
        attn_mask = token.get("attention_mask", None)  # [B,T_token] or None

        # Ensure mask length matches pred_logits T if possible
        if attn_mask is not None and attn_mask.shape[-1] != T:
            # If mismatch, drop mask to avoid wrong masking
            attn_mask = None

        # 2) Build label_map_list per sample: [P, T]
        label_map_list: List[torch.Tensor] = []
        for j in range(B):
            phrases = cat_list[j]  # list of phrases for sample j
            P = len(phrases)
            if P <= 0:
                # fallback: 1 pseudo phrase
                P = 1
            maps = []
            for i in range(P):
                maps.append(
                    create_positive_map_exemplar_from_input_ids(
                        input_ids[j],
                        label=i,
                        special_tokens=self._special_tokens,
                        max_len=T,
                    )
                )
            label_map_j = torch.stack(maps, dim=0)  # [P, T]
            label_map_list.append(label_map_j)

        # 3) Matcher per-sample (CountGD style) :contentReference[oaicite:3]{index=3}
        indices: List[tuple[torch.Tensor, torch.Tensor]] = []
        for j in range(B):
            for_match = {
                "pred_logits": outputs_without_aux["pred_logits"][j].unsqueeze(0),
                "pred_boxes": outputs_without_aux["pred_boxes"][j].unsqueeze(0),
            }
            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)

        # 4) Build one_hot: [B,Q,T] (CountGD) :contentReference[oaicite:4]{index=4}
        one_hot = torch.zeros((B, Q, T), dtype=torch.int64, device=device)
        for j in range(B):
            src_idx, tgt_idx = indices[j]
            if tgt_idx.numel() == 0:
                continue
            tgt_labels = targets[j]["labels"][tgt_idx].to(torch.long)  # labels after match
            one_hot[j, src_idx] = label_map_list[j][tgt_labels].to(torch.long)

        outputs["one_hot"] = one_hot
        outputs["text_mask"] = attn_mask  # can be None
        outputs["input_ids"] = input_ids

        # normalize num_boxes
        num_boxes = sum(len(t.get("boxes", [])) for t in targets)
        num_boxes = float(max(num_boxes, 1))

        # 5) Compute losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 6) Aux losses (optional): recompute matcher+one_hot per aux layer
        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_wo = {k: v for k, v in aux_out.items() if k != "aux_outputs"}
                B2, Q2, T2 = aux_wo["pred_logits"].shape
                assert B2 == B and T2 == T, "aux_outputs must match batch & token dims"

                aux_indices: List[tuple[torch.Tensor, torch.Tensor]] = []
                for j in range(B):
                    for_match = {
                        "pred_logits": aux_wo["pred_logits"][j].unsqueeze(0),
                        "pred_boxes": aux_wo["pred_boxes"][j].unsqueeze(0),
                    }
                    inds = self.matcher(for_match, [targets[j]], label_map_list[j])
                    aux_indices.extend(inds)

                aux_one_hot = torch.zeros((B, Q2, T2), dtype=torch.int64, device=device)
                for j in range(B):
                    src_idx, tgt_idx = aux_indices[j]
                    if tgt_idx.numel() == 0:
                        continue
                    tgt_labels = targets[j]["labels"][tgt_idx].to(torch.long)
                    aux_one_hot[j, src_idx] = label_map_list[j][tgt_labels].to(torch.long)

                aux_out["one_hot"] = aux_one_hot
                aux_out["text_mask"] = attn_mask
                aux_out["input_ids"] = input_ids

                for loss in self.losses:
                    if loss == "cardinality":
                        continue
                    l_dict = self.get_loss(loss, aux_out, targets, aux_indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            return losses, indices
        return losses
