# scripts/criterion_detect.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from scripts.matcher import HungarianMatcher
from scripts.set_criterion_countgd import SetCriterion


@dataclass
class CriterionDetectConfig:
    num_classes: int = 1
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    lambda_cls: float = 1.0
    lambda_bbox: float = 5.0
    lambda_giou: float = 2.0
    eos_coef: float = 0.1
    focal_alpha: float = 0.25


def build_criterion_detect(
    tokenizer,
    num_classes: int = 1,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
    lambda_cls: float = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
    eos_coef: float = 0.1,
    focal_alpha: float = 0.25,
) -> SetCriterion:
    """
    Criterion detect theo đúng CountGD:
      - HungarianMatcher trên (cls/token cost + L1 bbox + GIoU)
      - SetCriterion dùng create_positive_map_exemplar để map nhãn ↔ tokens
    """
    matcher = HungarianMatcher(
        cost_class=class_cost,
        cost_bbox=bbox_cost,
        cost_giou=giou_cost,
    )

    weight_dict = {
        "loss_ce": float(lambda_cls),
        "loss_bbox": float(lambda_bbox),
        "loss_giou": float(lambda_giou),
    }

    losses = ["labels", "boxes", "cardinality"]

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
        focal_alpha=focal_alpha,
        tokenizer=tokenizer,
    )
    return criterion
