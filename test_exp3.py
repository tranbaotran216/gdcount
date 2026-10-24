"""Offline Inference Script for GDCount + Exp3 Pipeline.

Usage:
    python test_exp3.py --image path/to/image.jpg --prompt "strawberries"
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.gdcount_model_calib import GDCountConfig, build_gdcount_model
from models.exp3_postprocessor import Exp3Config, Exp3PostProcessor, PrototypeBankLite
from models.feature_extractor import CandidateFeatureExtractor
from app_exp3 import (
    sanitize_caption,
    preprocess_image_for_model,
    pick_raw_candidates,
    load_gdcount_model,
)


def main():
    parser = argparse.ArgumentParser(description="Run GDCount + Exp3 Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Category name query (e.g. 'apples')")
    parser.add_argument("--config", type=str, default="./weights/GroundingDINO_SwinT_OGC.cfg.py")
    parser.add_argument("--gdino_ckpt", type=str, default="./weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--model_ckpt", type=str, default="./checkpoints/gdcount_epoch_013_best.pth")
    parser.add_argument("--cache_path", type=str, default=r"D:\projects\zsc\data\syn_FSC147_classes\.dinov3_cache")
    parser.add_argument("--dinov3_repo", type=str, default=r"D:\projects\zsc\dinov3")
    parser.add_argument("--dinov3_weights", type=str, default=r"D:\projects\zsc\dinov3\dinov3\weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--adapt_threshold", type=float, default=0.60)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="./output_exp3.jpg")
    args = parser.parse_args()

    print(f"Loading image from: {args.image}")
    img = Image.open(args.image).convert("RGB")
    W, H = img.size

    # Fallback checkpoint if 013 doesn't exist locally
    model_ckpt = args.model_ckpt
    if not os.path.isfile(model_ckpt):
        fallback = "./checkpoints/gdcount_epoch_011_best_fp16.pth"
        if os.path.isfile(fallback):
            model_ckpt = fallback

    print(f"Loading GDCount model with checkpoint: {model_ckpt}")
    model, epoch = load_gdcount_model(args.config, args.gdino_ckpt, model_ckpt, args.device)

    # Prototype bank
    bank = None
    if os.path.isdir(args.cache_path):
        bank = PrototypeBankLite.from_cache_dir(args.cache_path)
        print(f"Loaded Prototype Bank with {bank.num_classes} classes.")

    extractor = CandidateFeatureExtractor(args.dinov3_repo, args.dinov3_weights, device=args.device)

    cap = sanitize_caption(args.prompt)
    print(f"Running inference for query: '{cap}' (class: '{args.prompt}')")

    t0 = time.time()
    x = preprocess_image_for_model(img).unsqueeze(0).to(args.device)
    with torch.no_grad():
        outputs = model(x, captions=[cap])
    det_time = time.time() - t0

    boxes_norm, scores = pick_raw_candidates(outputs, threshold=args.threshold, nms_iou=0.50)
    raw_count = len(boxes_norm)
    print(f"Raw detector found {raw_count} candidates in {det_time*1000:.1f}ms")

    # Exp3 Post-Processing
    boxes_px = np.zeros_like(boxes_norm)
    if raw_count > 0:
        boxes_px[:, 0] = boxes_norm[:, 0] * W
        boxes_px[:, 1] = boxes_norm[:, 1] * H
        boxes_px[:, 2] = boxes_norm[:, 2] * W
        boxes_px[:, 3] = boxes_norm[:, 3] * H

    roi_features = extractor.extract_box_features(img, boxes_px, expected_dim=bank.embedding_dim if bank else 1024)

    exp3_cfg = Exp3Config(
        countgd_threshold=args.threshold,
        adapt_threshold=args.adapt_threshold,
        use_competitor_filter=True,
    )
    postprocessor = Exp3PostProcessor(config=exp3_cfg, prototype_bank=bank)

    res = postprocessor.verify_and_refine(
        boxes_xyxy=boxes_norm,
        scores=scores,
        features=roi_features,
        query=args.prompt,
        presence_gate=True,
    )

    print("\n" + "=" * 50)
    print(f"EXP3 INFERENCE RESULTS:")
    print(f"  - Target Query: '{args.prompt}'")
    print(f"  - Target Present: {res['presence']}")
    print(f"  - Raw Detector Count: {raw_count}")
    print(f"  - Exp3 Verified Count: {res['final_count']}")
    print(f"  - Hallucinations Eliminated: {res['eliminated_hallucinations']}")
    print("=" * 50 + "\n")

    # Print breakdown of diagnostics
    for d in res["diagnostics"]:
        print(f"Candidate #{d['candidate_id']}: Score={d['score']:.3f}, Status={d['status']}" +
              (f", Competitor={d.get('competitor')}" if 'competitor' in d else ""))


if __name__ == "__main__":
    main()
