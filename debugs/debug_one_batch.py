import argparse
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from datasets.fsc147_dataset import FSC147Dataset, fsc147_collate
from models.gdcount_model_calib import GDCountConfig, build_gdcount_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Debug 1 batch GDCount + FSC147")

    # GroundingDINO
    parser.add_argument("--config", type=str, required=True,
                        help="Path tới file config GroundingDINO (vd GroundingDINO_SwinT_OGC.py)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path tới file .pth GroundingDINO pretrain")

    # Data
    parser.add_argument("--ann", type=str, required=True,
                        help="annotation_FSC147_384.json")
    parser.add_argument("--img-root", type=str, required=True,
                        help="thư mục images_384_VarV2")
    parser.add_argument("--split-file", type=str, required=True,
                        help="Train_Test_Val_FSC_147.json")
    parser.add_argument("--class-map", type=str, required=True,
                        help="ImageClasses_FSC147.txt")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])

    parser.add_argument("--device", type=str, default="cuda")

    # GDCount
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--soa-level", type=int, default=-1)
    parser.add_argument("--freeze-keywords", type=str, nargs="+",
                        default=["backbone.0", "bert"])

    return parser.parse_args()


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = FSC147Dataset(
        ann_path=args.ann,
        img_root=args.img_root,
        split=args.split,
        split_file=args.split_file,
        class_map_file=args.class_map,
        img_size=None,          # không resize lại; dùng size thật của ảnh resized
        normalize=True,
        density_root=None,
    )
    print(f"Tổng số mẫu {args.split}: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,          # debug trên Windows: để 0 cho chắc
        pin_memory=True,
        collate_fn=fsc147_collate,
    )
    return loader


def build_model(args: argparse.Namespace, device: str) -> torch.nn.Module:
    cfg = GDCountConfig(
        threshold=args.threshold,
        soa_level=args.soa_level,
        feature_dim=256,
        freeze_keywords=args.freeze_keywords,
    )
    model = build_gdcount_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        gdcount_cfg=cfg,
    )

    # In 1 vài thông tin freeze / trainable
    total, trainable = 0, 0
    print("==== Các tham số trainable ====")
    for name, p in model.named_parameters():
        numel = p.numel()
        total += numel
        if p.requires_grad:
            trainable += numel
            print(f"[TRAIN] {name}  ({numel} params)")
        # else:
        #     print(f"[FROZEN] {name}  ({numel} params)")
    print(f"Tổng params: {total}, trainable: {trainable}")
    return model


def debug_one_batch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> None:
    model.eval()

    batch = next(iter(loader))
    images = batch["images"].to(device)       # (1,3,H,W)
    prompts = batch["prompts"]               # list length 1
    gt_counts = batch["gt_counts"]           # (1,)
    meta = batch["meta"]

    print("\n==== Thông tin batch ====")
    print("image_ids:  ", meta["image_ids"])
    print("class_names:", meta["class_names"])
    print("prompt:     ", prompts[0])
    print("GT count:   ", gt_counts.item())
    print("image_size: ", batch["image_sizes"][0])
    print("num_points: ", meta["points"][0].shape)

    with torch.no_grad():
        outputs: Dict[str, Any] = model(images, captions=prompts)

        soft = outputs["soft_counts"]      # (1,)
    hard = outputs["hard_counts"]      # (1,)
    qlogits = outputs["query_logits"]  # (1, Q)

    print("\n==== Kết quả model ====")
    print("soft_counts:", soft.cpu().numpy())
    print("hard_counts:", hard.cpu().numpy())
    print("query_logits shape:", qlogits.shape)
    print("query_logits min/max:",
          float(qlogits.min().item()), float(qlogits.max().item()))

    # L1 error
    l1_err = torch.abs(soft - gt_counts.to(soft.device))
    print("L1 error (|soft - GT|):", l1_err.cpu().numpy())

    # Debug hs
    hs = model.base_model.transformer.last_hs
    print("\n==== Debug hs (decoder output) ====")
    if hs is None:
        print("hs is None (cần kiểm tra lại TransformerWrapper / forward GroundingDINO).")
    elif isinstance(hs, (list, tuple)):
        print(f"hs là list, length = {len(hs)}")
        if len(hs) > 0 and isinstance(hs[0], torch.Tensor):
            print("một phần tử hs[0].shape =", hs[0].shape)
    elif isinstance(hs, torch.Tensor):
        print("hs tensor shape:", hs.shape)
    else:
        print(f"hs type = {type(hs)}")


def main():
    args = parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = args.device
    else:
        device = "cpu"
    print(f"Device: {device}")

    loader = build_dataloader(args)
    model = build_model(args, device=device)

    debug_one_batch(model, loader, device=device)


if __name__ == "__main__":
    main()
