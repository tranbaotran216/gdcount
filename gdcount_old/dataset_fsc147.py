import json
import os
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FSC147Dataset(Dataset):
    def __init__(
        self,
        ann_path: str,
        img_root: str,
        split: str,
        split_file: str,
        class_map_file: str,
        img_size: int = None,
        normalize: bool = True,
        density_root: str = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]

        self.ann_path = ann_path
        self.img_root = img_root
        self.img_size = img_size        # nếu None -> không resize lại ảnh
        self.normalize = normalize
        self.density_root = density_root

        # 1) Load annotation
        with open(ann_path, "r") as f:
            self.ann: Dict[str, Any] = json.load(f)

        # 2) Load split, chuẩn hoá id về "xxx.jpg"
        with open(split_file, "r") as f:
            split_dict = json.load(f)
        raw_ids = split_dict[split]

        self.image_ids: List[str] = []
        for v in raw_ids:
            # v có thể là "1050", "1050.jpg", hoặc "images_384_VarV2/1050.jpg"
            name = os.path.basename(v)
            base, ext = os.path.splitext(name)
            if ext.lower() != ".jpg":
                img_id = base + ".jpg"
            else:
                img_id = name
            if img_id in self.ann:
                self.image_ids.append(img_id)

        # 3) Load file ánh xạ image_id -> class_name, chuẩn hoá về "xxx.jpg"
        self.img2class: Dict[str, str] = {}
        with open(class_map_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                raw_id, cls_name = parts[0], parts[1]
                name = os.path.basename(raw_id)
                base, ext = os.path.splitext(name)
                if ext.lower() != ".jpg":
                    img_id = base + ".jpg"
                else:
                    img_id = name
                self.img2class[img_id] = cls_name

        # 4) Build samples list từ annotation nested
        self.samples: List[Dict[str, Any]] = []
        for img_id in self.image_ids:
            rec = self.ann[img_id]

            H = rec.get("H", None)
            W = rec.get("W", None)
            points = rec.get("points", [])
            ex_boxes = rec.get("box_examples_coordinates", [])
            img_path = rec.get("img_path", img_id)
            density_path = rec.get("density_path", None)

            # img_path trong JSON là /nfs/... -> dùng basename + img_root local
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.img_root, os.path.basename(img_path))
            else:
                img_path = os.path.join(self.img_root, os.path.basename(img_path))

            # density_path tương tự (nếu bạn muốn dùng)
            if density_path is not None and self.density_root is not None:
                density_path = os.path.join(
                    self.density_root, os.path.basename(density_path)
                )

            cls_name = self.img2class.get(img_id, "")

            self.samples.append(
                {
                    "image_id": img_id,
                    "image_path": img_path,
                    "points": points,
                    "ex_boxes": ex_boxes,
                    "class_name": cls_name,
                    "height": H,
                    "width": W,
                    "density_path": density_path,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        img = Image.open(path).convert("RGB")
        w, h = img.size  # kích thước đã resize (vd 384 x 469)

        # Nếu bạn muốn ép về kích thước cố định, đặt img_size != None
        if self.img_size is not None and (h, w) != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
            h, w = self.img_size, self.img_size

        arr = np.asarray(img).astype("float32") / 255.0   # H, W, C
        arr = np.transpose(arr, (2, 0, 1))                # C, H, W
        tensor = torch.from_numpy(arr)

        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor, (h, w)

    @staticmethod
    def _points_to_tensor(
        points: List[List[float]],
        orig_hw: Tuple[int, int],
        new_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if not points:
            return torch.empty(0, 2, dtype=torch.float32)

        orig_h, orig_w = orig_hw
        new_h, new_w = new_hw
        scale_x = new_w / max(float(orig_w), 1e-6)
        scale_y = new_h / max(float(orig_h), 1e-6)

        pts = []
        for x, y in points:
            xs = x * scale_x
            ys = y * scale_y
            pts.append([xs, ys])
        return torch.tensor(pts, dtype=torch.float32)

    @staticmethod
    def _ex_boxes_to_tensor(
        ex_boxes: List[Any],
        orig_hw: Tuple[int, int],
        new_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if not ex_boxes:
            return torch.empty(0, 4, dtype=torch.float32)

        orig_h, orig_w = orig_hw
        new_h, new_w = new_hw
        scale_x = new_w / max(float(orig_w), 1e-6)
        scale_y = new_h / max(float(orig_h), 1e-6)

        out_boxes = []
        for box in ex_boxes:
            if not box:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x_min = min(xs) * scale_x
            y_min = min(ys) * scale_y
            x_max = max(xs) * scale_x
            y_max = max(ys) * scale_y
            out_boxes.append([x_min, y_min, x_max, y_max])

        if not out_boxes:
            return torch.empty(0, 4, dtype=torch.float32)

        return torch.tensor(out_boxes, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]

        image_tensor, (new_h, new_w) = self._load_image(rec["image_path"])

        # orig_hw dùng H,W gốc từ annotation
        orig_h = rec["height"]
        orig_w = rec["width"]
        orig_hw = (orig_h, orig_w)

        points_tensor = self._points_to_tensor(
            rec["points"], orig_hw, (new_h, new_w)
        )
        gt_count = float(points_tensor.shape[0])

        ex_boxes_tensor = self._ex_boxes_to_tensor(
            rec["ex_boxes"], orig_hw, (new_h, new_w)
        )

        cls_name = rec["class_name"] if rec["class_name"] else "object"
        prompt = cls_name.strip()
        if not prompt.endswith("."):
            prompt = prompt + " ."

        return {
            "image": image_tensor,
            "prompt": prompt,
            "exemplar_boxes": ex_boxes_tensor,
            "image_size": (new_h, new_w),
            "points": points_tensor,
            "gt_count": torch.tensor(gt_count, dtype=torch.float32),
            "image_id": rec["image_id"],
            "class_name": rec["class_name"],
        }


def fsc147_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    ex_boxes = [b["exemplar_boxes"] for b in batch]
    image_sizes = [b["image_size"] for b in batch]
    gt_counts = torch.stack([b["gt_count"] for b in batch], dim=0)

    meta = {
        "image_ids": [b["image_id"] for b in batch],
        "class_names": [b["class_name"] for b in batch],
        "points": [b["points"] for b in batch],
    }

    return {
        "images": images,
        "prompts": prompts,
        "exemplar_boxes": ex_boxes,
        "image_sizes": image_sizes,
        "gt_counts": gt_counts,
        "meta": meta,
    }
