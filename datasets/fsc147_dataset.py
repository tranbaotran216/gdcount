import json
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# def _scale_points_with_ratio(points_xy, ratio_w, ratio_h, img_w, img_h) -> torch.Tensor:
#     """
#     points_xy: list[[x,y],...] hoặc Tensor(N,2) theo hệ toạ độ trước-resize (annotation gốc)
#     trả về Tensor(N,2) theo hệ toạ độ ảnh thực tế trong img_root.
#     """
#     if points_xy is None or (isinstance(points_xy, list) and len(points_xy) == 0):
#         return torch.zeros((0, 2), dtype=torch.float32)

#     if isinstance(points_xy, list):
#         pts = torch.tensor(points_xy, dtype=torch.float32)
#     else:
#         pts = points_xy.float()

#     out = pts.clone()
#     out[:, 0] = out[:, 0] * float(ratio_w)
#     out[:, 1] = out[:, 1] * float(ratio_h)

#     # clamp theo ảnh thật (img_root)
#     out[:, 0] = out[:, 0].clamp(0, img_w - 1)
#     out[:, 1] = out[:, 1].clamp(0, img_h - 1)
#     return out


# def _scale_poly_boxes_with_ratio(box_examples_coordinates, ratio_w, ratio_h, img_w, img_h) -> List[torch.Tensor]:
#     """
#     box_examples_coordinates: list exemplars; mỗi exemplar là 4 điểm [[x,y],...]
#     trả về list[Tensor(4,2)] theo hệ toạ độ ảnh trong img_root.
#     """
#     if not box_examples_coordinates:
#         return []

#     scaled: List[torch.Tensor] = []
#     for poly in box_examples_coordinates:
#         if not poly:
#             continue
#         pts = torch.tensor(poly, dtype=torch.float32)  # (4,2)
#         pts[:, 0] = pts[:, 0] * float(ratio_w)
#         pts[:, 1] = pts[:, 1] * float(ratio_h)
#         pts[:, 0] = pts[:, 0].clamp(0, img_w - 1)
#         pts[:, 1] = pts[:, 1].clamp(0, img_h - 1)
#         scaled.append(pts)
#     return scaled


def poly_to_xyxy(poly_4x2: torch.Tensor) -> torch.Tensor:
    x1 = poly_4x2[:, 0].min()
    y1 = poly_4x2[:, 1].min()
    x2 = poly_4x2[:, 0].max()
    y2 = poly_4x2[:, 1].max()
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


class FSC147Dataset(Dataset):
    def __init__(
        self,
        ann_path: str,
        img_root: str,
        split: str,
        split_file: str,
        class_map_file: str,
        img_size: Optional[int] = None,   # nếu None: giữ nguyên kích thước ảnh trong folder
        normalize: bool = True,
        density_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]

        self.ann_path = ann_path
        self.img_root = img_root
        self.img_size = img_size
        self.normalize = normalize
        self.density_root = density_root

        # 1) Load annotation
        with open(ann_path, "r") as f:
            self.ann: Dict[str, Any] = json.load(f)

        # 2) Load split -> list image_ids chuẩn hoá "xxx.jpg"
        with open(split_file, "r") as f:
            split_dict = json.load(f)
        raw_ids = split_dict[split]

        self.image_ids: List[str] = []
        for v in raw_ids:
            name = os.path.basename(v)
            base, ext = os.path.splitext(name)
            img_id = name if ext.lower() == ".jpg" else (base + ".jpg")
            if img_id in self.ann:
                # chỉ lấy nếu ảnh tồn tại trong img_root (tránh missing file)
                img_path = os.path.join(self.img_root, img_id)
                if os.path.exists(img_path):
                    self.image_ids.append(img_id)

        # 3) Load map image_id -> class_name
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
                img_id = name if ext.lower() == ".jpg" else (base + ".jpg")
                self.img2class[img_id] = cls_name

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image_tensor(self, img_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if self.img_size is not None and (w != self.img_size or h != self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
            w, h = img.size

        elif max(w,h) > 384:
            img = img.resize((384, 384), Image.BICUBIC)
            w, h = img.size

        arr = np.asarray(img).astype("float32") / 255.0  # (H,W,C)
        arr = np.transpose(arr, (2, 0, 1))               # (C,H,W)
        x = torch.from_numpy(arr)

        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            x = (x - mean) / std
        #<class 'torch.Tensor'> torch.float32 torch.Size([3, 384, 384])
        # <class 'tuple'> 384 384

        return x, (h, w)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        rec = self.ann[img_id]
        img_path = os.path.join(self.img_root, img_id)
        img_tensor, (img_h, img_w) = self._load_image_tensor(img_path)

        cls_name = self.img2class.get(img_id, "")
        prompt = cls_name if cls_name else rec.get("class", "")
        prompt = (prompt.strip() if prompt else "object").strip()
        if not prompt.endswith("."):
            prompt = prompt + " ."

        ratio_w = rec.get("ratio_w", 1.0)
        ratio_h = rec.get("ratio_h", 1.0)

        # points + exemplar boxes scale về hệ toạ độ ảnh trong folder
        points_scaled = torch.tensor(
            rec.get("points", []),
            dtype=torch.float32
        )

        # exemplar boxes
        polys_scaled = [
            torch.tensor(poly, dtype=torch.float32)
            for poly in rec.get("box_examples_coordinates", [])
        ]
        exemplar_xyxy = (
            torch.stack([poly_to_xyxy(p) for p in polys_scaled], dim=0)
            if len(polys_scaled) > 0
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        exemplar_xyxy = exemplar_xyxy[:3]
        # print("img_w,img_h:", img_w, img_h)
        # print("points min/max x:", points_scaled[:,0].min(), points_scaled[:,0].max())
        # print("points min/max y:", points_scaled[:,1].min(), points_scaled[:,1].max())

        gt_count = torch.tensor(float(points_scaled.shape[0]), dtype=torch.float32)

        sample = {
            "images": img_tensor,         # (3,H,W)
            "prompts": prompt,            # str
            "gt_counts": gt_count,        # scalar float32
            "meta": {
                "image_ids": img_id,
                "class_names": cls_name,
                "points": points_scaled,          # (N,2) đúng hệ toạ độ ảnh trong folder
                "exemplar_xyxy": exemplar_xyxy,   # (K,4) nếu cần
                "ratio_w": ratio_w,
                "ratio_h": ratio_h,
                "img_h": img_h,
                "img_w": img_w,
                "ann_H": rec.get("H", None),
                "ann_W": rec.get("W", None),
            },
        }
        return sample


def collate_batch(batch):
    B = len(batch)

    images = torch.stack([b["images"] for b in batch], dim=0)  # (B,3,H,W)
    prompts = [b["prompts"] for b in batch]
    gt_counts = torch.stack([b["gt_counts"].float().view(1) for b in batch], dim=0).squeeze(1)  # (B,)

    metas = [b["meta"] for b in batch]

    # Variable-length: points
    points_list = [m.get("points") for m in metas]
    n_list = [p.shape[0] if isinstance(p, torch.Tensor) else 0 for p in points_list]
    max_n = max(n_list) if n_list else 0

    if max_n > 0:
        points = images.new_full((B, max_n, 2), -1.0)
        points_mask = torch.zeros((B, max_n), dtype=torch.bool)
        for i, p in enumerate(points_list):
            if isinstance(p, torch.Tensor) and p.numel() > 0:
                ni = p.shape[0]
                points[i, :ni] = p.to(points.dtype)
                points_mask[i, :ni] = True
    else:
        points = images.new_empty((B, 0, 2))
        points_mask = torch.zeros((B, 0), dtype=torch.bool)

    # Variable-length: exemplar boxes
    ex_list = [m.get("exemplar_xyxy") for m in metas]
    m_list = [e.shape[0] if isinstance(e, torch.Tensor) else 0 for e in ex_list]
    max_m = max(m_list) if m_list else 0

    if max_m > 0:
        exemplar_xyxy = images.new_full((B, max_m, 4), -1.0)
        exemplar_mask = torch.zeros((B, max_m), dtype=torch.bool)
        for i, e in enumerate(ex_list):
            if isinstance(e, torch.Tensor) and e.numel() > 0:
                mi = e.shape[0]
                exemplar_xyxy[i, :mi] = e.to(exemplar_xyxy.dtype)
                exemplar_mask[i, :mi] = True
    else:
        exemplar_xyxy = images.new_empty((B, 0, 4))
        exemplar_mask = torch.zeros((B, 0), dtype=torch.bool)

    meta = {
        "image_ids":   [m.get("image_ids") for m in metas],
        "class_names": [m.get("class_names") for m in metas],
        "ratio_w": torch.tensor([float(m.get("ratio_w", 1.0)) for m in metas], dtype=torch.float32),
        "ratio_h": torch.tensor([float(m.get("ratio_h", 1.0)) for m in metas], dtype=torch.float32),
        "img_h":   torch.tensor([int(m.get("img_h", 0)) for m in metas], dtype=torch.int64),
        "img_w":   torch.tensor([int(m.get("img_w", 0)) for m in metas], dtype=torch.int64),
        "ann_H":   torch.tensor([int(m.get("ann_H", 0)) for m in metas], dtype=torch.int64),
        "ann_W":   torch.tensor([int(m.get("ann_W", 0)) for m in metas], dtype=torch.int64),
    }

    return {
        "images": images,
        "prompts": prompts,
        "gt_counts": gt_counts,
        "points": points,
        "points_mask": points_mask,
        "exemplar_xyxy": exemplar_xyxy,
        "exemplar_mask": exemplar_mask,
        "meta": meta,
    }
