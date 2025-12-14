import os
import cv2
import torch
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _to_tensor_and_normalize(img_rgb, img_size: int):
    """
    img_rgb: numpy (H, W, 3), uint8, range [0,255]
    img_size: kích thước target (H=W=img_size), ở đây dùng 384.
    """
    # resize về img_size x img_size (384 x 384)
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    img = img_resized.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img - mean) / std

    return img, img_resized.shape[:2]  # (tensor, (H_resized, W_resized))


# ======================================================
# 1. PREPROCESSING CHO 1 SAMPLE FSC147 (TRAIN / VAL)
# ======================================================

def preprocess_fsc147_sample(sample: dict,
                             img_root: str,
                             img_size: int = 384):
    """
    sample: 1 dict từ annotation FSC147 (json).
      Key thường gặp:
        - 'image_path' : đường dẫn tương đối, ví dụ 'images_384_VarV2/36.jpg'
        - 'points'     : list toạ độ GT [(x,y), ...] trên ảnh gốc
        - 'box_examples_coordinates': list 3 boxes, mỗi box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        - 'class' hoặc 'class_name' : tên lớp (prompt text)

    img_root: thư mục gốc FSC147 (chứa folder ảnh).
    img_size: kích thước input cho GDCount (384).

    return:
      img_tensor        : (3, img_size, img_size)
      prompt_text       : string
      exemplar_boxes    : Tensor (N_box, 4) trên ảnh RESIZED [x1,y1,x2,y2]
      points_rescaled   : Tensor (N_points, 2) GT trên ảnh RESIZED
      meta              : dict (H_orig, W_orig, image_path, scale_x, scale_y, ...)
    """

    # ----- 1) Load ảnh gốc từ images_384_VarV2 -----
    rel_path = sample.get("image_path") or sample.get("img_path")
    img_path = os.path.join(img_root, rel_path)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H_orig, W_orig = img_rgb.shape[:2]

    # ----- 2) Resize về 384x384 + normalize -----
    img_tensor, (H_resized, W_resized) = _to_tensor_and_normalize(img_rgb, img_size)

    # Do đã ép về (384,384), nên:
    scale_x = W_resized / W_orig
    scale_y = H_resized / H_orig

    # ----- 3) Prompt text -----
    prompt_text = sample.get("class", None) or sample.get("class_name", "")
    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0]

    # ----- 4) Exemplar boxes -----
    raw_boxes = sample.get("box_examples_coordinates", [])
    box_list = []
    for box in raw_boxes:
        box = np.array(box, dtype=np.float32)  # (4,2)
        xs = box[:, 0]
        ys = box[:, 1]
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        # scale từ gốc -> resized (384x384)
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        box_list.append([x1, y1, x2, y2])

    if len(box_list) > 0:
        exemplar_boxes = torch.tensor(box_list, dtype=torch.float32)
    else:
        exemplar_boxes = torch.empty((0, 4), dtype=torch.float32)

    # ----- 5) GT points -----
    raw_points = sample.get("points", [])
    pts_list = []
    for p in raw_points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = float(p[0]), float(p[1])
        elif isinstance(p, dict):
            x, y = float(p["x"]), float(p["y"])
        else:
            continue
        x *= scale_x
        y *= scale_y
        pts_list.append([x, y])

    if len(pts_list) > 0:
        points_rescaled = torch.tensor(pts_list, dtype=torch.float32)
    else:
        points_rescaled = torch.empty((0, 2), dtype=torch.float32)

    meta = {
        "image_path": img_path,
        "H_orig": H_orig,
        "W_orig": W_orig,
        "H_resized": H_resized,
        "W_resized": W_resized,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "class_name": prompt_text,
    }

    return img_tensor, prompt_text, exemplar_boxes, points_rescaled, meta

# ======================================================
# 2. PREPROCESSING CHO ẢNH INFERENCE BẤT KỲ
# ======================================================

def preprocess_inference_image(img_path: str,
                               img_size: int = 384):
    """
    img_path: đường dẫn ảnh bất kỳ (jpg/png/...).
    img_size: kích thước input cho GDCount (384).

    return:
      img_tensor      : (3,img_size,img_size), normalized
      resized_size    : (H_resized, W_resized) = (img_size, img_size)
      orig_size       : (H_orig, W_orig)
      img_bgr_orig    : ảnh gốc BGR (để vẽ bbox, heatmap)
      scale_x,scale_y : factor từ ảnh gốc -> ảnh resized
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H_orig, W_orig = img_rgb.shape[:2]

    img_tensor, (H_resized, W_resized) = _to_tensor_and_normalize(img_rgb, img_size)

    scale_x = W_resized / W_orig
    scale_y = H_resized / H_orig

    return img_tensor, (H_resized, W_resized), (H_orig, W_orig), img_bgr, scale_x, scale_y
