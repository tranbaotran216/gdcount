import os
import cv2
import torch
from gdcount.model import GDCount, GDCountConfig

def load_and_preprocess(img_path, size=800):
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # resize sao cho cạnh ngắn = size (giống paper) :contentReference[oaicite:7]{index=7}
    if min(h, w) != size:
        scale = size / min(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (3,H,W)

    # chuẩn hoá tương tự GroundingDINO
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0), (new_h, new_w), img_bgr