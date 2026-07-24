"""
model_runtime.py
-------------------
YOLOP model loading + single-frame inference, ported from
yolop final/lane_pipeline.py unchanged. Only the lane-line segmentation head
is used; the object-detection and drivable-area heads are computed by the
model (it's multi-task) but not consumed here.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

YOLOP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "YOLOP")
sys.path.insert(0, YOLOP_DIR)

from lib.config import cfg              # noqa: E402
from lib.models import get_net          # noqa: E402
from lib.utils.utils import select_device  # noqa: E402,F401  (re-exported for node convenience)
from lib.utils.augmentations import letterbox_for_img  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(weights_path: str, device):
    model = get_net(cfg)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, frame_bgr, img_size: int, device, half: bool):
    """Returns (ll_mask, ll_confidence_map) at the frame's original resolution."""
    h0, w0 = frame_bgr.shape[:2]
    img, _, (pad_w, pad_h) = letterbox_for_img(frame_bgr, new_shape=img_size, auto=True)
    img = np.ascontiguousarray(img)

    x = to_tensor(img).unsqueeze(0).to(device)
    x = x.half() if half else x.float()

    with torch.no_grad():
        _, _, ll_seg_out = model(x)

    _, _, hp, wp = x.shape
    pad_w, pad_h = int(pad_w), int(pad_h)
    ll_cropped = ll_seg_out[:, :, pad_h:hp - pad_h, pad_w:wp - pad_w]
    ll_resized = F.interpolate(ll_cropped.float(), size=(h0, w0), mode="bilinear", align_corners=False)
    ll_prob = torch.softmax(ll_resized, dim=1)[0, 1]  # P(lane) per pixel, (h0, w0)

    confidence_map = ll_prob.cpu().numpy()
    mask = (confidence_map > 0.5).astype(np.uint8)
    return mask, confidence_map
