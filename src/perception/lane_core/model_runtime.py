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


class TRTLaneModel:
    """Runs the lane-only TensorRT engine produced by export_yolop_onnx.py +
    trtexec. Exposes the same (1,2,H,W) ll_seg logits that the PyTorch model's
    third head returns, so run_inference()'s pre/post-processing is unchanged.

    Engine I/O is FP32 by default (trtexec --fp16 keeps network I/O in fp32 and
    reformats internally), so this consumes an fp32 input and yields an fp32
    output regardless of the checkpoint's `half` flag."""
    is_trt = True

    def __init__(self, engine_path: str, device):
        import tensorrt as trt  # imported lazily so the pure-PyTorch path never needs TRT
        self.trt = trt
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        # Resolve the single input / single output tensor names by I/O mode.
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))

    def warmup(self):
        """Run one inference on a correctly-shaped zero tensor so the first real
        frame doesn't pay lazy-init cost. Uses the engine's own input shape --
        the node's square warmup frame would letterbox to the wrong size."""
        self.infer(torch.zeros(self.in_shape, dtype=torch.float32, device=self.device))

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1,3,H,W) fp32 CUDA tensor -> (1,2,H,W) fp32 CUDA tensor."""
        x = x.contiguous()
        out = torch.empty(self.out_shape, dtype=torch.float32, device=self.device)
        self.context.set_input_shape(self.in_name, tuple(x.shape))
        self.context.set_tensor_address(self.in_name, x.data_ptr())
        self.context.set_tensor_address(self.out_name, out.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out


def load_model(weights_path: str, device):
    # A .engine path selects the TensorRT runtime (see README "TensorRT
    # acceleration"); anything else is the PyTorch checkpoint.
    if weights_path.endswith(".engine"):
        if device.type == "cpu":
            raise ValueError("TensorRT engine requires a CUDA device, got cpu")
        return TRTLaneModel(weights_path, device)
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

    if getattr(model, "is_trt", False):
        # Engine I/O is fp32; `half` does not apply to the TRT path.
        ll_seg_out = model.infer(x.float())
    else:
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
