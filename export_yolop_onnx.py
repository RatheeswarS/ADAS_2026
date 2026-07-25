"""
export_yolop_onnx.py
--------------------
Export ONLY the YOLOP lane-line segmentation head to ONNX, so the resulting
graph skips the object-detection Detect layer's grid/anchor decoding (the part
that makes full YOLOP a pain to export) and the unused drivable-area head.

Our runtime only consumes ll_seg (see model_runtime.run_inference), so a
lane-only graph is smaller, exports cleanly, and builds a faster TRT engine.

Static input shape (1,3,384,640) matches a 1280x720 camera through
letterbox_for_img(new_shape=640, auto=True). Change H/W below if your camera
resolution differs (run a frame through letterbox and print img.shape).
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "YOLOP"))

from perception.lane_core.model_runtime import load_model  # noqa: E402

WEIGHTS = os.path.join(HERE, "YOLOP", "weights", "End-to-end.pth")
OUT = os.path.join(HERE, "YOLOP", "weights", "yolop_ll_384x640.onnx")
H, W = 384, 640            # letterboxed input for a 1280x720 camera


class LaneOnly(nn.Module):
    """Wraps YOLOP so forward() returns only the raw lane-seg logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        det, da_seg, ll_seg = self.model(x)   # multi-task heads
        return ll_seg                          # raw (1,2,H,W) logits only


def main():
    device = torch.device("cpu")               # export on CPU for portability
    model = load_model(WEIGHTS, device).float().eval()
    wrapped = LaneOnly(model).eval()

    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)
    torch.onnx.export(
        wrapped, dummy, OUT,
        input_names=["images"], output_names=["ll_seg"],
        opset_version=12, do_constant_folding=True,
        dynamic_axes=None,                     # static shapes -> fastest engine
    )
    print(f"[ok] wrote {OUT}")

    # sanity-check the graph loads and runs
    import onnx
    onnx.checker.check_model(onnx.load(OUT))
    print("[ok] onnx.checker passed")


if __name__ == "__main__":
    main()
