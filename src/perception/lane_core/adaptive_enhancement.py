"""
adaptive_enhancement.py
--------------------------
Gamma + CLAHE brightness/contrast lift, applied ONLY to frames measured as
dark (mean pixel value below a threshold) -- well-lit frames pass through
untouched. This is a genuine photo-domain transform (brightness/contrast),
not a binarization, so it stays close enough to what YOLOP was trained on
to be safe to feed the model, unlike thresholding the image down to "white
lines only" (see color_filter.py's docstring for why that idea was
rejected as network input).

A prior attempt at exactly this kind of enhancement (a related project's
--enhance-low-light flag) found it only marginally recovers detections on
genuinely near-black footage -- a real but partial mitigation, not a fix
for severe low light. Kept here for the more common case of merely dim
(not pitch-black) frames.
"""
import numpy as np
import cv2

ENHANCE_MEAN_THRESHOLD = 60.0  # only enhance frames darker than this (0-255)
ENHANCE_GAMMA = 2.2
ENHANCE_CLIP_LIMIT = 3.0
ENHANCE_TILE_GRID = (8, 8)


def _apply_gamma_lut(l_channel: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(l_channel, lut)


def enhance_if_dark(frame_bgr: np.ndarray, mean_threshold: float = ENHANCE_MEAN_THRESHOLD,
                     gamma: float = ENHANCE_GAMMA) -> np.ndarray:
    if frame_bgr.mean() >= mean_threshold:
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _apply_gamma_lut(l, gamma)
    clahe = cv2.createCLAHE(clipLimit=ENHANCE_CLIP_LIMIT, tileGridSize=ENHANCE_TILE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Night mode: a stronger low-light front-end than enhance_if_dark
# ---------------------------------------------------------------------------
# Rationale (see refs): the SOTA nighttime-lane-detection pipelines (FLLENet,
# MDPI "Combining Low-Light Scene Enhancement...") put a low-light ENHANCER in
# front of the lane network rather than changing the detector. They repeatedly
# note that a naive brightness lift also amplifies sensor noise into false
# lane-line texture, so denoising after the lift is part of the recipe -- which
# enhance_if_dark above does NOT do. This variant fixes the three concrete gaps
# for genuinely night footage while staying a pure photo-domain transform (safe
# as YOLOP input, unlike binarizing to "white only"):
#   1. engages on a much wider brightness range (dim, not just near-black), and
#      near the boundary the strength tapers to ~identity, so no on/off flicker;
#   2. adapts gamma to how dark the frame actually is (gentle when dim, strong
#      when near-black) instead of one fixed gamma;
#   3. edge-preserving denoise AFTER the lift, to kill amplified luminance noise
#      while keeping lane edges crisp.
# This is a real but PARTIAL mitigation (as the module note above already warns
# for the milder version): it recovers a lot of merely-dim footage, but a
# learned enhancer (Zero-DCE++) or a night-fine-tuned YOLOP is the higher
# ceiling for near-black scenes.
NIGHT_MEAN_THRESHOLD = 120.0   # engage on merely-dim frames too, not just near-black
NIGHT_GAMMA_MIN = 1.6          # gentle lift for a dim frame (mean near the threshold)
NIGHT_GAMMA_MAX = 3.0          # strong lift for a near-black frame
NIGHT_DARK_REF = 20.0          # frame-mean at/below which the strongest gamma is used
NIGHT_CLIP_LIMIT = 2.0         # gentler CLAHE than the dim-mode version (noise control)
NIGHT_TILE_GRID = (8, 8)


def _adaptive_gamma(mean_val: float) -> float:
    """Darker frame -> larger gamma (stronger brightening). Linear in frame
    mean between NIGHT_DARK_REF (=> NIGHT_GAMMA_MAX) and NIGHT_MEAN_THRESHOLD
    (=> NIGHT_GAMMA_MIN, i.e. ~identity so the engage boundary doesn't flicker)."""
    span = max(NIGHT_MEAN_THRESHOLD - NIGHT_DARK_REF, 1e-6)
    t = float(np.clip((mean_val - NIGHT_DARK_REF) / span, 0.0, 1.0))
    return NIGHT_GAMMA_MAX + t * (NIGHT_GAMMA_MIN - NIGHT_GAMMA_MAX)


def enhance_for_night(frame_bgr: np.ndarray, mean_threshold: float = NIGHT_MEAN_THRESHOLD,
                       denoise: bool = True) -> np.ndarray:
    """Stronger low-light enhancement for night footage. No-op on frames already
    brighter than mean_threshold. Pass denoise=False to drop the (edge-
    preserving but non-trivial-cost) bilateral filter on a slow platform."""
    mean_val = float(frame_bgr.mean())
    if mean_val >= mean_threshold:
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _apply_gamma_lut(l, _adaptive_gamma(mean_val))
    clahe = cv2.createCLAHE(clipLimit=NIGHT_CLIP_LIMIT, tileGridSize=NIGHT_TILE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if denoise:
        # Edge-preserving: removes the luminance noise the brightening lift
        # amplifies, without smearing the lane edges the way a Gaussian would.
        # Bilateral is real-time-affordable per frame; fastNlMeans is higher
        # quality but too slow for a live Jetson path.
        out = cv2.bilateralFilter(out, d=5, sigmaColor=60, sigmaSpace=60)
    return out
