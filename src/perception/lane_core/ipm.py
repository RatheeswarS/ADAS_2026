"""
ipm.py
--------
Approximate, UNCALIBRATED bird's-eye (inverse perspective mapping) view.
Ported from yolop final/lane_pipeline.py unchanged.

No real camera intrinsics/extrinsics are available -- this assumes a flat
road and picks a plausible forward-facing trapezoid, so the warped space's
absolute scale is arbitrary. It removes row-dependent perspective
compression, which is what let a small fit-curvature difference swing the
raw heading term from a few degrees to well past the steering clamp (see
lane_core/steering.py) -- the warped slope should be read as "proportional
to real curvature," not as a calibrated real-world angle.

RECALIBRATED for the real stereo camera (d_dash.mp4, ZED left eye). The
original fractions (top_frac 0.60, top_half 0.10) were tuned to an earlier
dashcam and are BROKEN for this camera: measured over 40 frames spanning the
clip, that trapezoid warped BOTH lane lines fully out of the bird's-eye canvas
95% of the time (only 5% of frames had a left+right lane pair visible in the
warp). The lane lines on this camera live in the ~35-75% vertical band, spread
across most of the width -- the old trapezoid (top edge at 60% height, only 20%
of the width wide) looked below and inside all of it, capturing empty centre
asphalt. Raising the top edge to ~42% and widening it to ~80% of the width
brought both-lanes-visible up to ~68% (a 14x improvement). See the
BirdEyeLaneDetectionNode auto-calibration below, which searches this axis at
runtime and locks the zoom level that best reveals the lanes.
"""
import numpy as np
import cv2

# Base (recalibrated) trapezoid -- also the fixed transform used when adaptive
# IPM zoom is off. top_frac = how far DOWN the image the top edge sits (smaller
# = looks further ahead / higher up); *_half_width_frac = half-width of that
# edge as a fraction of frame width.
IPM_SRC_TOP_FRAC = 0.42
IPM_SRC_TOP_HALF_WIDTH_FRAC = 0.40
IPM_SRC_BOTTOM_HALF_WIDTH_FRAC = 0.50
IPM_WARPED_SIZE = (400, 600)  # (width, height)

# --- adaptive "zoom" auto-calibration (see BirdEyeLaneDetectionNode) ---------
# The source trapezoid IS the zoom. Two axes matter, and the dominant one for
# this camera turned out to be VERTICAL, not horizontal: the lanes sit ABOVE
# the old trapezoid's top edge, so widening alone never reached them -- the top
# edge has to be RAISED (smaller top_frac) as well. This ladder sweeps both,
# from the legacy zoomed-in framing (index 0) to a wide, high, nearly-full-road
# capture (last index). The node warps the live YOLOP probability map through
# each rung, scores how well both lanes show up, and locks the best rung.
IPM_ZOOM_LADDER = [
    (0.60, 0.10),  # 0  legacy zoomed-in (kept so a well-framed camera can pick it)
    (0.54, 0.22),  # 1
    (0.50, 0.30),  # 2
    (0.46, 0.36),  # 3
    (0.42, 0.40),  # 4  ~ recalibrated base
    (0.40, 0.45),  # 5  most zoomed-out (widest/highest capture)
]
IPM_BASE_RUNG = 4      # matches the recalibrated base fractions above; fallback while calibrating
IPM_CALIB_FRAMES = 12  # accumulate rung scores over this many frames, then lock the best

# NOTE on scoring (done in BirdEyeLaneDetectionNode, not here): a rung is scored
# by whether the REAL boundary extractor finds BOTH lanes through it -- NOT by a
# raw warped-pixel count. A raw count is biased toward zoomed-IN rungs, which
# magnify the same pixels into a larger count; that is exactly backwards, and an
# earlier version of this that used min(left_px, right_px) locked onto a too-
# zoomed-in rung that detected worse than the fixed base. "Did extraction find a
# left AND a right boundary" is scale-invariant and matches what the pipeline
# actually needs.


def build_ipm_transform(frame_w: int, frame_h: int,
                        top_frac: float = IPM_SRC_TOP_FRAC,
                        top_half_width_frac: float = IPM_SRC_TOP_HALF_WIDTH_FRAC,
                        bottom_half_width_frac: float = IPM_SRC_BOTTOM_HALF_WIDTH_FRAC) -> np.ndarray:
    src = np.float32([
        [frame_w / 2 - frame_w * top_half_width_frac, frame_h * top_frac],
        [frame_w / 2 + frame_w * top_half_width_frac, frame_h * top_frac],
        [frame_w / 2 + frame_w * bottom_half_width_frac, frame_h - 1],
        [frame_w / 2 - frame_w * bottom_half_width_frac, frame_h - 1],
    ])
    ww, wh = IPM_WARPED_SIZE
    dst = np.float32([[0, 0], [ww - 1, 0], [ww - 1, wh - 1], [0, wh - 1]])
    return cv2.getPerspectiveTransform(src, dst)


def _warp_points(ys, xs, M):
    pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    return warped[:, 1], warped[:, 0]  # (ys_warped, xs_warped)


def _ipm_slope(curve, M):
    if curve.raw_ys is None or len(curve.raw_ys) < 3:
        return None
    ys_w, xs_w = _warp_points(curve.raw_ys, curve.raw_xs, M)
    valid = np.isfinite(ys_w) & np.isfinite(xs_w)
    ys_w, xs_w = ys_w[valid], xs_w[valid]
    if len(ys_w) < 3 or (ys_w.max() - ys_w.min()) < 5:
        return None
    a, b, _ = np.polyfit(ys_w, xs_w, 2)
    eval_y_w = float(ys_w.max())  # nearest-the-vehicle row, in warped space
    return 2 * a * eval_y_w + b


def invert_transform(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)


def warp_to_birdeye(frame_bgr: np.ndarray, prob_map: np.ndarray, M: np.ndarray):
    """
    Dense warp of BOTH the color frame and the lane-line probability map
    into the same fixed bird's-eye canvas (IPM_WARPED_SIZE), so downstream
    boundary extraction, color/whiteness checks, and curve fitting all
    operate in one consistent coordinate system instead of perspective-
    space extraction plus a separate after-the-fact warp used only for
    heading (the earlier, narrower use of IPM in this project).

    Returns (warped_frame_bgr, warped_prob_map), both shaped IPM_WARPED_SIZE
    (height, width order matches cv2's (width, height) size convention
    handled internally).
    """
    ww, wh = IPM_WARPED_SIZE
    warped_frame = cv2.warpPerspective(frame_bgr, M, (ww, wh))
    warped_prob = cv2.warpPerspective(prob_map.astype(np.float32), M, (ww, wh), flags=cv2.INTER_LINEAR)
    return warped_frame, warped_prob
