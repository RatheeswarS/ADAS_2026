"""
color_filter.py
------------------
Checks whether a candidate boundary blob's pixels actually look like painted
white lane paint in the ORIGINAL color frame -- bright and desaturated --
rather than trusting position alone.

Built to fix a real, confirmed failure mode (see diagnose_curb_confusion.py):
boundary_extraction.py's existing gates (reach-bottom, long-enough,
nearest-to-center) are all purely positional and have no visibility into
color at all, so a strong-enough edge in YOLOP's segmentation mask -- e.g. a
bare-asphalt/dirt-shoulder edge, or a curb -- can pass every positional gate
and get chosen as "the" boundary even though it isn't painted at all. A
24-frame, 1870-frame diagnostic scan (worst case: 0% white pixels in the
chosen boundary) confirmed this happens in practice, not just in theory.

This deliberately operates on the ORIGINAL frame at candidate-selection time
-- NOT as a preprocessing step feeding a transformed image into YOLOP.
Feeding the network a thresholded/binarized "white lines only" image instead
of a natural photo would push the input far outside what the model was
trained on (BDD100K dashcam photos), almost certainly making segmentation
worse across the board rather than fixing this one failure mode. Filtering
candidates by real pixel color *after* segmentation keeps the network's
input untouched and only affects which already-detected blob gets trusted.
"""
import numpy as np
import cv2

WHITENESS_MIN_VALUE = 120       # HSV V (brightness), 0-255
WHITENESS_MAX_SATURATION = 90   # HSV S (saturation), 0-255 -- low = white/gray, not colored
MIN_WHITE_FRACTION = 0.5        # a blob needs at least this fraction of "white-like" pixels


def frame_to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Compute once per frame and reuse across every candidate blob's check,
    rather than reconverting the whole frame per-candidate."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)


def whiteness_fraction(hsv_frame: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> float:
    """0-1 fraction of the given pixel locations that are bright and
    desaturated enough to plausibly be painted white/light-colored lane
    marking, as opposed to bare asphalt, dirt, or a curb edge."""
    if len(ys) == 0:
        return 0.0
    sat = hsv_frame[ys, xs, 1].astype(np.float32)
    val = hsv_frame[ys, xs, 2].astype(np.float32)
    white_like = (val >= WHITENESS_MIN_VALUE) & (sat <= WHITENESS_MAX_SATURATION)
    return float(white_like.mean())


def is_white_enough(hsv_frame: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> bool:
    return whiteness_fraction(hsv_frame, ys, xs) >= MIN_WHITE_FRACTION
