"""
roi_mask.py
-------------
Zeroes out the sky/building region of a lane-line probability map before
boundary extraction -- the road is never up there, so any model activation
in the top portion of frame (a building edge, a tree line, sky glare) is
definitely not a lane and should never be able to compete with a real
near-road detection.

Applied to the model's OUTPUT (the probability map), not the input frame --
consistent with this project's established rule (see color_filter.py's
docstring) of never feeding YOLOP a transformed image that departs from
what it was trained on. Masking its output costs nothing on that front.
"""
import numpy as np

ROI_TOP_FRAC = 0.35  # ignore the top 35% of the frame (sky, distant buildings/tree canopy)


def apply_roi_mask(prob_map: np.ndarray, top_frac: float = ROI_TOP_FRAC) -> np.ndarray:
    h = prob_map.shape[0]
    cutoff = int(h * top_frac)
    masked = prob_map.copy()
    masked[:cutoff, :] = 0.0
    return masked
