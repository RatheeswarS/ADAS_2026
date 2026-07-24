"""
lane_continuity.py
---------------------
"Lane Continuity Prediction": uses the previous frame's fitted boundary
curve to predict where that same boundary should be this frame, so
candidate selection can search near the prediction instead of re-deciding
from scratch (a fixed left/right-of-image-center partition) every single
frame. This is the role a classical sliding-window pipeline's "search
within a margin of the previous fit" trick plays -- here it scores the
model's already-computed candidate blobs the same way.

Falls back to center-based cold-start selection (in bird_eye_extraction.py)
when no previous curve exists -- the very first frame, or after track loss
-- which this module has no effect on.
"""
from typing import Optional

from perception.lane_core.curves import LaneCurve


def predicted_position(prev_curve: Optional[LaneCurve], y: float) -> Optional[float]:
    if prev_curve is None:
        return None
    return float(prev_curve.x_at(y))
