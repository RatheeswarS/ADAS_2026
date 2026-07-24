"""
calibration.py
-----------------
Learns pixel-width(y) from frames where both lane boundaries are seen, so a
missing boundary can be reconstructed later using the real 3m lane width.
Ported from yolop final/lane_pipeline.py unchanged.
"""
from typing import Optional

import numpy as np

from perception.lane_core.curves import LaneCurve

MIN_CALIBRATION_FRAMES = 15  # distinct both-boundaries-seen frames, not just raw point count


class LaneWidthCalibrator:
    def __init__(self, max_samples: int = 4000, poly_order: int = 1):
        # Linear, not quadratic: perspective narrowing over the vertical
        # extent actually visible in a dashcam-style frame is close enough
        # to affine that a line is a good fit, and a line is far less prone
        # to the "sparse/noisy data at one end -> the fit swings wildly
        # there" overfitting a quadratic showed in practice (a calibrated
        # width of ~50px near the bottom row of an 832px-wide frame, clearly
        # implausible, traced back to this).
        self.poly_order = poly_order
        self.max_samples = max_samples
        self._ys: list = []
        self._ws: list = []
        self._n_frames = 0
        self.coeffs: Optional[np.ndarray] = None

    def observe(self, left: LaneCurve, right: LaneCurve, n_rows: int = 20):
        y0, y1 = max(left.y_min, right.y_min), min(left.y_max, right.y_max)
        if y1 <= y0:
            return
        ys = np.linspace(y0, y1, n_rows)
        widths = right.x_at(ys) - left.x_at(ys)
        keep = widths > 5  # guard against a degenerate/crossed fit
        self._ys.extend(ys[keep].tolist())
        self._ws.extend(widths[keep].tolist())
        self._n_frames += 1
        if len(self._ys) > self.max_samples:
            self._ys = self._ys[-self.max_samples:]
            self._ws = self._ws[-self.max_samples:]
        # A single frame already contributes up to n_rows=20 point-samples,
        # which used to be enough by itself to fit the quadratic (needs only
        # 9). That let one noisy frame's fit get extrapolated as "the"
        # width(y) model for every reconstruction until later frames diluted
        # it. Require several distinct frames of agreement first; until then,
        # fall back to the plain running mean (see width_at), which one bad
        # frame skews far less.
        if self._n_frames >= MIN_CALIBRATION_FRAMES and len(self._ys) >= (self.poly_order + 1) * 3:
            self.coeffs = np.polyfit(self._ys, self._ws, self.poly_order)

    def width_at(self, y, frame_width: int):
        """Vectorized: `y` may be a scalar or an array. The result is always
        clamped to a plausible pixel range -- the quadratic width(y) model is
        fit only over the rows where both boundaries have actually been seen
        together, and reconstructing a boundary can need widths at rows well
        outside that observed range (e.g. close to the camera, where only
        one line has stayed in frame for a long stretch). Quadratic
        extrapolation outside its fitted domain is unbounded and blew up in
        practice; clamping to a fraction of the frame width keeps the
        reconstructed boundary on-screen and physically sane regardless."""
        min_w, max_w = frame_width * 0.05, frame_width * 0.95
        if self.coeffs is not None:
            w = np.polyval(self.coeffs, y)
        elif self._ws:
            w = np.full_like(np.asarray(y, dtype=float), np.mean(self._ws))
        else:
            # Cold start, no calibration data yet: a rough placeholder guess.
            w = np.full_like(np.asarray(y, dtype=float), frame_width * 0.35)
        return np.clip(w, min_w, max_w)
