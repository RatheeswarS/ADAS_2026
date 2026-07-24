"""
curves.py
-----------
Quadratic lane-boundary curve fit, ported from yolop final/lane_pipeline.py
unchanged (same math, same thresholds).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_FIT_POINTS = 40        # min pixels needed before trusting a quadratic fit
MIN_FIT_ROW_SPAN = 15       # those pixels must span at least this many rows


@dataclass
class LaneCurve:
    coeffs: np.ndarray   # np.polyfit(y, x, 2) -> [a, b, c], x = a*y^2+b*y+c
    y_min: int
    y_max: int
    confidence: float
    reconstructed: bool
    # The (y, x) sample points the fit was actually derived from -- kept so
    # alternate heading strategies can re-derive a slope from the raw data
    # instead of only the already-fitted coefficients (e.g. refitting in a
    # warped bird's-eye coordinate system -- see lane_core/ipm.py). None only
    # if a curve somehow gets built without them, which shouldn't happen in
    # practice.
    raw_ys: Optional[np.ndarray] = None
    raw_xs: Optional[np.ndarray] = None

    def x_at(self, y):
        return np.polyval(self.coeffs, y)

    def slope_at(self, y):
        a, b, _ = self.coeffs
        return 2 * a * y + b

    def shifted(self, dx_at):
        """New curve offset by dx_at(y) (a callable), refit as a quadratic
        over this curve's own y-range. Returns (coeffs, ys, xs) so the
        caller can build a LaneCurve with raw_ys/raw_xs populated too."""
        ys = np.linspace(self.y_min, self.y_max, 25)
        xs = self.x_at(ys) + dx_at(ys)
        coeffs = np.polyfit(ys, xs, 2)
        return coeffs, ys, xs


def fit_lane_curve(points_y: np.ndarray, points_x: np.ndarray,
                    confidence_values: np.ndarray) -> Optional[LaneCurve]:
    if len(points_y) < MIN_FIT_POINTS:
        return None
    y_min, y_max = int(points_y.min()), int(points_y.max())
    if y_max - y_min < MIN_FIT_ROW_SPAN:
        return None
    coeffs = np.polyfit(points_y, points_x, 2)
    return LaneCurve(coeffs=coeffs, y_min=y_min, y_max=y_max,
                      confidence=float(confidence_values.mean()), reconstructed=False,
                      raw_ys=points_y, raw_xs=points_x)
