#!/usr/bin/env python3
"""
test_lane_hold_gate.py
------------------------
Unit tests for the temporal-consistency gate + time-bounded coast added to
BirdEyeLaneDetectionNode (the "hold-through-dropout" change): stop a road-
pavement / shoulder edge from masquerading as a real `direct` lane when a real
line is missing, and coast on trusted history for a bounded buffer instead.

These exercise the ISOLATED gate helpers only -- no YOLOP / torch model load.
The two instance-method gates (`_lateral_jump_ok`, `_pair_width_ok`) depend on
just `self.calibrator` and `self.lane_width_m`, so we build a bare node via
`object.__new__` and attach those two attributes rather than loading a model.
`_is_partial` is a staticmethod and is called directly.

Run:  python tests/test_lane_hold_gate.py    (plain asserts, no pytest needed)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))

from perception.lane_core.curves import LaneCurve
from perception.lane_core.calibration import LaneWidthCalibrator, MIN_CALIBRATION_FRAMES
from perception.lane_core.bird_eye_extraction import MIN_BOTTOM_REACH_FRAC
from perception.lane_detection_node_birdeye import (
    BirdEyeLaneDetectionNode, LATERAL_JUMP_MARGIN_M,
)

WW = 400          # warped canvas width  (IPM_WARPED_SIZE[0])
WH = 600          # warped canvas height (IPM_WARPED_SIZE[1])
BOTTOM_Y = WH - 1


def _line(x_at_all_rows, y_min=0, y_max=WH - 1):
    """A perfectly vertical LaneCurve at constant x = x_at_all_rows."""
    return LaneCurve(coeffs=np.array([0.0, 0.0, float(x_at_all_rows)]),
                     y_min=int(y_min), y_max=int(y_max),
                     confidence=0.9, reconstructed=False,
                     raw_ys=None, raw_xs=None)


def _bare_node(calibrator=None, lane_width_m=3.0):
    node = object.__new__(BirdEyeLaneDetectionNode)
    node.calibrator = calibrator or LaneWidthCalibrator()
    node.lane_width_m = lane_width_m
    return node


def _warm_calibrator(width_px=140.0, n=MIN_CALIBRATION_FRAMES + 5):
    """Return a calibrator whose width(y) model has locked onto `width_px`."""
    cal = LaneWidthCalibrator()
    for _ in range(n):
        cal.observe(_line(100.0), _line(100.0 + width_px))
    assert cal.coeffs is not None, "calibrator should be warmed"
    return cal


# ---------------------------------------------------------------------------
# _is_partial: keyed on bottom-REACH, not raw span
# ---------------------------------------------------------------------------
def test_is_partial_reaching_bottom_is_not_partial():
    # A short blob that still reaches the near-vehicle band (e.g. a dashed
    # segment near the car) must NOT be flagged partial. The extractor's reach
    # band starts at (1 - MIN_BOTTOM_REACH_FRAC)*h; a fit whose y_max is at/below
    # it counts as bottom-reaching -> direct.
    reach_y = int(WH * (1.0 - MIN_BOTTOM_REACH_FRAC))
    dash = _line(120.0, y_min=reach_y + 10, y_max=WH - 1)   # short but at the bottom
    assert BirdEyeLaneDetectionNode._is_partial(dash, WH) is False


def test_is_partial_high_arc_is_partial():
    # An arc that stops short of the extractor's reach band (outer line on a
    # turn) is partial: reading CTE at the bottom would extrapolate past its data.
    reach_y = int(WH * (1.0 - MIN_BOTTOM_REACH_FRAC))
    arc = _line(300.0, y_min=40, y_max=reach_y - 40)
    assert BirdEyeLaneDetectionNode._is_partial(arc, WH) is True


# ---------------------------------------------------------------------------
# _lateral_jump_ok: reject a pavement-edge-sized lateral jump
# ---------------------------------------------------------------------------
def test_lateral_jump_none_prev_is_noop():
    node = _bare_node()
    assert node._lateral_jump_ok(_line(100.0), None, WW, BOTTOM_Y, dt=0.033, speed_mps=5.0) is True


def test_lateral_jump_small_jitter_accepted():
    node = _bare_node()
    prev = _line(100.0)
    fresh = _line(105.0)   # ~5 px ~ 0.1 m at cold-start scale -> well under the margin
    assert node._lateral_jump_ok(fresh, prev, WW, BOTTOM_Y, dt=0.033, speed_mps=5.0) is True


def test_lateral_jump_full_lane_jump_rejected():
    # A ~one-lane-width lateral jump at low speed is the pavement latch we reject.
    cal = _warm_calibrator(width_px=140.0)   # 140 px ~ 3.0 m -> ~0.0214 m/px
    node = _bare_node(cal)
    prev = _line(100.0)
    fresh = _line(240.0)   # +140 px ~ 3 m, far beyond speed*dt + margin
    assert node._lateral_jump_ok(fresh, prev, WW, BOTTOM_Y, dt=0.033, speed_mps=5.0) is False


def test_lateral_jump_loosens_with_speed_and_dt():
    cal = _warm_calibrator(width_px=140.0)
    node = _bare_node(cal)
    prev, fresh = _line(100.0), _line(140.0)   # +40 px ~ 0.86 m jump
    # Tight budget (slow, short frame) rejects it...
    assert node._lateral_jump_ok(fresh, prev, WW, BOTTOM_Y, dt=0.02, speed_mps=1.0) is False
    # ...a large elapsed time / high speed loosens the bound and accepts it.
    assert node._lateral_jump_ok(fresh, prev, WW, BOTTOM_Y, dt=1.0, speed_mps=30.0) is True


# ---------------------------------------------------------------------------
# _pair_width_ok: no-op until warmed, then rejects an implausible pair width
# ---------------------------------------------------------------------------
def test_pair_width_noop_until_calibrated():
    node = _bare_node(LaneWidthCalibrator())   # cold: coeffs is None
    assert node.calibrator.coeffs is None
    # Even a wildly wrong pair passes while the model is un-warmed.
    assert node._pair_width_ok(_line(100.0), _line(400.0), WW, BOTTOM_Y) is True


def test_pair_width_rejects_too_wide_pair():
    cal = _warm_calibrator(width_px=140.0)
    node = _bare_node(cal)
    left, right = _line(100.0), _line(100.0 + 210.0)   # 1.5x calibrated width
    assert node._pair_width_ok(left, right, WW, BOTTOM_Y) is False


def test_pair_width_accepts_in_tolerance():
    cal = _warm_calibrator(width_px=140.0)
    node = _bare_node(cal)
    left, right = _line(100.0), _line(100.0 + 160.0)   # ~1.14x -> within 0.35 tol
    assert node._pair_width_ok(left, right, WW, BOTTOM_Y) is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
