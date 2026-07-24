"""
conditioner.py
-----------------
Signal conditioning between perception and temporal filtering, per the
architecture's "temporal filtering / outlier rejection / dropout handling /
confidence gating" block.

This is deliberately a gate that runs BEFORE the Kalman filter
(kalman_tracker.py), not a replacement for it: the filter is good at
smoothing plausible noisy measurements and extrapolating through brief
gaps, but it isn't designed to catch a single wildly-wrong frame (e.g. a
momentarily crossed/garbage curve fit that slips past the extraction-layer
checks) -- feeding that straight into the filter lets its innovation
partially corrupt the tracked state. SignalConditioner's job is narrower:
decide, per frame, whether this raw measurement should be trusted at all;
a rejection is treated exactly like a dropout (None, None into the
filter), reusing the tracker's own already-validated predict-through-gap
behavior rather than inventing a second one.

Two independent checks, either one rejects the frame:
  1. Confidence gate: raw confidence below MIN_CONFIDENCE.
  2. Plausibility gate: cte/heading changed further than the vehicle could
     plausibly have moved in dt at MAX_CTE_RATE_M_S / MAX_HEADING_RATE_DPS
     (each generous, plus a fixed margin for measurement noise).

Thresholds below are placeholders in the same spirit as every other
hand-picked gain in this project (K_GAIN, Q_CTE_INTENSITY, ...): reasonable
starting points, not bench-tuned against the real vehicle.
"""
from typing import Optional

MIN_CONFIDENCE = 0.1

MAX_CTE_RATE_M_S = 3.0
CTE_JUMP_MARGIN_M = 0.3

MAX_HEADING_RATE_DPS = 120.0
HEADING_JUMP_MARGIN_DEG = 5.0


class SignalConditioner:
    def __init__(self):
        self._last_cte_m: Optional[float] = None
        self._last_heading_deg: Optional[float] = None
        self._last_t: Optional[float] = None

    def check(self, raw_cte_m: Optional[float], raw_heading_deg: Optional[float],
              confidence: float, valid: bool, now_s: float) -> dict:
        """Returns {"accept": bool, "reason": str}. On accept, updates the
        internal last-accepted state used for the next call's plausibility
        check. A rejection does NOT update that state, so a run of garbage
        frames is judged against the last genuinely trusted measurement,
        not against another rejected one."""
        if not valid or raw_cte_m is None or raw_heading_deg is None:
            return {"accept": False, "reason": "invalid"}

        if confidence < MIN_CONFIDENCE:
            return {"accept": False, "reason": "low_confidence"}

        if self._last_cte_m is not None and self._last_t is not None:
            dt = max(now_s - self._last_t, 0.0)
            max_cte_jump = MAX_CTE_RATE_M_S * dt + CTE_JUMP_MARGIN_M
            max_heading_jump = MAX_HEADING_RATE_DPS * dt + HEADING_JUMP_MARGIN_DEG
            if abs(raw_cte_m - self._last_cte_m) > max_cte_jump:
                return {"accept": False, "reason": "implausible_cte_jump"}
            if abs(raw_heading_deg - self._last_heading_deg) > max_heading_jump:
                return {"accept": False, "reason": "implausible_heading_jump"}

        self._last_cte_m = raw_cte_m
        self._last_heading_deg = raw_heading_deg
        self._last_t = now_s
        return {"accept": True, "reason": "ok"}
