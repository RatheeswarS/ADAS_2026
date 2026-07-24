"""
ego_state_estimator.py
-------------------------
IMU/GPS -> ego motion, Design A (per the architecture discussion this was
built from): NOT a global/mapped position estimate -- purely a short-
horizon dead-reckoning bridge answering "how far has the vehicle moved
relative to where it was at the last camera update," so a lane-relative
reference can be advanced sensibly across a dropped/occluded frame instead
of either freezing or blindly trusting stale geometry.

    heading_bridge_deg = -yaw_rate_dps * dt
    lateral_bridge_m   = -speed_mps * sin(heading_error_rad) * dt   (small-
                         angle: sin(x) ~= x for the near-field headings this
                         pipeline deals with, so this reduces to
                         -speed_mps * heading_error_rad * dt)

This class only computes the delta; deciding what to DO with it (fuse into
the Kalman-tracked reference when the camera drops out, versus trust the
camera directly when it's available) is reference_generator.py's job.
"""
from typing import Optional

import numpy as np

from ego_state.imu_source import ImuSource, NullImuSource


class EgoStateEstimator:
    def __init__(self, imu_source: Optional[ImuSource] = None, default_speed_mps: float = 5.0):
        self.imu = imu_source or NullImuSource()
        self.default_speed_mps = default_speed_mps
        self._last_t: Optional[float] = None

    @property
    def imu_active(self) -> bool:
        return not isinstance(self.imu, NullImuSource)

    def predict_delta(self, now_s: float, heading_error_rad: float = 0.0) -> dict:
        """
        Returns {"delta_heading_deg", "delta_lateral_m", "yaw_rate_dps",
        "speed_mps", "dt_s", "have_imu_yaw_rate"}. When no real IMU is
        attached, yaw_rate_dps is None and delta_heading_deg is 0.0 (a
        request for the caller to fall back to whatever extrapolation it
        already trusts, e.g. the Kalman-tracked rate) -- NullImuSource never
        fabricates a yaw rate. `speed_mps` falls back to the placeholder
        constant (same status as every other speed placeholder in this
        project) when no fused speed reading exists either.
        """
        dt = 0.0 if self._last_t is None else max(now_s - self._last_t, 0.0)
        self._last_t = now_s

        yaw_rate_dps = self.imu.read_yaw_rate_dps()
        speed_mps = self.imu.read_speed_mps()
        if speed_mps is None:
            speed_mps = self.default_speed_mps

        if yaw_rate_dps is None:
            return {"delta_heading_deg": 0.0, "delta_lateral_m": 0.0,
                    "yaw_rate_dps": None, "speed_mps": speed_mps, "dt_s": dt,
                    "have_imu_yaw_rate": False}

        delta_heading_deg = -yaw_rate_dps * dt
        delta_lateral_m = -speed_mps * np.sin(heading_error_rad) * dt
        return {"delta_heading_deg": float(delta_heading_deg), "delta_lateral_m": float(delta_lateral_m),
                "yaw_rate_dps": yaw_rate_dps, "speed_mps": speed_mps, "dt_s": dt,
                "have_imu_yaw_rate": True}
