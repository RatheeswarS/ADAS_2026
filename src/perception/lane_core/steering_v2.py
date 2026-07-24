"""
steering_v2.py
----------------
Stanley control law as a stateless function, paired with the Kalman-
filtered (cte, heading) state from kalman_tracker.py -- the "Centerline
Generation -> Steering Controller (Stanley)" tail of the bird's-eye
pipeline.

This replaces steering.py's SteeringEstimator's own internal exponential
smoothing (LaneKalmanTracker now does that job, on the actual physical
state rather than on the already-nonlinear-combined steering angle), but
keeps a slew-rate limiter on the FINAL command as a final actuator-safety
backstop -- a statistically optimal state estimate is not the same
guarantee as "the physical actuator can move this fast," so that clamp
stays independent of whatever produced the state estimate.

Two vehicle-dynamics inputs feed the law (both routed through the ego-state
bridge, see ego_state/ego_state_estimator.py -- so they are the vehicle's
*measured* speed/yaw when a real IMU is attached, and graceful fallbacks
otherwise):

  * speed_mps  -- the Stanley cross-track term is speed-scaled by design
    (K_GAIN*e / (K_SOFT + v)): the faster the vehicle, the gentler a given
    lateral error is corrected, so the same CTE doesn't yank the wheel at
    speed the way it should nudge it at a crawl. Feeding the *actual* speed
    here (instead of a fixed constant) is what makes that scaling real.

  * yaw_rate_dps -- an optional yaw-rate *damping* term. The bare Stanley
    law is proportional-only on heading error; on a vehicle with actuator
    and yaw-inertia lag that invites overshoot/weave. Damping subtracts a
    term proportional to how fast the vehicle is actually rotating, so the
    command eases off as the buggy swings toward the target instead of
    driving hard until the error is already gone. This is a no-op (term ==
    0) whenever yaw_rate_dps is None -- i.e. NullImuSource, the current
    reality on this dev machine -- so vision-only behavior is unchanged
    until a real gyro is wired in.

Gains below are the same placeholders as steering.py's (K_GAIN, K_SOFT,
MAX_STEERING_DEG, MAX_STEER_RATE_DPS) -- duplicated rather than imported so
this module has no dependency on the legacy per-frame-smoothing estimator
it's meant to replace. K_YAW_DAMP is new and in the same status: a
reasonable, conservative starting point, NOT bench-tuned against the real
vehicle's yaw response, and its SIGN assumes the same (still-unverified)
gyro axis convention ego_state_estimator.py already commits to
(delta_heading = -yaw_rate*dt) -- verify on a bench before trusting it
closed-loop, exactly as imu_source.py flags for the raw gyro.
"""
from typing import Optional

import numpy as np

K_GAIN = 0.8
K_SOFT = 1.0
K_YAW_DAMP = 0.15   # deg of counter-steer per (deg/s) of measured yaw rate.
                    # Derivative/damping gain; units are effectively seconds.
                    # Small on purpose -- too large fights legitimate cornering
                    # yaw (see note in stanley_steering_deg on raw-vs-desired
                    # yaw damping). Placeholder, not bench-tuned.
MAX_STEERING_DEG = 30.0
MAX_STEER_RATE_DPS = 60.0


def stanley_steering_deg(cte_m: float, heading_deg: float, speed_mps: float,
                         yaw_rate_dps: Optional[float] = None,
                         k_yaw_damp: float = K_YAW_DAMP) -> float:
    """Augmented Stanley steering command (degrees, +right), clamped to
    +/-MAX_STEERING_DEG.

    speed_mps scales the cross-track term; guarded so a spurious zero/
    negative speed reading can't blow up or invert the denominator (it only
    ever softens the correction, never sharpens it past the v=0 case).

    yaw_rate_dps, when provided, adds a damping term that opposes the
    vehicle's current rotation. Sign convention matches the ego-state
    bridge (ego_state_estimator.py: the heading-error rate is -yaw_rate_dps),
    so the damping term is -k_yaw_damp*yaw_rate_dps: as the buggy yaws toward
    closing the error, that eases the command off and curbs overshoot. Left
    None it contributes exactly zero (vision-only fallback, unchanged from
    the pre-yaw-rate behavior)."""
    v_eff = max(float(speed_mps), 0.0)
    steering_rad = np.radians(heading_deg) + np.arctan(
        (K_GAIN * cte_m) / (K_SOFT + v_eff)
    )
    steering_deg = float(np.degrees(steering_rad))
    if yaw_rate_dps is not None:
        # heading-error rate = -yaw_rate_dps (ego_state_estimator.py
        # convention); PD-style damping adds a term along that rate, i.e.
        # subtracts k*yaw_rate. A raw-yaw-rate damper does slightly resist a
        # sustained cornering yaw (steady-state bias on a constant-radius
        # turn); acceptable at this gain, and the alternative -- damping only
        # (yaw_rate - v*tan(delta)/L), the discrepancy from the commanded
        # kinematic yaw -- is a future refinement if that bias ever matters.
        steering_deg += -k_yaw_damp * float(yaw_rate_dps)
    return float(np.clip(steering_deg, -MAX_STEERING_DEG, MAX_STEERING_DEG))


class SlewRateLimiter:
    def __init__(self, max_rate_dps: float = MAX_STEER_RATE_DPS):
        self.max_rate_dps = max_rate_dps
        self._last_value: Optional[float] = None
        self._last_t: Optional[float] = None

    def limit(self, target_deg: float, now_s: float) -> float:
        if self._last_value is None:
            self._last_value = target_deg
            self._last_t = now_s
            return self._last_value
        dt = max(now_s - self._last_t, 0.0) if self._last_t is not None else 0.0
        self._last_t = now_s
        max_delta = self.max_rate_dps * dt
        delta = float(np.clip(target_deg - self._last_value, -max_delta, max_delta))
        self._last_value = self._last_value + delta
        return self._last_value
