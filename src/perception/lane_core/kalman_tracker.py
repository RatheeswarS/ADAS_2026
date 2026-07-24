"""
kalman_tracker.py
--------------------
Constant-velocity Kalman filter tracking lane state
[cte_m, cte_rate_m_s, heading_deg, heading_rate_deg_s], replacing the ad-hoc
exponential smoothing in steering.py's SteeringEstimator with a principled
state estimator that smooths the actual physical quantities (cross-track
error, heading) rather than the already-nonlinear-combined steering angle.

This is deliberately scoped to the two scalars the rest of the pipeline
actually needs (they're what feeds centerline generation and the Stanley
law), not the full left/right polynomial coefficient vectors -- tracking
6+ raw curve coefficients would be a meaningfully larger state for no
practical benefit here, since cte/heading are the only quantities the
controller consumes.

On a frame with no valid detection, `step()` predicts forward using the
tracked rate instead of freezing outright -- the filter's own principled
version of "hold the last estimate," but one that keeps extrapolating
plausibly (e.g. continuing a turn) rather than assuming zero rate of
change during a brief dropout.

Process/measurement noise below are placeholders, same status as every
other hand-picked gain in this project (K_GAIN, K_SOFT, MAX_STEERING_DEG,
...): a reasonable starting point, not bench-tuned against the real
vehicle -- but NOT arbitrary first guesses either. The first-guess values
(Q_HEADING_INTENSITY=0.5, R_HEADING_VAR=9.0) were caught, during testing,
lagging a realistic changing heading by 13 frames (over 200ms at 60fps) --
technically smooth, but too slow to be a usable replacement for the old
exponential smoother (~6-7 frames to 90% of a step). The values below were
picked by sweeping Q/R and choosing the point that (a) meets an 8-frame
90%-settling bound (see tests/test_kalman_tracker.py) and (b) keeps
frame-to-frame filtered-output jitter small under realistic measurement
noise (~0.34deg heading, ~0.03m cte) -- a real, if coarse, tuning pass, not
a first guess left unchecked.
"""
from typing import Optional

import numpy as np

Q_CTE_INTENSITY = 5.0         # process noise intensity, cte/cte_rate pair
Q_HEADING_INTENSITY = 200.0   # process noise intensity, heading/heading_rate pair
R_CTE_VAR = 0.02              # measurement noise variance, cte (m^2)
R_HEADING_VAR = 4.0           # measurement noise variance, heading (deg^2)
R_HEADING_RATE_VAR = 1.0      # trusted uncertainty (deg/s)^2 assigned to heading_rate
                               # when a real IMU yaw-rate reading overrides it during
                               # blind prediction -- a placeholder tighter than the
                               # filter's own default rate uncertainty (P init 90.0),
                               # since a real gyro reading is presumed more trustworthy
                               # than a vision-derived rate the filter hasn't corrected
                               # in a while; not bench-verified against a real IMU's
                               # actual noise spec.

MAX_PREDICT_DT = 0.5  # cap how far the filter will extrapolate blind (seconds)
                      # before a fresh measurement -- long past this, propagating
                      # on a stale rate estimate is more likely wrong than useful

MAX_CONSECUTIVE_PREDICTS = 15  # ~0.25s at 60fps. Capping each individual predict
                               # step's dt (MAX_PREDICT_DT) does not bound the
                               # CUMULATIVE effect of many consecutive predict-only
                               # steps: if the tracked rate is nonzero when
                               # measurements stop arriving, the state keeps
                               # advancing in that direction indefinitely with
                               # nothing to pull it back -- confirmed in practice,
                               # where a single bad-fit frame with no measurement
                               # for 200+ subsequent frames let a stale heading
                               # rate carry the filter's heading from ~24deg to
                               # 56+deg, straight through the steering clamp,
                               # before "valid" detections ever recovered. Past
                               # this many consecutive misses, `step()` reports
                               # stale=True so callers can stop trusting the
                               # ever-extrapolating estimate for control.


def _cv_block(dt: float, q: float) -> np.ndarray:
    """2x2 discretized process-noise block for an [x, x_dot] constant-
    velocity pair under continuous white-noise acceleration of intensity q."""
    return q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                         [dt ** 2 / 2.0, dt]])


class LaneKalmanTracker:
    def __init__(self):
        self.x = np.zeros(4)
        self.P = np.diag([1.0, 1.0, 90.0, 90.0])  # large initial uncertainty
        self._initialized = False
        self._last_t: Optional[float] = None
        self._consecutive_predicts = 0

    @property
    def cte_m(self) -> float:
        return float(self.x[0])

    @property
    def heading_deg(self) -> float:
        return float(self.x[2])

    def _predict(self, dt: float, imu_heading_rate_dps: Optional[float] = None):
        dt = min(dt, MAX_PREDICT_DT)
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])
        Q = np.zeros((4, 4))
        Q[0:2, 0:2] = _cv_block(dt, Q_CTE_INTENSITY)
        Q[2:4, 2:4] = _cv_block(dt, Q_HEADING_INTENSITY)
        if imu_heading_rate_dps is not None:
            # A real gyro reading is available for this blind-prediction
            # step: trust it over the filter's own vision-derived rate
            # estimate (x[3]) before propagating, and shrink that state's
            # uncertainty accordingly -- this is what actually bridges a
            # camera dropout under Design A instead of extrapolating on a
            # possibly-stale vision rate.
            self.x[3] = imu_heading_rate_dps
            self.P[3, 3] = min(self.P[3, 3], R_HEADING_RATE_VAR)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _update(self, cte_m: float, heading_deg: float):
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])
        R = np.diag([R_CTE_VAR, R_HEADING_VAR])
        z = np.array([cte_m, heading_deg])
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P

    def step(self, cte_m: Optional[float], heading_deg: Optional[float], now_s: float,
             imu_heading_rate_dps: Optional[float] = None) -> dict:
        """One filter tick. cte_m/heading_deg may be None (no valid
        detection this frame) -- the filter still predicts forward using
        its tracked rate, it just skips the measurement-update half.

        `imu_heading_rate_dps` is an optional real gyro yaw-rate (Design A's
        dead-reckoning bridge, see ego_state_estimator.py) that, when given,
        overrides the filter's own vision-derived heading-rate state before
        this step's prediction -- it's a no-op when None (NullImuSource,
        the current reality on this dev machine), so behavior is unchanged
        from the vision-only case unless a real IMU is actually attached.

        Returns a "stale" flag once too many consecutive ticks have passed
        without a real measurement (MAX_CONSECUTIVE_PREDICTS) -- callers
        should stop trusting cte_m/heading_deg for control once stale,
        since blind extrapolation on a possibly-nonzero rate is unbounded
        and will keep drifting further from reality the longer it goes
        uncorrected."""
        if not self._initialized:
            if cte_m is not None and heading_deg is not None:
                self.x = np.array([cte_m, 0.0, heading_deg, 0.0])
                self._initialized = True
                self._consecutive_predicts = 0
            self._last_t = now_s
            return {"cte_m": self.cte_m, "heading_deg": self.heading_deg,
                    "initialized": self._initialized, "stale": not self._initialized}

        dt = max(now_s - self._last_t, 0.0) if self._last_t is not None else 0.0
        self._last_t = now_s
        if dt > 0:
            self._predict(dt, imu_heading_rate_dps)
        if cte_m is not None and heading_deg is not None:
            self._update(cte_m, heading_deg)
            self._consecutive_predicts = 0
        else:
            self._consecutive_predicts += 1

        return {"cte_m": self.cte_m, "heading_deg": self.heading_deg, "initialized": True,
                "stale": self._consecutive_predicts > MAX_CONSECUTIVE_PREDICTS}
