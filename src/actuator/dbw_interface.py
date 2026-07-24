"""
dbw_interface.py
-------------------
DBW actuator interface: the architecture's "Actuator Interface -> DBW
Steering" block. Takes a reference steering angle (from the Stanley law,
already slew-limited once upstream in steering_v2.SlewRateLimiter) and
turns it into an actual outgoing command to a DbwBackend.

Whether the real DBW ECU wants an angle or a torque command is one of this
project's genuinely open questions (no real DBW hardware is connected to
verify it against) -- `CommandMode` makes this an explicit, switchable
choice rather than a hardcoded assumption, and ANGLE is the default only
because it needs some default, not because it's confirmed correct.

Two things happen here that are deliberately NOT already done upstream:

  1. A SECOND, actuator-level rate limiter. The Stanley controller's output
     is already slew-limited once (steering_v2.SlewRateLimiter) before it
     ever reaches this interface -- this is intentional defense-in-depth,
     not redundancy for its own sake: a bug anywhere between the
     controller and here (a bad reference generator edge case, a future
     caller that bypasses the controller's own limiter) still can't command
     an actuator-destroying instantaneous jump, because THIS layer enforces
     the same physical limit independently, right at the actuator
     boundary.

  2. A command watchdog: `last_command_age_s(now_s)` lets the safety
     monitor (task #34) detect "no command has actually been sent in too
     long" -- distinct from perception going stale, since a healthy
     perception/control chain that simply isn't being driven (e.g. the
     caller stopped ticking this interface) is its own failure mode.

ANGLE_TO_TORQUE_GAIN_PLACEHOLDER below is exactly that -- a placeholder,
not a measured actuator characteristic. There is no real DBW hardware on
this dev machine to bench-characterize a genuine angle/torque relationship
against, so TORQUE mode is wired up and testable, but the specific gain
must be replaced once real hardware exists (see Q3 in the architecture
discussion this was built from: DBW command type is explicitly unresolved).
"""
from enum import Enum
from typing import Optional

from actuator.dbw_backend import DbwBackend, NullDbwBackend
from perception.lane_core.steering_v2 import SlewRateLimiter

ANGLE_TO_TORQUE_GAIN_PLACEHOLDER = 0.5  # Nm per degree -- UNVERIFIED, see module docstring
COMMAND_TIMEOUT_S = 0.5  # no successful send within this long -> watchdog trips


class CommandMode(str, Enum):
    ANGLE = "angle"
    TORQUE = "torque"


class ActuatorInterface:
    def __init__(self, backend: Optional[DbwBackend] = None, mode: CommandMode = CommandMode.ANGLE):
        self.backend = backend or NullDbwBackend()
        self.mode = mode
        self._rate_limiter = SlewRateLimiter()
        self._last_send_t: Optional[float] = None

    def _to_command_value(self, steering_deg: float) -> float:
        if self.mode == CommandMode.ANGLE:
            return steering_deg
        return steering_deg * ANGLE_TO_TORQUE_GAIN_PLACEHOLDER

    def send(self, steering_deg: float, now_s: float) -> dict:
        """Rate-limits `steering_deg` a second time (actuator-level
        defense-in-depth), converts to this interface's configured
        CommandMode, and forwards to the backend. Returns
        {"commanded_value", "mode", "success"}."""
        limited_deg = self._rate_limiter.limit(steering_deg, now_s)
        command_value = self._to_command_value(limited_deg)
        success = self.backend.send_command(command_value, self.mode.value, now_s)
        if success:
            self._last_send_t = now_s
        return {"commanded_value": command_value, "mode": self.mode.value, "success": success}

    def last_command_age_s(self, now_s: float) -> float:
        """Seconds since the last SUCCESSFUL send, or +inf if none has ever
        succeeded -- the safety monitor's watchdog input (task #34)."""
        if self._last_send_t is None:
            return float("inf")
        return max(now_s - self._last_send_t, 0.0)

    def is_watchdog_tripped(self, now_s: float) -> bool:
        return self.last_command_age_s(now_s) > COMMAND_TIMEOUT_S

    def read_actual_angle_deg(self) -> Optional[float]:
        return self.backend.read_actual_angle_deg()
