"""
safety_monitor.py
---------------------
The architecture's Safety Monitor block, supervising the perception ->
reference -> control -> actuator chain end to end (per the diagram this was
built from: the Safety Monitor sits alongside/over the whole chain, not
inside the main control-loop path).

Four responsibilities, matching the architecture discussion directly:

  1. Plausibility check: commanded steering angle vs. what's kinematically
     reasonable given speed and yaw rate, via the standard bicycle-model
     relation yaw_rate ~= v * tan(delta) / L (WHEELBASE_M below is this
     project's real measured wheelbase, not a placeholder). This check
     only runs when a real yaw-rate reading actually exists -- on this dev
     machine (NullImuSource, see ego_state/imu_source.py) it's always
     skipped rather than being faked against a rate nothing ever measured.
  2. Sensor/module health watchdogs: perception staleness
     (ReferenceState.stale, sourced from the Kalman tracker's own
     MAX_CONSECUTIVE_PREDICTS bound) and actuator command staleness
     (ActuatorInterface.is_watchdog_tripped).
  3. Driver override: a placeholder hook. No real override switch/sensor
     is attached on this dev machine, so it always reads False (never
     overridden) until real hardware defines what "override" means here --
     the parameter exists so a real caller has somewhere to plug it in
     without an interface change.
  4. Engagement conditions: speed within an operating band, confidence
     above threshold, tracker initialized and not stale, no active faults.
     ALL must hold, and must hold for ENGAGE_HOLD_TICKS consecutive ticks,
     before (re-)engaging -- deliberately asymmetric with disengagement
     (any single failing condition disengages immediately) so a momentary
     recovery frame can't cause an immediate re-engage.

All thresholds below are placeholders in the same spirit as every other
hand-picked gain in this project: reasonable starting points, not
bench-tuned against the real vehicle.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from actuator.dbw_interface import ActuatorInterface
from reference.reference_generator import ReferenceState

WHEELBASE_M = 1.3716  # this project's real measured buggy wheelbase

MIN_ENGAGE_SPEED_MPS = 0.5
MAX_ENGAGE_SPEED_MPS = 15.0
MIN_ENGAGE_CONFIDENCE = 0.3
ENGAGE_HOLD_TICKS = 5

YAW_RATE_PLAUSIBILITY_MARGIN_DPS = 15.0


@dataclass
class SafetyState:
    engaged: bool
    faults: List[str] = field(default_factory=list)


class SafetyMonitor:
    def __init__(self):
        self._consecutive_good_ticks = 0
        self._was_engaged = False

    def _plausibility_fault(self, commanded_steering_deg: float, speed_mps: float,
                             measured_yaw_rate_dps: Optional[float]) -> Optional[str]:
        if measured_yaw_rate_dps is None or speed_mps <= 0.0:
            return None  # nothing real to check the command against
        predicted_yaw_rate_dps = float(np.degrees(
            speed_mps * np.tan(np.radians(commanded_steering_deg)) / WHEELBASE_M))
        if abs(predicted_yaw_rate_dps - measured_yaw_rate_dps) > YAW_RATE_PLAUSIBILITY_MARGIN_DPS:
            return "implausible_command_vs_kinematics"
        return None

    def evaluate(self, reference: ReferenceState, commanded_steering_deg: float, speed_mps: float,
                 actuator: ActuatorInterface, now_s: float,
                 measured_yaw_rate_dps: Optional[float] = None, driver_override: bool = False) -> SafetyState:
        faults: List[str] = []

        if driver_override:
            faults.append("driver_override")
        if not reference.initialized:
            faults.append("perception_not_initialized")
        if reference.stale:
            faults.append("perception_stale")
        if actuator.is_watchdog_tripped(now_s):
            faults.append("actuator_command_watchdog")
        if reference.confidence < MIN_ENGAGE_CONFIDENCE:
            faults.append("low_confidence")
        if not (MIN_ENGAGE_SPEED_MPS <= speed_mps <= MAX_ENGAGE_SPEED_MPS):
            faults.append("speed_out_of_engage_band")
        plaus_fault = self._plausibility_fault(commanded_steering_deg, speed_mps, measured_yaw_rate_dps)
        if plaus_fault:
            faults.append(plaus_fault)

        if faults:
            self._consecutive_good_ticks = 0
            self._was_engaged = False
            return SafetyState(engaged=False, faults=faults)

        self._consecutive_good_ticks += 1
        if self._was_engaged or self._consecutive_good_ticks >= ENGAGE_HOLD_TICKS:
            self._was_engaged = True
            return SafetyState(engaged=True, faults=[])
        return SafetyState(engaged=False, faults=["engage_hold_pending"])
