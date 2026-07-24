"""
sda_serial_backend.py
------------------------
DbwBackend implementation that speaks the vehicle's Arduino steering
protocol directly: one ASCII line per command,

    SDA <value>\n

where <value> is the commanded steering angle in degrees, signed float,
2 decimal places (e.g. "SDA -12.34\n" for a 12.34 degree left turn, "SDA
30.00\n" for a 30 degree right turn) -- negative = left, positive = right,
no colon/comma separator. This is the exact wire format given for this
project's Arduino sketch; it is NOT the same as dbw_backend.SerialDbwBackend
(that one speaks an unrelated, unverified "CMD:<mode>:<value>" placeholder
protocol for a different, not-yet-benched DBW ECU).

This link is bidirectional in ONE respect: while the Jetson -> Arduino
direction carries the SDA steering commands, the Arduino -> Jetson direction
carries the vehicle's SPEED telemetry -- the motor controller reports speed
to the Arduino, which echoes it back as an ASCII line

    SPD:<meters_per_second>\n     (e.g. "SPD:4.20\n")

on this same port. A background reader thread parses those lines and
`read_speed_mps()` exposes the latest fresh value (older than
`stale_after_s` -> None), which `ego_state.imu_source.ArduinoSpeedSource`
adapts into the ego-state/steering path so the Stanley law runs on the
vehicle's real speed. The `SPD:` line format and its rate are a firmware
contract that is UNVERIFIED against the real Arduino sketch here -- confirm
on the bench (and that the sketch actually emits it) before trusting it.
STEERING-ANGLE feedback, by contrast, still has no specified wire format,
so `read_actual_angle_deg()` remains None (no fabricated read protocol) --
it exists only to satisfy DbwBackend's abstract interface, which
ActuatorInterface.read_actual_angle_deg() exposes for an optional future
feedback path.

Engagement gating: ActuatorInterface.send() must be called every tick
regardless of SafetyMonitor's engagement state -- ActuatorInterface's own
command watchdog (is_watchdog_tripped) and SafetyMonitor's engagement
decision both depend on a steady stream of sends (see safety_monitor.py's
`actuator.is_watchdog_tripped(now_s)` check, which needs `actuator.send()`
to keep running even while disengaged, or the watchdog fault would never
clear and the system could never (re-)engage in the first place). So the
"only actuate while engaged" requirement is enforced HERE, at the hardware
boundary, not by skipping calls upstream: the caller tells this backend the
current engagement state via `set_engaged()` before each `actuator.send()`
call, and `send_command()` transmits the real commanded value only while
engaged, substituting a neutral "SDA 0.00" (straight-ahead, fail-safe)
command every tick it is not.
"""
import logging
import threading
import time
from typing import Optional

from actuator.dbw_backend import DbwBackend

logger = logging.getLogger(__name__)


class SdaSerialBackend(DbwBackend):
    def __init__(self, port: str, baud: int = 115200, stale_after_s: float = 0.5):
        import serial  # pyserial; imported lazily so this module still
        # imports cleanly on a machine without pyserial or a real serial
        # port attached (same lazy-import pattern as dbw_backend.py /
        # imu_source.py elsewhere in this project).

        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._engaged = False
        # Speed telemetry (Arduino -> Jetson) read off this same link by a
        # background thread. Concurrent readline() here + write() in
        # send_command() on one pyserial port is the established pattern in
        # this project (see dbw_backend.SerialDbwBackend). stale_after_s is
        # looser than the IMU's since motor-controller speed telemetry is
        # typically slower than the gyro stream.
        self._stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._last_speed_mps: Optional[float] = None
        self._last_speed_time: float = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("SdaSerialBackend open on %s @ %d baud", port, baud)

    def _run(self):
        while not self._stop:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            if line.startswith("SPD:"):
                try:
                    speed = float(line.split(":", 1)[1])
                except ValueError:
                    continue
                with self._lock:
                    self._last_speed_mps = speed
                    self._last_speed_time = time.monotonic()

    def set_engaged(self, engaged: bool) -> None:
        """Called once per tick, BEFORE ActuatorInterface.send(), so
        send_command() knows whether to transmit the real commanded value
        or the neutral fail-safe command this same tick."""
        self._engaged = engaged

    def send_command(self, value: float, mode: str, now_s: float) -> bool:
        outgoing = value if self._engaged else 0.0
        line = f"SDA {outgoing:.2f}\n"
        try:
            self._ser.write(line.encode("ascii"))
            return True
        except Exception:
            logger.exception("SdaSerialBackend: command write failed")
            return False

    def read_speed_mps(self) -> Optional[float]:
        """Latest motor-controller speed (m/s) echoed by the Arduino, or None
        if none fresh -- consumed via ego_state.imu_source.ArduinoSpeedSource."""
        with self._lock:
            if self._last_speed_mps is None or time.monotonic() - self._last_speed_time > self._stale_after_s:
                return None
            return self._last_speed_mps

    def read_actual_angle_deg(self) -> Optional[float]:
        return None

    def close(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self._ser.close()
