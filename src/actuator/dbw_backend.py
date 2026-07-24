"""
dbw_backend.py
-----------------
Pluggable DBW (drive-by-wire) hardware abstraction, same
ABC-plus-Null-plus-real-hardware-placeholder pattern as
ego_state/imu_source.py: no real DBW ECU is connected to this dev machine,
so `NullDbwBackend` is what's actually active everywhere right now -- it
logs every commanded value (so a run's output is inspectable) but never
touches real hardware, and always reports success.

`SerialDbwBackend` is a real-hardware placeholder for a serial-attached DBW
ECU, but it is UNTESTED -- wire format, baud rate, command units (angle vs.
torque -- see dbw_interface.py's CommandMode, genuinely unresolved per this
project's own open questions), and feedback frame layout all need bench
verification before this drives a real steering rack.
"""
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class DbwBackend(ABC):
    @abstractmethod
    def send_command(self, value: float, mode: str, now_s: float) -> bool:
        """Send one commanded value (angle-deg or torque-Nm, per `mode`) to
        the actuator. Returns True on a successful send (not necessarily a
        confirmed actuation -- just that the write didn't fail outright)."""
        raise NotImplementedError

    @abstractmethod
    def read_actual_angle_deg(self) -> Optional[float]:
        """Latest actual steering angle reported back by the actuator, or
        None if unavailable -- feeds the safety monitor's commanded-vs-
        actual divergence check (task #34), not used for control."""
        raise NotImplementedError


class NullDbwBackend(DbwBackend):
    """Active on this dev machine (no DBW ECU attached). Logs the fallback
    once, then logs every commanded value at debug level so a run's output
    remains inspectable without ever touching real hardware."""

    def __init__(self):
        self._warned = False

    def _warn_once(self):
        if not self._warned:
            logger.warning("NullDbwBackend active: no DBW ECU attached -- commands are logged only, "
                            "never sent to real hardware.")
            self._warned = True

    def send_command(self, value: float, mode: str, now_s: float) -> bool:
        self._warn_once()
        logger.debug("NullDbwBackend command: %.3f (%s) at t=%.3f", value, mode, now_s)
        return True

    def read_actual_angle_deg(self) -> Optional[float]:
        self._warn_once()
        return None


class SerialDbwBackend(DbwBackend):
    """
    Real hardware path: writes ASCII lines "CMD:<mode>:<value>\n" and reads
    ASCII feedback lines "ANGLE:<deg>\n" over a serial port, background-
    threaded so a slow/blocked port never stalls the control loop -- same
    structure as ego_state.imu_source.SerialImuSource.

    UNVERIFIED: line format, baud rate, command units, and feedback layout
    are placeholders -- confirm on a bench before trusting this on a
    vehicle.
    """

    def __init__(self, port: str, baud: int = 115200, stale_after_s: float = 0.2):
        import serial  # pyserial; imported lazily so this module still
        # imports cleanly on a machine (like this one) without pyserial or
        # a real serial port attached.

        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._last_actual_angle: Optional[float] = None
        self._last_actual_time: float = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            if line.startswith("ANGLE:"):
                try:
                    with self._lock:
                        self._last_actual_angle = float(line.split(":", 1)[1])
                        self._last_actual_time = time.monotonic()
                except ValueError:
                    continue

    def send_command(self, value: float, mode: str, now_s: float) -> bool:
        try:
            self._ser.write(f"CMD:{mode}:{value:.4f}\n".encode("ascii"))
            return True
        except Exception:
            logger.exception("SerialDbwBackend: command write failed")
            return False

    def read_actual_angle_deg(self) -> Optional[float]:
        with self._lock:
            if self._last_actual_angle is None or time.monotonic() - self._last_actual_time > self._stale_after_s:
                return None
            return self._last_actual_angle

    def close(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self._ser.close()
