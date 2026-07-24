"""
imu_source.py
---------------
Pluggable IMU/GPS abstraction for ego-state estimation. No sensor hardware
is attached to this dev machine, so `NullImuSource` is what's actually
active in a bare offline run -- it always reports "no reading," letting
Design A's dead-reckoning bridge (see ego_state_estimator.py) gracefully
degrade to vision-only extrapolation, exactly as it already behaves today.

The `ImuSource` interface deliberately exposes BOTH read_yaw_rate_dps() and
read_speed_mps(), but on the real vehicle those two quantities come from two
DIFFERENT sensors, so they are provided by two separate concrete sources
that are then fused with `CompositeImuSource`:

  * `Hwt905ImuSource`  -- vehicle YAW RATE from the WitMotion HWT905 IMU,
    read in its native binary protocol on the IMU's own serial link. Speed
    is None (an IMU measures no ground speed).
  * `ArduinoSpeedSource` -- vehicle SPEED from the motor controller, echoed
    to the Jetson by the Arduino over the SAME serial link that already
    carries the SDA steering commands (owned by
    actuator.sda_serial_backend.SdaSerialBackend, which parses the telemetry).
    Yaw rate is None.
  * `CompositeImuSource(yaw_source=..., speed_source=...)` -- presents the
    pair as the single ImuSource the ego-state estimator consumes.

`SerialImuSource` is the older generic single-port ASCII reader (one device
publishing both `GYRO_Z:`/`SPEED:` lines); it is kept as a simple option but
is NOT this vehicle's wiring -- the HWT905 + Arduino split above is.

All real-hardware paths here are UNTESTED against the actual sensors on this
dev machine: wire/packet format, baud rate, and -- critically -- the yaw
axis and sign convention all need bench verification before feeding a real
vehicle (the sign in particular ties into ego_state_estimator.py's
-yaw_rate heading-rate convention and the steering damping term in
steering_v2.py; get it backwards and damping becomes positive feedback).
"""
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ImuSource(ABC):
    @abstractmethod
    def read_yaw_rate_dps(self) -> Optional[float]:
        """Latest gyro yaw-rate in degrees/second, or None if no fresh
        reading is available (no hardware, port not open, or stale)."""
        raise NotImplementedError

    @abstractmethod
    def read_speed_mps(self) -> Optional[float]:
        """Latest fused/wheel speed in meters/second, or None if
        unavailable. Design A only ever uses this for the Stanley law's `v`
        term and for scaling the dead-reckoning bridge -- never for global
        position."""
        raise NotImplementedError


class NullImuSource(ImuSource):
    """Active on this dev machine (no IMU/GPS hardware attached). Logs the
    fallback once so it's visible in a run's output, then stays silent."""

    def __init__(self):
        self._warned = False

    def _warn_once(self):
        if not self._warned:
            logger.warning("NullImuSource active: no IMU/GPS attached -- ego state estimation "
                            "runs on vision-only extrapolation (ego_state_estimator.py).")
            self._warned = True

    def read_yaw_rate_dps(self) -> Optional[float]:
        self._warn_once()
        return None

    def read_speed_mps(self) -> Optional[float]:
        self._warn_once()
        return None


class SerialImuSource(ImuSource):
    """
    Real hardware path: expects ASCII lines "GYRO_Z:<deg_per_sec>\n" and
    "SPEED:<m_per_sec>\n" over a serial port. Runs a background reader
    thread so a slow/blocked port never stalls the perception loop; a
    reading older than `stale_after_s` is treated as unavailable, same
    graceful-degradation contract as NullImuSource.

    UNVERIFIED: line format, baud rate, and sign convention are
    placeholders -- confirm on a bench before trusting this on a vehicle.
    """

    def __init__(self, port: str, baud: int = 115200, stale_after_s: float = 0.2):
        import serial  # pyserial; imported lazily so this module still
        # imports cleanly on a machine (like this one) without pyserial or
        # a real serial port attached.

        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._last_yaw_rate: Optional[float] = None
        self._last_speed: Optional[float] = None
        self._last_yaw_time: float = 0.0
        self._last_speed_time: float = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            now = time.monotonic()
            if line.startswith("GYRO_Z:"):
                try:
                    with self._lock:
                        self._last_yaw_rate = float(line.split(":", 1)[1])
                        self._last_yaw_time = now
                except ValueError:
                    continue
            elif line.startswith("SPEED:"):
                try:
                    with self._lock:
                        self._last_speed = float(line.split(":", 1)[1])
                        self._last_speed_time = now
                except ValueError:
                    continue

    def read_yaw_rate_dps(self) -> Optional[float]:
        with self._lock:
            if self._last_yaw_rate is None or time.monotonic() - self._last_yaw_time > self._stale_after_s:
                return None
            return self._last_yaw_rate

    def read_speed_mps(self) -> Optional[float]:
        with self._lock:
            if self._last_speed is None or time.monotonic() - self._last_speed_time > self._stale_after_s:
                return None
            return self._last_speed

    def close(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self._ser.close()


class Hwt905ImuSource(ImuSource):
    """
    The vehicle's WitMotion HWT905 IMU, read over serial in WitMotion's
    native binary protocol. Provides YAW RATE only -- read_speed_mps() is
    always None, since an IMU measures no ground speed (on this vehicle
    speed comes from the motor controller via the Arduino, see
    `ArduinoSpeedSource`).

    WitMotion frame format: fixed 11-byte frames, 0x55 <type> followed by
    four little-endian int16 payloads and a 1-byte checksum (the low 8 bits
    of the sum of the first 10 bytes). The angular-velocity frame (type
    0x52) carries wx, wy, wz, temp; each axis decodes as
    raw / 32768 * FS_dps, with FS = +/-2000 deg/s at the HWT905's default
    gyro range. Yaw is the wz axis for the usual flat mounting.

    `yaw_axis` and `yaw_sign` are exposed because BOTH the axis assignment
    and the sign depend on how the unit is physically mounted, and -- like
    every other IMU sign in this project -- are UNVERIFIED until bench-
    checked against the real installed orientation. `yaw_sign` must end up
    consistent with ego_state_estimator.py's convention (heading-error rate
    = -yaw_rate) so the steering_v2.py damping term stays negative feedback.

    Background reader thread + staleness gate: same graceful-degradation
    contract as NullImuSource/SerialImuSource -- a reading older than
    `stale_after_s`, a closed port, or a not-yet-seen gyro frame all read as
    "no reading" (None) rather than a stale or fabricated value.
    """

    _HEADER = 0x55
    _FRAME_LEN = 11
    _ID_ANGULAR_VELOCITY = 0x52
    _GYRO_FS_DPS = 2000.0                      # HWT905 default angular-velocity full scale
    _AXIS_OFFSET = {"x": 2, "y": 4, "z": 6}    # byte offset of each int16 payload in the frame

    def __init__(self, port: str, baud: int = 9600, stale_after_s: float = 0.2,
                 yaw_axis: str = "z", yaw_sign: float = 1.0):
        import serial  # pyserial; lazy-imported, same pattern as SerialImuSource
        if yaw_axis not in self._AXIS_OFFSET:
            raise ValueError(f"yaw_axis must be one of 'x'/'y'/'z', got {yaw_axis!r}")

        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._stale_after_s = stale_after_s
        self._yaw_off = self._AXIS_OFFSET[yaw_axis]
        self._yaw_sign = float(yaw_sign)
        self._lock = threading.Lock()
        self._last_yaw_rate: Optional[float] = None
        self._last_yaw_time: float = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        buf = bytearray()
        while not self._stop:
            try:
                chunk = self._ser.read(self._FRAME_LEN * 4)  # read a few frames' worth at a time
            except Exception:
                continue
            if not chunk:
                continue
            buf.extend(chunk)
            # Frame-sync + parse: the stream is a continuous run of 11-byte
            # frames, but we may start mid-frame, so resync on the 0x55
            # header and validate every frame's checksum before trusting it.
            while len(buf) >= self._FRAME_LEN:
                if buf[0] != self._HEADER:
                    del buf[0]                       # not a header: drop one byte, re-scan
                    continue
                frame = bytes(buf[:self._FRAME_LEN])
                if (sum(frame[:10]) & 0xFF) != frame[10]:
                    del buf[0]                       # bad checksum (likely a false 0x55): resync by one
                    continue
                del buf[:self._FRAME_LEN]
                if frame[1] == self._ID_ANGULAR_VELOCITY:
                    off = self._yaw_off
                    raw = int.from_bytes(frame[off:off + 2], "little", signed=True)
                    yaw_rate = self._yaw_sign * raw / 32768.0 * self._GYRO_FS_DPS
                    with self._lock:
                        self._last_yaw_rate = yaw_rate
                        self._last_yaw_time = time.monotonic()

    def read_yaw_rate_dps(self) -> Optional[float]:
        with self._lock:
            if self._last_yaw_rate is None or time.monotonic() - self._last_yaw_time > self._stale_after_s:
                return None
            return self._last_yaw_rate

    def read_speed_mps(self) -> Optional[float]:
        return None  # an IMU measures no ground speed (see class docstring)

    def close(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self._ser.close()


class ArduinoSpeedSource(ImuSource):
    """
    Speed-only ImuSource backed by the Arduino link that already carries the
    SDA steering commands. The motor controller reports vehicle speed to the
    Arduino, which echoes it to the Jetson over that same serial connection;
    the connection (and the telemetry parsing) is owned by
    actuator.sda_serial_backend.SdaSerialBackend, so this class is a thin
    adapter -- it takes anything exposing `read_speed_mps()` (duck-typed on
    purpose so it works with the real backend, a DryRun stub, or a test
    double) and presents it through the ImuSource interface.

    Yaw rate is always None here: on this vehicle yaw comes from the HWT905
    IMU (`Hwt905ImuSource`), not the Arduino -- combine the two with
    `CompositeImuSource`. This adapter does NOT own the backend's serial
    port (the backend does), so it intentionally has no close().
    """

    def __init__(self, speed_reader):
        self._speed_reader = speed_reader

    def read_yaw_rate_dps(self) -> Optional[float]:
        return None

    def read_speed_mps(self) -> Optional[float]:
        reader = self._speed_reader
        if reader is None or not hasattr(reader, "read_speed_mps"):
            return None
        return reader.read_speed_mps()


class CompositeImuSource(ImuSource):
    """
    Fuses the vehicle's two independent ego-motion sensors into the single
    ImuSource the ego-state estimator consumes, matching the real hardware
    split: YAW RATE from the HWT905 IMU (`Hwt905ImuSource`) and SPEED from
    the motor controller via the Arduino (`ArduinoSpeedSource`). Either half
    may be None -- a missing yaw source degrades to vision-only heading-rate,
    a missing speed source degrades to the ego-state's configured speed
    fallback -- so a partial sensor set still runs, it just informs the
    controller with fewer real measurements.
    """

    def __init__(self, yaw_source: Optional[ImuSource] = None,
                 speed_source: Optional[ImuSource] = None):
        self._yaw_source = yaw_source
        self._speed_source = speed_source

    def read_yaw_rate_dps(self) -> Optional[float]:
        return self._yaw_source.read_yaw_rate_dps() if self._yaw_source is not None else None

    def read_speed_mps(self) -> Optional[float]:
        return self._speed_source.read_speed_mps() if self._speed_source is not None else None

    def close(self):
        # Close only sources this composite owns that expose close(). The
        # ArduinoSpeedSource intentionally has none (it borrows the actuator
        # backend's port, which the backend closes itself).
        for src in (self._yaw_source, self._speed_source):
            if src is not None and hasattr(src, "close"):
                try:
                    src.close()
                except Exception:
                    logger.exception("CompositeImuSource: error closing %r", src)
