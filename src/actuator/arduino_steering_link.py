"""
arduino_steering_link.py
--------------------------
Steering-ONLY link from the Jetson to the low-level Arduino controller
(arduinocode.ino), used by run_video_annotated.py for on-vehicle LKA.

It speaks that firmware's REAL protocol:

  Jetson -> Arduino:  STA<±deg>\\n      absolute steering angle, + = right / - = left
  Arduino -> Jetson:  $TEL,millis,STATE,thrMode,speedKmh,wheelRPM,throttle%,
                      steerActualDeg,steerTargetDeg,brakeActual%,brakeTarget%,
                      rampedRPM,pidErr,corr,faultFlags,dryRun     (~20 Hz)

It sends ONLY `STA`. It never sends `S<kmh>` (throttle), `BRK` (brake), or
`LOCK`, so the controller stays in CONFIG mode with throttle + brake DISARMED
-- steering-only operation with a human safety driver handling speed/braking.

Why this is NOT sda_serial_backend.py: that module speaks a `SDA <value>`
protocol this firmware does not implement. `"SDA ..."` does not match the
sketch's `STA` prefix and falls through to its `S<kmh>` throttle branch, which
would spin up the throttle PID instead of steering. This module is the correct
one for arduinocode.ino.

A background thread parses `$TEL` telemetry to expose:
  * read_speed_mps()  -- the vehicle's real speed (ArduinoSpeedSource adapts
                         this into the Stanley speed term), None if stale.
  * centered          -- True once boot self-centering finishes (the firmware
                         SILENTLY DROPS every STA until then), None until the
                         first telemetry line arrives.
  * faults / state / actual_deg -- for the HUD / diagnostics.
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

FAULT_NOT_CENTERED = 0x0020  # arduinocode.ino faultFlags bit


class ArduinoSteeringLink:
    def __init__(self, port: str, baud: int = 115200, flip_sign: bool = False,
                 stale_after_s: float = 1.0):
        import serial  # pyserial; lazy import so the module loads without it
        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._sign = -1.0 if flip_sign else 1.0  # firmware convention is + = right
        self._stale_after_s = stale_after_s

        self._lock = threading.Lock()
        self._speed_mps: Optional[float] = None
        self._speed_t = 0.0
        self._fault_flags: Optional[int] = None
        self._state: Optional[str] = None
        self._actual_deg: Optional[float] = None
        self._target_deg: Optional[float] = None   # steerTargetDeg echoed in $TEL
        self._dry_run: Optional[bool] = None        # dryRun flag echoed in $TEL
        self._last_line_t = 0.0
        self._last_sent_deg = 0.0

        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("ArduinoSteeringLink open on %s @ %d (flip_sign=%s)", port, baud, flip_sign)

    # ---- telemetry reader -------------------------------------------------
    def _run(self):
        while not self._stop:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            self._last_line_t = time.monotonic()
            if line.startswith("$TEL,"):
                self._parse_tel(line)

    def _parse_tel(self, line: str):
        # Field order matches arduinocode.ino sendTelemetry(); see module docstring.
        p = line.split(",")
        if len(p) < 16:
            return
        try:
            state = p[2]
            speed_kmh = float(p[4])
            actual_deg = float(p[7])
            target_deg = float(p[8])
            fault_flags = int(p[14])
            dry_run = p[15].strip() == "1"
        except (ValueError, IndexError):
            return
        with self._lock:
            self._state = state
            self._speed_mps = speed_kmh / 3.6
            self._speed_t = time.monotonic()
            self._actual_deg = actual_deg
            self._target_deg = target_deg
            self._fault_flags = fault_flags
            self._dry_run = dry_run

    # ---- commands ---------------------------------------------------------
    def send_steering_deg(self, deg: float) -> bool:
        """Transmit one absolute steering command. `deg` is in the pipeline's
        convention (+ = right); the sign flip (if any) is applied here."""
        out = self._sign * float(deg)
        try:
            self._ser.write(f"STA{out:.2f}\n".encode("ascii"))
            self._last_sent_deg = out
            return True
        except Exception:
            logger.exception("ArduinoSteeringLink: STA write failed")
            return False

    def send_command(self, line: str) -> bool:
        """Send an arbitrary command line (e.g. 'SET:DRY_RUN=1', 'PING'). Used by
        the bench self-test; the LKA run path only ever calls send_steering_deg."""
        try:
            self._ser.write((line.rstrip("\n") + "\n").encode("ascii"))
            return True
        except Exception:
            logger.exception("ArduinoSteeringLink: command write failed")
            return False

    # ---- telemetry accessors ---------------------------------------------
    def read_speed_mps(self) -> Optional[float]:
        """Latest telemetry speed (m/s), or None if stale -- consumed via
        ego_state.imu_source.ArduinoSpeedSource, duck-typed on this method."""
        with self._lock:
            if self._speed_mps is None or time.monotonic() - self._speed_t > self._stale_after_s:
                return None
            return self._speed_mps

    @property
    def centered(self) -> Optional[bool]:
        """True once boot centering is done (STA is ignored by the firmware
        until then). None until the first telemetry line arrives."""
        with self._lock:
            if self._fault_flags is None:
                return None
            return (self._fault_flags & FAULT_NOT_CENTERED) == 0

    @property
    def fault_flags(self) -> Optional[int]:
        with self._lock:
            return self._fault_flags

    @property
    def state(self) -> Optional[str]:
        with self._lock:
            return self._state

    @property
    def last_sent_deg(self) -> float:
        return self._last_sent_deg

    @property
    def target_deg(self) -> Optional[float]:
        """steerTargetDeg echoed by the firmware in $TEL -- used to confirm a
        sent STA command actually landed (the round-trip)."""
        with self._lock:
            return self._target_deg

    @property
    def dry_run(self) -> Optional[bool]:
        """dryRun flag from $TEL: True = hardware outputs suppressed (safe)."""
        with self._lock:
            return self._dry_run

    @property
    def link_alive(self) -> bool:
        return bool(self._last_line_t) and (time.monotonic() - self._last_line_t) < self._stale_after_s

    def close(self):
        # Best-effort: return the wheels to centre before dropping the link.
        try:
            self.send_steering_deg(0.0)
        except Exception:
            pass
        self._stop = True
        self._thread.join(timeout=1.0)
        try:
            self._ser.close()
        except Exception:
            pass


class SecondMeanSteering:
    """Consolidates the per-frame steering command into a 1-second MEAN, which
    is the value actually commanded to the wheels (smooths out per-frame vision
    jitter -- a position controller like arduinocode.ino would otherwise hunt).

    Feed each ENGAGED frame's steering with add(); call update() every tick.
    When a window closes it returns (target_deg, True): the mean of that
    window's samples, or -- if the window had NO engaged samples (perception
    lost) -- the previous target ramped toward centre so a dropout can't hold a
    stale turn. Between window closes update() returns (held_target, False)."""

    def __init__(self, output_period_s: float = 1.0, loss_ramp_dps: float = 20.0,
                 max_deg: float = 30.0):
        self.output_period_s = output_period_s
        self.loss_ramp_dps = loss_ramp_dps
        self.max_deg = max_deg
        self._samples = []
        self._window_start: Optional[float] = None
        self._target_deg = 0.0

    def add(self, steering_deg: float):
        self._samples.append(float(steering_deg))

    def update(self, now_s: float):
        if self._window_start is None:
            self._window_start = now_s
            return self._target_deg, False
        elapsed = now_s - self._window_start
        if elapsed < self.output_period_s:
            return self._target_deg, False

        if self._samples:
            self._target_deg = sum(self._samples) / len(self._samples)
        else:
            # No engaged frames this window -> perception lost / disengaged.
            # Ramp the last command back toward centre rather than holding it.
            step = self.loss_ramp_dps * elapsed
            if self._target_deg > 0:
                self._target_deg = max(0.0, self._target_deg - step)
            else:
                self._target_deg = min(0.0, self._target_deg + step)

        self._target_deg = max(-self.max_deg, min(self.max_deg, self._target_deg))
        self._samples = []
        self._window_start = now_s
        return self._target_deg, True

    @property
    def target_deg(self) -> float:
        return self._target_deg
