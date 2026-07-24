# Lane-Keep Assist (LKA) — Operator Guide

Vision-based lane keeping for the BAJA buggy. A camera on the Jetson feeds a
YOLOP lane pipeline; the resulting steering angle is smoothed into a **1-second
mean** and streamed to the low-level Arduino controller as **steering-only**
commands. Throttle and braking are **not** touched — a safety driver controls
speed and stops.

---

## 1. Data flow

```
 ZED camera ──(live, side-by-side)──► run_video_annotated.py ──STA<±deg>──► Arduino (arduinocode.ino) ──► steering stepper
   3840×1080                          │  crop left eye                        (CONFIG mode, steering only)
                                      │  YOLOP → bird's-eye → CTE/heading
                                      │  Stanley steering law
                                      │  1-second MEAN
                                      ▲  speed  ◄──$TEL telemetry──┘
```

- **Perception → steering** is the existing pipeline (unchanged). Per frame it
  produces `steering_deg` (Stanley law, ±30° clamp).
- **New:** `run_video_annotated.py` averages those per-frame commands over each
  1-second window and sends the mean as `STA<±deg>` to the Arduino.
- The Arduino's `$TEL` telemetry reports real vehicle speed back, which feeds the
  Stanley speed term automatically.

---

## 2. The Arduino interface (what the firmware actually expects)

From `arduinocode.ino`. **We use CONFIG mode and send only `STA`.**

| Command | Meaning | Used by LKA? |
|---|---|---|
| `STA<±deg>` | absolute steering angle, **+ = right, − = left** | **YES — the only command we send** |
| `S<kmh>` | throttle PID target | no |
| `BRK<0-100>` | brake percent | no |
| `LOCK` / `UNLOCK` | enter/leave RUN mode | no (we stay in CONFIG) |
| `STCAL` | steering pot calibration | one-time setup only |
| `PING` | watchdog heartbeat | not needed in CONFIG |

Telemetry back (`$TEL`, ~20 Hz): `millis, STATE, thrMode, speedKmh, wheelRPM,
throttle%, steerActualDeg, steerTargetDeg, brakeActual%, brakeTarget%,
rampedRPM, pidErr, corr, faultFlags, dryRun`.

**Two firmware facts that matter:**
1. **STA is silently dropped until boot self-centering completes.** The script
   holds steering at centre and shows `CENTERING` until telemetry reports the
   `FAULT_NOT_CENTERED` bit cleared.
2. **CONFIG mode has no comms-loss watchdog-brake.** Steering only; the safety
   driver is the fallback. (RUN mode would auto-brake on comms loss but also
   arms throttle — we deliberately don't use it.)

> ⚠️ Do **not** use the old `--port` / `sda_serial_backend.py` path with this
> firmware. It speaks `SDA <value>`, which this Arduino misparses as a **throttle**
> command. Always use `--steer-port` (this guide).

---

## 3. One-time setup

### 3a. Steering pot calibration (`STCAL`)
Do this once (or whenever the steering pot/linkage changes), from a serial
terminal (`screen /dev/ttyUSB0 115200`, Arduino Serial Monitor, etc.) — **not**
the LKA script:

1. Send `STCAL`. Manually turn the wheels to **centre**, send any line → captures centre.
2. Turn wheels **full left**, send any line → captures left.
3. Turn wheels **full right**, send any line → captures right.
4. Send `LOCK` then `UNLOCK` to save the calibration to EEPROM and return to CONFIG.

The wheel then re-centres automatically. Verify `$TEL` `steerActualDeg` reads ~0.

### 3b. Confirm steering sign (2 minutes, safety-critical)
Firmware convention is **+ = right** (you confirmed this). Re-verify on the bench:
send `STA10` → front wheels must turn **RIGHT**; `STA-10` → **LEFT**.
If backwards, run the script with `--flip-steer` (do **not** rewire).

---

## 4. Bench validation before driving (no motion)

Prove the whole Jetson→Arduino chain with the vehicle safely on stands / wheels
off the ground:

```bash
# On the Arduino serial terminal, enable dry-run (control math + telemetry run,
# hardware outputs suppressed):
SET:DRY_RUN=1

# On the Jetson — headless, no video file, just watch the numbers:
python run_video_annotated.py --source 0 --stereo-eye left --adaptive-ipm \
    --steer-port /dev/ttyUSB0 --print-readings --no-video
```

You should see per-frame `CTE / heading / steering / confidence` plus an
`arduino  STA … | centered | v=…` line. Confirm `STA` tracks the lane sign
(car left of centre → steer right, etc.). When happy: `SET:DRY_RUN=0`.

---

## 5. On-vehicle run

```bash
python run_video_annotated.py \
    --source 0 \              # camera index (or /dev/videoN)
    --stereo-eye left \       # ZED side-by-side → left eye only
    --adaptive-ipm \          # auto-tune the bird's-eye zoom to the lanes
    --night-mode \            # only if lighting is dim; harmless in daylight
    --steer-port /dev/ttyUSB0 # STA steering-only link to the Arduino
```

- `--source` is a **camera index** (`0`) or device path (`/dev/video0`), not a file.
- Steering is the **1-second mean** of the per-frame command, re-sent every 0.5 s.
- Add `--show` for a live window (needs a display), `--print-readings` for terminal.
- Add `--no-video` on the vehicle unless you want a saved recording (uses disk).

### Tunables
| Flag | Default | Effect |
|---|---|---|
| `--steer-output-period` | 1.0 | seconds per steering-mean window |
| `--steer-heartbeat` | 0.5 | how often the current target is re-sent |
| `--steer-loss-ramp-dps` | 20 | on lane loss, ramp back to centre at this rate |
| `--flip-steer` | off | invert sign if the bench test showed it backwards |
| `--speed` | 5.0 | fallback speed for the Stanley gain if telemetry speed is stale |

---

## 6. Pre-drive checklist (run through every time)

- [ ] Steering calibrated (`STCAL` done, `steerActualDeg ≈ 0` at rest).
- [ ] Sign verified: `STA10` → wheels right.
- [ ] Bench dry-run passed (§4); `STA` follows the lane correctly.
- [ ] `SET:DRY_RUN=0`.
- [ ] Camera framed on the lane; script shows `centered` (not `CENTERING`) and lanes detected (`L:direct`/`R:direct`, confidence > ~0.4).
- [ ] Safety driver ready on throttle **and** brake (CONFIG mode = no auto-brake).
- [ ] Clear e-stop / kill path to cut Jetson↔Arduino or power.
- [ ] Start slow. Confirm the wheel follows the lane before adding speed.

---

## 7. If the steering fluctuates too much

The per-frame vision signal is naturally jittery; several knobs tame it (the
1-second mean is the biggest one, already on):

1. **Longer window** — `--steer-output-period 1.5` (smoother, slightly laggier).
2. **Gentler gains** — in `src/perception/lane_core/steering_v2.py`, lower
   `K_GAIN` (0.8) or raise `K_SOFT` (1.0) to soften the cross-track response.
3. **More Kalman smoothing** — increase the measurement-noise on cte/heading in
   `src/perception/lane_core/kalman_tracker.py`.
4. **Deadband** — small angles (< the Arduino's ~0.27° tolerance) are already
   ignored by the firmware; if it still dithers, raise `--steer-loss-ramp` is *not*
   the lever — use knobs 1–3.

Why it happens: heading error enters the Stanley command 1:1 (undamped), CTE is
noisy per frame, and low speed inflates the CTE gain. See the notes in
`steering_v2.py`.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| HUD shows `CENTERING` forever | boot centering not finishing | check steering pot wiring / `STCAL`; `$TEL faultFlags` bit `0x20` |
| HUD shows `no telem` / `LINK?` | wrong port / baud / cable | check `--steer-port`, `--steer-baud 115200`, USB cable |
| Wheels turn the **wrong way** | sign convention | add `--flip-steer` |
| Wheels barely move / wrong angle | pot not calibrated | run `STCAL` (§3a) |
| Throttle spins up unexpectedly | used `--port` (SDA) not `--steer-port` | use `--steer-port` |
| Steering never engages | low confidence / speed band | check camera framing; lanes must be detected & confidence gate passed |
| Bird's-eye shows blank / no lanes | IPM framing | keep `--adaptive-ipm` on (auto-zooms to the lanes) |
| `Source not found` | passed a file path for a camera | use a camera index (`0`) or `/dev/videoN` |

---

## 9. Known limitations / not-yet-verified

- **CONFIG mode has no comms-loss auto-brake** — the safety driver is the only
  fallback if the Jetson stalls or the link drops. Keep a hand on the brake.
- **Steering sign** is the firmware default (`+ = right`); re-verify per §3b.
- **Speed** comes from `$TEL` when the link is up, else the `--speed` fallback —
  a wrong speed only changes how *aggressively* it steers, not the direction.
- The IPM bird's-eye is uncalibrated (no camera intrinsics), so curvature is
  *proportional* to real curvature, not metric. Fine for lane keeping.
- **Steering-range mismatch:** the firmware maps **24° = full lock**
  (`recalcCPD` divides the pot travel by 24), but the pipeline clamps at ±30°
  (`MAX_STEERING_DEG` in `steering_v2.py`), so any command 24–30° saturates to
  full lock. If your buggy's real max road-wheel angle isn't ~24°, set
  `MAX_STEERING_DEG` (and/or the firmware's `/24.0`) to match so commanded
  degrees equal actual degrees.
