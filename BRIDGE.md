# The bridge — `run_video_annotated.py`

One command that takes **any** recorded video and produces **two deliverables**:

1. an **annotated video** (lane overlay + bird's-eye picture-in-picture + a
   CTE / heading / steering / confidence / engagement HUD + a steering gauge), and
2. a **per-second steering CSV** (`elapsed_s`, the SDA steering command that
   second, the exact SDA wire string, and diagnostics).

It runs **log-only by default — no Arduino / no serial port required.**

```bash
cd ~/steertest
python run_video_annotated.py                        # uses the bundled dashcam.mp4
python run_video_annotated.py --source /path/any.mp4 # any other video
```

Outputs land next to the source (or in `--out-dir`):
`<name>.annotated.mp4` and `<name>.steering_per_second.csv`.

---

## How the two projects connect (the thing that was unclear)

They are **two independent lane-keeping stacks**, not a perception + serial
pair. This bridge wires the video-file workflow to the **steertest** stack,
which is the one that *generates the steering command*:

| | `baja_lane_control` (the ROS repo) | **`steertest`** (used here) |
|---|---|---|
| Lane detection | classical OpenCV sliding-window | **YOLOP** deep-learning segmentation |
| Runtime | C++ / ROS 2 | pure Python |
| Steering law | Stanley | Stanley |
| Arduino protocol | `STR/STL/STC` | **`SDA <signed 2-dp>`** |

Per the project decision, **steertest/YOLOP is the steering generator**; this
bridge is the offline (video-in, no-hardware) counterpart to
`run_video_to_arduino.py`, adding the annotated video and per-second logging
that the serial entry point doesn't produce.

Pipeline the bridge drives (unchanged steertest code):

```
video frame → YOLOP → IPM bird's-eye → boundary extract → quadratic fit
            → Kalman track → Stanley steering → slew-limit → safety gate → SDA command
```

## What each per-second CSV column means

| column | meaning |
|---|---|
| `elapsed_s` | integer second of the **video** timeline (0,1,2,…) |
| `sda_steering_deg` | that second's command = **mean of the steering values while ENGAGED**; `0.00` if the whole second was disengaged (the neutral value really put on the wire) |
| `sda_command` | the exact wire string, e.g. `SDA -3.20` (negative = left, positive = right, **no leading `+`** — byte-identical to the real `SdaSerialBackend`) |
| `engaged_frac` | fraction of that second's frames the safety monitor was ENGAGED |
| `mean_confidence` / `mean_cte_m` / `mean_heading_deg` | per-second averages |
| `n_frames` | frames aggregated into that second |
| `faults` | union of safety faults seen (e.g. `low_confidence`, `engage_hold_pending`) |

> First ~5 frames of any run are DISENGAGED by design (`ENGAGE_HOLD_TICKS`),
> and low-confidence/dark footage stays disengaged and emits `SDA 0.00`
> (fail-safe). That is correct behaviour, not a fault of the tool.

## Options

| flag | default | meaning |
|---|---|---|
| `--source` | bundled `dashcam.mp4` | input video (any file) |
| `--out-dir` | next to `--source` | where outputs go |
| `--frame-csv PATH` | *(off)* | also write a per-frame CSV |
| `--device` | `0` (CUDA) | `cpu` to force CPU; auto-falls back to CPU if CUDA is unavailable **or** errors (e.g. out-of-memory) |
| `--speed` | `5.0` | assumed vehicle speed m/s (Stanley + engage band; no speed sensor exists) |
| `--lane-width` | `3.0` | assumed lane width m |
| `--port /dev/ttyACM0` | *(off)* | also stream real SDA to an Arduino (implies real-time pacing) |
| `--no-video` | off | skip the annotated video (CSV only, faster) |
| `--max-frames N` | 0 (all) | stop after N frames (smoke test) |

## Timing model

- **Offline (default):** the pipeline clock is `frame_index / fps` — correct
  per-tick `dt` for the Kalman filter and slew limiters, but no real-time
  wait, so it renders as fast as the GPU allows (~14 fps on this Jetson).
- **`--port` / `--realtime`:** paces to the wall clock and drives the
  pipeline off real elapsed time, exactly like `run_video_to_arduino.py`, so
  a real processing stall trips the actuator watchdog instead of being hidden.

## Robustness (validated)

No crash on: any resolution/aspect ratio (tiny 64×48 … 1080p), portrait,
grayscale or BGRA frames, missing/implausible FPS (bounded, defaults to 30),
3-frame clips, mid-stream resolution changes (frames coerced to keep one
playable stream), transient decode/inference hiccups (bad frame skipped),
and CUDA out-of-memory (falls back to CPU). Empty/corrupt files exit non-zero
with a clear message. Partial results (video + CSV) are always flushed, even
on `Ctrl-C` or a mid-run error.

## Setup note

Requires the steertest deps plus `yacs` (needed by YOLOP's config loader):

```bash
pip install -r requirements.txt   # torch/torchvision installed separately, see README.md
pip install yacs                  # if not already present
```
