# steertest

Offline (video-file-in, not live-camera) lane-keeping pipeline for Jetson deployment. Reads a recorded video, runs the full perception -> reference -> safety -> actuator stack from the `baja_lka` project, and streams the resulting steering command to a real Arduino over serial as an `SDA` command.

This folder is a **trimmed, self-contained copy** of only the files needed to run this specific pipeline -- it does not depend on the parent `baja_lka` checkout at all and can be copied to a Jetson on its own.

## What's in here

```
steertest/
  run_video_to_arduino.py   <- entry point, run this
  requirements.txt
  README.md
  src/
    perception/             <- BirdEyeLaneDetectionNode + its lane_core/ dependencies
    ego_state/              <- IMU/dead-reckoning bridge (NullImuSource active -- no IMU wired up)
    reference/              <- ReferenceGenerator (perception -> control handoff contract)
    actuator/                <- ActuatorInterface, DbwBackend, and the new SdaSerialBackend
    safety/                  <- SafetyMonitor (engagement/fault gating)
    signal_conditioning/     <- gates raw perception readings before they reach the Kalman filter
  YOLOP/
    lib/                     <- trimmed vendored YOLOP (only what get_net()/inference actually needs --
                                 training code, dataset loaders, ONNX export, TensorRT deploy kit, and
                                 all matplotlib/scipy/prefetch_generator-only code paths were removed)
    weights/End-to-end.pth   <- the actual PyTorch checkpoint used at inference (92MB)
```

Nothing here is a placeholder or a stub -- every module is the real, working code copied from `baja_lka`, minus files that are genuinely unreachable from this pipeline (training scripts, dataset loaders, ONNX/TensorRT export tooling, demo/visualization code, matplotlib-only plotting utilities). See "What was left out and why" below if you want the exact reasoning.

## The SDA protocol (Jetson -> Arduino)

One ASCII line per command, newline-terminated:

```
SDA <value>\n
```

- `<value>` is the commanded steering angle in **degrees**, signed float, **2 decimal places**.
- **Negative = left turn, positive = right turn** (e.g. `SDA -12.34` = 12.34 degrees left, `SDA 30.00` = 30 degrees right).
- No colon or comma -- just `SDA`, one space, the number.
- This connection is **one-way** (Jetson -> Arduino only). No feedback read-back format was specified, so nothing is read back from the Arduino by this code.

This is the exact format given for this project's Arduino sketch. If your sketch expects a different terminator (e.g. `\r\n`) or framing, the only place this is written is `src/actuator/sda_serial_backend.py`'s `send_command()` -- change the `f"SDA {outgoing:.2f}\n"` line there.

### Sign convention -- please double-check this against your hardware before a real drive

`steering_deg` (the value sent as `<value>`) comes from this project's Stanley steering-law controller. Tracing the sign through the code:

- `src/reference/reference_generator.py` documents cross-track error as `+ = right of center` (i.e. positive means the lane center is to the right of the vehicle -- the vehicle needs to steer right to correct).
- `src/perception/lane_core/steering_v2.py`'s `stanley_steering_deg()` adds that cross-track-error term into the output angle with the **same sign** (`K_GAIN * cte_m` inside the `arctan`, added positively).

So a positive cross-track error (vehicle left of lane center, needs to correct right) produces a positive `steering_deg`, which matches the requested convention (positive = right) with **no sign flip applied anywhere in this code**. This was derived from the existing project's own documented invariant, not asserted from scratch -- but it has not been bench-verified against real steering hardware, so confirm a positive `SDA` value actually swings your steering rack right (and negative left) on the bench, at low command magnitude, before trusting this on a moving vehicle.

## Engagement gating -- when does the Arduino actually get a real command?

The Arduino receives a real (possibly non-zero) steering command **only while `SafetyMonitor` reports the system ENGAGED**. Engagement requires, simultaneously and for `ENGAGE_HOLD_TICKS` (5) consecutive frames:

- speed within `[MIN_ENGAGE_SPEED_MPS, MAX_ENGAGE_SPEED_MPS]` = `[0.5, 15.0]` m/s (the `--speed` you pass in)
- lane tracking confidence above `MIN_ENGAGE_CONFIDENCE` (0.3)
- the Kalman tracker initialized and not stale
- the actuator command watchdog not tripped (i.e. commands are actually flowing -- see below)
- no other active fault

**Every tick it is not engaged, the Arduino receives an explicit `SDA 0.00`** (neutral/straight-ahead) instead of going silent. This is deliberate: `run_video_to_arduino.py` always calls `ActuatorInterface.send()` every frame (engaged or not), because `SafetyMonitor` itself needs that steady stream of sends to know the actuator layer is alive (`ActuatorInterface.is_watchdog_tripped()` -- if sends stopped whenever disengaged, the watchdog could never clear and the system could never engage in the first place). The engaged/disengaged gate is applied one layer down, inside `SdaSerialBackend.send_command()`, which substitutes the neutral value on the wire when not engaged. Expect the first ~5 frames of any run to show `SDA 0.00` while the engage-hold counter builds up, even on a perfectly good video.

## Requirements

### 1. PyTorch for Jetson (install this FIRST, separately)

`torch`/`torchvision` are **not** in `requirements.txt` and must not be installed from PyPI on a Jetson -- PyPI has no CUDA-enabled wheel for Jetson's ARM64 + JetPack combination. Install NVIDIA's JetPack-matched wheel instead:

1. Check your JetPack/L4T version: `cat /etc/nv_tegra_release` or `sudo apt-cache show nvidia-jetpack`.
2. Find the matching PyTorch wheel on NVIDIA's Jetson PyTorch forum/index for that exact JetPack version (the wheel URL is version-specific, e.g. differs between JetPack 5.x and 6.x -- there isn't one fixed link that works for all of them).
3. `pip3 install <the matching torch .whl URL>`, then either use the matching prebuilt `torchvision` wheel from the same source, or build `torchvision` from source against the installed `torch` (see the `torchvision` README's "from source" instructions for the version compatible with your torch version).
4. Verify before continuing: `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"` should print `True`.

### 2. Everything else

```
pip install -r requirements.txt
```

`opencv-python` here is the plain pip wheel (no GStreamer/CSI-camera support, no CUDA) -- fine for this pipeline since it only reads a video file via `cv2.VideoCapture` and never touches a camera or does GPU-accelerated OpenCV ops.

### 3. Serial port permissions (Linux/Jetson)

The user running this script needs permission to open the Arduino's serial device (commonly `/dev/ttyACM0` for a native-USB Arduino, `/dev/ttyUSB0` behind a CH340/FTDI adapter):

```
sudo usermod -a -G dialout $USER
# log out and back in for the group change to take effect
```

Confirm the device exists and note its exact path: `ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null`.

## Optional: TensorRT acceleration (Jetson)

Eager PyTorch inference of YOLOP is the pipeline's dominant cost -- on a Jetson Orin the forward pass (plus pre/post-processing) measures ~113 ms/frame (~9 FPS) at `--img-size 640`, FP16. Converting the lane head to a TensorRT FP16 engine typically brings this to ~15-35 ms (a 3-6x speedup, ~25-40 FPS). This is optional -- the `.pth` path above works as-is -- but recommended before any real drive to cut sensorimotor latency.

We only consume YOLOP's lane-line segmentation head, so we export a **lane-only wrapper** to ONNX. This deliberately skips the object-detection `Detect` layer's grid/anchor decoding (the part that makes full-YOLOP ONNX export painful) and the unused drivable-area head, producing a smaller graph and a faster engine. The bundled `export_yolop_onnx.py` is self-contained and does **not** need the ONNX/TensorRT export tooling that was stripped from the vendored `YOLOP/` copy.

**Prerequisites:** TensorRT + `trtexec` (ships with JetPack, at `/usr/src/tensorrt/bin/trtexec`) and the `onnx` pip package. Build the engine **on the same Jetson it will run on** -- engines are device- and TensorRT-version-specific and are not portable between different Jetsons or JetPack/TRT versions.

### 1. PyTorch -> ONNX (lane-only wrapper)

```
python export_yolop_onnx.py
```

Writes `YOLOP/weights/yolop_ll_384x640.onnx`. The static input shape `(1,3,384,640)` matches a **1280x720** camera through `letterbox_for_img(new_shape=640, auto=True)`. If your camera resolution differs, run one frame through `letterbox_for_img` and print `img.shape`, then set `H, W` at the top of `export_yolop_onnx.py` to match.

### 2. ONNX -> TensorRT FP16 engine

```
/usr/src/tensorrt/bin/trtexec \
  --onnx=YOLOP/weights/yolop_ll_384x640.onnx \
  --saveEngine=YOLOP/weights/yolop_ll_384x640.fp16.engine \
  --fp16 --useCudaGraph --noDataTransfers \
  --iterations=100 --avgRuns=100
```

The first build takes ~1-5 min while TensorRT searches kernels (one-time). `trtexec` prints the mean GPU latency at the end -- that is your new per-frame inference number.

### 3. Run with the engine (now the default)

Once `YOLOP/weights/yolop_ll_384x640.fp16.engine` exists, **both entry points use it automatically on a CUDA device** -- no flag needed. The default weights resolver picks the engine when it's present and you're on GPU, and falls back to the `.pth` checkpoint otherwise (including any `--device cpu` run, since the engine can't run on CPU). Startup prints which one it chose, e.g. `[info] YOLOP weights: .../yolop_ll_384x640.fp16.engine (TensorRT engine)`.

To force a specific one, pass `--weights` explicitly: any path ending in `.engine` selects TensorRT, anything else loads the checkpoint.

```
# uses the engine automatically:
python run_video_to_arduino.py --device 0 --source /path/to/video.mp4 --port /dev/ttyACM0
# force the PyTorch checkpoint instead:
python run_video_to_arduino.py --device 0 --weights YOLOP/weights/End-to-end.pth ...
```

The pre/post-processing (letterbox, normalize, softmax, resize-to-full) is unchanged -- only the `model(x)` forward is replaced by an engine execution returning the raw `ll_seg` logits, so nothing downstream in the lane-fit/control path changes. `--img-size` is ignored for the geometry (the engine's input shape is baked in at export), so keep the camera resolution consistent with what you exported for.

### 4. Verify numerics before trusting it

TensorRT FP16 is not bit-identical to the PyTorch path. Run one real frame through both the `.pth` and the engine and confirm the lane-probability maps agree within FP16 tolerance (max abs diff around 1e-2) before relying on the engine on a moving vehicle.

## Running it

```
python run_video_to_arduino.py \
  --source /path/to/your/video.mp4 \
  --port /dev/ttyACM0 \
  --baud 115200 \
  --device 0 \
  --speed 5.0 \
  --lane-width 3.0 \
  --log-csv /path/to/run_log.csv
```

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--source` | *(required)* | Path to the input video file. |
| `--port` | *(required)* | Arduino serial device path, e.g. `/dev/ttyACM0`. |
| `--baud` | `115200` | Serial baud rate. |
| `--weights` | *(auto)* | YOLOP weights: a `.engine` (TensorRT) or `.pth` (checkpoint). Defaults to the bundled TensorRT engine when it exists and you're on GPU, else the `.pth` checkpoint. See "TensorRT acceleration". |
| `--device` | `0` | `0` selects CUDA:0 (the Jetson's GPU) -- pass `cpu` to force CPU inference (much slower, but useful for a sanity-check run without a GPU). |
| `--img-size` | `640` | YOLOP inference resolution. |
| `--lane-width` | `3.0` | Assumed lane width in meters. A hand-picked starting value carried over from `baja_lka`, not a measurement of your actual track/course -- tune to match reality. |
| `--speed` | `5.0` | Assumed vehicle speed in m/s. There is no real speed sensor wired into this project (see `src/ego_state/imu_source.py` -- `NullImuSource` is always active), so this is a fixed operator-supplied value feeding both the Stanley controller's speed term and `SafetyMonitor`'s engage-speed-band check, not a live measurement. Set it to the actual speed you intend to drive at when replaying this video. |
| `--speed-factor` | `1.0` | Video playback pacing multiplier. `1.0` plays the video at its own real frame rate (recommended -- see "Why frames are paced to real time" below). Values `>1.0` speed up playback for a quicker non-hardware test run, but then `--port`'s Arduino would receive commands faster than a real drive would produce them. |
| `--log-csv` | *(none)* | Optional path to write a per-frame CSV log (cross-track error, heading, steering angle, engagement state, fault list, and exactly what was sent to the Arduino each tick). |

### Why frames are paced to real time

Every time-dependent block in this pipeline -- the Kalman tracker's `dt`, the Stanley controller's slew-rate limiter, `ActuatorInterface`'s second-stage rate limiter, and `SafetyMonitor`'s 0.5s command watchdog -- assumes the `now_s` passed into it advances at real wall-clock speed, exactly as it would from a live camera. `run_video_to_arduino.py` paces frame delivery to the video's own FPS (adjustable via `--speed-factor`) rather than decoding and sending commands as fast as the CPU/GPU allows, so these rate limits and timeouts behave the same way they would during an actual drive.
