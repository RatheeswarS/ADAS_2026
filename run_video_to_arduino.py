#!/usr/bin/env python3
"""
run_video_to_arduino.py
--------------------------
Offline pipeline entry point: reads a recorded video file (NOT a live
camera feed), runs it through the full perception -> reference ->
safety -> actuator stack, and streams the resulting steering command to
a real Arduino over serial using the "SDA <value>" protocol:

    SDA <signed float, 2 decimals>\n

negative = left turn, positive = right turn (e.g. "SDA -12.34" for a
12.34 degree left turn, "SDA 30.00" for a 30 degree right turn).

Commands are only transmitted with a real (non-zero) steering value while
SafetyMonitor reports the system ENGAGED (speed in band, tracking
confident and not stale, no active faults, held for
safety.ENGAGE_HOLD_TICKS consecutive ticks). Every other tick, the
Arduino receives an explicit neutral "SDA 0.00" instead -- see
src/actuator/sda_serial_backend.py's docstring for why this is enforced
at the backend rather than by skipping ActuatorInterface.send() itself
(skipping it would starve the actuator command watchdog that engagement
depends on in the first place).

Frames are paced to the video's own real elapsed time (not decoded as
fast as the CPU/GPU allows), because every time-dependent block in this
stack -- the Kalman tracker's dt, the Stanley controller's slew-rate
limiter, ActuatorInterface's second-stage rate limiter, and
SafetyMonitor's command watchdog -- assumes `now_s` is real elapsed
seconds between calls, exactly as it would be fed by a live camera. See
src/perception/lane_core/steering.py's TIMING NOTE for the same point
made about the legacy estimator.

USAGE
    python run_video_to_arduino.py --source path/to/video.mp4 --port /dev/ttyACM0
"""
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "YOLOP"))

import cv2  # noqa: E402

from perception.lane_detection_node_birdeye import BirdEyeLaneDetectionNode  # noqa: E402
from perception.lane_core.model_runtime import select_device  # noqa: E402
from reference.reference_generator import ReferenceGenerator  # noqa: E402
from actuator.dbw_interface import ActuatorInterface, CommandMode  # noqa: E402
from actuator.sda_serial_backend import SdaSerialBackend  # noqa: E402
from safety.safety_monitor import SafetyMonitor  # noqa: E402

DEFAULT_ENGINE = os.path.join(HERE, "YOLOP", "weights", "yolop_ll_384x640.fp16.engine")
DEFAULT_CHECKPOINT = os.path.join(HERE, "YOLOP", "weights", "End-to-end.pth")


def default_weights(device: str) -> str:
    """Prefer the TensorRT engine (~3.5x faster, see README) when it exists AND
    we're on a CUDA device; otherwise fall back to the PyTorch checkpoint. The
    engine can't run on CPU, so a `--device cpu` run always uses the checkpoint."""
    if device != "cpu" and os.path.exists(DEFAULT_ENGINE):
        return DEFAULT_ENGINE
    return DEFAULT_CHECKPOINT


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="input video file path (a recorded file, not a camera index)")
    ap.add_argument("--weights", default=None,
                    help="YOLOP weights: a .engine (TensorRT) or .pth (checkpoint). "
                         "Default: the bundled TensorRT engine when present and on GPU, "
                         "else the .pth checkpoint.")
    ap.add_argument("--device", default="0", help="'0' for CUDA:0 (Jetson GPU), or 'cpu'")
    ap.add_argument("--img-size", type=int, default=640, help="YOLOP inference resolution")
    ap.add_argument("--lane-width", type=float, default=3.0, help="assumed lane width in meters")
    ap.add_argument("--speed", type=float, default=5.0,
                     help="FALLBACK vehicle speed in m/s -- feeds the Stanley law's speed term and "
                          "SafetyMonitor's engage-speed-band check when no IMU/wheel speed is "
                          "available. A live --imu-port reading overrides it; without one this "
                          "fixed operator-supplied value is used, not a measurement")
    ap.add_argument("--imu-port", default=None,
                     help="serial port of the WitMotion HWT905 IMU (its own link, e.g. /dev/ttyUSB1); "
                          "when given, the vehicle's real yaw rate feeds the Stanley law's damping term "
                          "and the safety plausibility check. Speed comes separately from the motor "
                          "controller via the --port Arduino link")
    ap.add_argument("--imu-baud", type=int, default=9600,
                     help="HWT905 serial baud (match the unit's configured rate; WitMotion default 9600)")
    ap.add_argument("--imu-yaw-sign", type=float, default=1.0,
                     help="+1 or -1: flips the HWT905 yaw-rate sign to match this vehicle's mounting "
                          "and ego_state_estimator's -yaw_rate convention; verify on the bench")
    ap.add_argument("--speed-factor", type=float, default=1.0,
                     help="video playback pacing multiplier relative to the source's own frame rate; "
                          "1.0 = real-time playback (recommended -- see module docstring on timing)")
    ap.add_argument("--port", required=True, help="Arduino serial port, e.g. /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200, help="Arduino serial baud rate")
    ap.add_argument("--log-csv", default=None, help="optional per-frame CSV log path")
    return ap.parse_args()


def main():
    opt = parse_args()

    device = select_device(None, opt.device)

    if opt.weights is None:
        opt.weights = default_weights(opt.device)
    kind = "TensorRT engine" if opt.weights.endswith(".engine") else "PyTorch checkpoint"
    print(f"[info] YOLOP weights: {opt.weights} ({kind})")

    # Arduino link first: it owns the serial port AND reads the motor
    # controller's speed telemetry off it -- so the ego-state's speed source
    # is derived from this backend (hence built before the node).
    backend = SdaSerialBackend(opt.port, opt.baud)

    # Ego-motion sensors, matching the vehicle's real hardware split:
    #   yaw rate <- HWT905 IMU (its own serial link, --imu-port)
    #   speed    <- motor controller, via the Arduino backend above
    # Both feed the steering-angle calculation (steering_v2.py); each falls
    # back gracefully if its sensor is absent (vision-only yaw / --speed).
    from ego_state.imu_source import Hwt905ImuSource, ArduinoSpeedSource, CompositeImuSource
    yaw_source = None
    if opt.imu_port:
        yaw_source = Hwt905ImuSource(opt.imu_port, opt.imu_baud, yaw_sign=opt.imu_yaw_sign)
        print(f"[info] HWT905 IMU on {opt.imu_port} @ {opt.imu_baud} -- live yaw rate feeds the steering law")
    else:
        print("[warn] no --imu-port: yaw rate falls back to vision-only heading rate (no HWT905)")
    imu_source = CompositeImuSource(yaw_source=yaw_source, speed_source=ArduinoSpeedSource(backend))
    print("[info] vehicle speed sourced from the motor controller via the Arduino (SPD telemetry); "
          f"--speed {opt.speed} is the fallback until real telemetry arrives")

    node = BirdEyeLaneDetectionNode(opt.weights, device, img_size=opt.img_size,
                                     lane_width_m=opt.lane_width, speed_mps=opt.speed,
                                     imu_source=imu_source)
    refgen = ReferenceGenerator()
    actuator = ActuatorInterface(backend=backend, mode=CommandMode.ANGLE)
    safety = SafetyMonitor()

    cap = cv2.VideoCapture(opt.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source video: {opt.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval_s = 1.0 / (fps * opt.speed_factor)

    csv_file = None
    csv_writer = None
    if opt.log_csv:
        csv_file = open(opt.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "time_s", "confidence", "cross_track_error_m",
                              "heading_deg", "steering_deg_raw", "actuator_commanded_value",
                              "engaged", "wire_value_sent_to_arduino", "faults"])

    t0 = time.monotonic()
    next_due = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            if next_due is not None and now < next_due:
                time.sleep(next_due - now)
            next_due = time.monotonic() + frame_interval_s

            frame_idx += 1
            now_s = time.monotonic() - t0

            pf = node.process(frame, now_s)
            ref = refgen.update(pf)
            # Evaluate safety using the actuator's state as of the END of the
            # PREVIOUS tick's send (is_watchdog_tripped only looks backward in
            # time) -- this lets us tell the backend this tick's engagement
            # state before it transmits, without ever skipping the send that
            # the watchdog itself depends on. See sda_serial_backend.py.
            # Use the speed/yaw the pipeline actually ran on this tick (measured
            # when an IMU is attached, --speed fallback otherwise); passing the
            # measured yaw rate also arms the monitor's kinematic plausibility
            # check, which stays dormant while it's None (no IMU).
            safety_state = safety.evaluate(ref, pf["steering_deg"], pf["speed_mps"], actuator, now_s,
                                           measured_yaw_rate_dps=pf["yaw_rate_dps"])
            backend.set_engaged(safety_state.engaged)
            act_result = actuator.send(pf["steering_deg"], now_s)
            wire_value = act_result["commanded_value"] if safety_state.engaged else 0.0

            if frame_idx % 30 == 0 or frame_idx == frame_count:
                status = "ENGAGED" if safety_state.engaged else "DISENGAGED"
                print(f"\rframe {frame_idx}/{frame_count}  steering={pf['steering_deg']:+.1f} deg  "
                      f"wire=SDA {wire_value:+.2f}  {status}"
                      + (f"  faults={','.join(safety_state.faults)}" if safety_state.faults else ""),
                      end="")

            if csv_writer:
                csv_writer.writerow([
                    frame_idx, f"{now_s:.3f}", f"{pf['confidence']:.4f}",
                    f"{pf['cross_track_error_m']:.4f}", f"{pf['heading_deg']:.4f}",
                    f"{pf['steering_deg']:.4f}", f"{act_result['commanded_value']:.4f}",
                    safety_state.engaged, f"{wire_value:.2f}", "|".join(safety_state.faults),
                ])
    finally:
        cap.release()
        backend.close()
        imu_source.close()   # stops the HWT905 reader thread (no-op for the borrowed Arduino speed link)
        if csv_file:
            csv_file.close()

    print(f"\nDone: {frame_idx} frames processed from {opt.source}")


if __name__ == "__main__":
    main()
