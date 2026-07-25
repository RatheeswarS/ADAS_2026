#!/usr/bin/env python3
"""
run_video_annotated.py  --  the "bridge"
=========================================
Single, self-contained offline tool that takes ANY recorded video, runs it
through the steertest YOLOP lane-keeping stack (the exact same
perception -> reference -> safety -> actuator chain as
run_video_to_arduino.py), and writes TWO deliverables:

  1. an ANNOTATED video  (lane overlay + bird's-eye picture-in-picture +
     CTE/heading/steering/confidence/engagement HUD + a steering gauge), and
  2. a PER-SECOND steering CSV  (one row per elapsed second of video:
     elapsed_s, the SDA steering command that second, the SDA wire string,
     and diagnostics).

Why this exists separately from run_video_to_arduino.py
------------------------------------------------------
run_video_to_arduino.py REQUIRES a live Arduino serial port (--port) and
logs PER FRAME with no video output. This tool is the offline/analysis
counterpart:

  * No hardware needed.  By default it runs "log-only" (a DryRunSdaBackend
    that reproduces SdaSerialBackend's exact engagement gating -- real
    command while ENGAGED, neutral "SDA 0.00" otherwise, byte-identical wire
    formatting -- but never opens a serial port).  Pass --port to also
    stream real SDA commands to a connected Arduino (this forces real-time
    pacing; see below).

  * Two clocks, kept separate on purpose:
      - video_s = frame_index / fps  -> WHERE we are in the video; drives the
        per-second CSV bucketing and the HUD "t=" readout, so every row maps
        to the video's own timeline regardless of processing speed.
      - now_s   = the PIPELINE clock fed to the Kalman tracker's dt, both
        slew-rate limiters, and the safety watchdog.  In the default offline
        mode now_s == video_s (deterministic, correct per-tick dt, no
        real-time wait -- the annotated video renders as fast as the GPU
        allows).  When streaming to a real Arduino (--port, or explicit
        --realtime) now_s is the real monotonic wall clock, exactly like
        run_video_to_arduino.py, so a real processing stall correctly trips
        the actuator watchdog and neutralizes instead of being masked by a
        simulated clock.

Robustness: handles any resolution/aspect ratio, grayscale/BGRA frames,
missing/implausible FPS, mid-stream resolution changes, transient decode
hiccups (skips the bad frame), and CUDA out-of-memory (falls back to CPU).
Partial results (video + per-second CSV) are always flushed, even on Ctrl-C
or a mid-run error.

Sign / protocol convention is inherited unchanged from the steertest stack:
SDA <signed float, 2 decimals>, negative = left, positive = right, and (as
on the real wire) NO leading '+' for non-negative values.

USAGE
    python run_video_annotated.py                       # defaults to bundled dashcam.mp4
    python run_video_annotated.py --source my_clip.mp4
    python run_video_annotated.py --source clip.mp4 --out-dir /tmp/out --device cpu

    # Live bench-check before uploading to the vehicle: watch the annotated
    # video in a window AND stream the 4 outputs to the terminal, saving nothing.
    python run_video_annotated.py --source clip.mp4 --show --print-readings --no-video

    # Side-by-side STEREO source (e.g. d_dash.mp4 at 3840x1080): crop to the
    # left eye so the pipeline gets a single-camera feed. Stereo eyes are
    # cropped, never merged (blending ghosts the lane lines).
    python run_video_annotated.py --source d_dash.mp4 --stereo-eye left

    # ON-VEHICLE LKA: live ZED (side-by-side) in, STEERING-ONLY STA commands out
    # to the Arduino as the 1-second mean. Throttle/brake untouched (CONFIG mode);
    # a safety driver controls speed/braking. --source is a camera index or /dev
    # path. See LKA_README.md for the full pre-drive checklist.
    python run_video_annotated.py --source 0 --stereo-eye left --adaptive-ipm \
        --steer-port /dev/ttyUSB0 --print-readings
"""
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "YOLOP"))

import cv2          # noqa: E402
import numpy as np  # noqa: E402

from perception.lane_detection_node_birdeye import BirdEyeLaneDetectionNode  # noqa: E402
from perception.lane_core.model_runtime import select_device                 # noqa: E402
from perception.lane_core.steering_v2 import (                                # noqa: E402
    MAX_STEERING_DEG, stanley_steering_deg, SlewRateLimiter,
)
from reference.reference_generator import ReferenceGenerator                 # noqa: E402
from actuator.dbw_interface import ActuatorInterface, CommandMode            # noqa: E402
from actuator.dbw_backend import DbwBackend                                  # noqa: E402
from safety.safety_monitor import SafetyMonitor                              # noqa: E402

# >>> STEREO-1L: robust single-lane recovery via ZED stereo depth + rulebook
# lane width, used when a sharp turn leaves only one boundary detected (see
# single_lane_stereo_recovery.py for the geometry and a worked example).
from single_lane_stereo_recovery import (                                    # noqa: E402
    StereoIntrinsics, ZedDepthProvider, StereoBMDepthProvider, recover_single_lane,
)

DEFAULT_WEIGHTS = os.path.join(HERE, "YOLOP", "weights", "End-to-end.pth")
DEFAULT_SOURCE = os.path.join(HERE, "fintest.mp4")
LIVE_WINDOW_NAME = "run_video_annotated -- live (q/ESC to stop)"

# Per-source colours (BGR) for the warped-back lane boundary, keyed by how
# that side's curve was obtained this frame (see lane_detection_node_birdeye).
SOURCE_COLOURS = {
    "direct":        (0, 255, 0),      # green   -- a real detection this frame
    "reconstructed": (0, 215, 255),    # amber   -- inferred from the other side + width model
    "held":          (0, 165, 255),    # orange  -- coasting on the last good fit
    "none":          (120, 120, 120),  # grey    -- no curve
}


# --------------------------------------------------------------------------
# Log-only actuator backend (no serial port)
# --------------------------------------------------------------------------
class DryRunSdaBackend(DbwBackend):
    """DbwBackend that behaves EXACTLY like SdaSerialBackend on the wire --
    same engagement gating (real value while engaged, neutral 0.0 otherwise),
    same byte-identical "SDA <value>" 2-decimal formatting (no forced '+'
    sign, matching sda_serial_backend.py) -- but writes nothing to a serial
    port. It always reports success so ActuatorInterface's command watchdog
    stays satisfied (the safety monitor's engagement decision depends on that
    steady send stream; see sda_serial_backend.py's docstring)."""

    def __init__(self):
        self._engaged = False
        self.last_outgoing = 0.0
        self.last_line = "SDA 0.00"

    def set_engaged(self, engaged: bool) -> None:
        self._engaged = engaged

    def send_command(self, value: float, mode: str, now_s: float) -> bool:
        self.last_outgoing = value if self._engaged else 0.0
        self.last_line = f"SDA {self.last_outgoing:.2f}"   # byte-identical to SdaSerialBackend
        return True

    def read_actual_angle_deg(self):
        return None

    def close(self):
        pass


# --------------------------------------------------------------------------
# Overlay / annotation helpers
# --------------------------------------------------------------------------
def _as_bgr(frame):
    """Guarantee a 3-channel BGR uint8 frame for any decoded input, or None
    if the frame is empty/undecodable."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def crop_stereo_eye(frame, eye):
    """Crop a side-by-side stereo frame ([left | right], two eyes concatenated
    horizontally -- e.g. a ZED read as a plain UVC webcam, or a recorded stereo
    mp4 like d_dash.mp4 at 3840x1080) down to ONE eye, so everything downstream
    sees a single-camera feed.

    The correct stereo->mono operation is a CROP, not a merge/blend: the two
    eyes are the same scene from the stereo baseline apart, so blending them
    just ghosts (double lane lines) -- which is exactly the failure the crop
    exists to avoid. 'left' is the right choice (matches zed_publisher.py's
    rectified-left-eye output in the ROS stack). eye='none' returns the frame
    unchanged (source already single-eye)."""
    if eye == "none":
        return frame
    h, w = frame.shape[:2]
    half = w // 2
    if eye == "left":
        return frame[:, :half].copy()   # .copy() so the right half's pixels don't exist downstream
    return frame[:, w - half:].copy()   # 'right'


def _curve_points_warped(curve, n=60):
    """Sample (x, y) points of a LaneCurve in bird's-eye (warped) space."""
    ys = np.linspace(curve.y_min, curve.y_max, n)
    xs = curve.x_at(ys)
    return xs, ys


def _project_clean(xs, ys, M_inv, w, h):
    """Map bird's-eye (x, y) samples back to original-frame pixel coords,
    dropping non-finite and absurdly-out-of-range points (near-horizon back-
    projection can blow up), so cv2 never receives NaN/inf or int32-overflow
    coordinates."""
    pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2)
    bound = 10 * max(w, h)
    keep = np.isfinite(out).all(axis=1) & (np.abs(out) < bound).all(axis=1)
    return out[keep]


def _lane_heading_deg_at(curve, y):
    """Lane heading (deg) at bird's-eye row y == atan(local slope). The warp is
    set up so a bird's-eye slope already reads as the perspective-corrected
    heading (see lane_detection_node_birdeye.py), so this is directly comparable
    to pf['heading_deg'] and to the Stanley heading term."""
    yy = float(np.clip(y, curve.y_min, curve.y_max))
    return float(np.degrees(np.arctan(curve.slope_at(yy))))


def curve_preview_deg(pf, lookahead_frac):
    """TURN ANTICIPATION signal: how much the lane heading CHANGES between the
    vehicle (near/bottom row of the bird's-eye) and a look-ahead point
    `lookahead_frac` of the way to the farthest fitted point. Degrees, sign
    matching pf['heading_deg'] (+ = the road bends right ahead). This is a
    curvature proxy read off the lane fit's far field -- it swings BEFORE the
    near-vehicle heading does, which is what lets us pre-steer into a sharp turn
    instead of only reacting at the apex.

    Uses the centerline when both boundaries exist, else the single detected
    boundary (locally-parallel lanes share the same heading change). Returns 0.0
    when there's no usable curve or too little forward span to read."""
    curves = [pf[s] for s in ("left", "right") if pf.get(s) is not None]
    if not curves:
        return 0.0
    y_near = min(float(c.y_max) for c in curves)   # bottom of the warp == nearest the vehicle
    y_far = max(float(c.y_min) for c in curves)    # top of the warp == farthest ahead
    span = y_near - y_far
    if span <= 1.0:
        return 0.0
    y_ahead = y_near - float(np.clip(lookahead_frac, 0.05, 0.95)) * span
    near = float(np.mean([_lane_heading_deg_at(c, y_near) for c in curves]))
    ahead = float(np.mean([_lane_heading_deg_at(c, y_ahead) for c in curves]))
    return ahead - near


def draw_lane_overlay(frame, pf, M_inv):
    """Draw both lane boundaries (warped back onto the original perspective)
    and, when both exist, a translucent fill of the lane area."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    drawn = {}
    for side in ("left", "right"):
        curve = pf[side]
        if curve is None:
            continue
        xs, ys = _curve_points_warped(curve)
        pts = _project_clean(xs, ys, M_inv, w, h)
        if len(pts) < 2:
            continue
        drawn[side] = pts
        colour = SOURCE_COLOURS.get(pf[f"{side}_source"], (255, 255, 255))
        cv2.polylines(frame, [pts.astype(np.int32)], False, colour, 4, cv2.LINE_AA)

    if "left" in drawn and "right" in drawn:
        poly = np.vstack([drawn["left"], drawn["right"][::-1]]).astype(np.int32)
        cv2.fillPoly(overlay, [poly], (0, 200, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame


def draw_birdeye_pip(frame, pf, pip_w=170, margin=12):
    """Bird's-eye picture-in-picture: warped frame + lane-probability mask +
    fitted curves, drawn top-right. Sized to fit ANY frame (never silently
    omitted on small frames)."""
    bev = pf["warped_frame"].copy()
    mask = pf["warped_mask"]
    tint = np.zeros_like(bev)
    tint[mask > 0] = (0, 255, 255)  # cyan where the model says "lane line"
    cv2.addWeighted(tint, 0.5, bev, 1.0, 0, bev)
    for side in ("left", "right"):
        curve = pf[side]
        if curve is None:
            continue
        xs, ys = _curve_points_warped(curve)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        colour = SOURCE_COLOURS.get(pf[f"{side}_source"], (255, 255, 255))
        cv2.polylines(bev, [pts], False, colour, 2, cv2.LINE_AA)

    bh, bw = bev.shape[:2]
    h, w = frame.shape[:2]
    # Clamp PiP width to the frame, then clamp height and re-derive width so
    # the aspect ratio holds and the box always fits.
    pw = max(40, min(pip_w, w - 2 * margin))
    ph = int(bh * (pw / bw))
    max_h = h - 2 * margin
    if ph > max_h:
        ph = max(1, max_h)
        pw = max(1, int(bw * (ph / bh)))
    if pw < 8 or ph < 8:
        return frame  # frame too small for any meaningful PiP
    pip = cv2.resize(bev, (pw, ph))
    x0, y0 = max(0, w - pw - margin), margin
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + pw + 2, y0 + ph + 2), (255, 255, 255), 1)
    frame[y0:y0 + ph, x0:x0 + pw] = pip
    cv2.putText(frame, "bird's-eye", (x0, min(h - 4, y0 + ph + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, pf, wire_value, engaged, faults, video_s, frame_idx, sta_status=None,
             stereo_status=None):
    """Translucent text panel, top-left."""
    yaw = pf.get("yaw_rate_dps")
    yaw_str = f"{yaw:+.1f} dps" if yaw is not None else "n/a (no IMU)"
    lines = [
        f"t = {video_s:6.2f} s   frame {frame_idx}",
        f"CTE       {pf['cross_track_error_m']:+.3f} m",
        f"heading   {pf['heading_deg']:+.2f} deg",
        f"steering  {pf['steering_deg']:+.2f} deg",
        f"SDA wire  {wire_value:.2f}",             # wire form: no forced '+'
        f"speed     {pf.get('speed_mps', 0.0):.2f} m/s",
        f"yaw rate  {yaw_str}",
        f"conf      {pf['confidence']:.2f}   L:{pf['left_source']} R:{pf['right_source']}",
        f"state     {'ENGAGED' if engaged else 'DISENGAGED'}",
    ]
    if pf.get("ipm_adaptive"):
        lock = "LOCKED" if pf.get("ipm_zoom_locked") else "searching"
        lines.append(f"ipm zoom  step {pf['ipm_zoom_step']} ({lock})")
    if pf.get("night_mode"):
        lines.append(f"night     ON  (lane thr {pf.get('lane_prob_threshold', 0.5):.2f})")
    if pf.get("curve_preview_ff_deg") is not None:
        lines.append(f"preview   d{pf.get('curve_preview_deg', 0.0):+.1f} -> ff {pf['curve_preview_ff_deg']:+.1f} deg")
    if sta_status:
        lines.append(f"arduino   {sta_status}")
    if stereo_status:
        lines.append(f"stereo1L  {stereo_status}")
    if faults:
        lines.append("faults    " + ",".join(faults))

    pad, lh = 8, 20
    box_w = 340
    box_h = pad * 2 + lh * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    y = 8 + pad + 14
    for i, line in enumerate(lines):
        colour = (255, 255, 255)
        if line.startswith("state"):
            colour = (0, 255, 0) if engaged else (0, 165, 255)
        if line.startswith("faults"):
            colour = (0, 165, 255)
        if line.startswith("stereo1L"):
            colour = (255, 200, 0)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        y += lh
    return frame


def draw_steering_gauge(frame, steering_deg, engaged):
    """Horizontal steering bar at the bottom centre: centre tick + a filled
    marker proportional to steering / MAX_STEERING_DEG (left = negative). The
    numeric value is drawn ABOVE the bar so it never collides with the L/R
    end labels."""
    h, w = frame.shape[:2]
    bar_w, bar_h = min(360, max(80, w - 60)), 16
    x0 = (w - bar_w) // 2
    y0 = h - 30
    cx = x0 + bar_w // 2
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    cv2.line(frame, (cx, y0 - 4), (cx, y0 + bar_h + 4), (255, 255, 255), 1)
    frac = float(np.clip(steering_deg / MAX_STEERING_DEG, -1.0, 1.0))
    mx = int(cx + frac * (bar_w // 2))
    colour = (0, 255, 0) if engaged else (0, 165, 255)
    cv2.rectangle(frame, (min(cx, mx), y0), (max(cx, mx), y0 + bar_h), colour, -1)
    cv2.putText(frame, f"{steering_deg:+.1f} deg", (cx - 34, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.putText(frame, "L", (max(2, x0 - 16), y0 + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "R", (min(w - 14, x0 + bar_w + 6), y0 + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------
def gui_available():
    """True only if cv2 can actually open a window here -- i.e. a real display
    is reachable AND OpenCV was built with GUI (highgui) support. Lets --show
    degrade to readings-only on a headless box / SSH session instead of raising
    cv2.error mid-run."""
    # On Linux, no DISPLAY/WAYLAND env => no window is possible; skip the probe
    # (which would otherwise just fail) and answer immediately.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
            and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(LIVE_WINDOW_NAME)
        return True
    except cv2.error:
        # OpenCV built without GUI support, or the display rejected the window.
        return False


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="input video file (default: bundled dashcam.mp4)")
    ap.add_argument("--stereo-eye", choices=["auto", "none", "left", "right"], default="auto",
                    help="handle a side-by-side STEREO source ([left|right] in one frame, e.g. a raw "
                         "ZED or fintest.mp4 at 3840x1080) by CROPPING to one eye before the pipeline "
                         "(stereo eyes are never merged/blended -- that ghosts the lanes). "
                         "'auto' (default) = crop the LEFT eye when the frame looks side-by-side "
                         "(wide aspect), otherwise pass through; 'left'/'right' force a crop; "
                         "'none' forces no crop.")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: alongside --source)")
    ap.add_argument("--out-video", default=None, help="annotated video path (overrides --out-dir)")
    ap.add_argument("--out-csv", default=None, help="per-second CSV path (overrides --out-dir)")
    ap.add_argument("--frame-csv", default=None, help="optional extra per-frame CSV path")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOP checkpoint path")
    ap.add_argument("--device", default="0",
                    help="'0' for CUDA:0, or 'cpu'. Auto-falls back to cpu if CUDA is "
                         "unavailable OR raises at load/inference (e.g. out-of-memory).")
    ap.add_argument("--img-size", type=int, default=640, help="YOLOP inference resolution")
    ap.add_argument("--lane-width", type=float, default=3.0, help="assumed lane width in meters")
    ap.add_argument("--speed", type=float, default=5.0,
                    help="FALLBACK vehicle speed m/s (Stanley speed term + engage-band) used only "
                         "when no IMU/wheel speed is available; overridden by a live --imu-port reading")
    ap.add_argument("--imu-port", default=None,
                    help="serial port of the WitMotion HWT905 IMU (its own link, e.g. /dev/ttyUSB1); "
                         "when given, the vehicle's real yaw rate feeds the Stanley damping term and "
                         "the safety plausibility check. Speed comes from the motor controller via the "
                         "--port Arduino link (SPD telemetry), else the --speed fallback is used")
    ap.add_argument("--imu-baud", type=int, default=9600,
                    help="HWT905 serial baud (match the unit's configured rate; WitMotion default 9600)")
    ap.add_argument("--imu-yaw-sign", type=float, default=1.0,
                    help="+1 or -1: flips HWT905 yaw-rate sign to match this vehicle's mounting and "
                         "ego_state_estimator's -yaw_rate convention; verify on the bench")
    ap.add_argument("--port", default=None,
                    help="optional Arduino serial port; if omitted, runs log-only (no hardware). "
                         "Implies --realtime.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--realtime", action="store_true",
                    help="pace to the wall clock and drive the pipeline off real elapsed time "
                         "(automatic when --port is given)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all); handy for smoke tests")
    ap.add_argument("--no-video", action="store_true", help="skip annotated-video output")
    ap.add_argument("--show", action="store_true",
                    help="play the annotated frames in a live window as they are processed so you "
                         "can watch the lane marks track the road (press q or ESC to stop early). "
                         "Combine with --no-video to watch without writing any file.")
    ap.add_argument("--print-readings", action="store_true",
                    help="print the 4 outputs (CTE, heading, steering, confidence) for every "
                         "processed frame, line by line, so the terminal readings track the video.")
    ap.add_argument("--adaptive-ipm", action="store_true",
                    help="adaptive bird's-eye zoom: while no lane is detected, widen the warp "
                         "trapezoid step by step (zoom out); lock the zoom on the first detection "
                         "and hold it for the rest of the run. Off = fixed framing (unchanged).")
    ap.add_argument("--night-mode", action="store_true",
                    help="stronger low-light enhancement front-end (adaptive gamma + CLAHE + "
                         "edge-preserving denoise) for nighttime footage, in place of the default "
                         "mild dim-frame lift. Off = unchanged.")
    ap.add_argument("--no-night-denoise", action="store_true",
                    help="with --night-mode, skip the bilateral denoise step (cheaper per frame "
                         "on a slow platform, at the cost of more amplified sensor noise).")
    ap.add_argument("--lane-prob-threshold", type=float, default=0.5,
                    help="YOLOP lane-probability cutoff for the warped mask (default 0.5). Lower "
                         "(e.g. 0.35-0.45) recovers faint night lines at the cost of more noise; "
                         "the whiteness/reach gates downstream reject most of the extra.")
    # ---- on-vehicle steering link to the low-level Arduino (arduinocode.ino) ----
    ap.add_argument("--steer-port", default=None,
                    help="serial port of the low-level Arduino steering controller (arduinocode.ino) "
                         "for on-vehicle LKA. Streams STEERING-ONLY commands (STA<deg>, + = right) as "
                         "the 1-second MEAN of the per-frame steering; throttle & brake are never "
                         "touched (CONFIG mode). Implies --realtime. This is NOT the --port SDA path.")
    ap.add_argument("--steer-baud", type=int, default=115200, help="Arduino steering-link baud (default 115200)")
    ap.add_argument("--steer-output-period", type=float, default=1.0,
                    help="seconds per steering window; the MEAN over each window is the angle "
                         "commanded to the wheels (default 1.0 s).")
    ap.add_argument("--steer-heartbeat", type=float, default=0.5,
                    help="re-send the current steering target at least this often (s) to keep the "
                         "actuator target fresh (default 0.5 s).")
    ap.add_argument("--steer-loss-ramp-dps", type=float, default=20.0,
                    help="when a 1-s window has no engaged frames (lane lost), ramp steering back "
                         "toward centre at this rate (deg/s) instead of holding a stale turn.")
    ap.add_argument("--flip-steer", action="store_true",
                    help="invert steering sign before sending. Firmware convention is + = right; "
                         "use ONLY if a bench test shows the wheels turn the wrong way.")

    # ---- STEREO-1L: single-lane recovery via ZED stereo depth ----
    ap.add_argument("--stereo-depth-fallback", action="store_true",
                    help="when only ONE lane boundary is detected (typical mid-sharp-turn), recover "
                         "CTE/heading from real-world stereo depth + the rulebook lane width instead "
                         "of the image-space width-model reconstruction. Falls back to the existing "
                         "behavior automatically whenever the stereo read isn't confident (too few "
                         "valid depth points, or they don't fit a consistent line).")
    ap.add_argument("--zed-depth", action="store_true",
                    help="use the ZED SDK's own depth (requires pyzed.sl and a live camera opened via "
                         "the SDK -- see the ZED_CAMERA placeholder in main()). Without this flag, "
                         "--stereo-depth-fallback uses classical block-matching on the LEFT+RIGHT crop "
                         "pair instead, which is fine for bench-testing this feature on a recorded "
                         "side-by-side file but is NOT the on-vehicle depth source.")
    ap.add_argument("--stereo-baseline-m", type=float, default=0.12,
                    help="stereo baseline in meters, for the StereoBM fallback depth (ZED 2i default "
                         "baseline ~0.12 m; verify against your unit's spec sheet). Unused with --zed-depth.")
    ap.add_argument("--camera-fx", type=float, default=None,
                    help="left-camera focal length in pixels (fx), AT THE RESOLUTION fed to the "
                         "pipeline (i.e. after --stereo-eye crop). REQUIRED for --stereo-depth-fallback. "
                         "Get from ZED SDK calibration_parameters.left_cam.fx, or scale_intrinsics() "
                         "in single_lane_stereo_recovery.py if your calibration was taken at a "
                         "different resolution.")
    ap.add_argument("--camera-fy", type=float, default=None, help="left-camera fy in pixels; defaults to --camera-fx if omitted")
    ap.add_argument("--camera-cx", type=float, default=None, help="left-camera principal point cx in pixels; defaults to frame_width/2")
    ap.add_argument("--camera-cy", type=float, default=None, help="left-camera principal point cy in pixels; defaults to frame_height/2")
    ap.add_argument("--stereo-min-points", type=int, default=4,
                    help="minimum valid stereo-depth samples along the visible boundary required to "
                         "trust the recovery; below this, fall back to the existing behavior.")
    ap.add_argument("--stereo-max-rms-m", type=float, default=0.35,
                    help="max allowed line-fit residual (meters) through the sampled 3D boundary "
                         "points; a worse fit means inconsistent/noisy depth, so fall back instead "
                         "of trusting it.")
    ap.add_argument("--stereo-cte-sign", type=float, default=1.0,
                    help="+1 or -1: flip if the recovered CTE comes out with the wrong sign vs your "
                         "reference_generator's convention -- verify on the bench, same pattern as "
                         "--imu-yaw-sign.")
    ap.add_argument("--stereo-heading-sign", type=float, default=1.0,
                    help="+1 or -1: flip if the recovered heading comes out with the wrong sign vs "
                         "your reference_generator's convention -- verify on the bench.")

    # ---- CURVE-PREVIEW: sharp-turn anticipation (look-ahead curvature feed-forward) ----
    ap.add_argument("--curve-preview-gain", type=float, default=0.0,
                    help="TURN ANTICIPATION (default 0.0 = OFF, behavior byte-for-byte unchanged). "
                         "Adds a feed-forward pre-steer of this many degrees per degree of lane-"
                         "heading CHANGE between the vehicle and a look-ahead point -- so the buggy "
                         "starts turning BEFORE a sharp turn's apex instead of only reacting at it. "
                         "Read off the bird's-eye lane fit's far field and fused with the Stanley "
                         "angle. Start small (~0.3) and bench-tune; negative flips the sign.")
    ap.add_argument("--curve-preview-lookahead-frac", type=float, default=0.6,
                    help="how far ahead to read the turn, as a fraction (0-1) of the visible lane "
                         "span in the bird's-eye view (0.6 = 60%% of the way to the farthest fitted "
                         "point). Larger anticipates earlier but reads a noisier far-field fit.")
    ap.add_argument("--curve-preview-max-deg", type=float, default=8.0,
                    help="hard cap (deg) on the preview feed-forward's contribution, so a noisy "
                         "far-field fit can't command a large pre-steer on its own.")
    return ap.parse_args()


def resolve_outputs(opt):
    stem = os.path.splitext(os.path.basename(opt.source))[0]
    out_dir = opt.out_dir or os.path.dirname(os.path.abspath(opt.source))
    os.makedirs(out_dir, exist_ok=True)
    out_video = opt.out_video or os.path.join(out_dir, f"{stem}.annotated.mp4")
    out_csv = opt.out_csv or os.path.join(out_dir, f"{stem}.steering_per_second.csv")
    return out_video, out_csv


def open_video_writer(base_path, fps, size):
    """Open a VideoWriter with a codec/container that reliably FINALIZES on this
    platform, returning (writer, tag, actual_path).

    AVI (XVID/MJPG) is tried FIRST on purpose: OpenCV's mp4v writer on this
    Jetson leaves a large .mp4 without its trailing 'moov' atom, making the whole
    file unreadable ("moov atom not found") even though every frame was written.
    AVI has no such end-of-file index dependency and finalizes reliably. mp4v is
    kept only as a last resort. The extension of the returned path may differ
    from base_path so it matches the codec actually used."""
    stem = os.path.splitext(base_path)[0]
    for tag, ext in (("XVID", ".avi"), ("MJPG", ".avi"), ("mp4v", ".mp4")):
        path = stem + ext
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, size)
        if writer.isOpened():
            return writer, tag, path
        writer.release()
    return None, None, None


def build_node(opt, device, imu_source=None):
    return BirdEyeLaneDetectionNode(opt.weights, device, img_size=opt.img_size,
                                    lane_width_m=opt.lane_width, speed_mps=opt.speed,
                                    imu_source=imu_source, adaptive_zoom=opt.adaptive_ipm,
                                    night_mode=opt.night_mode,
                                    night_denoise=not opt.no_night_denoise,
                                    lane_prob_threshold=opt.lane_prob_threshold)


def main():
    opt = parse_args()
    # source may be a video FILE, a camera INDEX ("0", "1", ...), or a device
    # path (/dev/videoN) for a live on-vehicle feed.
    is_camera = str(opt.source).isdigit() or str(opt.source).startswith("/dev/")
    if not is_camera and not os.path.isfile(opt.source):
        raise SystemExit(f"Source not found (not a file, camera index, or /dev path): {opt.source}")
    if not os.path.isfile(opt.weights):
        raise SystemExit(f"YOLOP weights not found: {opt.weights}")

    # Live window needs a reachable display; on a headless box / SSH session
    # fall back to readings-only instead of crashing on cv2.imshow.
    if opt.show and not gui_available():
        print("[warn] --show requested but no display is available (headless / SSH session, or "
              "OpenCV built without GUI support).\n"
              "       Continuing WITHOUT the live window."
              + ("" if opt.print_readings else
                 " Add --print-readings to watch the 4 outputs in the terminal instead."))
        opt.show = False

    if opt.port:
        opt.realtime = True  # never blast commands at a real Arduino faster than real time
    if opt.steer_port:
        opt.realtime = True  # pace to wall clock so the 1-second steering windows are real seconds
    hardware_clock = bool(opt.realtime or opt.port or opt.steer_port)

    out_video_path, out_csv_path = resolve_outputs(opt)

    # ---- device selection (graceful CPU fallback if CUDA is unavailable) ----
    try:
        device = select_device(None, opt.device)
    except Exception as e:
        print(f"[warn] could not select device '{opt.device}' ({e}); falling back to cpu")
        device = select_device(None, "cpu")

    # ---- Arduino link (built before the node: it is also the speed source) ----
    # With --port it opens the real Arduino, which BOTH receives SDA commands
    # AND reports the motor controller's speed telemetry back; without it, the
    # DryRun stub logs SDA only (offline analysis) and there is no live speed,
    # so the controller uses the --speed fallback.
    if opt.port:
        from actuator.sda_serial_backend import SdaSerialBackend
        backend = SdaSerialBackend(opt.port, opt.baud)
        print(f"[info] streaming SDA to {opt.port} @ {opt.baud} (real-time paced); "
              "vehicle speed read back from the motor controller via the Arduino (SPD telemetry)")
    else:
        backend = DryRunSdaBackend()
        print("[info] log-only mode (no serial port) -- SDA commands are logged, not transmitted")

    # ---- steering-only link to the low-level Arduino controller (arduinocode.ino) ----
    # Speaks STA<deg> (steering only, CONFIG mode) -- NOT the SDA protocol of --port
    # above. Its $TEL telemetry also carries the vehicle's real speed, which feeds
    # the Stanley speed term below (via ArduinoSpeedSource, duck-typed on read_speed_mps).
    steer_link = None
    if opt.steer_port:
        try:
            from actuator.arduino_steering_link import ArduinoSteeringLink
            steer_link = ArduinoSteeringLink(opt.steer_port, opt.steer_baud, flip_sign=opt.flip_steer)
        except Exception as e:
            raise SystemExit(f"Could not open steering link on {opt.steer_port}: {e}")
        print(f"[info] steering link on {opt.steer_port} @ {opt.steer_baud} -- STA steering-only, "
              f"CONFIG mode, + = right{' (FLIPPED)' if opt.flip_steer else ''}; "
              f"{opt.steer_output_period:.1f}s mean. Steering is HELD at centre until the Arduino "
              "reports boot-centering complete.")

    # ---- ego-motion sensors (vehicle's real split: HWT905 yaw + Arduino speed) ----
    # Built once and reused across a possible CUDA->CPU node rebuild so each
    # serial port is opened only once. On the bare dev machine (no --imu-port,
    # no --port) both halves are absent -> NullImuSource behavior, i.e. the
    # speed/yaw-rate path stays a no-op and offline output is unchanged.
    from ego_state.imu_source import Hwt905ImuSource, ArduinoSpeedSource, CompositeImuSource
    yaw_source = None
    if opt.imu_port:
        yaw_source = Hwt905ImuSource(opt.imu_port, opt.imu_baud, yaw_sign=opt.imu_yaw_sign)
        print(f"[info] HWT905 IMU on {opt.imu_port} @ {opt.imu_baud} -- live yaw rate feeds the steering law")
    if opt.port:
        speed_source = ArduinoSpeedSource(backend)
    elif steer_link is not None:
        speed_source = ArduinoSpeedSource(steer_link)  # real vehicle speed from the STA link's $TEL
    else:
        speed_source = None
    imu_source = CompositeImuSource(yaw_source=yaw_source, speed_source=speed_source)

    # ---- model load + warmup (graceful CPU fallback on a CUDA runtime error) ----
    print(f"[info] loading YOLOP on {device} ...")
    try:
        node = build_node(opt, device, imu_source)
    except RuntimeError as e:
        if device.type == "cpu":
            raise
        print(f"[warn] CUDA failure while building/warming the model ({e}); falling back to cpu")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        device = select_device(None, "cpu")
        node = build_node(opt, device, imu_source)

    refgen = ReferenceGenerator()
    actuator = ActuatorInterface(backend=backend, mode=CommandMode.ANGLE)
    safety = SafetyMonitor()

    cap = cv2.VideoCapture(int(opt.source) if str(opt.source).isdigit() else opt.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {opt.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 1e-3 or fps > 1000.0:
        print(f"[warn] source reports no/implausible FPS ({fps}); assuming 30.0")
        fps = 30.0
    frame_total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    # Resolve --stereo-eye into the eye actually cropped each frame. A side-by-side
    # stereo frame ([left|right]) fed in as if it were mono is the exact failure
    # that produces plausible-but-wrong CTE/heading, so 'auto' crops the LEFT eye
    # whenever the frame is much wider than tall (two eyes side by side). Stereo
    # eyes are CROPPED, never merged/blended -- blending ghosts the lane lines.
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    looks_stereo = bool(src_h) and (src_w / src_h) > 2.5   # e.g. 3840x1080 = 3.56
    stereo_eye_eff = opt.stereo_eye
    if opt.stereo_eye == "auto":
        stereo_eye_eff = "left" if looks_stereo else "none"
        if looks_stereo:
            print(f"[info] auto: side-by-side stereo detected ({src_w}x{src_h}, aspect "
                  f"{src_w / src_h:.2f}) -> cropping to the LEFT eye ({src_w // 2}x{src_h}). "
                  "Override with --stereo-eye none/left/right.")
    elif stereo_eye_eff == "none" and looks_stereo:
        print(f"[warn] source is {src_w}x{src_h} (aspect {src_w / src_h:.2f}) -- looks like a "
              "side-by-side STEREO frame but --stereo-eye is 'none'; proceeding WITHOUT cropping "
              "(the pipeline will see BOTH eyes as one image and mis-detect). Use 'auto' or 'left'.")
    elif stereo_eye_eff != "none":
        print(f"[info] stereo source: cropping to the {stereo_eye_eff.upper()} eye "
              f"({src_w}x{src_h} -> {src_w // 2}x{src_h}) before the pipeline")

    # ---- STEREO-1L: depth provider + intrinsics (built once) ----
    depth_provider = None
    intr = None
    zed_cam = None  # placeholder: wire this to your pyzed capture object if --zed-depth
    if opt.stereo_depth_fallback:
        if opt.camera_fx is None:
            raise SystemExit("--stereo-depth-fallback requires --camera-fx (and ideally --camera-fy/"
                             "--camera-cx/--camera-cy) -- see single_lane_stereo_recovery.py docstring "
                             "for where to get these from the ZED SDK calibration.")
        fx = opt.camera_fx
        fy = opt.camera_fy if opt.camera_fy is not None else fx
        crop_w = (src_w // 2) if stereo_eye_eff != "none" else src_w
        crop_h = src_h
        cx = opt.camera_cx if opt.camera_cx is not None else crop_w / 2.0
        cy = opt.camera_cy if opt.camera_cy is not None else crop_h / 2.0
        intr = StereoIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        if opt.zed_depth:
            if zed_cam is None:
                raise SystemExit("--zed-depth needs a live ZED SDK camera object (zed_cam) -- open one "
                                 "with pyzed.sl.Camera() where this placeholder is, matching --source; "
                                 "the plain cv2.VideoCapture path above only gives raw pixels, not depth.")
            depth_provider = ZedDepthProvider(zed_cam)
        else:
            depth_provider = StereoBMDepthProvider(baseline_m=opt.stereo_baseline_m, fx=fx)
            if stereo_eye_eff == "none":
                raise SystemExit("--stereo-depth-fallback without --zed-depth needs a side-by-side "
                                 "stereo SOURCE (both eyes) to block-match depth from -- --stereo-eye "
                                 "resolved to 'none' for this source, so there's no second eye to use. "
                                 "Either point --source at a stereo file/camera, or add --zed-depth.")
            print(f"[info] stereo-1L fallback: StereoBM depth (baseline={opt.stereo_baseline_m:.3f} m, "
                  f"fx={fx:.1f}px) -- bench/offline only; use --zed-depth on the vehicle")

    frame_interval_s = 1.0 / fps
    t_wall = time.monotonic()

    writer = None
    writer_size = None
    buckets = {}   # video-second -> accumulator

    # 1-second steering consolidator (the value actually commanded to the wheels).
    steer_consolidator = None
    if steer_link is not None:
        from actuator.arduino_steering_link import SecondMeanSteering
        steer_consolidator = SecondMeanSteering(output_period_s=opt.steer_output_period,
                                                loss_ramp_dps=opt.steer_loss_ramp_dps)
    steer_last_send_t = 0.0
    steer_sent_deg = 0.0
    steer_status = ""  # HUD line

    # Single authority for the steering command. The node computes its own
    # pf["steering_deg"] INSIDE process(), i.e. BEFORE the stereo single-lane
    # recovery below can correct cte/heading -- so on a recovered frame that
    # baked-in value is stale. We recompute the Stanley command here from the
    # FINAL (possibly recovered) cte/heading and push it through ONE slew
    # limiter, then overwrite pf["steering_deg"] so every downstream consumer
    # (safety, SDA actuator.send, the STA 1-s mean, HUD, CSV) uses it. On
    # frames where recovery never fires this reproduces the node's command
    # exactly (same inputs, same law, same default slew rate), so behavior is
    # unchanged whenever --stereo-depth-fallback is off or inactive.
    cmd_slew = SlewRateLimiter()
    last_trusted_raw_deg = 0.0

    frame_csv_file = frame_csv_writer = None
    if opt.frame_csv:
        frame_csv_file = open(opt.frame_csv, "w", newline="")
        frame_csv_writer = csv.writer(frame_csv_file)
        frame_csv_writer.writerow(["frame", "elapsed_s", "confidence", "cross_track_error_m",
                                   "heading_deg", "steering_deg", "wire_sda_value", "engaged", "faults"])

    frame_idx = 0        # frames read (incl. skipped) -- also the video-time index
    processed = 0        # frames that produced a pipeline result
    skipped = 0
    interrupted = False

    def flush_and_report():
        # Always write the per-second CSV (even on interruption / mid-run
        # error) so the primary deliverable survives alongside the partial
        # video. steering column = mean of the ENGAGED wire commands that
        # second (a value the controller actually commanded), or 0.00 for a
        # fully-disengaged second (the neutral value really on the wire).
        with open(out_csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["elapsed_s", "sda_steering_deg", "sda_command", "engaged_frac",
                         "mean_confidence", "mean_cte_m", "mean_heading_deg", "n_frames", "faults"])
            for sec in sorted(buckets):
                b = buckets[sec]
                eng = [w for w, e in zip(b["wire"], b["eng"]) if e]
                steering = round(float(np.mean(eng)), 2) if eng else 0.0
                wr.writerow([
                    sec, f"{steering:.2f}", f"SDA {steering:.2f}",
                    f"{b['engaged'] / b['n']:.2f}", f"{float(np.mean(b['conf'])):.3f}",
                    f"{float(np.mean(b['cte'])):.3f}", f"{float(np.mean(b['head'])):.3f}",
                    b["n"], "|".join(sorted(b["faults"])),
                ])
        print(f"\n[done] {processed} frames processed"
              + (f", {skipped} skipped" if skipped else "")
              + (" (interrupted)" if interrupted else ""))
        if writer is not None:
            print(f"[out ] annotated video : {out_video_path}")
        print(f"[out ] per-second CSV  : {out_csv_path}  ({len(buckets)} seconds)")
        if opt.frame_csv:
            print(f"[out ] per-frame CSV   : {opt.frame_csv}")

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = _as_bgr(raw)
            if frame is None:                 # (True, None) / empty decode hiccup
                skipped += 1
                frame_idx += 1
                continue
            # Stereo -> mono: crop to a single eye before ANYTHING else touches
            # the frame, so the pipeline (and the annotated output) only ever
            # see a single-camera feed.
            # >>> STEREO-1L: grab the OTHER eye too, before it's discarded, if
            # block-matching depth needs it this run (unused/None otherwise).
            other_eye_frame = None
            if isinstance(depth_provider, StereoBMDepthProvider):
                other_eye = "right" if stereo_eye_eff == "left" else "left"
                other_eye_frame = crop_stereo_eye(frame, other_eye)
            frame = crop_stereo_eye(frame, stereo_eye_eff)

            video_s = frame_idx / fps         # position in the video timeline
            if opt.realtime and not is_camera:
                # A live camera already self-paces (cap.read blocks until a frame
                # is ready); only sleep-pace when replaying a FILE in real time.
                due = t_wall + frame_idx * frame_interval_s
                dt = due - time.monotonic()
                if dt > 0:
                    time.sleep(dt)
            now_s = (time.monotonic() - t_wall) if hardware_clock else video_s

            try:
                pf = node.process(frame, now_s)
            except RuntimeError as e:
                # Transient CUDA OOM / inference error on one frame: free what
                # we can, skip it, keep going (partial results still flushed).
                if device.type == "cuda":
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                print(f"\n[warn] frame {frame_idx} inference failed ({e}); skipping")
                skipped += 1
                frame_idx += 1
                continue

            # >>> STEREO-1L: exactly one boundary visible -- try real-world
            # stereo recovery before falling through to refgen's own
            # (image-space width-model) reconstruction. Only overrides
            # pf's CTE/heading when the stereo read is actually confident;
            # otherwise pf is untouched and behavior is unchanged.
            stereo_status = None
            recovered = False   # did the stereo single-lane recovery override cte/heading this frame?
            if depth_provider is not None:
                # Fire when exactly ONE side is a REAL detection this frame and
                # the other is only an image-space width-model guess
                # ("reconstructed") or absent ("none") -- that guess is precisely
                # what stereo geometry should replace on a sharp turn. Checking
                # only for the other side being literally None misses almost
                # every real case: the pipeline RECONSTRUCTS a missing boundary
                # rather than dropping it, so it's a non-None "reconstructed"
                # curve, not None. recover_single_lane() still self-gates on
                # depth quality and returns None (-> unchanged) when unsure.
                GUESS = ("reconstructed", "none")
                left_direct = pf["left"] is not None and pf["left_source"] == "direct"
                right_direct = pf["right"] is not None and pf["right_source"] == "direct"
                single_side = None
                if left_direct and pf["right_source"] in GUESS:
                    single_side = "left"
                elif right_direct and pf["left_source"] in GUESS:
                    single_side = "right"
                if single_side is not None:
                    if isinstance(depth_provider, StereoBMDepthProvider) and other_eye_frame is not None:
                        left_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if stereo_eye_eff == "left" \
                            else cv2.cvtColor(other_eye_frame, cv2.COLOR_BGR2GRAY)
                        right_gray = cv2.cvtColor(other_eye_frame, cv2.COLOR_BGR2GRAY) if stereo_eye_eff == "left" \
                            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        depth_provider.update(left_gray, right_gray)
                    elif isinstance(depth_provider, ZedDepthProvider):
                        depth_provider.refresh()
                    rec = recover_single_lane(
                        pf, node.ipm_M_inv, single_side, depth_provider, intr, opt.lane_width,
                        min_points=opt.stereo_min_points, max_rms_m=opt.stereo_max_rms_m,
                        cte_sign=opt.stereo_cte_sign, heading_sign=opt.stereo_heading_sign)
                    if rec is not None:
                        pf["cross_track_error_m"] = rec.cte_m
                        pf["heading_deg"] = rec.heading_deg
                        pf["curvature_1pm"] = rec.curvature_1pm
                        recovered = True
                        stereo_status = (f"{rec.side_used[0].upper()} n={rec.n_points} "
                                         f"rms={rec.rms_residual_m:.2f}m -> "
                                         f"CTE {rec.cte_m:+.2f}m hd {rec.heading_deg:+.1f}deg "
                                         f"k={rec.curvature_1pm:+.3f}")
                    else:
                        stereo_status = f"{single_side[0].upper()} low-confidence -> unchanged"

            # ---- recompute the steering command from the FINAL cte/heading ----
            # This is what makes the stereo recovery actually change the angle
            # sent to the wheels (the node baked pf["steering_deg"] before the
            # override). Mirrors the node's own gating: 0 until the tracker has
            # locked; hold the last trusted command while stale -- UNLESS a fresh
            # stereo recovery fired this frame, which is a real measurement and
            # so overrides the hold. Single cmd_slew limiter => continuous output.
            if not pf["tracker_initialized"]:
                raw_cmd_deg = 0.0
            elif pf["tracker_stale"] and not recovered:
                raw_cmd_deg = last_trusted_raw_deg
            else:
                raw_cmd_deg = stanley_steering_deg(
                    pf["cross_track_error_m"], pf["heading_deg"],
                    pf["speed_mps"], yaw_rate_dps=pf["yaw_rate_dps"])
                # Turn anticipation: fuse a bounded look-ahead curvature feed-
                # forward so the buggy pre-steers into a sharp turn. OFF by
                # default (gain 0.0) -> this whole block is skipped and the
                # command is exactly the reactive Stanley value above.
                if opt.curve_preview_gain != 0.0:
                    preview_deg = curve_preview_deg(pf, opt.curve_preview_lookahead_frac)
                    ff_deg = float(np.clip(opt.curve_preview_gain * preview_deg,
                                           -opt.curve_preview_max_deg, opt.curve_preview_max_deg))
                    pf["curve_preview_deg"] = preview_deg
                    pf["curve_preview_ff_deg"] = ff_deg
                    raw_cmd_deg = float(np.clip(raw_cmd_deg + ff_deg,
                                                -MAX_STEERING_DEG, MAX_STEERING_DEG))
                last_trusted_raw_deg = raw_cmd_deg
            pf["steering_deg_raw"] = raw_cmd_deg
            pf["steering_deg"] = cmd_slew.limit(raw_cmd_deg, now_s)

            ref = refgen.update(pf)
            # Evaluate safety using the actuator state as of the END of the
            # PREVIOUS tick's send (watchdog only looks backward), then tell
            # the backend this tick's engagement before it transmits -- same
            # ordering as run_video_to_arduino.py.
            # Feed the safety monitor the speed/yaw the pipeline actually used
            # this tick (measured when an IMU is attached, configured fallback
            # otherwise) rather than the static --speed: the engage-speed band
            # now tracks real speed, and passing the measured yaw rate activates
            # the monitor's bicycle-model plausibility check (dormant while it's
            # None on the IMU-less dev machine).
            safety_state = safety.evaluate(ref, pf["steering_deg"], pf["speed_mps"], actuator, now_s,
                                           measured_yaw_rate_dps=pf["yaw_rate_dps"])
            backend.set_engaged(safety_state.engaged)
            act = actuator.send(pf["steering_deg"], now_s)
            engaged = safety_state.engaged
            wire_value = act["commanded_value"] if engaged else 0.0

            # ---- steering to the low-level Arduino (STA, 1-second mean) ----
            if steer_link is not None:
                centered = steer_link.centered  # True/False/None(no telemetry yet)
                if engaged and centered:
                    steer_consolidator.add(pf["steering_deg"])
                target_deg, updated = steer_consolidator.update(now_s)
                # Send a new mean each window, plus a heartbeat re-send; but stay
                # silent until the firmware finishes boot-centering (it drops STA
                # until then, and we don't want to command a turn pre-centre).
                if centered and (updated or (now_s - steer_last_send_t) >= opt.steer_heartbeat):
                    steer_link.send_steering_deg(target_deg)
                    steer_last_send_t = now_s
                    steer_sent_deg = target_deg
                tv = steer_link.read_speed_mps()
                steer_status = (
                    f"STA {steer_sent_deg:+.1f} | "
                    + ("centered" if centered else ("CENTERING" if centered is False else "no telem"))
                    + (f" v={tv:.1f}" if tv is not None else "")
                    + ("" if steer_link.link_alive else " LINK?"))

            # ---- per-second aggregation (keyed by video time) ----
            sec = int(video_s)
            b = buckets.setdefault(sec, {"wire": [], "eng": [], "conf": [], "cte": [],
                                         "head": [], "engaged": 0, "n": 0, "faults": set()})
            b["wire"].append(wire_value)
            b["eng"].append(engaged)
            b["conf"].append(pf["confidence"])
            b["cte"].append(pf["cross_track_error_m"])
            b["head"].append(pf["heading_deg"])
            b["engaged"] += int(engaged)
            b["n"] += 1
            b["faults"].update(safety_state.faults)

            if frame_csv_writer:
                frame_csv_writer.writerow([
                    frame_idx, f"{video_s:.3f}", f"{pf['confidence']:.4f}",
                    f"{pf['cross_track_error_m']:.4f}", f"{pf['heading_deg']:.4f}",
                    f"{pf['steering_deg']:.4f}", f"{wire_value:.2f}", engaged,
                    "|".join(safety_state.faults)])

            # ---- annotate (built once, then optionally saved and/or shown live) ----
            if not opt.no_video or opt.show:
                vis = frame.copy()
                M_inv = node.ipm_M_inv
                if M_inv is not None:
                    vis = draw_lane_overlay(vis, pf, M_inv)
                vis = draw_birdeye_pip(vis, pf)
                vis = draw_hud(vis, pf, wire_value, engaged, safety_state.faults, video_s, frame_idx,
                               sta_status=steer_status, stereo_status=stereo_status)
                vis = draw_steering_gauge(vis, pf["steering_deg"], engaged)

                if not opt.no_video:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer_size = (w, h)
                        writer, tag, out_video_path = open_video_writer(out_video_path, fps, writer_size)
                        if writer is None:
                            raise SystemExit("Could not open a VideoWriter with any tried codec (XVID/MJPG/mp4v).")
                        print(f"[info] writing annotated video ({w}x{h} @ {fps:.2f} fps, "
                              f"codec={tag}, path={out_video_path})")
                    to_write = vis
                    if (vis.shape[1], vis.shape[0]) != writer_size:
                        # Source changed resolution mid-stream: keep one continuous,
                        # playable stream by coercing to the writer's frame size.
                        # Coerce a copy so the live window still shows native size.
                        to_write = cv2.resize(vis, writer_size)
                    writer.write(to_write)

                if opt.show:
                    cv2.imshow(LIVE_WINDOW_NAME, vis)
                    # 1 ms poll keeps the window responsive without pacing the run;
                    # add --realtime for wall-clock playback speed.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):  # q or ESC
                        interrupted = True
                        print("\n[warn] stopped from live window -- flushing partial results")
                        break

            # ---- per-frame readings: the 4 outputs, one per line ----
            if opt.print_readings:
                zoom = ""
                if pf.get("ipm_adaptive"):
                    zoom = (f"  zoom step {pf['ipm_zoom_step']} "
                            f"({'LOCKED' if pf['ipm_zoom_locked'] else 'searching'})")
                print(
                    f"frame {frame_idx:>6}  t={video_s:6.2f}s   "
                    f"{'ENGAGED' if engaged else 'DISENGAGED'}  "
                    f"[L:{pf['left_source']} R:{pf['right_source']}]{zoom}\n"
                    f"  CTE         {pf['cross_track_error_m']:+8.3f} m\n"
                    f"  heading     {pf['heading_deg']:+8.2f} deg\n"
                    f"  steering    {pf['steering_deg']:+8.2f} deg\n"
                    f"  confidence  {pf['confidence']:8.2f}"
                    + (f"\n  arduino     {steer_status}" if steer_status else "")
                    + (f"\n  stereo1L    {stereo_status}" if stereo_status else "")
                    + (f"\n  preview     d{pf.get('curve_preview_deg', 0.0):+.1f}deg -> ff {pf['curve_preview_ff_deg']:+.1f}deg"
                       if pf.get('curve_preview_ff_deg') is not None else "")
                )

            processed += 1
            frame_idx += 1
            # The one-line \r progress bar is suppressed while --print-readings is on,
            # so it doesn't overwrite / interleave with the per-frame reading blocks.
            if not opt.print_readings and (frame_idx % 30 == 0 or frame_idx == frame_total):
                tail = f"/{frame_total}" if frame_total else ""
                print(f"\r  processed {processed}{tail} frames  "
                      f"steering={pf['steering_deg']:+6.1f}  SDA {wire_value:.2f}  "
                      f"{'ENGAGED   ' if engaged else 'DISENGAGED'}", end="")
            if opt.max_frames and frame_idx >= opt.max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[warn] interrupted -- flushing partial results")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if opt.show:
            cv2.destroyAllWindows()
        if frame_csv_file:
            frame_csv_file.close()
        if steer_link is not None:
            steer_link.close()   # centres the wheels (STA0), then closes the port
        if hasattr(backend, "close"):
            backend.close()
        imu_source.close()   # stops the HWT905 reader thread (no-op for the borrowed Arduino speed link)

    if processed == 0 and not interrupted:
        raise SystemExit(f"No decodable frames from {opt.source} "
                         f"(opened but yielded 0 usable frames -- truncated, empty, or unsupported codec)")

    flush_and_report()


if __name__ == "__main__":
    main()
