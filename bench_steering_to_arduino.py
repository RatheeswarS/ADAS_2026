#!/usr/bin/env python3
"""
bench_steering_to_arduino.py  --  BENCH-TEST HARNESS (temporary, deletable)
==========================================================================
Sends steering commands from the LKA algorithm to the real Arduino over
serial and PRINTS every command it sends (TX) plus anything the Arduino
sends back (RX), so *this Python terminal is your serial monitor*.

Why the Python terminal and not the Arduino IDE Serial Monitor: only one
process can own /dev/ttyACM0 at a time. While this script is connected, the
IDE monitor cannot be. So this script does the monitoring for you.

Protocols
  --protocol str   ardunio.ino's language:  STR<deg> / STL<deg> / STC
                   (DEFAULT -- this is the sketch you have on the board)
  --protocol sda   SDA <signed 2-decimals>  (only if you flashed an SDA sketch)

Modes (pick one; default = sweep)
  --sweep          scripted left/right sweep -- START HERE to verify wiring
                   and the steering SIGN on the bench (does STR actually turn
                   the wheels RIGHT?). Nothing else is trustworthy until this
                   is confirmed.
  --manual         type a steering angle (deg, + = right / - = left), Enter
                   sends it; 'q' quits.
  --csv PATH       replay a per-second steering CSV produced by
                   run_video_annotated.py (real algorithm output, no YOLOP
                   needed here -- the lightweight way to bench the serial path)
  --video PATH     live full chain: run the YOLOP bridge and stream its
                   steering to the Arduino, throttled to --rate

No Arduino plugged in? add --dry-run : opens no port, just prints the packets
(works on any machine, exactly like the current log-only bridge).

USAGE
  python bench_steering_to_arduino.py --sweep
  python bench_steering_to_arduino.py --sweep --dry-run
  python bench_steering_to_arduino.py --csv output/dashcam.steering_per_second.csv
  python bench_steering_to_arduino.py --manual --port /dev/ttyACM0
"""
import argparse
import csv
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BAUD = 115200   # matches ardunio.ino's Serial.begin(115200)


# --------------------------------------------------------------------------
def find_port():
    for pat in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def build_packet(deg, protocol):
    """Signed steering angle (deg, + = right / - = left) -> wire string.

    str  : ardunio.ino protocol. Integer magnitude only; it rejects deg<=0 for
           STR/STL, so 0 becomes STC (center). Positive -> STR (right),
           negative -> STL (left). Mirrors serial_bridge.py's build_packet()
           and the Arduino's own degToADC().
    sda  : SDA <signed float, 2 decimals>, no forced '+' (matches
           SdaSerialBackend / the real wire).
    """
    if protocol == "sda":
        return f"SDA {deg:.2f}\n"
    d = int(round(deg))
    if d == 0:
        return "STC\n"
    if d > 0:
        return f"STR{d}\n"
    return f"STL{-d}\n"


class ArduinoLink:
    """Thin serial wrapper that logs TX and drains/prints RX. In --dry-run it
    opens no port and only prints (so it runs with no hardware)."""

    def __init__(self, port, baud, dry_run):
        self.dry_run = dry_run or port is None
        self.port = port
        self.ser = None
        if not self.dry_run:
            import serial  # pyserial; imported lazily so --dry-run needs nothing
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"[info] opened {port} @ {baud}")
        else:
            print("[info] DRY-RUN: no serial port opened; packets are only printed")

    def drain(self):
        if not self.ser:
            return
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode("ascii", errors="ignore").strip()
                if line:
                    print(f"              RX <- {line}")
        except Exception as e:
            print(f"[warn] RX read error: {e}")

    def send(self, packet, note=""):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] TX -> {packet.strip():<9} {note}")
        if self.ser:
            try:
                self.ser.write(packet.encode("ascii"))
            except Exception as e:
                print(f"[warn] TX write error: {e}")
        self.drain()

    def sleep(self, seconds):
        """Sleep while still draining RX, so late Arduino replies still show."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.drain()
            time.sleep(0.05)

    def warmup(self, protocol, secs=3.0):
        """Opening the port resets most Arduinos -> setup() re-runs and auto-
        centers the steering. Wait for that, then nudge to center once."""
        if self.ser:
            print(f"[info] waiting {secs:.1f}s for Arduino reset + auto-center "
                  f"(STR/STL/STC are ignored until centering completes) ...")
            self.sleep(secs)
        self.send(build_packet(0, protocol), note="(center / warm-up)")

    def close(self):
        if self.ser:
            self.ser.close()


# --------------------------------------------------------------------------
def mode_sweep(link, protocol):
    print("\n=== SWEEP: verify wiring + steering SIGN ===")
    print("Watch the front wheels. Each step below says the direction it SHOULD go.")
    print("If STR turns LEFT (or STL turns RIGHT), the sign is inverted -- fix it in")
    print("build_packet() (swap STR/STL) or swap the motor DIR wiring. Do NOT trust")
    print("closed-loop steering until this matches.\n")
    steps = [(0, "CENTER", 1.5),
             (5, "RIGHT (small)", 2.0), (10, "RIGHT (more)", 2.0), (15, "RIGHT (max-ish)", 2.0),
             (0, "CENTER", 2.0),
             (-5, "LEFT (small)", 2.0), (-10, "LEFT (more)", 2.0), (-15, "LEFT (max-ish)", 2.0),
             (0, "CENTER", 2.0)]
    for deg, expect, hold in steps:
        link.send(build_packet(deg, protocol), note=f"expect wheels -> {expect}")
        link.sleep(hold)
    print("\n[done] sweep complete. Confirm each expected direction matched reality.")


def mode_manual(link, protocol):
    print("\n=== MANUAL: type a steering angle in degrees (+ = right, - = left). 'q' to quit. ===")
    while True:
        try:
            raw = input("steer deg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw.lower() in ("q", "quit", "exit"):
            break
        if not raw:
            continue
        try:
            deg = float(raw)
        except ValueError:
            print("  (enter a number like 12, -8, 0)")
            continue
        link.send(build_packet(deg, protocol), note="(manual)")
        link.sleep(0.3)


def mode_csv(link, protocol, path, column, rate):
    if not os.path.isfile(path):
        raise SystemExit(f"CSV not found: {path}")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"CSV has no rows: {path}")
    if column not in rows[0]:
        raise SystemExit(f"column '{column}' not in CSV. Columns: {', '.join(rows[0].keys())}")
    period = 1.0 / rate if rate > 0 else 0.0
    print(f"\n=== CSV REPLAY: {os.path.basename(path)}  column='{column}'  "
          f"{len(rows)} rows @ {rate} Hz ===")
    for r in rows:
        deg = float(r[column])
        tcol = r.get("elapsed_s", "?")
        link.send(build_packet(deg, protocol), note=f"t={tcol}s  algo={deg:+.2f}deg")
        link.sleep(period)
    print("\n[done] CSV replay complete.")


def mode_video(link, protocol, opt):
    """Live full chain: YOLOP bridge steering -> Arduino, throttled to --rate.
    Kept minimal (mirrors run_video_to_arduino.py's control path). Uses the
    real monotonic clock because we are driving hardware in real time."""
    sys.path.insert(0, os.path.join(HERE, "src"))
    sys.path.insert(0, os.path.join(HERE, "YOLOP"))
    import cv2
    from perception.lane_detection_node_birdeye import BirdEyeLaneDetectionNode
    from perception.lane_core.model_runtime import select_device
    from reference.reference_generator import ReferenceGenerator
    from actuator.dbw_interface import ActuatorInterface, CommandMode
    from actuator.dbw_backend import NullDbwBackend
    from safety.safety_monitor import SafetyMonitor
    from ego_state.imu_source import Hwt905ImuSource, CompositeImuSource

    if not os.path.isfile(opt.video):
        raise SystemExit(f"video not found: {opt.video}")
    weights = os.path.join(HERE, "YOLOP", "weights", "End-to-end.pth")
    try:
        device = select_device(None, opt.device)
    except Exception as e:
        print(f"[warn] device '{opt.device}' unavailable ({e}); using cpu")
        device = select_device(None, "cpu")
    # Yaw rate from the HWT905 IMU feeds the steering damping term when its
    # own serial link is given. (Speed here stays the --speed fallback: this
    # bench harness drives the Arduino through ArduinoLink, not the
    # SdaSerialBackend that reads the motor-controller SPD telemetry, so the
    # live-speed path is only wired in run_video_to_arduino.py.)
    imu_source = None
    if getattr(opt, "imu_port", None):
        imu_source = CompositeImuSource(
            yaw_source=Hwt905ImuSource(opt.imu_port, opt.imu_baud, yaw_sign=opt.imu_yaw_sign))
        print(f"[info] HWT905 IMU on {opt.imu_port} @ {opt.imu_baud} -- live yaw rate feeds the steering law")
    print(f"[info] loading YOLOP on {device} ...")
    node = BirdEyeLaneDetectionNode(weights, device, img_size=opt.img_size,
                                    lane_width_m=opt.lane_width, speed_mps=opt.speed,
                                    imu_source=imu_source)
    refgen = ReferenceGenerator()
    actuator = ActuatorInterface(backend=NullDbwBackend(), mode=CommandMode.ANGLE)
    safety = SafetyMonitor()

    cap = cv2.VideoCapture(opt.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {opt.video}")
    period = 1.0 / opt.rate if opt.rate > 0 else 0.0
    print(f"\n=== LIVE VIDEO: {os.path.basename(opt.video)} -> Arduino @ {opt.rate} Hz ===")
    t0 = time.monotonic()
    last_send = -1e9
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now_s = time.monotonic() - t0
            pf = node.process(frame, now_s)
            ref = refgen.update(pf)
            st = safety.evaluate(ref, pf["steering_deg"], pf["speed_mps"], actuator, now_s,
                                 measured_yaw_rate_dps=pf["yaw_rate_dps"])
            safety_engaged = st.engaged
            act = actuator.send(pf["steering_deg"], now_s)
            wire = act["commanded_value"] if safety_engaged else 0.0
            if now_s - last_send >= period:
                last_send = now_s
                tag = "ENG" if safety_engaged else "DIS"
                link.send(build_packet(wire, protocol),
                          note=f"t={now_s:5.1f}s conf={pf['confidence']:.2f} {tag}"
                               + (f" {','.join(st.faults)}" if st.faults else ""))
            link.drain()
    finally:
        cap.release()
        if imu_source is not None:
            imu_source.close()
    print("\n[done] live video complete.")


# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--sweep", action="store_true", help="scripted L/R sign-check sweep (default)")
    m.add_argument("--manual", action="store_true", help="type angles interactively")
    m.add_argument("--csv", metavar="PATH", help="replay a per-second steering CSV")
    m.add_argument("--video", metavar="PATH", help="live: YOLOP bridge -> Arduino")
    ap.add_argument("--protocol", choices=["str", "sda"], default="str",
                    help="wire protocol (default: str = ardunio.ino's STR/STL/STC)")
    ap.add_argument("--port", default=None, help="serial device (default: auto-detect ttyACM*/ttyUSB*)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--dry-run", action="store_true", help="open no port; just print packets")
    ap.add_argument("--rate", type=float, default=1.0, help="send rate Hz for --csv/--video (default 1.0)")
    ap.add_argument("--column", default="sda_steering_deg", help="CSV column to send (default: sda_steering_deg)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the reset/auto-center wait")
    # --video knobs
    ap.add_argument("--device", default="0", help="[--video] '0'=CUDA or 'cpu'")
    ap.add_argument("--img-size", type=int, default=640, help="[--video] YOLOP resolution")
    ap.add_argument("--lane-width", type=float, default=3.0, help="[--video] lane width m")
    ap.add_argument("--speed", type=float, default=5.0, help="[--video] fallback speed m/s (no live motor-controller feed here)")
    ap.add_argument("--imu-port", default=None, help="[--video] HWT905 IMU serial port; live yaw rate feeds the steering law")
    ap.add_argument("--imu-baud", type=int, default=9600, help="[--video] HWT905 baud (WitMotion default 9600)")
    ap.add_argument("--imu-yaw-sign", type=float, default=1.0, help="[--video] +1/-1 HWT905 yaw-rate sign (bench-verify)")
    return ap.parse_args()


def main():
    opt = parse_args()
    port = None if opt.dry_run else (opt.port or find_port())
    if not opt.dry_run and port is None:
        raise SystemExit("No serial device found (ttyACM*/ttyUSB*). Plug in the Arduino, "
                         "pass --port, or use --dry-run.")

    link = ArduinoLink(port, opt.baud, opt.dry_run)
    try:
        if not opt.no_warmup:
            link.warmup(opt.protocol)
        if opt.csv:
            mode_csv(link, opt.protocol, opt.csv, opt.column, opt.rate)
        elif opt.video:
            mode_video(link, opt.protocol, opt)
        elif opt.manual:
            mode_manual(link, opt.protocol)
        else:  # default
            mode_sweep(link, opt.protocol)
        # leave the rack centered
        link.send(build_packet(0, opt.protocol), note="(center / shutdown)")
        link.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[warn] interrupted -- centering and closing")
        try:
            link.send(build_packet(0, opt.protocol), note="(center / interrupt)")
        except Exception:
            pass
    finally:
        link.close()


if __name__ == "__main__":
    main()
