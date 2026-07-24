#!/usr/bin/env python3
"""
steer_bench.py -- on-vehicle serial round-trip proof for the LKA steering link.

Verifies the Jetson -> Arduino (arduinocode.ino) STA path end to end, SAFELY:

  1. opens the link and confirms $TEL telemetry is flowing,
  2. forces DRY_RUN=1 (firmware suppresses all physical outputs) and CONFIRMS it
     from telemetry BEFORE sending any steering,
  3. waits for boot-centering (the firmware drops STA until centred),
  4. sends a sweep of STA targets and checks each is reflected back in
     $TEL steerTargetDeg -- the actual round-trip,
  5. returns the wheels to centre and closes.

By DEFAULT it refuses to command steering unless DRY_RUN is confirmed on, so it
cannot physically move the wheels. Only --allow-live (wheels OFF the ground!)
lets it drive the steering for a real direction/range check.

USAGE
    python steer_bench.py --port /dev/ttyUSB0                 # safe DRY_RUN round-trip
    python steer_bench.py --port /dev/ttyUSB0 --allow-live    # physical sweep (on stands!)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from actuator.arduino_steering_link import ArduinoSteeringLink  # noqa: E402


def wait_until(predicate, timeout_s, poll_s=0.05):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="Arduino serial port, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--flip-steer", action="store_true", help="invert steering sign (match your run)")
    ap.add_argument("--angles", default="0,5,-5,10,-10,0",
                    help="comma-separated STA test angles (deg). Kept small on purpose.")
    ap.add_argument("--settle", type=float, default=0.6, help="seconds to wait after each STA")
    ap.add_argument("--tol", type=float, default=0.6, help="deg tolerance for the round-trip match")
    ap.add_argument("--allow-live", action="store_true",
                    help="DANGEROUS: skip the DRY_RUN gate and actually MOVE the steering. "
                         "Only with the wheels OFF the ground.")
    opt = ap.parse_args()

    angles = [float(a) for a in opt.angles.split(",") if a.strip() != ""]

    print(f"[bench] opening {opt.port} @ {opt.baud} ...")
    try:
        link = ArduinoSteeringLink(opt.port, opt.baud, flip_sign=opt.flip_steer)
    except Exception as e:
        raise SystemExit(f"[bench] FAIL: could not open port: {e}")

    results = []
    passed = False
    try:
        # 1. telemetry alive
        if not wait_until(lambda: link.link_alive, 5.0):
            raise SystemExit("[bench] FAIL: no $TEL telemetry within 5s -- wrong port/baud/cable, "
                             "or firmware not running.")
        print(f"[bench] telemetry OK. state={link.state} dry_run={link.dry_run} "
              f"centered={link.centered} faults={link.fault_flags} "
              f"speed={link.read_speed_mps()}")

        # 2. SAFETY GATE: force DRY_RUN unless explicitly allowed to move.
        if not opt.allow_live:
            print("[bench] enabling DRY_RUN (no physical motion) ...")
            link.send_command("SET:DRY_RUN=1")
            if not wait_until(lambda: link.dry_run is True, 3.0):
                raise SystemExit("[bench] ABORT: could not confirm DRY_RUN=1 from telemetry; "
                                 "refusing to send STA (would risk physical motion). "
                                 "Use --allow-live only with wheels OFF the ground.")
            print("[bench] DRY_RUN confirmed -- steering commands will NOT move hardware.")
        else:
            print("[bench] !!! --allow-live: steering WILL physically move. Wheels off the ground? !!!")
            time.sleep(2.0)

        # 3. wait for centering (STA is dropped by firmware until then)
        if not wait_until(lambda: link.centered is True, 12.0):
            print("[bench] WARN: not centered after 12s (STA may be ignored). "
                  "Check steering pot / STCAL. Continuing.")

        # 4. round-trip sweep
        print("[bench] --- STA round-trip ---")
        for deg in angles:
            link.send_steering_deg(deg)
            wire = link.last_sent_deg   # value actually on the wire (after any flip)
            time.sleep(opt.settle)
            got = link.target_deg
            ok = got is not None and abs(got - wire) <= opt.tol
            results.append(ok)
            print(f"[bench]  sent STA{wire:+.2f}  ->  $TEL steerTargetDeg={got}  "
                  f"{'PASS' if ok else 'FAIL'}")

        passed = len(results) > 0 and all(results)
        print(f"[bench] === round-trip {'PASS' if passed else 'FAIL'} "
              f"({sum(results)}/{len(results)} commands echoed) ===")
        if passed and not opt.allow_live:
            print("[bench] Next: SET:DRY_RUN=0 and re-run with --allow-live (wheels OFF ground) "
                  "to confirm the wheels turn RIGHT for a positive command.")
    finally:
        link.close()  # sends STA0 (centre), then closes
        print("[bench] wheels commanded to centre; link closed.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
