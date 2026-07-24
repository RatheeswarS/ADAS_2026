# Bench test — Jetson → Arduino steering

Send steering commands from the LKA algorithm to the real Arduino over USB
serial, on a bench, and **see every command** (and the Arduino's reply) in
your terminal.

## Two things to know first

1. **Your Arduino speaks `STR/STL/STC`, not `SDA`.** `ardunio.ino`'s
   `handleSerial()` understands `STR<deg>` (right), `STL<deg>` (left,
   positive magnitude), `STC` (center). The bench tool sends exactly that
   (`--protocol str`, the default). `--protocol sda` exists only if you later
   flash an SDA-speaking sketch.

2. **Python and the Arduino IDE Serial Monitor cannot share the port.** Only
   one process can open `/dev/ttyACM0`. So while the bench script runs, keep
   the IDE Serial Monitor **closed** — the script's terminal *is* your
   monitor (it prints `TX ->` for every command it sends and `RX <-` for
   every line the Arduino echoes back).

## Flow

```
  JETSON (Linux)                                     ARDUINO (ardunio.ino @ 115200)
  ─────────────────────────────                      ──────────────────────────────
  bench_steering_to_arduino.py                       handleSerial():
    mode: sweep|manual|csv|video    STR/STL/STC\n       STR<deg> -> degToADC -> steer RIGHT
    build_packet(deg) ──────────────USB serial────►     STL<deg> -> degToADC -> steer LEFT
    prints  TX -> ...                                    STC      -> POT_CENTER
            RX <- ...   ◄───────────echo (BENCH-TEMP)─── moveToTarget() -> stepper pulses
```

The steering source is your choice of mode; the wire protocol and the
Arduino are the same in every mode.

## Temporary changes made (revert after bench)

- **`ardunio.ino`** — added echo lines, each tagged `// BENCH-TEMP`
  (`grep BENCH-TEMP ardunio.ino`). They make the (otherwise silent) sketch
  print `RX <line>`, `OK STR/STL/STC …`, `IGN … (still auto-centering)`, and
  a one-time `READY` when boot auto-center finishes. Delete those lines to
  restore the original. Nothing else changed — control logic is untouched.
- **`bench_steering_to_arduino.py`** — the whole harness is one throwaway
  file; delete it when done. It does not modify the pipeline.

## Run it

### 1. Arduino side (once)
- Open `ardunio.ino` in the Arduino IDE, **Verify/Compile**, **Upload** to
  the board (it now has the BENCH-TEMP echoes).
- **Close the IDE Serial Monitor.**

### 2. Jetson side
```bash
# one-time: serial permission (then log out/in once)
sudo usermod -a -G dialout $USER
python3 -c "import serial" || pip install --user pyserial

cd ~/steertest

# A) SANITY / SIGN CHECK — always start here. Verifies wiring + that STR
#    really turns the wheels RIGHT. Nothing else is trustworthy until this
#    passes. (auto-detects the port; add --port /dev/ttyACM0 to force it)
python3 bench_steering_to_arduino.py --sweep

# B) type angles by hand (+ = right, - = left)
python3 bench_steering_to_arduino.py --manual

# C) replay real algorithm output (the per-second CSV from the video run),
#    1 command/sec — the lightweight way to bench the serial path (no YOLOP)
python3 bench_steering_to_arduino.py --csv output/dashcam.steering_per_second.csv

# D) full live chain: YOLOP bridge steering -> Arduino at 1 Hz
python3 bench_steering_to_arduino.py --video dashcam.mp4 --rate 1

# No Arduino attached? add --dry-run to any of the above (prints packets only)
```

### What you'll see (with the BENCH-TEMP echo flashed)
```
[info] opened /dev/ttyACM0 @ 115200
[info] waiting 3.0s for Arduino reset + auto-center ...
              RX <- READY: auto-center complete, steering commands now accepted
[19:22:06] TX -> STR5      expect wheels -> RIGHT (small)
              RX <- RX STR5
              RX <-   OK STR right 5.00 deg -> ADC 638
```
Before `READY`, any `STR/STL/STC` you send is answered with
`IGN … (still auto-centering)` — that is the boot auto-center gate, not a bug.

## Sign check (the one thing to get right)

During `--sweep`, watch the front wheels. `STR#` must turn them **right**,
`STL#` **left**. If it's **reversed**, fix it in ONE place:

- Software: in `bench_steering_to_arduino.py` → `build_packet()`, swap the
  `STR`/`STL` strings (and mirror it later in `serial_bridge.py`), **or**
- Hardware: swap the steering motor's `DIR_PIN` wiring.

Do not run closed-loop steering until `STR = right` is confirmed.

## Safety (bench)

- Put the front wheels **off the ground** or give the rack clear travel — the
  stepper *will* move on every command and on power-up (auto-center).
- Steering range is ~±24° (`degToADC` clamps to `POT_LEFT..POT_RIGHT`);
  commands beyond that saturate, they don't over-travel.
- The throttle/brake commands (`T`, `S<kmh>`, `A1/A2/A3`) share this serial
  port but the bench tool never sends them — it only sends `STR/STL/STC`.
```
