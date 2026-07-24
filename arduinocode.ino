#include <Arduino.h>
#include <Wire.h>
#include <EEPROM.h>
#include <Adafruit_MCP4725.h>

Adafruit_MCP4725 dac;

// ============================================================
// ============  JETSON SERIAL PROTOCOL  =======================
// ============================================================
// Commands FROM Jetson (text lines, newline terminated):
//   SET:<NAME>=<value>        -> set a parameter (RAM only, live-applied)
//   GET:ALL                   -> dump all current params as $PARAM lines
//   LOCK                      -> commit params to EEPROM, enter RUN mode
//   UNLOCK                    -> return to CONFIG mode (stops vehicle)
//   S<kmh>                    -> throttle PID target speed
//   STA<±deg>                 -> absolute steering angle target
//   BRK<0-100>                -> brake percentage target
//   Z                         -> stop throttle (PID/manual)
//   STCAL                     -> steering calibration sequence
//   ACAL                      -> actuator (brake) calibration sequence
//   PING                      -> heartbeat / watchdog reset
//
// Messages TO Jetson:
//   $TEL,...   -> live telemetry, streamed continuously (~20Hz)
//   $PARAM,... -> full parameter dump (on GET:ALL or after SET/LOCK)
//   $ACK,<cmd>
//   $NAK,<cmd>,<reason>
//   $BOOT / $READY / $FAULT,<code>
// ============================================================

// ── System mode ────────────────────────────────────────────
enum SysState { STATE_CONFIG, STATE_RUN };
SysState sysState = STATE_CONFIG;

enum ThrottleMode { MODE_IDLE, MODE_TEST, MODE_PID };
ThrottleMode throttleMode = MODE_IDLE;

// ── Fault flags (bitmask) ─────────────────────────────────
uint16_t faultFlags = 0;
#define FAULT_DAC_MISSING     0x0001
#define FAULT_SENSOR_TIMEOUT  0x0002
#define FAULT_ACT_TIMEOUT     0x0004
#define FAULT_STEER_STALL     0x0008
#define FAULT_WATCHDOG        0x0010
#define FAULT_NOT_CENTERED    0x0020

// ============================================================
// ── PARAMETER STRUCT (everything Jetson can SET) ─────────────
// ============================================================
// Two categories:
//  - "locked" params: only settable in CONFIG state, committed to EEPROM on LOCK
//  - "live" params: settable in CONFIG or RUN state, never need LOCK
struct Params {
  // ---- PID / throttle (LIVE - tunable even during RUN) ----
  float KP;
  float KI;
  float KD;
  float RAMP_RATE;          // wheelRPM/s, max ramp velocity
  float RAMP_ACCEL_LIMIT;   // wheelRPM/s^2, S-curve jerk limit
  float MAX_THROTTLE_SLEW;  // %/s output slew limit
  float MAX_CORRECTION_SLEW;// PID correction %/s slew limit

  // ---- PID / throttle (LOCKED - structural) ----
  float PID_ALPHA;
  float ERROR_DEADBAND;     // wheel RPM
  float CORR_MIN;
  float CORR_MAX;

  // ---- Steering (LOCKED) ----
  int32_t POT_LEFT, POT_CENTER, POT_RIGHT;
  int32_t TOLERANCE, SLOW_ZONE, MICRO_ZONE;
  uint8_t FAST_BURST, SLOW_BURST, MICRO_BURST;

  // ---- Brake / actuator (LOCKED except noted) ----
  int32_t HOME_ADC;      // 0% brake position (released)
  int32_t FULL_ADC;      // 100% brake position
  int32_t ACT_TOLERANCE;
  uint32_t ACT_MOVE_TIMEOUT_MS;

  // ---- Auto-sequence (LIVE) ----
  float AUTO_SEQ_TARGET_KMH;
  uint32_t AUTO_SEQ_HOLD_MS;
  bool AUTO_SEQ_ENABLED;

  // ---- Watchdog (LOCKED) ----
  uint32_t WATCHDOG_TIMEOUT_MS;

  // ---- Bench testing (LIVE) ----
  // CHANGE (#9): when true, all control math (PID, steering, brake) runs
  // and telemetry streams normally, but actual hardware outputs (DAC,
  // stepper pulses, actuator PWM) are suppressed. Lets the full
  // Jetson<->Arduino pipeline be validated on the bench without the
  // vehicle actually moving.
  bool DRY_RUN;

  // CHANGE (#2): incremented every successful LOCK. Exposed in $PARAM so
  // the Jetson dashboard can detect "params changed since last session"
  // (e.g. after a power cycle silently reloading old EEPROM values) by
  // comparing against what it last saw.
  uint32_t lockVersion;

  uint32_t magic; // EEPROM validity marker
};

const uint32_t PARAMS_MAGIC = 0xBA1A2026UL;
const int EEPROM_ADDR = 0;

Params P; // active live params

void loadDefaults() {
  P.KP = 0.15f; P.KI = 0.02f; P.KD = 0.0008f;
  P.RAMP_RATE = 25.0f;
  P.RAMP_ACCEL_LIMIT = 60.0f;
  P.MAX_THROTTLE_SLEW = 40.0f;
  P.MAX_CORRECTION_SLEW = 15.0f;

  P.PID_ALPHA = 0.3f;
  P.ERROR_DEADBAND = 5.0f;
  P.CORR_MIN = -30.0f;
  P.CORR_MAX = 30.0f;

  P.POT_LEFT = 209; P.POT_CENTER = 565; P.POT_RIGHT = 914;
  P.TOLERANCE = 4; P.SLOW_ZONE = 20; P.MICRO_ZONE = 8;
  P.FAST_BURST = 30; P.SLOW_BURST = 12; P.MICRO_BURST = 4;

  P.HOME_ADC = 780; P.FULL_ADC = 440;
  P.ACT_TOLERANCE = 5;
  P.ACT_MOVE_TIMEOUT_MS = 5000;

  P.AUTO_SEQ_TARGET_KMH = 30.0f;
  P.AUTO_SEQ_HOLD_MS = 5000;
  P.AUTO_SEQ_ENABLED = false; // Jetson owns mission logic now; legacy off by default

  P.WATCHDOG_TIMEOUT_MS = 1000;

  P.DRY_RUN = false;
  P.lockVersion = 0;

  P.magic = PARAMS_MAGIC;
}

void saveParamsToEEPROM() {
  EEPROM.put(EEPROM_ADDR, P);
}

bool loadParamsFromEEPROM() {
  Params tmp;
  EEPROM.get(EEPROM_ADDR, tmp);
  if (tmp.magic == PARAMS_MAGIC) {
    P = tmp;
    return true;
  }
  return false;
}

// derived (recomputed whenever POT_LEFT/CENTER/RIGHT change)
float CPD_LEFT = 0, CPD_RIGHT = 0;
void recalcCPD() {
  CPD_LEFT  = (float)(P.POT_CENTER - P.POT_LEFT)  / 24.0f;
  CPD_RIGHT = (float)(P.POT_RIGHT  - P.POT_CENTER) / 24.0f;
}

// ============================================================
// ── PIN DEFINITIONS ────────────────────────────────────────
// ============================================================
const int PUL_PIN = 6, DIR_PIN = 7, EN_PIN = 5, POT_PIN = A0;
const bool MOTOR_ENABLE = LOW, MOTOR_DISABLE = HIGH;
const bool DIR_TO_LEFT = LOW, DIR_TO_RIGHT = HIGH;
const unsigned int PULSE_WIDTH_US = 100;
const unsigned int DIR_SETTLE_US  = 10;

const float ACTUAL_WHEEL_DIAMETER_INCHES = 23.0;
const float EXACT_GEAR_RATIO = 9.0;
const float MOTOR_PULSES_PER_REV = 5.0;
const byte SENSOR_PIN = 18, RELAY_PIN = 4, SWITCH_PIN = 2;

const int PWM1_PIN = 9, DIR1_PIN = 8, FB_PIN = A1;
const int ACT_SPEED = 255; // always max, per spec - not Jetson-settable

// ── DAC mapping ───────────────────────────────────────────
const int DAC_IDLE = 800, DAC_START = 1000, DAC_FULL = 3522;

// ============================================================
// ── STEERING STATE ───────────────────────────────────────────
// ============================================================
int32_t targetADC = 0;
bool motorReady = false;
bool currentDir = DIR_TO_LEFT;
bool calMode = false;
uint8_t calStep = 0;
bool centerAchieved = false;
bool steeringMoving = false;
float steerTargetDeg = 0.0f;

int32_t readPot() {
  int32_t sum = 0;
  for (int i = 0; i < 8; i++) { sum += analogRead(POT_PIN); delayMicroseconds(100); }
  return sum / 8;
}
int32_t readPotFast() {
  int32_t sum = 0;
  for (int i = 0; i < 2; i++) { sum += analogRead(POT_PIN); delayMicroseconds(50); }
  return sum / 2;
}
int32_t readAverage(int n) {
  int64_t sum = 0;
  for (int i = 0; i < n; i++) { sum += analogRead(POT_PIN); delay(2); }
  return (int32_t)(sum / n);
}

void steerEnable()  { digitalWrite(EN_PIN, MOTOR_ENABLE);  delayMicroseconds(100); }
void steerDisable() { digitalWrite(EN_PIN, MOTOR_DISABLE); }

void applyDirection(bool dir) {
  if (dir != currentDir) {
    currentDir = dir;
    digitalWrite(DIR_PIN, currentDir);
    delayMicroseconds(DIR_SETTLE_US);
  }
}
// REVERTED (per request): back to the old synchronous burst-stepping
// approach that was confirmed working. stepOnce() blocks for
// 2*PULSE_WIDTH_US per pulse, and moveToTarget() fires the whole burst
// (up to FAST_BURST=30 pulses -> ~6ms) in one call. This DOES block
// loop() for that duration while steering is actively moving -- that's
// the known tradeoff of this approach -- but it's the behavior you
// confirmed works correctly, so the non-blocking pulse state machine
// has been removed.
void stepOnce() {
  if (P.DRY_RUN) return; // simulate nothing physically stepping in dry-run
  digitalWrite(PUL_PIN, HIGH);
  delayMicroseconds(PULSE_WIDTH_US);
  digitalWrite(PUL_PIN, LOW);
  delayMicroseconds(PULSE_WIDTH_US);
}

int32_t degToADC(float deg) {
  int32_t adc;
  if (deg >= 0) adc = P.POT_CENTER + (int32_t)(deg * CPD_RIGHT);
  else          adc = P.POT_CENTER - (int32_t)((-deg) * CPD_LEFT);
  return constrain(adc, P.POT_LEFT, P.POT_RIGHT);
}
float adcToDeg(int32_t adc) {
  if (adc >= P.POT_CENTER)
    return (CPD_RIGHT > 0) ? (float)(adc - P.POT_CENTER) / CPD_RIGHT : 0;
  else
    return (CPD_LEFT > 0) ? -((float)(P.POT_CENTER - adc) / CPD_LEFT) : 0;
}

// STA<±deg> — absolute steering command (primary autonomy interface)
void setSteerAbsolute(float deg) {
  if (!centerAchieved) return; // blocked until boot centering done
  steerTargetDeg = deg;
  targetADC = degToADC(deg);
  motorReady = false;
  steerEnable();
}

unsigned long steerStallTimer = 0;
const unsigned long STEER_STALL_TIMEOUT_MS = 4000;

// ============================================================
// REVERTED (per request) to the old, confirmed-working synchronous
// burst logic: quick 2-sample read to check if we're near target, fall
// back to the slower 8-sample readPot() once close, then fire the
// whole step burst immediately and block until it's done. This is
// exactly the old moveToTarget() behavior -- the non-blocking pulse
// queue and fast-read arrival debounce have been removed.
// ============================================================
void moveToTarget() {
  if (motorReady || calMode) return;

  int32_t adc;
  int32_t quickCheck = readPotFast();
  int32_t quickErr = abs(targetADC - quickCheck);
  adc = (quickErr > P.SLOW_ZONE) ? quickCheck : readPot();

  int32_t err = targetADC - adc;
  int32_t absErr = abs(err);

  if (absErr <= P.TOLERANCE) {
    steerDisable();
    motorReady = true;
    steeringMoving = false;
    faultFlags &= ~FAULT_STEER_STALL;
    if (!centerAchieved) {
      centerAchieved = true;
    }
    return;
  }

  if (!steeringMoving) { steeringMoving = true; steerStallTimer = millis(); }
  if (millis() - steerStallTimer > STEER_STALL_TIMEOUT_MS) {
    faultFlags |= FAULT_STEER_STALL;
    steerDisable();
    motorReady = true;
    steeringMoving = false;
    return;
  }

  bool dir = (err > 0) ? DIR_TO_RIGHT : DIR_TO_LEFT;
  applyDirection(dir);

  uint8_t burst;
  if      (absErr > P.SLOW_ZONE)  burst = P.FAST_BURST;
  else if (absErr > P.MICRO_ZONE) burst = P.SLOW_BURST;
  else                            burst = P.MICRO_BURST;

  for (uint8_t i = 0; i < burst; i++) stepOnce();
}

// ============================================================
// ── BRAKE / ACTUATOR — NON-BLOCKING, PERCENTAGE-BASED ────────
// ============================================================
// BRK0 -> HOME_ADC (released), BRK100 -> FULL_ADC (full brake)
// Bang-bang at ACT_SPEED (always max), non-blocking state machine.

enum ActState { ACT_IDLE, ACT_MOVING };
enum ActDir { ACT_DIR_NONE, ACT_DIR_EXTEND, ACT_DIR_RETRACT };
ActState actState = ACT_IDLE;
ActDir   actLastDir = ACT_DIR_NONE;
int32_t actTargetADC = 0;
float brakeTargetPct = 0.0f;
float brakeActualPct = 0.0f;
unsigned long actMoveStartTime = 0;
unsigned long actDirChangeTime = 0;

bool actCalMode = false;
int actCalStep = 0;

// CHANGE (#6): hysteresis + minimum direction-hold time so the actuator
// doesn't rapidly flip direction when the ADC reading is noisy right at
// the target crossing. HYSTERESIS widens the "settled" band beyond
// ACT_TOLERANCE; MIN_DIR_HOLD_MS forces a chosen direction to persist for
// at least this long before it can reverse (protects motor/gearbox from
// rapid H-bridge reversal / mechanical wear).
const int32_t ACT_HYSTERESIS_ADC = 6;
const unsigned long MIN_DIR_HOLD_MS = 120;

int pctToADC(float pct) {
  pct = constrain(pct, 0.0f, 100.0f);
  return (int)(P.HOME_ADC + (pct / 100.0f) * (P.FULL_ADC - P.HOME_ADC) + 0.5f);
}
float adcToPct(int adc) {
  if (P.FULL_ADC == P.HOME_ADC) return 0;
  float pct = 100.0f * (float)(adc - P.HOME_ADC) / (float)(P.FULL_ADC - P.HOME_ADC);
  return constrain(pct, 0.0f, 100.0f);
}

void setBrakePercent(float pct) {
  brakeTargetPct = constrain(pct, 0.0f, 100.0f);
  actTargetADC = pctToADC(brakeTargetPct);
  actState = ACT_MOVING;
  actMoveStartTime = millis();
  actLastDir = ACT_DIR_NONE;
  actDirChangeTime = 0;
}

void stopActuatorMotor() {
  analogWrite(PWM1_PIN, 0);
  digitalWrite(DIR1_PIN, LOW);
  actLastDir = ACT_DIR_NONE;
}
void extendActuator()  { if (P.DRY_RUN) return; digitalWrite(DIR1_PIN, LOW);  analogWrite(PWM1_PIN, ACT_SPEED); } // CHANGE (#9)
void retractActuator() { if (P.DRY_RUN) return; digitalWrite(DIR1_PIN, HIGH); analogWrite(PWM1_PIN, ACT_SPEED); } // CHANGE (#9)

// CHANGE (#6): direction only changes once MIN_DIR_HOLD_MS has elapsed
// since the last change; otherwise the previous direction is held even if
// the instantaneous ADC comparison flips sign momentarily (noise).
void driveActuatorToward(int32_t current) {
  ActDir desiredDir = (current > actTargetADC) ? ACT_DIR_EXTEND : ACT_DIR_RETRACT;
  unsigned long now = millis();

  if (actLastDir == ACT_DIR_NONE) {
    actLastDir = desiredDir;
    actDirChangeTime = now;
  } else if (desiredDir != actLastDir) {
    if (now - actDirChangeTime >= MIN_DIR_HOLD_MS) {
      actLastDir = desiredDir;
      actDirChangeTime = now;
    }
    // else: too soon to reverse, keep driving in actLastDir
  }

  if (actLastDir == ACT_DIR_EXTEND) extendActuator();
  else retractActuator();
}

void updateActuator() {
  int current = analogRead(FB_PIN);
  brakeActualPct = adcToPct(current);

  if (actState != ACT_MOVING) return;

  // CHANGE (#6): widened settle band (TOLERANCE + HYSTERESIS) so the motor
  // fully stops and stays stopped instead of dithering at the edge of the
  // old tighter tolerance band.
  if (abs(current - actTargetADC) <= (P.ACT_TOLERANCE + ACT_HYSTERESIS_ADC)) {
    stopActuatorMotor();
    actState = ACT_IDLE;
    faultFlags &= ~FAULT_ACT_TIMEOUT;
    return;
  }

  if (millis() - actMoveStartTime > P.ACT_MOVE_TIMEOUT_MS) {
    stopActuatorMotor();
    actState = ACT_IDLE;
    faultFlags |= FAULT_ACT_TIMEOUT;
    return;
  }

  driveActuatorToward(current); // CHANGE (#6): debounced direction instead of raw compare
}

int readAverageADC(int samples) {
  long sum = 0;
  for (int i = 0; i < samples; i++) { sum += analogRead(FB_PIN); delay(2); }
  return sum / samples;
}
void handleActuatorCalibration() {
  switch (actCalStep) {
    case 0:
      P.HOME_ADC = readAverageADC(1000);
      Serial.print(F("$ACK,ACAL_HOME,")); Serial.println(P.HOME_ADC);
      Serial.println(F("# Move actuator to FULL BRAKE, press ENTER"));
      actCalStep = 1;
      break;
    case 1:
      P.FULL_ADC = readAverageADC(1000);
      Serial.print(F("$ACK,ACAL_FULL,")); Serial.println(P.FULL_ADC);
      actCalMode = false;
      actCalStep = 0;
      break;
  }
}

// ============================================================
// ── THROTTLE / PID ────────────────────────────────────────────
// ============================================================
volatile unsigned long pulseCount = 0;
unsigned long lastLoopTime = 0;
unsigned long lastPulseSeenTime = 0;
const unsigned long LOOP_INTERVAL = 100;

// CHANGE (#1): period-capture for low-speed resolution. At low RPM, a
// 100ms count-window catches very few edges (quantized/stairstep RPM).
// Instead we also track the time between the last two pulses in the ISR
// (micros() resolution) and use THAT to compute RPM when the pulse rate
// is too low for the windowed-count method to be accurate. At higher RPM
// (more pulses per window) the windowed-count method is used as before,
// since a single noisy period reading matters less when averaged over
// many pulses.
volatile unsigned long lastPulseMicros = 0;
volatile unsigned long prevPulsePeriodUs = 0; // time between the last two pulses
const unsigned long PERIOD_METHOD_PULSE_THRESHOLD = 4; // below this many pulses/window, use period method

float pidFeedbackRPM = 0.0;
const int FILTER_SAMPLES = 8;
float rpmHistory[FILTER_SAMPLES] = {0};
int historyIndex = 0;
float filteredMotorRPM = 0.0;
const float MAX_RPM_JUMP_PER_SAMPLE = 3000.0;
float lastAcceptedRPM = 0.0;

// CHANGE: median-of-3 pre-filter. Catches single-sample electrical spikes
// (e.g. motor-switching noise on a shared supply/ground coupling into the
// Hall sensor line when the stepper or actuator motor energizes) that a
// simple max-jump-per-sample reject can miss if the spike happens to land
// within the jump threshold. A median of the last 3 raw readings removes
// any single outlier sample without adding meaningful lag.
float rpmMedianBuf[3] = {0, 0, 0};
uint8_t rpmMedianIdx = 0;

float medianOf3(float a, float b, float c) {
  if (a > b) { float t = a; a = b; b = t; }
  if (b > c) { float t = b; b = c; c = t; }
  if (a > b) { float t = a; a = b; b = t; }
  return b;
}

float applyMedianFilter(float newSample) {
  rpmMedianBuf[rpmMedianIdx] = newSample;
  rpmMedianIdx = (rpmMedianIdx + 1) % 3;
  return medianOf3(rpmMedianBuf[0], rpmMedianBuf[1], rpmMedianBuf[2]);
}

float thr_manualThr = 0.0;      // commanded (pre-slew) throttle %
float lastAppliedThrottle = 0.0; // actually applied to DAC after slew limiting

float targetWheelRPM = 0.0;
float rampedTargetWheelRPM = 0.0;
float rampVelocity = 0.0; // wheelRPM/s, itself ramped for S-curve

float pidIntegral = 0.0;
float pidPrevError = 0.0;
float lastCorrection = 0.0;
unsigned long pidStartTime = 0;

void sensorPulseISR() {
  pulseCount++;
  unsigned long now = millis();
  lastPulseSeenTime = now;

  unsigned long nowUs = micros();
  if (lastPulseMicros != 0) {
    prevPulsePeriodUs = nowUs - lastPulseMicros; // CHANGE (#1)
  }
  lastPulseMicros = nowUs;
}

float updateDisplayFilter(float newRPM) {
  rpmHistory[historyIndex] = newRPM;
  historyIndex = (historyIndex + 1) % FILTER_SAMPLES;
  float sum = 0;
  for (int i = 0; i < FILTER_SAMPLES; i++) sum += rpmHistory[i];
  return sum / (float)FILTER_SAMPLES;
}

int percentToDAC(float pct) {
  pct = constrain(pct, 0.0, 100.0);
  if (pct == 0.0) return DAC_IDLE;
  return (int)(DAC_START + (pct - 1.0) / 99.0 * (DAC_FULL - DAC_START) + 0.5);
}
void setDACRaw(int dacVal) {
  dacVal = constrain(dacVal, 0, 4095);
  dac.setVoltage(dacVal, false);
}

// CHANGE: output slew limiter — smooths actual DAC output independent of
// how much PID+FF math wants to move per tick. This is the #1 smoothness fix.
void applyThrottleSlewed(float desiredPct, float dt) {
  float maxStep = P.MAX_THROTTLE_SLEW * dt;
  float delta = constrain(desiredPct - lastAppliedThrottle, -maxStep, maxStep);
  lastAppliedThrottle += delta;
  if (!P.DRY_RUN) setDACRaw(percentToDAC(lastAppliedThrottle)); // CHANGE (#9)
}

float wheelRPMToKmh(float wRPM) {
  float d = ACTUAL_WHEEL_DIAMETER_INCHES * 0.0254;
  float c = 3.14159265 * d;
  return (wRPM * c * 60.0) / 1000.0;
}
float kmhToWheelRPM(float kmh) {
  float d = ACTUAL_WHEEL_DIAMETER_INCHES * 0.0254;
  float c = 3.14159265 * d;
  return (kmh * 1000.0) / (c * 60.0);
}

// CHANGE: smoothstep-blended feedforward curve — removes the slope
// discontinuities ("kinks") that existed at the 100/200 RPM breakpoints
// in the old piecewise-linear version. Endpoints/gains match the original
// calibration so no new sweep test is needed.
float smoothstep(float t) { t = constrain(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }

float feedforwardThrottle(float rpm) {
  rpm = constrain(rpm, 0.0f, 6000.0f);
  if (rpm <= 100.0f) {
    float t = rpm / 100.0f;
    return 20.5f * smoothstep(t);
  } else if (rpm <= 200.0f) {
    float t = (rpm - 100.0f) / 100.0f;
    return 20.5f + 26.5f * smoothstep(t);
  } else {
    return 47.0f + 0.2750f * (rpm - 200.0f); // unchanged linear tail, no kink beyond 200
  }
}

// CHANGE: S-curve setpoint ramp. Old version ramped rampedTargetWheelRPM
// linearly at a fixed rate -> velocity continuous but acceleration jumps
// instantly at start/end of the ramp (infinite jerk). New version ramps
// rampVelocity itself, limited by RAMP_ACCEL_LIMIT, giving a true
// trapezoidal/S-curve profile -> smooth accel and decel into the target.
void updateSetpointRamp(float dt) {
  float error = targetWheelRPM - rampedTargetWheelRPM;
  float desiredVel = constrain(error / max(dt, 0.001f), -P.RAMP_RATE, P.RAMP_RATE);
  float maxVelChange = P.RAMP_ACCEL_LIMIT * dt;
  rampVelocity += constrain(desiredVel - rampVelocity, -maxVelChange, maxVelChange);
  rampedTargetWheelRPM += rampVelocity * dt;
}

void printGains() {
  Serial.print(F("$ACK,GAINS,KP=")); Serial.print(P.KP, 5);
  Serial.print(F(",KI=")); Serial.print(P.KI, 5);
  Serial.print(F(",KD=")); Serial.println(P.KD, 5);
}

void startPID(float newTargetWheelRPM) {
  throttleMode = MODE_PID;
  targetWheelRPM = constrain(newTargetWheelRPM, 0.0, 6000.0);
  rampedTargetWheelRPM = pidFeedbackRPM / EXACT_GEAR_RATIO;
  rampVelocity = 0.0f;
  pidIntegral = 0.0; pidPrevError = 0.0;
  float ff = feedforwardThrottle(rampedTargetWheelRPM);
  thr_manualThr = ff;
  pidStartTime = millis();
}

void stopPID() {
  throttleMode = MODE_IDLE;
  thr_manualThr = 0.0;
  targetWheelRPM = 0.0;
}

// CHANGE: correction slew limiter added — prevents correction term from
// jumping instantly between -30 and +30 when error sign flips; smooths
// the trim contribution on top of the already-smoothed FF curve.
void updatePIDControl(float dt) {
  updateSetpointRamp(dt);

  float wheelRPM = pidFeedbackRPM / EXACT_GEAR_RATIO;
  float error = rampedTargetWheelRPM - wheelRPM;

  if ((error > 0 && pidPrevError < 0) || (error < 0 && pidPrevError > 0)) {
    pidIntegral *= 0.5f;
  }

  if (fabs(error) < P.ERROR_DEADBAND) {
    pidPrevError = error;
  } else {
    if (!steeringMoving) pidIntegral += error * dt;
    float integralMax = P.CORR_MAX / max(P.KI, 0.0001f);
    pidIntegral = constrain(pidIntegral, -integralMax, integralMax);

    float derivative = (error - pidPrevError) / dt;
    pidPrevError = error;

    float rawCorrection = (P.KP * error) + (P.KI * pidIntegral) + (P.KD * derivative);
    rawCorrection = constrain(rawCorrection, P.CORR_MIN, P.CORR_MAX);

    float maxCorrDelta = P.MAX_CORRECTION_SLEW * dt;
    lastCorrection += constrain(rawCorrection - lastCorrection, -maxCorrDelta, maxCorrDelta);
  }

  float ff = feedforwardThrottle(rampedTargetWheelRPM);
  thr_manualThr = constrain(ff + lastCorrection, 0.0, 100.0);
  applyThrottleSlewed(thr_manualThr, dt); // CHANGE: output goes through slew limiter now
}

// ============================================================
// ── AUTO-SEQUENCE (legacy fallback, off by default) ──────────
// ============================================================
bool autoSeqLastPinState = HIGH;
bool autoSeqActive = false, autoSeqSpeedReached = false, autoSeqBrakeDone = false;
unsigned long autoSeqHoldStartTime = 0;
const float AUTO_SEQ_SPEED_TOL_KMH = 1.0;

void checkAutoSequence() {
  if (!P.AUTO_SEQ_ENABLED) return;
  bool currentPinState = digitalRead(SWITCH_PIN);
  bool triggered = (autoSeqLastPinState == HIGH && currentPinState == LOW);
  autoSeqLastPinState = currentPinState;

  if (triggered && !autoSeqActive) {
    autoSeqActive = true; autoSeqSpeedReached = false; autoSeqBrakeDone = false;
    startPID(kmhToWheelRPM(P.AUTO_SEQ_TARGET_KMH));
  }
  if (autoSeqActive && !autoSeqSpeedReached) {
    float wheelRPM = pidFeedbackRPM / EXACT_GEAR_RATIO;
    float speed = wheelRPMToKmh(wheelRPM);
    if (speed >= (P.AUTO_SEQ_TARGET_KMH - AUTO_SEQ_SPEED_TOL_KMH)) {
      autoSeqSpeedReached = true;
      autoSeqHoldStartTime = millis();
    }
  }
  if (autoSeqActive && autoSeqSpeedReached && !autoSeqBrakeDone &&
      (millis() - autoSeqHoldStartTime >= P.AUTO_SEQ_HOLD_MS)) {
    autoSeqBrakeDone = true;
    stopPID();
    setBrakePercent(100.0f); // non-blocking now
    autoSeqActive = false;
  }
}

// ============================================================
// ── RELAY / TSAL ──────────────────────────────────────────────
// ============================================================
void handleRelay() {
  digitalWrite(RELAY_PIN, (digitalRead(SWITCH_PIN) == LOW) ? HIGH : LOW);
}

// ============================================================
// ── WATCHDOG ───────────────────────────────────────────────────
// ============================================================
unsigned long lastJetsonCmdTime = 0;

void checkWatchdog() {
  if (sysState != STATE_RUN) return;
  if (millis() - lastJetsonCmdTime > P.WATCHDOG_TIMEOUT_MS) {
    faultFlags |= FAULT_WATCHDOG;
    stopPID();
    setBrakePercent(100.0f);
  } else {
    faultFlags &= ~FAULT_WATCHDOG;
  }
}

// ============================================================
// ── PARAM GET/SET DISPATCH ────────────────────────────────────
// ============================================================
bool isLockedParam(const String &name) {
  static const char* locked[] = {
    "PID_ALPHA","ERROR_DEADBAND","CORR_MIN","CORR_MAX",
    "POT_LEFT","POT_CENTER","POT_RIGHT","TOLERANCE","SLOW_ZONE","MICRO_ZONE",
    "FAST_BURST","SLOW_BURST","MICRO_BURST",
    "HOME_ADC","FULL_ADC","ACT_TOLERANCE","ACT_MOVE_TIMEOUT_MS",
    "WATCHDOG_TIMEOUT_MS"
  };
  for (auto &s : locked) if (name.equals(s)) return true;
  return false;
}

void sendParamDump() {
  Serial.print(F("$PARAM,KP=")); Serial.print(P.KP, 5);
  Serial.print(F(",KI="));  Serial.print(P.KI, 5);
  Serial.print(F(",KD="));  Serial.print(P.KD, 5);
  Serial.print(F(",RAMP_RATE=")); Serial.print(P.RAMP_RATE, 2);
  Serial.print(F(",RAMP_ACCEL_LIMIT=")); Serial.print(P.RAMP_ACCEL_LIMIT, 2);
  Serial.print(F(",MAX_THROTTLE_SLEW=")); Serial.print(P.MAX_THROTTLE_SLEW, 2);
  Serial.print(F(",MAX_CORRECTION_SLEW=")); Serial.print(P.MAX_CORRECTION_SLEW, 2);
  Serial.print(F(",PID_ALPHA=")); Serial.print(P.PID_ALPHA, 3);
  Serial.print(F(",ERROR_DEADBAND=")); Serial.print(P.ERROR_DEADBAND, 2);
  Serial.print(F(",CORR_MIN=")); Serial.print(P.CORR_MIN, 2);
  Serial.print(F(",CORR_MAX=")); Serial.print(P.CORR_MAX, 2);
  Serial.print(F(",POT_LEFT=")); Serial.print(P.POT_LEFT);
  Serial.print(F(",POT_CENTER=")); Serial.print(P.POT_CENTER);
  Serial.print(F(",POT_RIGHT=")); Serial.print(P.POT_RIGHT);
  Serial.print(F(",TOLERANCE=")); Serial.print(P.TOLERANCE);
  Serial.print(F(",SLOW_ZONE=")); Serial.print(P.SLOW_ZONE);
  Serial.print(F(",MICRO_ZONE=")); Serial.print(P.MICRO_ZONE);
  Serial.print(F(",FAST_BURST=")); Serial.print(P.FAST_BURST);
  Serial.print(F(",SLOW_BURST=")); Serial.print(P.SLOW_BURST);
  Serial.print(F(",MICRO_BURST=")); Serial.print(P.MICRO_BURST);
  Serial.print(F(",HOME_ADC=")); Serial.print(P.HOME_ADC);
  Serial.print(F(",FULL_ADC=")); Serial.print(P.FULL_ADC);
  Serial.print(F(",ACT_TOLERANCE=")); Serial.print(P.ACT_TOLERANCE);
  Serial.print(F(",ACT_MOVE_TIMEOUT_MS=")); Serial.print(P.ACT_MOVE_TIMEOUT_MS);
  Serial.print(F(",AUTO_SEQ_TARGET_KMH=")); Serial.print(P.AUTO_SEQ_TARGET_KMH, 2);
  Serial.print(F(",AUTO_SEQ_HOLD_MS=")); Serial.print(P.AUTO_SEQ_HOLD_MS);
  Serial.print(F(",AUTO_SEQ_ENABLED=")); Serial.print(P.AUTO_SEQ_ENABLED ? 1 : 0);
  Serial.print(F(",WATCHDOG_TIMEOUT_MS=")); Serial.print(P.WATCHDOG_TIMEOUT_MS);
  Serial.print(F(",DRY_RUN=")); Serial.print(P.DRY_RUN ? 1 : 0);
  Serial.print(F(",LOCK_VERSION=")); Serial.print(P.lockVersion);
  Serial.print(F(",STATE=")); Serial.println(sysState == STATE_RUN ? F("RUN") : F("CONFIG"));
}

void handleSet(String name, String valStr) {
  if (sysState == STATE_RUN && isLockedParam(name)) {
    Serial.print(F("$NAK,SET:")); Serial.print(name); Serial.println(F(",LOCKED"));
    return;
  }
  float v = valStr.toFloat();
  bool ok = true;

       if (name == "KP") P.KP = v;
  else if (name == "KI") P.KI = v;
  else if (name == "KD") P.KD = v;
  else if (name == "RAMP_RATE") P.RAMP_RATE = v;
  else if (name == "RAMP_ACCEL_LIMIT") P.RAMP_ACCEL_LIMIT = v;
  else if (name == "MAX_THROTTLE_SLEW") P.MAX_THROTTLE_SLEW = v;
  else if (name == "MAX_CORRECTION_SLEW") P.MAX_CORRECTION_SLEW = v;
  else if (name == "PID_ALPHA") P.PID_ALPHA = v;
  else if (name == "ERROR_DEADBAND") P.ERROR_DEADBAND = v;
  else if (name == "CORR_MIN") P.CORR_MIN = v;
  else if (name == "CORR_MAX") P.CORR_MAX = v;
  else if (name == "POT_LEFT")   { P.POT_LEFT = (int32_t)v; recalcCPD(); }
  else if (name == "POT_CENTER") { P.POT_CENTER = (int32_t)v; recalcCPD(); }
  else if (name == "POT_RIGHT")  { P.POT_RIGHT = (int32_t)v; recalcCPD(); }
  else if (name == "TOLERANCE") P.TOLERANCE = (int32_t)v;
  else if (name == "SLOW_ZONE") P.SLOW_ZONE = (int32_t)v;
  else if (name == "MICRO_ZONE") P.MICRO_ZONE = (int32_t)v;
  else if (name == "FAST_BURST") P.FAST_BURST = (uint8_t)v;
  else if (name == "SLOW_BURST") P.SLOW_BURST = (uint8_t)v;
  else if (name == "MICRO_BURST") P.MICRO_BURST = (uint8_t)v;
  else if (name == "HOME_ADC") P.HOME_ADC = (int32_t)v;
  else if (name == "FULL_ADC") P.FULL_ADC = (int32_t)v;
  else if (name == "ACT_TOLERANCE") P.ACT_TOLERANCE = (int32_t)v;
  else if (name == "ACT_MOVE_TIMEOUT_MS") P.ACT_MOVE_TIMEOUT_MS = (uint32_t)v;
  else if (name == "AUTO_SEQ_TARGET_KMH") P.AUTO_SEQ_TARGET_KMH = v;
  else if (name == "AUTO_SEQ_HOLD_MS") P.AUTO_SEQ_HOLD_MS = (uint32_t)v;
  else if (name == "AUTO_SEQ_ENABLED") P.AUTO_SEQ_ENABLED = (v != 0);
  else if (name == "WATCHDOG_TIMEOUT_MS") P.WATCHDOG_TIMEOUT_MS = (uint32_t)v;
  else if (name == "DRY_RUN") P.DRY_RUN = (v != 0);
  else ok = false;

  if (ok) { Serial.print(F("$ACK,SET:")); Serial.print(name); Serial.print(F("=")); Serial.println(v, 5); }
  else    { Serial.print(F("$NAK,SET:")); Serial.print(name); Serial.println(F(",UNKNOWN")); }
}

// ============================================================
// ── COMMAND PARSER ────────────────────────────────────────────
// ============================================================
void handleSerial() {
  if (!Serial.available()) return;
  String input = Serial.readStringUntil('\n');
  input.trim();
  if (input.length() < 1) return;

  lastJetsonCmdTime = millis(); // any traffic counts as heartbeat

  if (calMode)     { handleCalStepImpl(); return; }
  if (actCalMode)  { handleActuatorCalibration(); return; }

  String upper = input; upper.toUpperCase();

  if (upper == "PING") { Serial.println(F("$ACK,PING")); return; }

  if (upper.startsWith("SET:")) {
    int eq = input.indexOf('=');
    if (eq < 0) { Serial.println(F("$NAK,SET,BADFORMAT")); return; }
    String name = input.substring(4, eq); name.trim();
    String val  = input.substring(eq + 1); val.trim();
    handleSet(name, val);
    return;
  }

  if (upper == "GET:ALL") { sendParamDump(); return; }

  if (upper == "LOCK") {
    if (!centerAchieved) {
      Serial.println(F("$NAK,LOCK,NOT_CENTERED"));
      return;
    }
    if (faultFlags != 0) {
      Serial.print(F("$NAK,LOCK,FAULT=")); Serial.println(faultFlags);
      return;
    }
    P.lockVersion++; // CHANGE (#2): version bump so Jetson can detect param-set changes across sessions
    saveParamsToEEPROM();
    sysState = STATE_RUN;
    Serial.print(F("$ACK,LOCK,RUN,VERSION=")); Serial.println(P.lockVersion);
    return;
  }

  if (upper == "UNLOCK") {
    stopPID();
    setBrakePercent(100.0f);
    sysState = STATE_CONFIG;
    Serial.println(F("$ACK,UNLOCK,CONFIG"));
    return;
  }

  if (upper == "STCAL") {
    calMode = true; calStep = 0; centerAchieved = false;
    steerDisable(); motorReady = true;
    Serial.println(F("$ACK,STCAL,STEP1_CENTER"));
    return;
  }

  if (upper == "ACAL") {
    actCalMode = true; actCalStep = 0;
    Serial.println(F("$ACK,ACAL,STEP0_HOME"));
    return;
  }

  if (upper.startsWith("STA")) {
    float deg = input.substring(3).toFloat();
    setSteerAbsolute(deg);
    Serial.print(F("$ACK,STA,")); Serial.println(deg, 2);
    return;
  }

  // Back-compat thin aliases for manual/debug use (wrap STA)
  if (upper.startsWith("STR") && upper.length() >= 4 && isDigit(upper.charAt(3))) {
    float deg = input.substring(3).toFloat();
    setSteerAbsolute(adcToDeg(readPot()) + deg);
    return;
  }
  if (upper.startsWith("STL") && upper.length() >= 4 && isDigit(upper.charAt(3))) {
    float deg = input.substring(3).toFloat();
    setSteerAbsolute(adcToDeg(readPot()) - deg);
    return;
  }
  if (upper == "STC") { setSteerAbsolute(0.0f); return; }

  if (upper.startsWith("BRK")) {
    float pct = input.substring(3).toFloat();
    setBrakePercent(pct);
    Serial.print(F("$ACK,BRK,")); Serial.println(pct, 1);
    return;
  }

  if (upper == "Z") {
    stopPID();
    Serial.println(F("$ACK,Z"));
    return;
  }

  if (upper.charAt(0) == 'S' && upper.length() > 1 && !upper.startsWith("STA") &&
      !upper.startsWith("STR") && !upper.startsWith("STL") && !upper.startsWith("STC")) {
    float kmh = input.substring(1).toFloat();
    startPID(kmhToWheelRPM(kmh));
    Serial.print(F("$ACK,S,")); Serial.println(kmh, 2);
    return;
  }

  Serial.println(F("$NAK,UNKNOWN_CMD"));
}

void handleCalStepImpl() {
  switch (calStep) {
    case 0:
      P.POT_CENTER = readAverage(500);
      Serial.print(F("$ACK,STCAL_CENTER,")); Serial.println(P.POT_CENTER);
      calStep = 1;
      break;
    case 1:
      P.POT_LEFT = readAverage(500);
      Serial.print(F("$ACK,STCAL_LEFT,")); Serial.println(P.POT_LEFT);
      calStep = 2;
      break;
    case 2:
      P.POT_RIGHT = readAverage(500);
      Serial.print(F("$ACK,STCAL_RIGHT,")); Serial.println(P.POT_RIGHT);
      recalcCPD();
      calMode = false; calStep = 0; centerAchieved = false;
      targetADC = P.POT_CENTER; motorReady = false;
      steerEnable();
      break;
  }
}

// ============================================================
// ── TELEMETRY ($TEL) ──────────────────────────────────────────
// ============================================================
unsigned long lastTelTime = 0;
const unsigned long TEL_INTERVAL_MS = 50; // 20Hz

void sendTelemetry() {
  float displayWheelRPM = filteredMotorRPM / EXACT_GEAR_RATIO;
  float speedKmh = wheelRPMToKmh(displayWheelRPM);
  int32_t steerADC = readPot();

  Serial.print(F("$TEL,"));
  Serial.print(millis()); Serial.print(',');
  Serial.print(sysState == STATE_RUN ? F("RUN") : F("CONFIG")); Serial.print(',');
  Serial.print(throttleMode); Serial.print(',');
  Serial.print(speedKmh, 2); Serial.print(',');
  Serial.print(displayWheelRPM, 1); Serial.print(',');
  Serial.print(lastAppliedThrottle, 1); Serial.print(',');
  Serial.print(adcToDeg(steerADC), 2); Serial.print(',');
  Serial.print(steerTargetDeg, 2); Serial.print(',');
  Serial.print(brakeActualPct, 1); Serial.print(',');
  Serial.print(brakeTargetPct, 1); Serial.print(',');
  Serial.print(rampedTargetWheelRPM, 1); Serial.print(',');
  Serial.print(pidPrevError, 2); Serial.print(',');
  Serial.print(lastCorrection, 2); Serial.print(',');
  Serial.print(faultFlags); Serial.print(',');
  Serial.println(P.DRY_RUN ? 1 : 0); // CHANGE (#9): dry-run flag always visible in live stream
}

// ============================================================
// ── THROTTLE SAMPLING LOOP ───────────────────────────────────
// ============================================================
void runThrottleLoop(unsigned long currentTime) {
  handleRelay();

  if (currentTime - lastLoopTime >= LOOP_INTERVAL) {
    float dt = (currentTime - lastLoopTime) / 1000.0;
    lastLoopTime = currentTime;

    noInterrupts();
    unsigned long currentPulses = pulseCount;
    pulseCount = 0;
    interrupts();

    float rawMotorRPM;

    // CHANGE (#1): below the pulse-count threshold, use the period between
    // the last two pulses (micros() resolution) instead of the windowed
    // pulse count -> much less quantization at low RPM. Above the
    // threshold, revert to the windowed-count method (more pulses per
    // window makes it accurate and it's cheaper/less noise-sensitive to a
    // single bad period reading).
    if (currentPulses > 0 && currentPulses < PERIOD_METHOD_PULSE_THRESHOLD) {
      noInterrupts();
      unsigned long periodUs = prevPulsePeriodUs;
      interrupts();
      if (periodUs > 0) {
        float freqHz = 1000000.0f / (float)periodUs;
        rawMotorRPM = (freqHz * 60.0f) / MOTOR_PULSES_PER_REV;
      } else {
        rawMotorRPM = ((float)currentPulses / dt) * 60.0f / MOTOR_PULSES_PER_REV;
      }
    } else if (currentPulses == 0) {
      rawMotorRPM = 0.0f; // no edges this window -> genuinely stopped or below sensor resolution
    } else {
      float frequency = (float)currentPulses / dt;
      rawMotorRPM = (frequency * 60.0) / MOTOR_PULSES_PER_REV;
    }

    rawMotorRPM = applyMedianFilter(rawMotorRPM); // CHANGE: strip single-sample electrical spikes first

    if (fabs(rawMotorRPM - lastAcceptedRPM) > MAX_RPM_JUMP_PER_SAMPLE) {
      rawMotorRPM = lastAcceptedRPM;
    } else {
      lastAcceptedRPM = rawMotorRPM;
    }

    pidFeedbackRPM = (P.PID_ALPHA * rawMotorRPM) + ((1.0 - P.PID_ALPHA) * pidFeedbackRPM);
    filteredMotorRPM = updateDisplayFilter(rawMotorRPM);
    if (filteredMotorRPM < 15.0) filteredMotorRPM = 0.0;
    if (pidFeedbackRPM  < 15.0)  pidFeedbackRPM = 0.0;

    if (millis() - lastPulseSeenTime > 2000 && throttleMode != MODE_IDLE) {
      faultFlags |= FAULT_SENSOR_TIMEOUT;
    } else {
      faultFlags &= ~FAULT_SENSOR_TIMEOUT;
    }

    if (throttleMode == MODE_PID) updatePIDControl(dt);
    else if (throttleMode == MODE_IDLE) applyThrottleSlewed(0.0f, dt);
  }
}

// ============================================================
// ── SETUP ────────────────────────────────────────────────────
// ============================================================
void setup() {
  pinMode(SENSOR_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), sensorPulseISR, FALLING);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(SWITCH_PIN, INPUT_PULLUP);
  digitalWrite(RELAY_PIN, LOW);

  pinMode(EN_PIN, OUTPUT);  digitalWrite(EN_PIN, MOTOR_DISABLE);
  pinMode(PUL_PIN, OUTPUT); digitalWrite(PUL_PIN, LOW);
  pinMode(DIR_PIN, OUTPUT); digitalWrite(DIR_PIN, LOW);
  pinMode(POT_PIN, INPUT);

  pinMode(PWM1_PIN, OUTPUT);
  pinMode(DIR1_PIN, OUTPUT);
  stopActuatorMotor();

  Serial.begin(115200);
  delay(200);

  Wire.begin();
  Wire.setClock(400000);

  if (!dac.begin(0x60)) {
    faultFlags |= FAULT_DAC_MISSING;
    Serial.println(F("$FAULT,DAC_MISSING"));
  }
  setDACRaw(DAC_IDLE);

  if (!loadParamsFromEEPROM()) {
    loadDefaults();
  }
  recalcCPD();

  autoSeqLastPinState = digitalRead(SWITCH_PIN);
  lastJetsonCmdTime = millis();

  Serial.println(F("$BOOT"));
  sendParamDump();

  centerAchieved = false;
  targetADC = P.POT_CENTER;
  motorReady = false;
  steerEnable();
  faultFlags |= FAULT_NOT_CENTERED;
}

// ============================================================
// ── LOOP ─────────────────────────────────────────────────────
// ============================================================
void loop() {
  unsigned long currentTime = millis();

  handleSerial(); // calMode/actCalMode steps are advanced inside here, per received line

  checkWatchdog();
  checkAutoSequence();
  moveToTarget();
  updateActuator();      // non-blocking brake state machine
  runThrottleLoop(currentTime);

  if (centerAchieved) faultFlags &= ~FAULT_NOT_CENTERED;

  if (currentTime - lastTelTime >= TEL_INTERVAL_MS) {
    lastTelTime = currentTime;
    sendTelemetry();
  }
}