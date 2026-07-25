"""
lane_detection_node_birdeye.py
---------------------------------
Full bird's-eye-first pipeline, following this order:

  camera frame
    -> camera calibration (undistort; pass-through, no calibration data yet)
    -> adaptive image enhancement (only touches genuinely dark frames)
    -> YOLOP inference                  [replaces "Edge + Color Lane Extraction"]
    -> ROI selection (ignore sky/buildings, on the probability map)
    -> inverse perspective mapping (warp frame + probability map to bird's-eye)
    -> boundary extraction on the warped map, continuity-aware   [replaces "Sliding Window Lane Search"]
    -> polynomial curve fit (in bird's-eye coordinates)
    -> missing-boundary reconstruction (warped-space width calibrator)
    -> cross-track-error + heading, read directly off the warped curves
       (no separate re-warp needed for heading -- unlike the perspective-
       space pipeline's heading_mode="ipm", the fit is ALREADY in bird's-
       eye space by construction, so this is simpler and more unified)
    -> Kalman filter (temporal tracking of [cte, cte_rate, heading, heading_rate])
    -> centerline (implicit in cte/heading) -> Stanley steering law -> slew-rate limiter

This is a parallel, newer pipeline alongside lane_detection_node.py (the
perspective-space version, already validated against a known-good
baseline) -- kept side by side rather than replacing it, so the earlier,
working system stays available for comparison.
"""
from typing import Optional

import numpy as np
import torch

from perception.lane_core.curves import LaneCurve, fit_lane_curve
from perception.lane_core.calibration import LaneWidthCalibrator
from perception.lane_core.camera_calibration import CameraCalibration
from perception.lane_core.adaptive_enhancement import enhance_if_dark, enhance_for_night
from perception.lane_core.roi_mask import apply_roi_mask
from perception.lane_core.ipm import (build_ipm_transform, invert_transform, warp_to_birdeye,
                                       IPM_ZOOM_LADDER, IPM_CALIB_FRAMES, IPM_BASE_RUNG,
                                       IPM_SRC_BOTTOM_HALF_WIDTH_FRAC)
from perception.lane_core.bird_eye_extraction import extract_birdeye_boundaries, MIN_BOTTOM_REACH_FRAC
from perception.lane_core.kalman_tracker import LaneKalmanTracker
from perception.lane_core.steering_v2 import stanley_steering_deg, SlewRateLimiter
from perception.lane_core.model_runtime import load_model, run_inference
from ego_state.ego_state_estimator import EgoStateEstimator
from ego_state.imu_source import ImuSource
from signal_conditioning.conditioner import SignalConditioner

RECONSTRUCTED_CONFIDENCE_SCALE = 0.5
MIN_WIDTH_FRAC_OF_CANVAS = 0.02  # reject a near-zero-width warped lane read as noise

# --- temporal consistency gate + time-bounded coast (hold-through-dropout) ----
# When one real lane line is absent, the boundary extractor can latch onto a
# road-pavement / shoulder / curb edge and report it as a legitimate "direct"
# detection, shifting the perceived lane laterally. The whiteness gate
# (color_filter.py) misses bright concrete, and the signal conditioner only
# rejects fast JUMPS of the fused cte/heading -- a slow drift onto pavement stays
# within its rate limit. These constants drive a per-side gate that instead
# compares each fresh fit against that side's own recent trusted history and, on
# rejection/absence, COASTS on the last good fit for a bounded buffer (labeled
# "held") before releasing for a clean cold-start reacquisition. All of it is a
# no-op on cold start / an un-warmed calibrator, so the both-lanes-visible case
# is unchanged. Placeholders in the same spirit as every other gain here.
HOLD_BUFFER_S = 1.8                 # max time to coast a side on its last good fit
LATERAL_JUMP_SPEED_FACTOR = 1.0     # scale on speed*dt (worst-case sideways motion)
LATERAL_JUMP_MARGIN_M = 0.6         # fixed slack for fit noise / curvature
WIDTH_CONSISTENCY_TOLERANCE = 0.35  # allowed fractional deviation of a fresh pair
                                    # from the calibrated width(y)
HELD_MIN_CONFIDENCE_SCALE = 0.2     # confidence floor a fully-aged hold decays to
                                    # (fraction of the last good confidence); the
                                    # decay crosses the safety monitor's 0.3 engage
                                    # floor before the buffer expires -> graceful
                                    # hand-back rather than blind coasting
# A fresh fit is "partial" when it does NOT reach the near-vehicle band -- the
# SAME band the extractor already uses to decide a blob "reaches the vehicle"
# (bird_eye_extraction.MIN_BOTTOM_REACH_FRAC). Reusing that one threshold (rather
# than a second, stricter one) is deliberate: a fit the extractor already trusts
# as bottom-reaching must stay `direct`, so only a fit selected from a lower tier
# -- a genuine short arc that never reached the vehicle row, e.g. the outer line
# on a turn while the inner one has left the frame -- is downgraded to `partial`.
# For such a fit, reading CTE at the bottom row extrapolates its shaky curvature
# far past its data, so it is trusted for heading only. A normal near-vehicle
# dashed segment DOES reach the band and is unaffected.
PARTIAL_CONFIDENCE_SCALE = 0.5      # confidence scale for a partial fresh fit

# --- anti-flicker display smoothing (curves-only, no effect on control) -------
# The Kalman filter smooths the cte/heading SCALARS that drive steering, but it
# deliberately does not track the lane-curve coefficients -- so the left/right
# LaneCurves returned for the overlay are the raw per-frame polynomial fits and
# visibly jitter. These smooth the DISPLAYED geometry only: the controller still
# runs on the raw fit via the Kalman, so there is zero added steering lag.
CURVE_SMOOTH_TAU_S = 0.20           # EMA time constant for the displayed lane
                                    # curves. Time-based (alpha = 1 - exp(-dt/tau))
                                    # so smoothing is framerate-independent. Since
                                    # x_at = polyval(coeffs, y) is LINEAR in coeffs,
                                    # an EMA of the coefficients is exactly an EMA of
                                    # the lane's x-position at every row.
LABEL_HYSTERESIS_FRAMES = 3         # a side's DISPLAYED source label (overlay colour)
                                    # only switches after the new label has held for
                                    # this many consecutive frames -- stops colour
                                    # strobing on rapid direct<->reconstructed<->held
                                    # toggling. A switch TO "none" is immediate (never
                                    # linger a colour on a lane that isn't there). The
                                    # real per-frame decision is unchanged.


class BirdEyeLaneDetectionNode:
    def __init__(self, weights_path: str, device, img_size: int = 640,
                 lane_width_m: float = 3.0, speed_mps: float = 5.0,
                 camera_calibration: Optional[CameraCalibration] = None,
                 imu_source: Optional[ImuSource] = None,
                 adaptive_zoom: bool = False,
                 night_mode: bool = False, night_denoise: bool = True,
                 lane_prob_threshold: float = 0.5):
        self.img_size = img_size
        self.device = device
        self.model = load_model(weights_path, device)
        # TensorRT engines carry their own precision (fp32 I/O, internal fp16);
        # only the PyTorch path honours the `half` flag / needs .half().
        self.is_trt = getattr(self.model, "is_trt", False)
        self.half = device.type != "cpu" and not self.is_trt
        if self.half:
            self.model.half()
        if self.is_trt:
            self.model.warmup()   # square dummy frame would letterbox to the wrong size
        elif device.type != "cpu":
            dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            with torch.no_grad():
                run_inference(self.model, dummy, img_size, device, self.half)

        self.calibration = camera_calibration or CameraCalibration()
        self.lane_width_m = lane_width_m
        self.speed_mps = speed_mps

        self.calibrator = LaneWidthCalibrator()  # warped-space width(y) calibration
        self.tracker = LaneKalmanTracker()
        self.slew_limiter = SlewRateLimiter()
        # Design A dead-reckoning bridge: on this dev machine imu_source is
        # None -> EgoStateEstimator defaults to NullImuSource, so
        # ego_delta["have_imu_yaw_rate"] is always False and the tracker's
        # blind prediction is untouched (see kalman_tracker.py's
        # imu_heading_rate_dps no-op contract) -- a real ImuSource plugged
        # in later starts informing the filter with no other code change.
        self.ego_state = EgoStateEstimator(imu_source, default_speed_mps=speed_mps)
        self.conditioner = SignalConditioner()

        # Adaptive IPM "zoom" auto-calibration: when enabled, for the first
        # IPM_CALIB_FRAMES frames, score every rung of IPM_ZOOM_LADDER (a sweep
        # of trapezoids from zoomed-in to zoomed-out) against the live YOLOP
        # probability map -- how well each reveals a BALANCED left+right lane
        # pair in the bird's-eye -- accumulate the scores, then LOCK the best
        # rung for the rest of the run. This "zooms out until the lanes are
        # visible" and adapts to the actual camera framing. Off by default: the
        # live-vehicle path and other entry points use the fixed recalibrated
        # base trapezoid (build_ipm_transform's defaults).
        self.adaptive_zoom = adaptive_zoom
        self._zoom_step = 0                 # chosen ladder rung (0 until calibrated)
        self._zoom_locked = False
        self._calib_scores = [0] * len(IPM_ZOOM_LADDER)
        self._calib_frames = 0

        # Night mode: stronger low-light enhancement front-end (adaptive gamma +
        # CLAHE + edge-preserving denoise) plus a settable lane-probability
        # threshold, so faint night lane lines that read below the default 0.5
        # can still be picked up. Both off/at-default by default -> the live
        # vehicle path and other entry points are unchanged.
        self.night_mode = night_mode
        self.night_denoise = night_denoise
        self.lane_prob_threshold = float(lane_prob_threshold)
        self._ipm_M = None
        self._ipm_M_inv = None
        self._ipm_key = None  # (frame_w, frame_h) the cached base M was built for
        self.last_left: Optional[LaneCurve] = None
        self.last_right: Optional[LaneCurve] = None
        self._last_trusted_raw_deg = 0.0
        self._last_tracked_heading_deg = 0.0
        # Temporal-consistency gate + coast state (see the module constants
        # above). _<side>_good_t = now_s of the last ACCEPTED DIRECT observation
        # for that side -- what bounds its coast. Refreshed ONLY on a direct
        # (never partial/reconstructed/held) commit, so a side can't coast
        # forever off its own inferred value. _last_process_t drives per-frame
        # dt; _last_good_conf is the last committed direct-pair confidence the
        # held-confidence decay interpolates down from.
        self._left_good_t: Optional[float] = None
        self._right_good_t: Optional[float] = None
        self._last_process_t: Optional[float] = None
        self._last_good_conf = 0.0
        # Anti-flicker DISPLAY state (curves-only; never touches control). The
        # EMA-smoothed curves shown in the overlay, and per-side displayed-label
        # hysteresis (a pending candidate label + how many consecutive frames it
        # has persisted) so the overlay colour only switches once a new label has
        # held for LABEL_HYSTERESIS_FRAMES.
        self._smoothed_left: Optional[LaneCurve] = None
        self._smoothed_right: Optional[LaneCurve] = None
        self._disp_left_source = "none"
        self._disp_right_source = "none"
        self._pend_left_source = "none"
        self._pend_right_source = "none"
        self._pend_left_count = 0
        self._pend_right_count = 0

    @property
    def ipm_M_inv(self):
        return self._ipm_M_inv

    def _set_ipm(self, frame_w, frame_h, top_frac=None, top_half=None):
        """Build and cache the IPM transform + its inverse. With no fractions,
        uses the recalibrated base defaults (build_ipm_transform's own)."""
        if top_frac is None:
            self._ipm_M = build_ipm_transform(frame_w, frame_h)
        else:
            self._ipm_M = build_ipm_transform(frame_w, frame_h, top_frac, top_half,
                                              IPM_SRC_BOTTOM_HALF_WIDTH_FRAC)
        self._ipm_M_inv = invert_transform(self._ipm_M)

    def _rung_finds_both_lanes(self, prob_map, enhanced, w, h, tf, thw) -> bool:
        """Warp through one ladder rung and run the REAL boundary extractor:
        True iff it yields BOTH a left and a right boundary. Scale-invariant --
        unlike a raw warped-pixel count, which is biased toward zoomed-in rungs
        (they magnify the same pixels). This is what the pipeline actually needs."""
        M = build_ipm_transform(w, h, tf, thw, IPM_SRC_BOTTOM_HALF_WIDTH_FRAC)
        wf, wp = warp_to_birdeye(enhanced, prob_map, M)
        wm = (wp > self.lane_prob_threshold).astype(np.uint8)
        (ly, _, _), (ry, _, _) = extract_birdeye_boundaries(wm, wp, wf,
                                                            prev_left=None, prev_right=None)
        return len(ly) > 0 and len(ry) > 0

    def _select_ipm(self, prob_map, enhanced, frame_w, frame_h) -> None:
        """Choose the IPM transform for this frame. Non-adaptive: the fixed
        recalibrated base, built once. Adaptive: until locked, test every rung of
        IPM_ZOOM_LADDER (does it reveal BOTH lanes?), run this frame live on the
        most zoomed-out rung that works, accumulate a both-lanes vote per rung,
        and after IPM_CALIB_FRAMES lock the rung with the most votes (ties ->
        more zoomed-out)."""
        if not self.adaptive_zoom:
            if self._ipm_key != (frame_w, frame_h):
                self._set_ipm(frame_w, frame_h)
                self._ipm_key = (frame_w, frame_h)
            return
        if self._zoom_locked:
            return  # locked rung's M already set

        both = [self._rung_finds_both_lanes(prob_map, enhanced, frame_w, frame_h, tf, thw)
                for (tf, thw) in IPM_ZOOM_LADDER]
        for i, b in enumerate(both):
            self._calib_scores[i] += int(b)
        # Live this frame: most zoomed-out rung that reveals both lanes, else base.
        live = max((i for i, b in enumerate(both) if b), default=IPM_BASE_RUNG)
        self._zoom_step = live
        self._set_ipm(frame_w, frame_h, *IPM_ZOOM_LADDER[live])

        self._calib_frames += 1
        if self._calib_frames >= IPM_CALIB_FRAMES:
            # Most both-lanes votes; tie -> higher index (more zoomed out).
            best = max(range(len(self._calib_scores)),
                       key=lambda i: (self._calib_scores[i], i))
            self._zoom_step = best
            self._set_ipm(frame_w, frame_h, *IPM_ZOOM_LADDER[best])
            self._zoom_locked = True

    @staticmethod
    def _is_partial(curve: LaneCurve, canvas_h: int) -> bool:
        """A fresh fit is 'partial' when it does NOT reach the near-vehicle band
        the extractor itself uses (MIN_BOTTOM_REACH_FRAC) -- i.e. it was picked
        from a lower quality tier and its nearest point stops short of the
        vehicle row, so reading CTE at the bottom extrapolates its shaky
        curvature far past its data (the outer arc on a turn while the inner line
        has left the frame). Reusing the extractor's own reach threshold keeps a
        fit it already treats as bottom-reaching -- including a normal near-
        vehicle dashed segment -- classified `direct`, not `partial`."""
        min_reach_y = canvas_h * (1.0 - MIN_BOTTOM_REACH_FRAC)
        return curve.y_max < min_reach_y

    def _lateral_jump_ok(self, fresh: LaneCurve, prev_good: Optional[LaneCurve],
                         ww: int, bottom_y: int, dt: float, speed_mps: float) -> bool:
        """True iff `fresh` hasn't jumped further laterally than the vehicle
        could plausibly shift the lane in `dt`. Compared at an anchor row INSIDE
        the fresh fit's observed span (never an extrapolation past its data -- a
        short arc high in the frame blows up if evaluated at the bottom row).
        No-op True when `prev_good` is None (cold start / just released): there's
        no trusted history to judge against yet."""
        if prev_good is None:
            return True
        # Anchor where BOTH curves actually have support, biased toward the
        # bottom (nearest the car) but clamped into each curve's real y-range.
        anchor = float(np.clip(bottom_y, max(fresh.y_min, prev_good.y_min),
                               min(fresh.y_max, prev_good.y_max)))
        calib_w_px = float(self.calibrator.width_at(anchor, ww))
        m_per_px = self.lane_width_m / max(calib_w_px, 1.0)
        jump_m = abs(float(fresh.x_at(anchor)) - float(prev_good.x_at(anchor))) * m_per_px
        allowed_m = speed_mps * dt * LATERAL_JUMP_SPEED_FACTOR + LATERAL_JUMP_MARGIN_M
        return jump_m <= allowed_m

    def _pair_width_ok(self, left: LaneCurve, right: LaneCurve,
                       ww: int, bottom_y: int) -> bool:
        """True iff the fresh L/R pair's width at a shared anchor row matches
        the calibrated width(y) within WIDTH_CONSISTENCY_TOLERANCE. No-op True
        until the calibrator is warmed (coeffs is None) -- no width model to
        judge against yet."""
        if self.calibrator.coeffs is None:
            return True
        anchor = float(np.clip(bottom_y, max(left.y_min, right.y_min),
                               min(left.y_max, right.y_max)))
        calib_w = float(self.calibrator.width_at(anchor, ww))
        if calib_w <= 0:
            return True
        pair_w = float(right.x_at(anchor)) - float(left.x_at(anchor))
        return abs(pair_w - calib_w) / calib_w <= WIDTH_CONSISTENCY_TOLERANCE

    @staticmethod
    def _smooth_curve(prev: Optional[LaneCurve], new: Optional[LaneCurve],
                      dt: float) -> Optional[LaneCurve]:
        """EMA-blend `new`'s coefficients toward `prev`'s for the DISPLAYED lane
        (overlay only -- control already ran on the raw fit). Because
        x_at = polyval(coeffs, y) is linear in coeffs, blending coefficients IS
        a pointwise EMA of the lane's x at every row. Snaps to `new` (alpha=1) on
        the first sample after a reset or a non-positive dt, so a reacquired lane
        appears immediately rather than fading in from stale geometry. Returns
        None when `new` is None (side absent -> resets the smoother)."""
        if new is None:
            return None
        if prev is None or dt <= 0:
            return new
        alpha = 1.0 - float(np.exp(-dt / CURVE_SMOOTH_TAU_S))
        coeffs = alpha * new.coeffs + (1.0 - alpha) * prev.coeffs
        return LaneCurve(coeffs, new.y_min, new.y_max, confidence=new.confidence,
                         reconstructed=new.reconstructed, raw_ys=new.raw_ys, raw_xs=new.raw_xs)

    @staticmethod
    def _stabilize_label(disp: str, pend: str, count: int, candidate: str):
        """Displayed-label hysteresis (overlay colour only). Returns the updated
        (disp, pend, count). The displayed label only switches to `candidate`
        once that same candidate has differed from `disp` for
        LABEL_HYSTERESIS_FRAMES consecutive frames -- except a switch TO "none"
        is immediate, so a colour never lingers on a lane that isn't there."""
        if candidate == disp:
            return disp, candidate, 0            # steady: clear any pending run
        if candidate == "none":
            return "none", "none", 0             # disappearance is committed at once
        if candidate == pend:
            count += 1
        else:
            pend, count = candidate, 1           # a new candidate starts a fresh run
        if count >= LABEL_HYSTERESIS_FRAMES:
            return candidate, candidate, 0       # persisted long enough -> commit
        return disp, pend, count                 # hold the old colour for now

    def process(self, frame_bgr: np.ndarray, now_s: float, return_debug: bool = False) -> dict:
        h0, w0 = frame_bgr.shape[:2]

        # 1. Camera calibration (undistort) -- pass-through until real
        #    calibration data exists.
        undistorted = self.calibration.undistort(frame_bgr)

        # 2. Adaptive enhancement -- only touches genuinely dark frames. Night
        #    mode swaps in the stronger low-light front-end (wider engage range,
        #    adaptive gamma, denoise) recommended for nighttime lane detection.
        enhanced = (enhance_for_night(undistorted, denoise=self.night_denoise)
                    if self.night_mode else enhance_if_dark(undistorted))

        # 3. YOLOP inference -- replaces classical "Edge + Color Lane
        #    Extraction"; the model's own probability map IS the extracted
        #    lane-line signal.
        _, prob_map = run_inference(self.model, enhanced, self.img_size, self.device, self.half)

        # 4. ROI selection -- ignore sky/buildings before anything else
        #    touches the probability map.
        prob_map = apply_roi_mask(prob_map)

        # 4b. Choose the IPM transform. Adaptive auto-calibration needs the
        #     probability map + frame (it warps + extracts through candidate zoom
        #     levels to score them), so it runs here, after ROI and before the
        #     warp uses self._ipm_M.
        self._select_ipm(prob_map, enhanced, w0, h0)

        # 5. Inverse perspective mapping -- warp both the color frame (for
        #    the whiteness check) and the probability map into one
        #    consistent bird's-eye canvas.
        warped_frame, warped_prob = warp_to_birdeye(enhanced, prob_map, self._ipm_M)
        warped_mask = (warped_prob > self.lane_prob_threshold).astype(np.uint8)
        wh, ww = warped_mask.shape[:2]

        # 6. Boundary extraction on the warped probability map -- replaces
        #    "Sliding Window Lane Search" with the model's already-computed
        #    output, cascaded through reach/length/whiteness gates, with
        #    lane-continuity-aware search when a previous curve exists.
        split_result = extract_birdeye_boundaries(
            warped_mask, warped_prob, warped_frame,
            prev_left=self.last_left, prev_right=self.last_right, return_debug=return_debug)
        if return_debug:
            (ly, lx, lc), (ry, rx, rc), split_debug = split_result
        else:
            (ly, lx, lc), (ry, rx, rc) = split_result
            split_debug = None

        # 7. Polynomial curve fitting, in bird's-eye coordinates.
        left = fit_lane_curve(ly, lx, lc) if len(ly) else None
        right = fit_lane_curve(ry, rx, rc) if len(ry) else None

        # 7b. Per-frame dt for the temporal gate/coast. speed_gate is the same
        #     speed the controller falls back to when no IMU is attached (step 9),
        #     so the lateral-jump bound tracks real speed on the vehicle and the
        #     configured fallback offline.
        dt = (now_s - self._last_process_t) if self._last_process_t is not None else 0.0
        self._last_process_t = now_s
        speed_gate = self.speed_mps
        bottom_y = wh - 1

        # 7c. Classify each fresh fit. A fit that doesn't reach the near-vehicle
        #     band (see _is_partial / MIN_BOTTOM_REACH_FRAC -- e.g. the outer line
        #     on a turn while the inner one has left the frame) is 'partial':
        #     trusted for heading within its own span, but its near-vehicle
        #     geometry is extrapolated, so it must not calibrate width or
        #     re-anchor the coast timer.
        left_partial = left is not None and self._is_partial(left, wh)
        right_partial = right is not None and self._is_partial(right, wh)

        # 7d. Lateral-jump gate: null a fresh fit that jumped laterally beyond
        #     what the vehicle could plausibly have moved vs this side's last good
        #     curve -- the pavement/shoulder latch this whole change exists to
        #     stop. Judged at an anchor INSIDE the fit's own span (never a bottom-
        #     row extrapolation of a short arc).
        left_gate_reason = right_gate_reason = "ok"
        if left is not None and not self._lateral_jump_ok(left, self.last_left, ww, bottom_y, dt, speed_gate):
            left, left_partial, left_gate_reason = None, False, "lateral_jump"
        if right is not None and not self._lateral_jump_ok(right, self.last_right, ww, bottom_y, dt, speed_gate):
            right, right_partial, right_gate_reason = None, False, "lateral_jump"

        # 7e. Pair-width gate: if both survived but the pair width disagrees with
        #     the calibrated width(y) AND exactly one side is freshly reacquired
        #     (no per-side history to vouch for it), that new side is the suspect
        #     -> drop it, keep the historied side. Both-historied / both-new is
        #     left to the crossed-pair + width-plausibility defenses downstream.
        if left is not None and right is not None and not self._pair_width_ok(left, right, ww, bottom_y):
            if self.last_left is None and self.last_right is not None:
                left, left_partial, left_gate_reason = None, False, "width_inconsistent"
            elif self.last_right is None and self.last_left is not None:
                right, right_partial, right_gate_reason = None, False, "width_inconsistent"

        def _src(curve, partial):
            return "none" if curve is None else ("partial" if partial else "direct")
        left_source = _src(left, left_partial)
        right_source = _src(right, right_partial)

        # 8. Missing-boundary handling. A FULL pair (both direct) calibrates the
        #    width model; a partial fit is trusted for heading but not allowed to
        #    poison the calibrator. A single present side (direct or partial)
        #    reconstructs the other from the width model.
        if left is not None and right is not None:
            if not left_partial and not right_partial:
                self.calibrator.observe(left, right)
        elif left is not None and right is None:
            right_coeffs, right_ys, right_xs = left.shifted(lambda y: self.calibrator.width_at(y, ww))
            right = LaneCurve(right_coeffs, left.y_min, left.y_max,
                               confidence=left.confidence * RECONSTRUCTED_CONFIDENCE_SCALE,
                               reconstructed=True, raw_ys=right_ys, raw_xs=right_xs)
            right_source = "reconstructed"
        elif right is not None and left is None:
            left_coeffs, left_ys, left_xs = right.shifted(lambda y: -self.calibrator.width_at(y, ww))
            left = LaneCurve(left_coeffs, right.y_min, right.y_max,
                              confidence=right.confidence * RECONSTRUCTED_CONFIDENCE_SCALE,
                              reconstructed=True, raw_ys=left_ys, raw_xs=left_xs)
            left_source = "reconstructed"

        # 8b. Time-bounded coast: any side STILL missing after reconstruction
        #     (i.e. both sides were gated/absent) coasts on its last good fit, but
        #     only within HOLD_BUFFER_S of that side's last DIRECT observation.
        #     Past the buffer it stays None (released) -> cold-start reacquire.
        #     `*_held` tracks a frozen hold so a fully-held frame (no fresh info)
        #     can be fed to the tracker as a dropout, letting it EXTRAPOLATE a turn
        #     on its heading-rate instead of being pinned flat by stale geometry.
        left_held = right_held = False
        if left is None and self.last_left is not None and self._left_good_t is not None \
                and (now_s - self._left_good_t) <= HOLD_BUFFER_S:
            left, left_source, left_held = self.last_left, "held", True
        if right is None and self.last_right is not None and self._right_good_t is not None \
                and (now_s - self._right_good_t) <= HOLD_BUFFER_S:
            right, right_source, right_held = self.last_right, "held", True
        both_frozen_held = left_held and right_held

        valid = left is not None and right is not None
        raw_cte_m: Optional[float] = None
        raw_heading_deg: Optional[float] = None
        if valid:
            eval_y_left = float(np.clip(bottom_y, left.y_min, left.y_max))
            eval_y_right = float(np.clip(bottom_y, right.y_min, right.y_max))
            width_px = right.x_at(eval_y_right) - left.x_at(eval_y_left)
            if width_px >= ww * MIN_WIDTH_FRAC_OF_CANVAS:
                # Only commit this pair as the continuity reference for
                # next frame once it's passed the plausibility check --
                # committing a degenerate/crossed pair here was exactly
                # what let a single bad fit start a self-reinforcing
                # "search near an increasingly wrong prediction" drift
                # (see bird_eye_extraction.py's crossed-pair fallback,
                # which catches most of this upstream; this is the
                # remaining defense-in-depth check).
                self.last_left, self.last_right = left, right
                # Refresh the coast timer ONLY for a side observed DIRECTLY this
                # frame -- a partial/reconstructed/held side is not a fresh trusted
                # observation and must not extend how long we'll coast on it.
                if left_source == "direct":
                    self._left_good_t = now_s
                if right_source == "direct":
                    self._right_good_t = now_s
                if left_source == "direct" and right_source == "direct":
                    self._last_good_conf = (left.confidence + right.confidence) / 2.0
                meters_per_px = self.lane_width_m / width_px
                center_x = (left.x_at(eval_y_left) + right.x_at(eval_y_right)) / 2.0
                raw_cte_m = float(np.clip((center_x - ww / 2.0) * meters_per_px,
                                           -2 * self.lane_width_m, 2 * self.lane_width_m))
                # No separate re-warp needed here (contrast with the
                # perspective-space pipeline's heading_mode="ipm"): this
                # curve was already fit in bird's-eye space, so its own
                # slope already IS the perspective-corrected heading.
                avg_slope = (left.slope_at(eval_y_left) + right.slope_at(eval_y_right)) / 2.0
                raw_heading_deg = float(np.degrees(np.arctan(avg_slope)))
            else:
                valid = False  # degenerate near-zero width read as noise, not a real lane

        # A partial side (short arc that doesn't reach the vehicle) reports
        # reduced confidence -- trusted for heading, but the pipeline should stay
        # cautious about a fit whose near-vehicle geometry is extrapolated.
        if valid:
            lconf = left.confidence * (PARTIAL_CONFIDENCE_SCALE if left_source == "partial" else 1.0)
            rconf = right.confidence * (PARTIAL_CONFIDENCE_SCALE if right_source == "partial" else 1.0)
            confidence = (lconf + rconf) / 2.0
        else:
            confidence = 0.0
        if valid and (left_source == "held" or right_source == "held"):
            # Decay a coasting hold's reported confidence with age so the safety
            # monitor (MIN_ENGAGE_CONFIDENCE) auto-disengages a hold that never
            # recovers -- graceful hand-back, not blind coasting to the buffer
            # edge. Interpolate from the last good direct-pair confidence down to
            # HELD_MIN_CONFIDENCE_SCALE of it over HOLD_BUFFER_S. An early hold
            # still reports near-full confidence and keeps steering.
            ages = []
            if left_source == "held" and self._left_good_t is not None:
                ages.append(now_s - self._left_good_t)
            if right_source == "held" and self._right_good_t is not None:
                ages.append(now_s - self._right_good_t)
            age = max(ages) if ages else 0.0
            frac = float(np.clip(age / HOLD_BUFFER_S, 0.0, 1.0))
            scale = 1.0 - frac * (1.0 - HELD_MIN_CONFIDENCE_SCALE)
            base = self._last_good_conf if self._last_good_conf > 0 else confidence
            confidence = base * scale

        # 8. Signal conditioning: gate the raw measurement BEFORE it ever
        #    reaches the Kalman filter -- confidence-based gating and
        #    outlier/implausible-jump rejection, distinct from (and
        #    upstream of) the filter's own noise smoothing. A rejection is
        #    treated exactly like a dropout (None, None) rather than
        #    letting a single bad frame's innovation partially corrupt the
        #    tracked state.
        condition_result = self.conditioner.check(raw_cte_m, raw_heading_deg, confidence, valid, now_s)
        # cte_for_filter/heading_for_filter (not raw_cte_m/raw_heading_deg)
        # feed the tracker -- raw_cte_m/raw_heading_deg stay as the true raw
        # geometric reading for diagnostics/logging even on a rejected frame.
        cte_for_filter, heading_for_filter = raw_cte_m, raw_heading_deg
        if not condition_result["accept"]:
            cte_for_filter, heading_for_filter = None, None
        # Turn-preserving feed (refinement C): when BOTH sides are a frozen hold
        # there is no fresh directional information this frame -- feeding the
        # tracker the held curves' constant heading would correct its tracked
        # heading-rate back toward zero and FLATTEN a turn it was mid-way through
        # extrapolating. Treat a fully-held frame as a dropout so the Kalman keeps
        # carrying the bend on its own rate (its designed predict-through-gap
        # behavior), until it goes stale or a real detection returns.
        if both_frozen_held:
            cte_for_filter, heading_for_filter = None, None

        # 9. Ego-state bridge (Design A): compute this tick's IMU-derived
        #    heading delta -- a no-op (have_imu_yaw_rate=False) on this dev
        #    machine's NullImuSource, since no real IMU is attached.
        ego_delta = self.ego_state.predict_delta(now_s, heading_error_rad=np.radians(self._last_tracked_heading_deg))
        imu_heading_rate_dps = ego_delta["yaw_rate_dps"] if ego_delta["have_imu_yaw_rate"] else None
        # Vehicle-dynamics inputs the controller now consumes directly:
        #   speed_mps    -- measured (IMU/wheel) speed, or the configured
        #                   fallback when no reading exists; scales Stanley's
        #                   cross-track term (see steering_v2.py).
        #   yaw_rate_dps -- measured gyro yaw rate, or None (NullImuSource);
        #                   feeds Stanley's damping term AND, downstream, the
        #                   safety monitor's kinematic plausibility check.
        # Both fall back so this is a no-op on the current IMU-less dev
        # machine and starts informing control the moment a real IMU is
        # plugged in -- same Design-A contract as the Kalman bridge above.
        speed_mps = ego_delta["speed_mps"]
        yaw_rate_dps = imu_heading_rate_dps

        # 10. Temporal tracking (Kalman) -- smooths the physical state
        #    rather than the already-nonlinear-combined steering angle,
        #    extrapolates sensibly through brief dropouts, and -- when a
        #    real IMU is attached -- lets a physical yaw-rate reading
        #    override the filter's own vision-derived rate during that
        #    blind prediction (the actual Design A bridge).
        tracked = self.tracker.step(cte_for_filter, heading_for_filter, now_s,
                                     imu_heading_rate_dps=imu_heading_rate_dps)
        self._last_tracked_heading_deg = tracked["heading_deg"]
        if tracked["stale"]:
            # Sustained track loss (not just one noisy frame): stop
            # letting the next frame's continuity search anchor on a
            # prediction that's had no real correction in a while, and
            # fall back to a fresh cold-start search instead.
            self.last_left, self.last_right = None, None
            # Also drop the coast timers so a post-stale recovery reacquires
            # genuinely cold, rather than resuming a coast against pre-loss
            # history that no longer describes the road.
            self._left_good_t = self._right_good_t = None
            # Reset the display smoother + label hysteresis too, so a post-stale
            # reacquisition snaps in visually instead of blending from pre-loss
            # geometry / lingering a stale colour.
            self._smoothed_left = self._smoothed_right = None
            self._disp_left_source = self._disp_right_source = "none"
            self._pend_left_source = self._pend_right_source = "none"
            self._pend_left_count = self._pend_right_count = 0

        # 11. Centerline generation is implicit in (cte, heading); 12.
        #     Steering controller (Stanley) + slew-rate-limited output.
        if tracked["initialized"] and not tracked["stale"]:
            raw_steering_deg = stanley_steering_deg(tracked["cte_m"], tracked["heading_deg"],
                                                    speed_mps, yaw_rate_dps=yaw_rate_dps)
            self._last_trusted_raw_deg = raw_steering_deg
            steering_deg = self.slew_limiter.limit(raw_steering_deg, now_s)
        elif tracked["initialized"]:
            # Stale: don't trust the ever-extrapolating state for control
            # -- hold the last trusted command instead (still slew-
            # limited, so a return to a real detection can't cause a
            # sudden jump either).
            raw_steering_deg = self._last_trusted_raw_deg
            steering_deg = self.slew_limiter.limit(raw_steering_deg, now_s)
        else:
            raw_steering_deg = 0.0
            steering_deg = 0.0

        # 13. DISPLAY-ONLY post-processing (never touches control, which already
        #     ran on the raw fit above): EMA-smooth the curves shown in the
        #     overlay to stop frame-to-frame flicker, and apply hysteresis to the
        #     displayed source labels so the overlay colour doesn't strobe.
        disp_left = self._smooth_curve(self._smoothed_left, left, dt)
        disp_right = self._smooth_curve(self._smoothed_right, right, dt)
        self._smoothed_left, self._smoothed_right = disp_left, disp_right
        (self._disp_left_source, self._pend_left_source, self._pend_left_count) = \
            self._stabilize_label(self._disp_left_source, self._pend_left_source,
                                  self._pend_left_count, left_source)
        (self._disp_right_source, self._pend_right_source, self._pend_right_count) = \
            self._stabilize_label(self._disp_right_source, self._pend_right_source,
                                  self._pend_right_count, right_source)

        return {
            "warped_frame": warped_frame, "warped_mask": warped_mask,
            "left": disp_left, "right": disp_right,
            "left_source": self._disp_left_source, "right_source": self._disp_right_source,
            "left_gate_reason": left_gate_reason, "right_gate_reason": right_gate_reason,
            "valid": valid, "confidence": confidence,
            "cross_track_error_raw_m": raw_cte_m, "heading_raw_deg": raw_heading_deg,
            "cross_track_error_m": tracked["cte_m"], "heading_deg": tracked["heading_deg"],
            "tracker_initialized": tracked["initialized"], "tracker_stale": tracked["stale"],
            "steering_deg_raw": raw_steering_deg, "steering_deg": steering_deg,
            "speed_mps": speed_mps, "yaw_rate_dps": yaw_rate_dps,
            "ipm_adaptive": self.adaptive_zoom,
            "ipm_zoom_step": self._zoom_step, "ipm_zoom_locked": self._zoom_locked,
            "night_mode": self.night_mode, "lane_prob_threshold": self.lane_prob_threshold,
            "split_debug": split_debug, "ego_delta": ego_delta,
            "condition_result": condition_result,
        }
