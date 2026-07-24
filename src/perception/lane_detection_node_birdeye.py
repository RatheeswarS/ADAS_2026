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
from perception.lane_core.bird_eye_extraction import extract_birdeye_boundaries
from perception.lane_core.kalman_tracker import LaneKalmanTracker
from perception.lane_core.steering_v2 import stanley_steering_deg, SlewRateLimiter
from perception.lane_core.model_runtime import load_model, run_inference
from ego_state.ego_state_estimator import EgoStateEstimator
from ego_state.imu_source import ImuSource
from signal_conditioning.conditioner import SignalConditioner

RECONSTRUCTED_CONFIDENCE_SCALE = 0.5
MIN_WIDTH_FRAC_OF_CANVAS = 0.02  # reject a near-zero-width warped lane read as noise


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
        self.half = device.type != "cpu"
        self.model = load_model(weights_path, device)
        if self.half:
            self.model.half()
        if device.type != "cpu":
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
        left_source = "direct" if left is not None else "none"
        right_source = "direct" if right is not None else "none"

        if left is not None and right is not None:
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

        if left is None and right is None:
            left, right = self.last_left, self.last_right
            left_source = right_source = "held"

        valid = left is not None and right is not None
        raw_cte_m: Optional[float] = None
        raw_heading_deg: Optional[float] = None
        if valid:
            bottom_y = wh - 1
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

        confidence = ((left.confidence + right.confidence) / 2.0) if valid else 0.0

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

        return {
            "warped_frame": warped_frame, "warped_mask": warped_mask,
            "left": left, "right": right,
            "left_source": left_source, "right_source": right_source,
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
