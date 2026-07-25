#!/usr/bin/env python3
"""
single_lane_stereo_recovery.py
===============================
Robust CTE/heading recovery for the ONE-LANE-VISIBLE failure mode (typically
mid-sharp-turn, when the far boundary leaves the frame or the bird's-eye
warp for it becomes numerically unstable).

WHY THIS EXISTS
---------------
The existing pipeline's "reconstructed" side (see SOURCE_COLOURS in
run_video_annotated.py) rebuilds a missing boundary purely in IMAGE /
bird's-eye space using a width model. That's fine on gentle curves, but on
a sharp 10-20 deg turn the IPM homography's small-error sensitivity is
worst exactly where the far boundary has already left the frame -- so the
reconstruction is least trustworthy exactly when you need it most.

This module replaces that image-space guess with REAL-WORLD geometry:
  1. Sample points along the ONE boundary that IS detected.
  2. Get each point's true 3D position via ZED stereo depth (not a pixel fit).
  3. Fit the boundary's LOCAL shape through those 3D ground-plane points --
     a QUADRATIC X = a*Z^2 + b*Z + c, not a straight line (see below).
  4. Read the boundary's position (c) and tangent (b) AT THE VEHICLE (Z=0),
     step perpendicular by lane_width/2 (from the rulebook) to the lane
     CENTER, and report CTE + heading there. Curvature (2a) falls out too.

WHY A QUADRATIC, AND WHY EVALUATED AT Z=0  (the sharp-turn fix)
--------------------------------------------------------------
The earlier version fit a straight line X = m*Z + b and then EXTRAPOLATED it
from the median sample distance back to the vehicle. Two failures on a real
sharp turn (the case this module exists for):

  * A genuine turn boundary is CURVED, so a straight-line fit has a large
    residual -> it trips the rms gate -> recover_single_lane() returns None
    -> the caller falls back to the very image-space guess this module was
    built to replace. The recovery switched ITSELF OFF on the sharpest turns.

  * Extrapolating a single global slope from ~4-6 m out back to Z=0 is a long
    lever arm: a small slope error is multiplied by the reference distance,
    and the global slope is the chord-average over the arc, not the tangent
    the wheels actually need NOW.

The quadratic fixes both: it FITS a curve (a clean arc passes the gate), and
its constant/linear terms ARE the position and tangent at the vehicle, so CTE
and heading are read locally with no long extrapolation. Samples are weighted
toward the near field (1/Z^2), because the at-vehicle terms are what we steer
on and near depth is the most reliable. It degrades gracefully to the linear
fit when there are too few points / too little forward span for a stable
curvature estimate, and still returns None (caller keeps its own value) when
the depth is genuinely inconsistent.

WORKED NUMERIC EXAMPLE (see also self-test at the bottom of this file)
------------------------------------------------------------------------
Flat (straight) LEFT boundary passing X=-1.0 m at Z=0 with a mild slope
b=0.05: rulebook lane width W=3.0 -> half=1.5. At the vehicle the lane centre
sits half a width to the RIGHT along the local normal:
    CTE = c + half*sqrt(1+b^2) = -1.0 + 1.5*sqrt(1.0025) = +0.502 m
    heading = atan(b) = +2.86 deg,  curvature = 2a = 0.0 (straight)
On a real arc a != 0 and the same formulas read the turn's near-field state
instead of a chord average -- the self-test demonstrates a case the old
straight-line gate would have rejected outright.

COORDINATE CONVENTION
----------------------
Camera/vehicle frame, meters: X = lateral (+ = right), Y = down, Z = forward
(vehicle's own forward axis == camera's optical axis; assumes a
roughly-level, roughly-forward ZED mount -- if yours is pitched down
noticeably, extend backproject_pixel() with a rotation before use).

Pixel space for depth lookups is the ORIGINAL (un-warped) camera image --
i.e. the same frame you pass to node.process(), post stereo-eye crop,
pre-IPM. Bird's-eye curve points are mapped back into that space with the
SAME node.ipm_M_inv your run_video_annotated.py already uses for overlay
drawing, so no new calibration path is needed for that half of the job.
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Camera intrinsics
# --------------------------------------------------------------------------
@dataclass
class StereoIntrinsics:
    """Pinhole intrinsics for the LEFT eye, AT THE RESOLUTION the pipeline
    actually runs at (i.e. after crop_stereo_eye(), before any YOLOP-internal
    resize). Get these from the ZED SDK's own calibration
    (zed.get_camera_information().camera_configuration.calibration_parameters
    .left_cam -> fx, fy, cx, cy) -- do not hand-guess them; a wrong fx/fy
    biases every distance this module computes."""
    fx: float
    fy: float
    cx: float
    cy: float


def scale_intrinsics(intr: StereoIntrinsics, calib_wh, frame_wh) -> StereoIntrinsics:
    """Scale intrinsics if the frame fed to the pipeline is a resized/cropped
    version of the resolution the calibration was taken at. calib_wh /
    frame_wh are (width, height) tuples."""
    sx = frame_wh[0] / calib_wh[0]
    sy = frame_wh[1] / calib_wh[1]
    return StereoIntrinsics(fx=intr.fx * sx, fy=intr.fy * sy, cx=intr.cx * sx, cy=intr.cy * sy)


# --------------------------------------------------------------------------
# Depth providers -- swap the source of depth without touching the geometry
# --------------------------------------------------------------------------
class DepthProvider:
    """depth_at(u, v) -> meters, or None if invalid/out of range."""
    def depth_at(self, u: float, v: float) -> Optional[float]:
        raise NotImplementedError


class ZedDepthProvider(DepthProvider):
    """On-vehicle path: wraps the ZED SDK's own hardware-accelerated depth
    (neural depth on a ZED 2i is far cleaner than DIY block matching).
    Requires pyzed.sl and a live camera opened via the SDK -- NOT the plain
    cv2.VideoCapture path your script currently uses for --source, since
    OpenCV's UVC read gives you raw stereo pixels but no depth. See the
    --zed-depth wiring note in the forked run_video_annotated.py.
    """
    def __init__(self, zed_camera):
        import pyzed.sl as sl
        self._sl = sl
        self.zed = zed_camera
        self._depth_mat = sl.Mat()

    def refresh(self):
        """Call once per frame, right after zed.grab(), before any depth_at() calls."""
        self.zed.retrieve_measure(self._depth_mat, self._sl.MEASURE.DEPTH)

    def depth_at(self, u, v):
        err, z = self._depth_mat.get_value(int(round(u)), int(round(v)))
        if err != self._sl.ERROR_CODE.SUCCESS or not np.isfinite(z):
            return None
        return float(z)


class StereoBMDepthProvider(DepthProvider):
    """Bench/offline fallback ONLY: classical block-matching depth from a
    LEFT+RIGHT crop pair (e.g. replaying a recorded side-by-side file like
    d_dash.mp4, where there's no live ZED SDK to ask). Noisier than the ZED
    SDK's own depth -- and on low-texture chalk-on-asphalt it returns few
    valid disparities, so expect min_points to gate many frames out. Use this
    to unit-test the recovery math, not as the on-vehicle depth source; on the
    vehicle use ZedDepthProvider.
    """
    def __init__(self, baseline_m: float, fx: float, num_disp=128, block_size=9):
        self.baseline_m = baseline_m
        self.fx = fx
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block_size,
            P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
        self._disp = None

    def update(self, left_gray, right_gray):
        """Call once per frame with grayscale left/right crops (same width/height)."""
        self._disp = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    def depth_at(self, u, v):
        if self._disp is None:
            return None
        u_i, v_i = int(round(u)), int(round(v))
        if not (0 <= v_i < self._disp.shape[0] and 0 <= u_i < self._disp.shape[1]):
            return None
        d = float(self._disp[v_i, u_i])
        if d <= 0.5:   # invalid / too-far / occluded disparity
            return None
        return self.baseline_m * self.fx / d


# --------------------------------------------------------------------------
# Core geometry
# --------------------------------------------------------------------------
def backproject_pixel(u: float, v: float, z_m: float, intr: StereoIntrinsics):
    """Pixel + depth -> camera-frame (X, Y, Z) meters."""
    X = (u - intr.cx) * z_m / intr.fx
    Y = (v - intr.cy) * z_m / intr.fy
    return X, Y, z_m


def sample_curve_points_xz(curve, ipm_M_inv, depth_provider: DepthProvider,
                            intr: StereoIntrinsics, n_samples: int = 12,
                            depth_min_m: float = 0.5, depth_max_m: float = 20.0):
    """Sample n_samples points along a bird's-eye LaneCurve (the SAME curve
    object your pipeline already fit for the visible side), map them back to
    original-image pixels with ipm_M_inv (node.ipm_M_inv), look up stereo
    depth at each, and return the valid ones as an (N, 2) array of ground-
    plane (X, Z) meters -- plus the (u, v) pixels used, for HUD/overlay
    debugging.
    """
    ys = np.linspace(curve.y_min, curve.y_max, n_samples)
    xs = curve.x_at(ys)
    pts_bev = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    pts_img = cv2.perspectiveTransform(pts_bev, ipm_M_inv).reshape(-1, 2)

    xz_pts, used_px = [], []
    for (u, v) in pts_img:
        z = depth_provider.depth_at(float(u), float(v))
        if z is None or not np.isfinite(z) or not (depth_min_m <= z <= depth_max_m):
            continue
        X, _Y, Z = backproject_pixel(float(u), float(v), z, intr)
        xz_pts.append((X, Z))
        used_px.append((float(u), float(v)))
    xz = np.array(xz_pts, dtype=np.float64) if xz_pts else np.zeros((0, 2))
    return xz, used_px


def fit_ground_curve(points_xz: np.ndarray, weights: Optional[np.ndarray] = None,
                     allow_quadratic: bool = True, min_z_span_m: float = 1.5):
    """Weighted least-squares fit of the detected boundary in the ground plane.

    Tries a QUADRATIC X = a*Z^2 + b*Z + c (captures a real turn) and falls back
    to a straight line X = b*Z + c (a == 0) only when there are too few points
    or too little forward (Z) span to estimate curvature stably. `weights`
    (per-point) let the caller lean the fit on the near field, whose at-vehicle
    terms drive steering; None = equal weights.

    Returns (a, b, c, rms) where rms is the WEIGHTED residual (meters) -- it
    judges the fit where the weight actually is, so a clean near field with a
    noisy far tail is not wrongly rejected. b is dX/dZ and c is X, BOTH AT
    Z=0, i.e. at the vehicle."""
    Z = np.asarray(points_xz[:, 1], dtype=np.float64)
    X = np.asarray(points_xz[:, 0], dtype=np.float64)
    n = len(X)
    if n < 2:
        return 0.0, 0.0, 0.0, float("inf")
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 0.0, None))

    z_span = float(Z.max() - Z.min())
    use_quad = allow_quadratic and n >= 3 and z_span >= min_z_span_m
    if use_quad:
        A = np.vstack([Z ** 2, Z, np.ones_like(Z)]).T
        coef, *_ = np.linalg.lstsq(A * w[:, None], X * w, rcond=None)
        a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    else:
        A = np.vstack([Z, np.ones_like(Z)]).T
        coef, *_ = np.linalg.lstsq(A * w[:, None], X * w, rcond=None)
        a, b, c = 0.0, float(coef[0]), float(coef[1])

    pred = a * Z ** 2 + b * Z + c
    wsum = float(np.sum(w ** 2))
    rms = float(np.sqrt(np.sum((w ** 2) * (pred - X) ** 2) / wsum)) if wsum > 0 else float("inf")
    return a, b, c, rms


def fit_ground_line(points_xz: np.ndarray):
    """Backward-compatible straight-line fit X = m*Z + b (rms). Kept for callers
    and tests that want the plain linear model; the recovery path uses
    fit_ground_curve()."""
    a, b, c, rms = fit_ground_curve(points_xz, allow_quadratic=False)
    return b, c, rms


@dataclass
class SingleLaneRecovery:
    cte_m: float
    heading_deg: float
    m_slope: float            # dX/dZ at the vehicle (tan of local heading), == b
    b_intercept: float        # boundary X at the vehicle (Z=0), == c
    n_points: int
    rms_residual_m: float
    side_used: str
    curvature_1pm: float = 0.0   # 2a: boundary d2X/dZ2 at the vehicle (signed);
                                  # exposed for an optional Stanley feed-forward,
                                  # not yet wired into the steering law.
    used_px: list = field(default_factory=list)   # for debug overlay


def recover_single_lane(pf, ipm_M_inv, side: str, depth_provider: DepthProvider,
                        intr: StereoIntrinsics, lane_width_m: float,
                        n_samples: int = 12, min_points: int = 4,
                        max_rms_m: float = 0.35, cte_sign: float = 1.0,
                        heading_sign: float = 1.0,
                        min_z_span_m: float = 1.5) -> Optional[SingleLaneRecovery]:
    """
    pf:   the perception-frame dict from node.process() -- must have pf[side]
          be a LaneCurve (not None) and pf[other_side] be None (single-lane
          case; caller checks this before calling).
    side: 'left' or 'right' -- which boundary IS detected this frame.

    Returns None (caller should keep whatever CTE/heading the existing
    pipeline already produced) when there isn't enough consistent stereo
    depth to trust the result -- this function never returns a low-
    confidence guess, only a good one or nothing.
    """
    assert side in ("left", "right")
    curve = pf.get(side)
    if curve is None:
        return None

    xz, used_px = sample_curve_points_xz(curve, ipm_M_inv, depth_provider, intr, n_samples=n_samples)
    if len(xz) < min_points:
        return None

    # Lean the fit on the near field: closer depth is more reliable AND the
    # at-vehicle (Z=0) position/tangent are what actually get steered on. A
    # curving-away far tail then can't drag the near-field terms around.
    z = np.clip(xz[:, 1], 0.5, None)
    weights = 1.0 / (z ** 2)

    a, b, c, rms = fit_ground_curve(xz, weights=weights, min_z_span_m=min_z_span_m)
    if not np.isfinite(rms) or rms > max_rms_m:
        # Depth samples don't lie on a consistent (even curved) boundary --
        # e.g. a bad stereo return, dust/glare, or a moving occluder crossing
        # the boundary. Don't trust it; let the caller fall back.
        return None

    # Read the boundary AT THE VEHICLE (Z=0): position c, tangent slope b.
    # The lane centre is half a lane-width along the inward normal; carried to
    # Z=0 this is c +/- half*sqrt(1+b^2) (+ for a LEFT boundary, whose centre is
    # to the vehicle's right; - for a RIGHT boundary). No long extrapolation --
    # c and b already describe the near field.
    half_w = lane_width_m / 2.0
    offset = half_w * float(np.hypot(b, 1.0))   # half-width projected to the Z=0 lateral axis
    center_x0 = c + offset if side == "left" else c - offset

    cte_m = cte_sign * center_x0
    heading_deg = heading_sign * float(np.degrees(np.arctan(b)))
    curvature_1pm = 2.0 * a   # d2X/dZ2 at the vehicle

    return SingleLaneRecovery(cte_m=cte_m, heading_deg=heading_deg, m_slope=b,
                              b_intercept=c, n_points=len(xz), rms_residual_m=rms,
                              side_used=side, curvature_1pm=curvature_1pm, used_px=used_px)


# --------------------------------------------------------------------------
# Self-test (run directly: python single_lane_stereo_recovery.py)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    def cte_heading_from_fit(a, b, c, side, half_w=1.5):
        offset = half_w * float(np.hypot(b, 1.0))
        cx0 = c + offset if side == "left" else c - offset
        return cx0, float(np.degrees(np.arctan(b))), 2.0 * a

    # ---- 1) flat-boundary sanity check (reduces to the old closed form) ----
    zs = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    b_true, c_true = 0.05, -1.0
    xz_flat = np.stack([b_true * zs + c_true, zs], axis=1)
    a, b, c, rms = fit_ground_curve(xz_flat)
    cte, hd, k = cte_heading_from_fit(a, b, c, "left")
    print("[flat]  fit a=%.4f b=%.4f c=%.4f rms=%.4f m" % (a, b, c, rms))
    print("[flat]  CTE=%+.3f m  heading=%+.2f deg  curvature=%+.4f 1/m" % (cte, hd, k))
    print("        (expect CTE ~ c + 1.5*sqrt(1+b^2) = %+.3f, curvature ~ 0)"
          % (c_true + 1.5 * np.hypot(b_true, 1.0)))
    assert abs(cte - (c_true + 1.5 * np.hypot(b_true, 1.0))) < 1e-6

    # ---- 2) SHARP TURN: the case the old straight-line gate rejected -------
    # A curved LEFT boundary of a turn: X = a*Z^2 + b*Z + c, sampled 1..8 m.
    a_t, b_t, c_t = 0.08, -0.15, -1.1
    zs = np.linspace(1.0, 8.0, 8)
    xs = a_t * zs ** 2 + b_t * zs + c_t
    xz_turn = np.stack([xs, zs], axis=1)

    m_lin, b_lin, rms_lin = fit_ground_line(xz_turn)        # OLD model
    a_q, b_q, c_q, rms_q = fit_ground_curve(xz_turn)        # NEW model
    print("\n[turn]  straight-line fit rms = %.3f m  -> %s (gate 0.35 m)"
          % (rms_lin, "REJECTED" if rms_lin > 0.35 else "accepted"))
    print("[turn]  quadratic   fit rms = %.3f m  -> %s (gate 0.35 m)"
          % (rms_q, "REJECTED" if rms_q > 0.35 else "ACCEPTED"))
    cte_q, hd_q, k_q = cte_heading_from_fit(a_q, b_q, c_q, "left")
    print("[turn]  NEW recovery -> CTE=%+.3f m  heading=%+.2f deg  curvature=%+.4f 1/m"
          % (cte_q, hd_q, k_q))
    assert rms_lin > 0.35 > rms_q, "expected the straight-line fit to fail the gate and the quadratic to pass"
    assert abs(c_q - c_t) < 1e-6 and abs(b_q - b_t) < 1e-6, "quadratic must recover the true near-field terms"
    print("\nOK: the sharp turn that the old straight-line recovery would have")
    print("    thrown away is now accepted, with CTE/heading read at the vehicle.")
