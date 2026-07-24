"""
bird_eye_extraction.py
-------------------------
Replaces the classical "sliding window search" stage: instead of a column-
histogram + moving-window search over a thresholded binary bird's-eye
image, the model's own per-pixel lane-line probability -- already warped to
bird's-eye alongside the color frame (see ipm.warp_to_birdeye) -- IS the
search result. This module turns that warped probability map into left/
right boundary point sets, using the same cascading-quality-gate
philosophy as boundary_extraction.py's perspective-space version (reach-
the-vehicle-edge, long-enough, white-enough), but now measured in bird's-
eye space, where a row/column of pixels corresponds to a roughly uniform
real-world distance everywhere in the warped canvas -- unlike perspective
space, where the same gate thresholds mean different physical distances
depending on how far a row is from the camera.

Also implements lane-continuity-aware selection: when a previous frame's
curve is available (see lane_continuity.py), each side searches for the
candidate closest to where THAT side's curve predicts it should be, rather
than re-partitioning everything by a fixed image-center every frame. Falls
back to the center-based partition (cold start) when no prediction exists.
"""
import numpy as np
import cv2

from perception.lane_core.color_filter import whiteness_fraction, MIN_WHITE_FRACTION
from perception.lane_core.lane_continuity import predicted_position

MIN_BLOB_AREA = 40            # smaller than perspective-space's 120: the warped canvas
                              # is a fixed IPM_WARPED_SIZE regardless of source resolution
MIN_BOTTOM_REACH_FRAC = 0.35  # a blob must reach at least this far towards the vehicle
                              # edge of the warped canvas (large-y side, see ipm.py)
BOTTOM_ANCHOR_ROWS = 15
MIN_ROW_SPAN_FRAC = 0.20


def extract_birdeye_boundaries(warped_mask: np.ndarray, warped_prob: np.ndarray,
                                warped_frame_bgr: np.ndarray,
                                prev_left=None, prev_right=None, return_debug: bool = False):
    """
    warped_mask: binary uint8 mask in bird's-eye space (warped_prob > 0.5)
    warped_prob: float probability map in bird's-eye space, same shape
    warped_frame_bgr: the color frame warped into the same bird's-eye
        canvas, used only for the whiteness/color check
    prev_left / prev_right: previous frame's LaneCurve, ALREADY expressed in
        this same warped coordinate system, if available -- see
        lane_continuity.py
    """
    h, w = warped_mask.shape[:2]
    center_x = w / 2.0
    min_y_max = h * (1.0 - MIN_BOTTOM_REACH_FRAC)
    min_row_span = h * MIN_ROW_SPAN_FRAC
    hsv = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2HSV)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(warped_mask, connectivity=8)
    candidates = []
    debug = {"labels": labels, "blobs": [], "chosen_left": None, "chosen_right": None} if return_debug else None

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        kept = area >= MIN_BLOB_AREA
        if return_debug:
            debug["blobs"].append({
                "label": i, "centroid_x": float(centroids[i][0]), "area": area, "kept": kept,
                "bbox": (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
            })
        if not kept:
            continue
        ys, xs = np.where(labels == i)
        y_min, y_max = ys.min(), ys.max()
        reaches_bottom = y_max >= min_y_max
        long_enough = (y_max - y_min) >= min_row_span
        white_enough = whiteness_fraction(hsv, ys, xs) >= MIN_WHITE_FRACTION
        near_rows = ys >= (y_max - BOTTOM_ANCHOR_ROWS)
        bottom_x = float(xs[near_rows].mean())
        near_y = float(y_max)
        candidates.append({"bottom_x": bottom_x, "near_y": near_y, "ys": ys, "xs": xs,
                            "conf": warped_prob[ys, xs], "label": i,
                            "reaches_bottom": reaches_bottom, "long_enough": long_enough,
                            "white_enough": white_enough})

    empty = (np.array([]), np.array([]), np.array([]))

    def finish(left_res, right_res):
        return (left_res, right_res, debug) if return_debug else (left_res, right_res)

    if not candidates:
        return finish(empty, empty)

    def select_side(pool, prev_curve, fallback_is_left):
        # Warm path: a previous curve exists -- search for the candidate
        # closest to where IT predicts this side should be, cascading
        # through the same quality tiers (best tier first).
        if prev_curve is not None:
            def dist(c):
                predicted = predicted_position(prev_curve, c["near_y"])
                return abs(c["bottom_x"] - predicted)

            for tier_test in (lambda c: c["reaches_bottom"] and c["long_enough"] and c["white_enough"],
                              lambda c: c["reaches_bottom"] and c["long_enough"],
                              lambda c: c["reaches_bottom"],
                              lambda c: True):
                tier = [c for c in pool if tier_test(c)]
                if tier:
                    return min(tier, key=dist)
            return None

        # Cold-start path: no prediction yet -- partition by image center,
        # same as the original center-based approach.
        side_pool = [c for c in pool if (c["bottom_x"] < center_x) == fallback_is_left]
        pick = max if fallback_is_left else min
        for tier_test in (lambda c: c["reaches_bottom"] and c["long_enough"] and c["white_enough"],
                          lambda c: c["reaches_bottom"] and c["long_enough"],
                          lambda c: c["reaches_bottom"],
                          lambda c: True):
            tier = [c for c in side_pool if tier_test(c)]
            if tier:
                return pick(tier, key=lambda c: c["bottom_x"])
        return None

    def pick_pair(use_continuity):
        # Left is picked from the full candidate pool; right is picked from
        # whatever's left afterwards, so the (rare, but possible in the
        # warm/continuity path where both sides search the same full pool)
        # case of both predictions converging on the same blob can't
        # silently assign one detection to both boundaries.
        lp = select_side(candidates, prev_left if use_continuity else None, fallback_is_left=True)
        rem = [c for c in candidates if lp is None or c["label"] != lp["label"]]
        rp = select_side(rem, prev_right if use_continuity else None, fallback_is_left=False)
        return lp, rp

    left_pick, right_pick = pick_pair(use_continuity=True)
    # Sanity check: a real left/right pair must have the right boundary
    # actually to the right of the left one. If continuity-based search
    # (scoring by distance to last frame's predicted position) produces a
    # crossed pair, the prediction itself has drifted off the real lane --
    # confirmed in practice, where one bad fit's prediction let tracking
    # lock onto an increasingly wrong detection for 200+ consecutive
    # frames, since nothing was pulling it back towards a fresh read. Fall
    # back to a cold-start (center-based) search on this SAME frame's
    # candidates instead of ever committing to a crossed result, so a
    # single bad frame can't start that self-reinforcing drift.
    if (left_pick is not None and right_pick is not None
            and right_pick["bottom_x"] <= left_pick["bottom_x"]):
        left_pick, right_pick = pick_pair(use_continuity=False)

    left_res = empty
    if left_pick is not None:
        left_res = (left_pick["ys"], left_pick["xs"], left_pick["conf"])
        if return_debug:
            debug["chosen_left"] = left_pick["label"]

    right_res = empty
    if right_pick is not None:
        right_res = (right_pick["ys"], right_pick["xs"], right_pick["conf"])
        if return_debug:
            debug["chosen_right"] = right_pick["label"]

    return finish(left_res, right_res)
