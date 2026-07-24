"""
reference_generator.py
--------------------------
Formalizes the architecture's "Reference Path" block: the single, named
handoff point between perception (BirdEyeLaneDetectionNode, which already
does temporal filtering, signal conditioning, and Design A's ego-state
bridging internally -- see lane_detection_node_birdeye.py) and everything
downstream that consumes a lane-relative reference (Stanley steering,
the DBW actuator interface, the safety monitor).

This is Design A throughout: (e, psi_e) is the vehicle's position/heading
*relative to the currently-perceived lane*, never a globally-mapped
waypoint -- there is no GPS/HD-map path anywhere in this project for a
Design-B global reference to come from.

Deliberately thin: it does not re-implement fusion/conditioning/bridging
(the node already owns all of that) -- it packages the node's already-
fused output into one clearly-named, stable-shaped object so downstream
modules depend on THIS contract rather than reaching into the node's
internal result dict directly.
"""
from dataclasses import dataclass


@dataclass
class ReferenceState:
    cross_track_error_m: float   # e: signed lateral offset from lane center (+ = right of center)
    heading_error_deg: float     # psi_e: signed heading error relative to lane direction
    initialized: bool            # False until the tracker has ever locked onto a trusted measurement
    stale: bool                  # True once too many consecutive ticks passed with no trusted
                                  # measurement (see kalman_tracker.MAX_CONSECUTIVE_PREDICTS) -- the
                                  # signal downstream (DBW/safety) should treat as "stop trusting this"
    rejected_this_tick: bool     # True if THIS tick's raw measurement specifically failed signal
                                  # conditioning (low confidence / implausible jump / no detection) --
                                  # a single-frame flag, weaker than `stale`, useful for diagnostics
    confidence: float


class ReferenceGenerator:
    def update(self, perception_result: dict) -> ReferenceState:
        """perception_result is whatever BirdEyeLaneDetectionNode.process()
        returned this tick."""
        return ReferenceState(
            cross_track_error_m=perception_result["cross_track_error_m"],
            heading_error_deg=perception_result["heading_deg"],
            initialized=perception_result["tracker_initialized"],
            stale=perception_result["tracker_stale"],
            rejected_this_tick=not perception_result["condition_result"]["accept"],
            confidence=perception_result["confidence"],
        )
