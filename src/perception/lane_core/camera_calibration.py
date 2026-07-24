"""
camera_calibration.py
------------------------
Undistort hook: applies cv2.remap using a real calibration (camera matrix +
distortion coefficients) if one has been computed and provided, otherwise
passes the frame through unchanged.

No physical checkerboard calibration has been performed against the real
camera yet -- this is a genuine placeholder, not a stand-in for something
already working. Run a standard OpenCV chessboard calibration on the actual
hardware and load the result via `CameraCalibration.from_npz(...)` before
Phase 5; until then every frame passes through this stage untouched, which
is functionally identical to not having the stage at all, but keeps the
pipeline's structure ready for when real calibration data exists.
"""
from typing import Optional

import numpy as np
import cv2


class CameraCalibration:
    def __init__(self, camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self._map1 = None
        self._map2 = None
        self._map_size = None

    @property
    def is_calibrated(self) -> bool:
        return self.camera_matrix is not None and self.dist_coeffs is not None

    @classmethod
    def from_npz(cls, path: str) -> "CameraCalibration":
        """Expects an .npz with 'camera_matrix' (3x3) and 'dist_coeffs' arrays,
        as produced by a standard cv2.calibrateCamera() chessboard routine."""
        data = np.load(path)
        return cls(camera_matrix=data["camera_matrix"], dist_coeffs=data["dist_coeffs"])

    def undistort(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.is_calibrated:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if self._map1 is None or self._map_size != (w, h):
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.0)
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, new_mtx, (w, h), cv2.CV_16SC2)
            self._map_size = (w, h)
        return cv2.remap(frame_bgr, self._map1, self._map2, cv2.INTER_LINEAR)
