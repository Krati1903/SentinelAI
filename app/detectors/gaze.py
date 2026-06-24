"""
Gaze estimation using iris position relative to the eye corners.
Falls back gracefully to CENTER/SCREEN if iris landmarks are unavailable
(e.g. refine_landmarks not producing iris points on a given frame).
"""

import numpy as np
from app.logger import logger
from app import config
from app.detectors.face_mesh_utils import (
    LEFT_EYE_RING, LEFT_IRIS, RIGHT_EYE_RING, RIGHT_IRIS,
)


def _eye_gaze_ratio(landmarks, eye_ring_ids, iris_ids):
    eye_points = landmarks[eye_ring_ids][:, :2]
    iris_points = landmarks[iris_ids][:, :2]

    x_min, x_max = eye_points[:, 0].min(), eye_points[:, 0].max()
    y_min, y_max = eye_points[:, 1].min(), eye_points[:, 1].max()

    iris_center = iris_points.mean(axis=0)

    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)

    horizontal_ratio = (iris_center[0] - x_min) / width
    vertical_ratio = (iris_center[1] - y_min) / height

    return horizontal_ratio, vertical_ratio


def estimate_gaze(landmarks_normalized: np.ndarray) -> str:
    """
    landmarks_normalized: Nx3 array of normalized landmarks for one face
    (must include refined iris landmarks, indices 468-477).
    Returns one of SCREEN, LEFT, RIGHT, UP, DOWN.
    """
    try:
        if landmarks_normalized.shape[0] < 478:
            return "SCREEN"

        l_h, l_v = _eye_gaze_ratio(landmarks_normalized, LEFT_EYE_RING, LEFT_IRIS)
        r_h, r_v = _eye_gaze_ratio(landmarks_normalized, RIGHT_EYE_RING, RIGHT_IRIS)

        avg_h = (l_h + r_h) / 2
        avg_v = (l_v + r_v) / 2

        if avg_h <= config.GAZE_LEFT_RATIO:
            return "LEFT"
        if avg_h >= config.GAZE_RIGHT_RATIO:
            return "RIGHT"
        if avg_v <= config.GAZE_VERTICAL_UP_RATIO:
            return "UP"
        if avg_v >= config.GAZE_VERTICAL_DOWN_RATIO:
            return "DOWN"
        return "SCREEN"

    except Exception as exc:
        logger.error(f"Gaze estimation failed: {exc}")
        return "SCREEN"
