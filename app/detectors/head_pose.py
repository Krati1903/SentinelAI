"""
Head pose estimation using solvePnP, not a simple nose-x comparison.
Returns yaw/pitch/roll in degrees plus a discrete label.
"""

import numpy as np
import cv2

from app.logger import logger
from app import config
from app.detectors.face_mesh_utils import PNP_LANDMARK_IDS, PNP_3D_MODEL_POINTS


def _rotation_matrix_to_euler_angles(rmat: np.ndarray):
    """Convert rotation matrix to yaw, pitch, roll in degrees."""
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = 0.0
        roll = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))

    return yaw, pitch, roll


def estimate_head_pose(landmarks_normalized: np.ndarray, frame_shape):
    """
    landmarks_normalized: Nx3 array of normalized (x, y, z) landmarks for one face.
    frame_shape: (h, w, channels) of the source frame.
    Returns dict with yaw, pitch, roll, label.
    """
    h, w = frame_shape[0], frame_shape[1]

    try:
        image_points = []
        for idx in PNP_LANDMARK_IDS:
            lm = landmarks_normalized[idx]
            image_points.append((lm[0] * w, lm[1] * h))
        image_points = np.array(image_points, dtype=np.float64)

        model_points = np.array(PNP_3D_MODEL_POINTS, dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "label": "CENTER"}

        rmat, _ = cv2.Rodrigues(rotation_vector)
        yaw, pitch, roll = _rotation_matrix_to_euler_angles(rmat)

        label = _classify_pose(yaw, pitch)

        return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll), "label": label}

    except Exception as exc:
        logger.error(f"Head pose estimation failed: {exc}")
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "label": "CENTER"}


def _classify_pose(yaw: float, pitch: float) -> str:
    if yaw <= config.HEAD_YAW_LEFT_THRESHOLD:
        return "LEFT"
    if yaw >= config.HEAD_YAW_RIGHT_THRESHOLD:
        return "RIGHT"
    if pitch <= config.HEAD_PITCH_UP_THRESHOLD:
        return "UP"
    if pitch >= config.HEAD_PITCH_DOWN_THRESHOLD:
        return "DOWN"
    return "CENTER"
