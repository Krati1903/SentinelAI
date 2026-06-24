"""
Webcam helper. Not required by the API itself (frames come in via
base64 over HTTP) but provided for local testing / demo scripts.
"""

import cv2
from app.logger import logger
from app.exceptions import CameraError
from app import config


def open_camera(index: int = None):
    index = index if index is not None else config.CAMERA_INDEX
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise CameraError(f"Could not open camera at index {index}")

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception as exc:
        logger.warning(f"Autofocus disable not supported on this camera: {exc}")

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    except Exception as exc:
        logger.warning(f"Could not set frame resolution: {exc}")

    return cap


def read_frame(cap):
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    except Exception as exc:
        logger.error(f"Failed to read frame from camera: {exc}")
        return None
