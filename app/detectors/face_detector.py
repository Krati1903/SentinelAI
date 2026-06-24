"""
Wraps MediaPipe Face Detection (counting faces) and Face Mesh
(landmark extraction only, never drawn on screen).
"""

import numpy as np
from app.logger import logger
from app.exceptions import ModelLoadError
from app import config

try:
    import mediapipe as mp
except ImportError as exc:
    raise ModelLoadError(f"mediapipe failed to import: {exc}")


class FaceDetector:
    def __init__(self):
        self.available = False
        self.face_detection = None
        self.face_mesh = None
        try:
            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=config.FACE_DETECTION_MIN_CONFIDENCE,
            )
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=config.MAX_NUM_FACES,
                refine_landmarks=True,
                min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
            )
            self.available = True
            logger.info("FaceDetector initialized (FaceDetection + FaceMesh)")
        except Exception as exc:
            logger.error(f"Failed to initialize MediaPipe FaceDetector: {exc}")
            self.available = False

    def detect_faces(self, frame_bgr: np.ndarray):
        """Returns list of face bounding boxes (x, y, w, h) in pixel coords."""
        if not self.available or self.face_detection is None:
            return []
        try:
            rgb = frame_bgr[:, :, ::-1]
            results = self.face_detection.process(rgb)
            boxes = []
            if results.detections:
                h, w, _ = frame_bgr.shape
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    boxes.append((x, y, bw, bh))
            return boxes
        except Exception as exc:
            logger.error(f"Face detection failed on frame: {exc}")
            return []

    def get_landmarks(self, frame_bgr: np.ndarray):
        """
        Returns a list of landmark arrays (Nx3, normalized) for each detected
        face, one per face. Used internally only — never rendered.
        """
        if not self.available or self.face_mesh is None:
            return []
        try:
            rgb = frame_bgr[:, :, ::-1]
            results = self.face_mesh.process(rgb)
            faces_landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    points = np.array(
                        [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
                    )
                    faces_landmarks.append(points)
            return faces_landmarks
        except Exception as exc:
            logger.error(f"Face mesh landmark extraction failed: {exc}")
            return []
