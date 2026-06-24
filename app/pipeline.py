"""
Orchestrates a single frame through all detectors and produces the
final warning list + risk score. This is the core detection logic
called by the /frame/analyze endpoint.
"""

import time
import numpy as np

from app.logger import logger
from app import config
from app.detectors.face_detector import FaceDetector
from app.detectors.object_detector import ObjectDetector
from app.detectors.head_pose import estimate_head_pose
from app.detectors.gaze import estimate_gaze
from app.risk_engine import calculate_risk
from app.session_manager import Session

# Models are loaded once at import time and reused across requests.
try:
    face_detector = FaceDetector()
except Exception as exc:
    logger.error(f"FaceDetector failed to initialize entirely: {exc}")
    face_detector = None

try:
    object_detector = ObjectDetector()
except Exception as exc:
    logger.error(f"ObjectDetector failed to initialize entirely: {exc}")
    object_detector = None


WARNING_MESSAGES = {
    "FACE_NOT_VISIBLE": ("Face not visible for more than {}s", "HIGH"),
    "MULTIPLE_PERSON_DETECTED": ("Multiple people detected in frame", "CRITICAL"),
    "PHONE_DETECTED": ("Mobile phone detected", "HIGH"),
    "BOOK_DETECTED": ("Book or reading material detected", "MODERATE"),
    "LAPTOP_DETECTED": ("Additional laptop/screen detected", "MODERATE"),
    "LOOKING_AWAY": ("Candidate looking away from screen", "LOW"),
    "EXCESSIVE_HEAD_MOVEMENT": ("Excessive head movement detected", "MODERATE"),
}

OBJECT_WARNING_MAP = {
    "cell phone": "PHONE_DETECTED",
    "book": "BOOK_DETECTED",
    "laptop": "LAPTOP_DETECTED",
}


def _make_warning(code: str):
    message_template, severity = WARNING_MESSAGES[code]
    if code == "FACE_NOT_VISIBLE":
        message = message_template.format(config.FACE_MISSING_THRESHOLD_SECONDS)
    else:
        message = message_template
    return {"code": code, "message": message, "severity": severity}


def analyze_frame(frame_bgr: np.ndarray, session: Session) -> dict:
    """
    Runs the full detection pipeline on a single BGR frame and updates
    session state. Returns a dict ready to map onto FrameAnalyzeResponse.
    """
    start_time = time.perf_counter()
    warning_codes = []
    warnings = []

    # --- Face detection ---
    face_boxes = []
    landmarks_list = []
    if face_detector is not None and face_detector.available:
        face_boxes = face_detector.detect_faces(frame_bgr)
        landmarks_list = face_detector.get_landmarks(frame_bgr)

    face_count = len(face_boxes)

    if face_count == 0:
        elapsed_missing = time.time() - session.last_face_seen_at
        if elapsed_missing >= config.FACE_MISSING_THRESHOLD_SECONDS:
            warning_codes.append("FACE_NOT_VISIBLE")
    else:
        session.last_face_seen_at = time.time()

    # --- Object / person detection (YOLO) ---
    person_count = 0
    detected_objects = []
    if object_detector is not None and object_detector.available:
        yolo_result = object_detector.detect(frame_bgr)
        person_count = yolo_result["person_count"]
        detected_objects = yolo_result["objects"]

    # --- Multiple person logic: YOLO person count OR MediaPipe face count ---
    if person_count > 1 or face_count > 1:
        warning_codes.append("MULTIPLE_PERSON_DETECTED")

    # --- Object based warnings ---
    for obj_label in detected_objects:
        code = OBJECT_WARNING_MAP.get(obj_label)
        if code and code not in warning_codes:
            warning_codes.append(code)

    # --- Head pose + gaze (only meaningful with exactly one tracked face) ---
    head_pose_label = "CENTER"
    gaze_label = "SCREEN"

    if landmarks_list:
        primary_landmarks = landmarks_list[0]
        pose_result = estimate_head_pose(primary_landmarks, frame_bgr.shape)
        head_pose_label = pose_result["label"]

        gaze_label = estimate_gaze(primary_landmarks)

        session.movement_tracker.update(pose_result["yaw"], pose_result["pitch"])

        if head_pose_label != "CENTER" or gaze_label != "SCREEN":
            if "LOOKING_AWAY" not in warning_codes:
                warning_codes.append("LOOKING_AWAY")

        if session.movement_tracker.is_excessive_movement():
            warning_codes.append("EXCESSIVE_HEAD_MOVEMENT")

    # --- Build warning objects + persist to session ---
    for code in warning_codes:
        warning_obj = _make_warning(code)
        warnings.append(warning_obj)
        session.add_warning(**warning_obj)

    # --- Risk scoring ---
    risk_result = calculate_risk(warning_codes)

    session.current_risk_score = risk_result["score"]
    session.current_risk_level = risk_result["level"]
    session.total_frames_analyzed += 1

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "face_count": face_count,
        "person_count": person_count,
        "head_pose": head_pose_label,
        "gaze": gaze_label,
        "objects_detected": detected_objects,
        "warnings": warnings,
        "risk_score": risk_result["score"],
        "risk_level": risk_result["level"],
        "processing_time_ms": processing_time_ms,
    }
