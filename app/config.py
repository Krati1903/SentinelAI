"""
Central configuration for the proctoring system.
Keep all tunable thresholds here so behaviour can be adjusted
without hunting through detector code.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Face visibility ---
FACE_MISSING_THRESHOLD_SECONDS = 2.0

# --- MediaPipe ---
FACE_DETECTION_MIN_CONFIDENCE = 0.5
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_FACES = 5

# --- YOLO ---
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35
YOLO_PERSON_CLASS_ID = 0
YOLO_TARGET_CLASSES = {
    0: "person",
    67: "cell phone",
    73: "book",
    63: "laptop",
    # tablet is not a native COCO class, closest practical proxy is cell phone
}

# --- Head pose thresholds (degrees) ---
HEAD_YAW_LEFT_THRESHOLD = -15.0
HEAD_YAW_RIGHT_THRESHOLD = 15.0
HEAD_PITCH_UP_THRESHOLD = -12.0
HEAD_PITCH_DOWN_THRESHOLD = 12.0

# --- Gaze thresholds (ratio based, 0.0 - 1.0 across eye) ---
GAZE_LEFT_RATIO = 0.38
GAZE_RIGHT_RATIO = 0.62
GAZE_VERTICAL_UP_RATIO = 0.38
GAZE_VERTICAL_DOWN_RATIO = 0.62

# --- Head movement tracking ---
MOVEMENT_HISTORY_SIZE = 30
EXCESSIVE_MOVEMENT_STD_DEGREES = 18.0
EXCESSIVE_MOVEMENT_MIN_SAMPLES = 10

# --- Risk engine weights ---
RISK_WEIGHTS = {
    "MULTIPLE_PERSON_DETECTED": 50,
    "PHONE_DETECTED": 30,
    "FACE_NOT_VISIBLE": 20,
    "LOOKING_AWAY": 10,
    "BOOK_DETECTED": 15,
    "EXCESSIVE_HEAD_MOVEMENT": 10,
    "LAPTOP_DETECTED": 10,
}

RISK_LEVELS = {
    "SAFE": (0, 19),
    "MODERATE": (20, 49),
    "HIGH": (50, 79),
    "CRITICAL": (80, 10_000),
}

# --- Session ---
SESSION_TTL_SECONDS = 3600
