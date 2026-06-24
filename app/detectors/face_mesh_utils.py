"""
Landmark index constants for MediaPipe Face Mesh (468/478 point model).
Kept in one place so head pose and gaze modules stay consistent.
"""

# 6-point model used for solvePnP head pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_LEFT_CORNER = 263
RIGHT_EYE_RIGHT_CORNER = 33
LEFT_MOUTH_CORNER = 287
RIGHT_MOUTH_CORNER = 57

PNP_LANDMARK_IDS = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_LEFT_CORNER,
    RIGHT_EYE_RIGHT_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
]

# Generic 3D face model points (mm) matching the order above.
# Values come from a standard anthropometric face model used widely for solvePnP.
PNP_3D_MODEL_POINTS = [
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
]

# Left eye (subject's left) landmark ring + iris
LEFT_EYE_RING = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]

# Right eye (subject's right) landmark ring + iris
RIGHT_EYE_RING = [33, 160, 158, 133, 153, 144]
RIGHT_IRIS = [473, 474, 475, 476, 477]
