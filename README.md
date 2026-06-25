# AI Interview Proctoring System

Real-time AI proctoring backend for interview platforms. Detects face
visibility, multiple people in frame, head pose, gaze direction, and
prohibited objects (phones, books, laptops), then produces a weighted
risk score per frame.



## Tech Stack

- FastAPI + Pydantic + Uvicorn (API layer)
- OpenCV (image decode/encode, webcam access)
- MediaPipe Face Detection (face counting)
- MediaPipe Face Mesh (landmark extraction only — **never rendered**)
- YOLOv8n / Ultralytics (person + object detection)
- NumPy
- Loguru (structured logging)

## Detection Capabilities

1. **Face visibility** — flags `FACE_NOT_VISIBLE` if no face is detected
   for more than 2 seconds.
2. **Multiple person detection** — combines YOLOv8 person count and
   MediaPipe face count (`person_count > 1 OR face_count > 1`). This is
   the highest-weighted risk signal.
3. **Partial person detection** — YOLO person class detects partial
   heads/bodies entering frame, not just full faces.
4. **Head pose** — CENTER / LEFT / RIGHT / UP / DOWN via OpenCV
   `solvePnP` using 6 MediaPipe landmarks (not a naive x-coordinate check).
5. **Gaze detection** — SCREEN / LEFT / RIGHT / UP / DOWN via iris
   position relative to eye corners.
6. **Head movement tracking** — rolling yaw/pitch history; flags
   excessive turning while ignoring brief, normal head motion.
7. **Object detection** — YOLOv8n detects phone, book, laptop, person
   in the COCO class set.
8. **Risk engine** — weighted score (Multiple Person 50, Phone 30,
   Face Missing 20, Looking Away 10, Book 15) mapped to
   SAFE / MODERATE / HIGH / CRITICAL.

## Project Structure

```
proctor/
├── main.py                       # FastAPI entrypoint
├── webcam_demo.py                 # optional local webcam test client
├── requirements.txt
├── app/
│   ├── config.py                  # all tunable thresholds
│   ├── logger.py                  # loguru setup + structured frame logs
│   ├── exceptions.py              # custom exceptions
│   ├── camera.py                  # webcam open/read with failsafes
│   ├── session_manager.py         # in-memory session store
│   ├── pipeline.py                # per-frame detection orchestration
│   ├── risk_engine.py             # weighted scoring
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── detectors/
│   │   ├── face_detector.py       # MediaPipe FaceDetection + FaceMesh wrapper
│   │   ├── face_mesh_utils.py     # landmark index constants
│   │   ├── head_pose.py           # solvePnP head pose estimation
│   │   ├── gaze.py                # iris-based gaze estimation
│   │   ├── object_detector.py     # YOLOv8 wrapper
│   │   └── movement_tracker.py    # head movement history
│   └── api/
│       └── routes.py              # all FastAPI endpoints
└── logs/
    └── proctoring.log             # generated at runtime
```

## Setup (Windows 10/11, Python 3.12.10)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The first run will automatically download `yolov8n.pt` (cached locally
afterward by Ultralytics).

## Run

```bash
python main.py
```

API will be available at:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health check: http://127.0.0.1:8000/health

## Optional: Live Webcam Demo

In a second terminal, with the API running:

```bash
python webcam_demo.py
```

This opens your webcam, sends frames to the API, and overlays the
latest risk result as plain text on the **clean video feed** (no face
mesh is ever drawn). Press `q` to quit.

## API Endpoints

| Method | Path                  | Description                          |
|--------|-----------------------|---------------------------------------|
| GET    | `/health`             | Service + model availability check    |
| POST   | `/session/start`       | Create a new proctoring session       |
| POST   | `/frame/analyze`       | Analyze one base64-encoded frame      |
| GET    | `/session/{id}`        | Get session summary                   |
| GET    | `/warnings/{id}`       | Get all warnings for a session        |

### Example: start a session

```bash
curl -X POST http://127.0.0.1:8000/session/start -H "Content-Type: application/json" -d "{\"candidate_name\": \"Jane Doe\"}"
```

### Example: analyze a frame

```bash
curl -X POST http://127.0.0.1:8000/frame/analyze -H "Content-Type: application/json" -d "{\"session_id\": \"<id>\", \"image_base64\": \"<base64 jpeg>\"}"
```

## Logging

Every analyzed frame is logged to `logs/proctoring.log` with:
timestamp, session_id, face_count, person_count, objects, warnings,
risk_score, processing_time_ms. Console output mirrors the same lines.

## Failsafe Behavior

- If YOLO fails to load, the system continues using MediaPipe only
  (face detection still works; object/person detection returns empty).
- If MediaPipe fails to load, the API still runs and returns
  structured results with `face_count: 0` instead of crashing.
- Camera autofocus disable (`cv2.CAP_PROP_AUTOFOCUS, 0`) is attempted
  inside a try/except in `app/camera.py` and skipped silently if the
  webcam driver doesn't support it.
- All decode/model/camera operations are wrapped in try/except and
  logged; a single bad frame never crashes the server.
