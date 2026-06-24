"""
Optional demo client: captures from the local webcam, sends each frame
to the running API for analysis, and prints results to the console.
The webcam output window itself stays clean — no mesh, no landmarks,
just the raw frame plus a text overlay of the latest result.

Run the API first (python main.py), then run this script separately:
    python webcam_demo.py
"""

import base64
import time
import requests
import cv2

from app.camera import open_camera, read_frame
from app.logger import logger

API_BASE = "http://127.0.0.1:8000"


def main():
    try:
        resp = requests.post(f"{API_BASE}/session/start", json={"candidate_name": "Demo Candidate"})
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        logger.info(f"Started demo session: {session_id}")
    except Exception as exc:
        logger.error(f"Could not start session, is the API running? {exc}")
        return

    try:
        cap = open_camera()
    except Exception as exc:
        logger.error(f"Could not open webcam: {exc}")
        return

    last_text = "Starting..."

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                logger.warning("Empty frame from webcam, skipping")
                time.sleep(0.1)
                continue

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            image_b64 = base64.b64encode(buffer).decode("utf-8")

            try:
                analyze_resp = requests.post(
                    f"{API_BASE}/frame/analyze",
                    json={"session_id": session_id, "image_base64": image_b64},
                    timeout=5,
                )
                if analyze_resp.status_code == 200:
                    data = analyze_resp.json()
                    last_text = (
                        f"Risk: {data['risk_level']} ({data['risk_score']}) "
                        f"Faces: {data['face_count']} Persons: {data['person_count']} "
                        f"Pose: {data['head_pose']} Gaze: {data['gaze']}"
                    )
                else:
                    last_text = f"Analyze error: {analyze_resp.status_code}"
            except Exception as exc:
                last_text = f"Request failed: {exc}"

            display_frame = frame.copy()
            cv2.putText(
                display_frame, last_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            cv2.imshow("Proctoring Demo (clean feed)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
