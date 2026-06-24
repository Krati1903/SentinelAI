"""
YOLOv8n wrapper for person/phone/book/laptop detection.
Person detection here is also used as the second signal (alongside
MediaPipe face detection) for multiple-person detection, since YOLO
can pick up partial bodies/heads that face detection might miss.
"""

import numpy as np
from app.logger import logger
from app import config


class ObjectDetector:
    def __init__(self):
        self.available = False
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(config.YOLO_MODEL_PATH)
            self.available = True
            logger.info(f"YOLO model loaded: {config.YOLO_MODEL_PATH}")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model, continuing without object detection: {exc}")
            self.available = False

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns dict:
        {
          "person_count": int,
          "person_boxes": [(x1,y1,x2,y2), ...],
          "objects": ["cell phone", "book", ...]   # excluding person
        }
        """
        result = {"person_count": 0, "person_boxes": [], "objects": []}

        if not self.available or self.model is None:
            return result

        try:
            predictions = self.model.predict(
                frame_bgr, conf=config.YOLO_CONF_THRESHOLD, verbose=False
            )
            if not predictions:
                return result

            boxes = predictions[0].boxes
            if boxes is None:
                return result

            names = self.model.names

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in config.YOLO_TARGET_CLASSES:
                    continue

                label = names.get(cls_id, config.YOLO_TARGET_CLASSES[cls_id])
                xyxy = box.xyxy[0].tolist()

                if cls_id == config.YOLO_PERSON_CLASS_ID:
                    result["person_count"] += 1
                    result["person_boxes"].append(tuple(xyxy))
                else:
                    result["objects"].append(label)

            return result

        except Exception as exc:
            logger.error(f"YOLO detection failed on frame: {exc}")
            return result
