"""
Logging setup. Uses loguru for simplicity but writes plain structured
lines so they stay easy to grep or parse later.
"""

import os
import sys
from loguru import logger

from app.config import LOG_DIR

LOG_FILE = os.path.join(LOG_DIR, "proctoring.log")

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add(
    LOG_FILE,
    level="INFO",
    rotation="10 MB",
    retention=10,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


def log_frame_event(
    session_id: str,
    face_count: int,
    person_count: int,
    objects: list,
    warnings: list,
    risk_score: int,
    processing_time_ms: float,
):
    """Write one structured log line per analyzed frame."""
    logger.info(
        "FRAME | session_id={} | face_count={} | person_count={} | objects={} | "
        "warnings={} | risk_score={} | processing_time_ms={:.2f}".format(
            session_id, face_count, person_count, objects, warnings, risk_score, processing_time_ms
        )
    )
