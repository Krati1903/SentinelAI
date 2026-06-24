import base64
import time
import numpy as np
import cv2

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    SessionStartRequest,
    SessionStartResponse,
    FrameAnalyzeRequest,
    FrameAnalyzeResponse,
    DetectionWarning,
    SessionInfoResponse,
    WarningsResponse,
)

from app.session_manager import session_manager
from app.exceptions import SessionNotFoundError, FrameDecodeError
from app.pipeline import analyze_frame, face_detector, object_detector
from app.logger import logger, log_frame_event

router = APIRouter()

SESSION_LOGS_DIR = Path("session_logs")
SESSION_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _get_session_log_path(session_id: str) -> Path:
    return SESSION_LOGS_DIR / f"{session_id}.txt"


def _write_session_header(session) -> None:
    log_path = _get_session_log_path(session.session_id)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("=" * 60 + "\n")
            log_file.write("AI INTERVIEW PROCTORING SESSION\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Session ID: {session.session_id}\n")
            log_file.write(f"Candidate: {session.candidate_name}\n")
            log_file.write(f"Interview ID: {session.interview_id}\n")
            log_file.write(f"Started Time: {session.started_at.isoformat()}\n")
            log_file.write("=" * 60 + "\n\n")
    except Exception as exc:
        logger.error(f"Failed to write session header for {session.session_id}: {exc}")


def _append_frame_log(session_id: str, result: dict) -> None:
    log_path = _get_session_log_path(session_id)
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("-" * 60 + "\n")
            log_file.write(f"Time: {datetime.now(timezone.utc).isoformat()}\n")
            log_file.write(f"Face Count: {result['face_count']}\n")
            log_file.write(f"Person Count: {result['person_count']}\n")
            log_file.write(f"Head Pose: {result['head_pose']}\n")
            log_file.write(f"Gaze: {result['gaze']}\n")
            log_file.write(f"Objects Detected: {result['objects_detected']}\n")
            log_file.write(
                f"Warnings: {[w['code'] for w in result['warnings']]}\n"
            )
            log_file.write(f"Risk Score: {result['risk_score']}\n")
            log_file.write(f"Risk Level: {result['risk_level']}\n")
            log_file.write(
                f"Processing Time: {result['processing_time_ms']:.2f} ms\n"
            )
            log_file.write("-" * 60 + "\n\n")
    except Exception as exc:
        logger.error(f"Failed to append frame log for session {session_id}: {exc}")


def _append_session_summary(session) -> None:
    log_path = _get_session_log_path(session.session_id)
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("=" * 60 + "\n")
            log_file.write("SESSION SUMMARY\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Total Frames: {session.total_frames_analyzed}\n")
            log_file.write(f"Total Warnings: {len(session.warnings)}\n")
            log_file.write(f"Final Risk Score: {session.current_risk_score}\n")
            log_file.write(f"Risk Level: {session.current_risk_level}\n")
            log_file.write(
                f"Session End Time: {datetime.now(timezone.utc).isoformat()}\n"
            )
            log_file.write("=" * 60 + "\n")
    except Exception as exc:
        logger.error(f"Failed to append session summary for {session.session_id}: {exc}")


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "face_detector_available": bool(face_detector and face_detector.available),
        "object_detector_available": bool(object_detector and object_detector.available),
        "active_sessions": len(session_manager._sessions),
    }


@router.post("/session/start", response_model=SessionStartResponse)
def start_session(payload: SessionStartRequest):
    session = session_manager.create_session(
        candidate_name=payload.candidate_name,
        interview_id=payload.interview_id,
    )

    _write_session_header(session)

    logger.info(f"Session started: {session.session_id}")

    return SessionStartResponse(
        session_id=session.session_id,
        started_at=session.started_at.isoformat(),
    )


def _decode_base64_image(image_base64: str) -> np.ndarray:
    try:
        if "," in image_base64 and image_base64.strip().startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise FrameDecodeError("Decoded frame is empty")
        return frame
    except Exception as exc:
        raise FrameDecodeError(f"Could not decode image: {exc}")


@router.post("/frame/analyze", response_model=FrameAnalyzeResponse)
def analyze_frame_endpoint(payload: FrameAnalyzeRequest):
    try:
        session = session_manager.get_session(payload.session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        frame = _decode_base64_image(payload.image_base64)
    except FrameDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = analyze_frame(frame, session)
    except Exception as exc:
        logger.error(f"Unhandled error during frame analysis: {exc}")
        raise HTTPException(status_code=500, detail="Frame analysis failed internally")

    session.last_activity_at = datetime.now(timezone.utc)

    log_frame_event(
        session_id=session.session_id,
        face_count=result["face_count"],
        person_count=result["person_count"],
        objects=result["objects_detected"],
        warnings=[w["code"] for w in result["warnings"]],
        risk_score=result["risk_score"],
        processing_time_ms=result["processing_time_ms"],
    )

    _append_frame_log(session.session_id, result)

    return FrameAnalyzeResponse(
        session_id=session.session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        face_count=result["face_count"],
        person_count=result["person_count"],
        head_pose=result["head_pose"],
        gaze=result["gaze"],
        objects_detected=result["objects_detected"],
        warnings=[DetectionWarning(**w) for w in result["warnings"]],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        processing_time_ms=result["processing_time_ms"],
    )


@router.get("/session/{session_id}", response_model=SessionInfoResponse)
def get_session(session_id: str):
    try:
        session = session_manager.get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfoResponse(
        session_id=session.session_id,
        candidate_name=session.candidate_name,
        interview_id=session.interview_id,
        started_at=session.started_at.isoformat(),
        last_activity_at=session.last_activity_at.isoformat() if session.last_activity_at else None,
        total_frames_analyzed=session.total_frames_analyzed,
        current_risk_score=session.current_risk_score,
        current_risk_level=session.current_risk_level,
        total_warnings=len(session.warnings),
    )


@router.get("/warnings/{session_id}", response_model=WarningsResponse)
def get_warnings(session_id: str):
    try:
        session = session_manager.get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return WarningsResponse(
        session_id=session.session_id,
        warnings=[
            DetectionWarning(code=w["code"], message=w["message"], severity=w["severity"])
            for w in session.warnings
        ],
    )


@router.post("/session/end/{session_id}")
def end_session(session_id: str):
    try:
        session = session_manager.get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    _append_session_summary(session)

    logger.info(f"Session ended: {session.session_id}")

    log_path = _get_session_log_path(session.session_id)

    return {
        "status": "ended",
        "session_id": session.session_id,
        "log_file": str(log_path.as_posix()),
    }
