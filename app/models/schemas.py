from typing import List, Optional
from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    candidate_name: Optional[str] = Field(default=None, description="Candidate display name")
    interview_id: Optional[str] = Field(default=None, description="External interview reference id")


class SessionStartResponse(BaseModel):
    session_id: str
    started_at: str
    message: str = "Session started"


class FrameAnalyzeRequest(BaseModel):
    session_id: str
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG frame")


class DetectionWarning(BaseModel):
    code: str
    message: str
    severity: str


class FrameAnalyzeResponse(BaseModel):
    session_id: str
    timestamp: str
    face_count: int
    person_count: int
    head_pose: str
    gaze: str
    objects_detected: List[str]
    warnings: List[DetectionWarning]
    risk_score: int
    risk_level: str
    processing_time_ms: float


class SessionInfoResponse(BaseModel):
    session_id: str
    candidate_name: Optional[str]
    interview_id: Optional[str]
    started_at: str
    last_activity_at: Optional[str]
    total_frames_analyzed: int
    current_risk_score: int
    current_risk_level: str
    total_warnings: int


class WarningsResponse(BaseModel):
    session_id: str
    warnings: List[DetectionWarning]
