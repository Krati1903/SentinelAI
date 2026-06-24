"""
Simple in-memory session store. No database — sessions live for the
life of the process, which matches the "no Postgres/Redis" requirement.
"""

import uuid
import time
from datetime import datetime, timezone

from app.exceptions import SessionNotFoundError
from app.detectors.movement_tracker import MovementTracker


class Session:
    def __init__(self, candidate_name=None, interview_id=None):
        self.session_id = str(uuid.uuid4())
        self.candidate_name = candidate_name
        self.interview_id = interview_id
        self.started_at = datetime.now(timezone.utc)
        self.last_activity_at = None
        self.last_face_seen_at = time.time()
        self.movement_tracker = MovementTracker()
        self.total_frames_analyzed = 0
        self.current_risk_score = 0
        self.current_risk_level = "SAFE"
        self.warnings = []  # list of dicts: code, message, severity, timestamp

    def add_warning(self, code: str, message: str, severity: str):
        self.warnings.append(
            {
                "code": code,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


class SessionManager:
    def __init__(self):
        self._sessions = {}

    def create_session(self, candidate_name=None, interview_id=None) -> Session:
        session = Session(candidate_name=candidate_name, interview_id=interview_id)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return session

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions


session_manager = SessionManager()
