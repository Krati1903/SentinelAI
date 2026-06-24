"""Custom exceptions so detector failures are explicit and catchable."""


class ProctoringError(Exception):
    """Base exception for the proctoring system."""


class ModelLoadError(ProctoringError):
    """Raised when a CV model fails to load."""


class CameraError(ProctoringError):
    """Raised when the webcam cannot be opened or read from."""


class FrameDecodeError(ProctoringError):
    """Raised when an incoming frame cannot be decoded."""


class SessionNotFoundError(ProctoringError):
    """Raised when a session id does not exist."""
