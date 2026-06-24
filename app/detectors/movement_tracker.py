"""
Tracks recent yaw/pitch history per session to detect excessive head
movement (e.g. repeated turning to look off-screen) without flagging
normal, occasional head motion.
"""

from collections import deque
import numpy as np

from app import config


class MovementTracker:
    def __init__(self):
        self.yaw_history = deque(maxlen=config.MOVEMENT_HISTORY_SIZE)
        self.pitch_history = deque(maxlen=config.MOVEMENT_HISTORY_SIZE)

    def update(self, yaw: float, pitch: float):
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

    def is_excessive_movement(self) -> bool:
        if len(self.yaw_history) < config.EXCESSIVE_MOVEMENT_MIN_SAMPLES:
            return False

        yaw_std = float(np.std(self.yaw_history))
        pitch_std = float(np.std(self.pitch_history))

        return (
            yaw_std >= config.EXCESSIVE_MOVEMENT_STD_DEGREES
            or pitch_std >= config.EXCESSIVE_MOVEMENT_STD_DEGREES
        )
