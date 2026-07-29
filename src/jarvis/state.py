"""State machine for Jarvis HUD states."""

from enum import IntEnum


class JarvisState(IntEnum):
    IDLE = 0
    LISTENING = 1
    THINKING = 2
    SPEAKING = 3

    @property
    def label(self) -> str:
        return {
            JarvisState.IDLE: "STANDBY",
            JarvisState.LISTENING: "LISTENING",
            JarvisState.THINKING: "THINKING",
            JarvisState.SPEAKING: "SPEAKING",
        }[self]

    @property
    def target_activity(self) -> float:
        """Target activity level for animation speed/color."""
        return {
            JarvisState.IDLE: 0.0,
            JarvisState.LISTENING: 0.5,
            JarvisState.THINKING: 0.7,
            JarvisState.SPEAKING: 1.0,
        }[self]
