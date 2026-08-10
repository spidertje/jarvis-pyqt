"""Tests for JarvisState enum."""

import pytest

from jarvis.state import JarvisState


class TestJarvisState:
    def test_states_exist(self):
        """All four states should exist."""
        assert hasattr(JarvisState, "IDLE")
        assert hasattr(JarvisState, "LISTENING")
        assert hasattr(JarvisState, "THINKING")
        assert hasattr(JarvisState, "SPEAKING")

    def test_idle_activity_is_zero(self):
        """IDLE state should have target_activity of 0.0."""
        assert JarvisState.IDLE.target_activity == 0.0

    def test_speaking_activity_is_one(self):
        """SPEAKING state should have target_activity of 1.0."""
        assert JarvisState.SPEAKING.target_activity == 1.0

    def test_target_activity_increases(self):
        """Activity levels should increase across states IDLE < LISTENING < THINKING < SPEAKING."""
        assert JarvisState.IDLE.target_activity < JarvisState.LISTENING.target_activity
        assert JarvisState.LISTENING.target_activity < JarvisState.THINKING.target_activity
        assert JarvisState.THINKING.target_activity < JarvisState.SPEAKING.target_activity

    @pytest.mark.parametrize(
        "state,label",
        [
            (JarvisState.IDLE, "STANDBY"),
            (JarvisState.LISTENING, "LISTENING"),
            (JarvisState.THINKING, "THINKING"),
            (JarvisState.SPEAKING, "SPEAKING"),
        ],
    )
    def test_labels(self, state, label):
        """Each state should have the correct label."""
        assert state.label == label
