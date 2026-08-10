"""Tests for FaceRecScreen widget."""

from unittest.mock import MagicMock

import pytest

from jarvis.face_screen import FaceRecScreen


@pytest.fixture
def screen(qtbot):
    """Create a FaceRecScreen widget."""
    widget = FaceRecScreen()
    widget.resize(800, 600)
    qtbot.addWidget(widget)
    yield widget
    widget.deleteLater()


class TestFaceRecScreenInit:
    def test_defaults(self, screen):
        assert screen._frame is None
        assert screen._faces == []
        assert screen._face_name == ""
        assert screen._recognized is False
        assert screen._fade_alpha == 0.0
        assert screen._status_text == "Initializing camera..."
        assert screen._enrolling is False

    def test_reg_overlay_exists(self, screen):
        assert screen._reg_overlay is not None

    def test_reg_overlay_hidden(self, screen):
        assert not screen._reg_overlay.isVisible()

    def test_name_input_exists(self, screen):
        assert screen._name_input is not None


class TestFaceRecScreenFrame:
    def test_set_frame_updates_state(self, screen):
        screen._frame = None
        faces = [(10, 20, 50, 60)]
        screen.set_frame(MagicMock(), faces)
        assert len(screen._faces) == 1
        assert screen._frame is not None

    def test_set_frame_no_faces(self, screen):
        screen.set_frame(MagicMock(), [])
        assert screen._faces == []

    def test_set_frame_updates_status(self, screen):
        screen.set_frame(MagicMock(), [(10, 20, 50, 60)])
        assert "face" in screen._status_text.lower()


class TestFaceRecScreenRecognition:
    def test_set_face_detected(self, screen):
        screen.set_face_detected("Alice", 85.5)
        assert screen._face_name == "Alice"
        assert screen._face_confidence == 85.5
        assert screen._recognized is True
        assert screen._status_text == "Welcome, Alice!"

    def test_start_fade(self, screen):
        screen.start_fade()
        assert screen._recognized is True
        assert screen._fade_alpha == 0.0

    def test_is_fade_complete_false(self, screen):
        screen.start_fade()
        assert screen.is_fade_complete is False

    def test_is_fade_complete_true(self, screen):
        screen._fade_alpha = 0.9
        assert screen.is_fade_complete is True


class TestFaceRecScreenEnrollment:
    def test_show_name_input(self, screen):
        screen.show_name_input()
        assert screen._enrolling is True
        assert screen._name_input.isEnabled() is True
        assert screen._name_input.text() == ""
        assert screen._reg_overlay is not None

    def test_show_sample_progress(self, screen):
        screen.show_name_input()
        screen.show_sample_progress(5, 20)
        assert screen._enrollment_progress == (5, 20)
        assert "5/20" in screen._progress_label.text()

    def test_clear_enrollment(self, screen):
        screen.show_name_input()
        screen.show_sample_progress(3, 20)
        screen.clear_enrollment()
        assert screen._enrolling is False
        assert screen._enrollment_name == ""
        assert not screen._reg_overlay.isVisible()

    def test_on_name_skipped(self, screen):
        screen.skip_requested.connect(lambda: None)
        # This should emit skip_requested and start fade
        screen._on_name_skipped()
        assert not screen._reg_overlay.isVisible()


class TestFaceRecScreenSignals:
    def test_name_submitted_signal(self, screen, qtbot):
        emitted = []
        screen.name_submitted.connect(lambda name: emitted.append(name))

        screen._name_input.setText("Alice")
        screen._on_name_submitted()
        assert emitted == ["Alice"]

    def test_name_submitted_empty(self, screen):
        screen._name_input.setText("")
        screen._on_name_submitted()
        assert not screen._enrolling


class TestFaceRecScreenTick:
    def test_tick_advances_pulse(self, screen):
        initial = screen._pulse
        screen._tick()
        assert abs(screen._pulse - initial) > 0.001 or screen._pulse < initial

    def test_tick_advances_scan(self, screen):
        initial = screen._scan_y
        screen._tick()
        assert screen._scan_y != initial

    def test_tick_fade_on_recognized(self, screen):
        screen.set_face_detected("Alice", 90.0)
        initial_alpha = screen._fade_alpha
        for _ in range(10):
            screen._tick()
        assert screen._fade_alpha > initial_alpha
