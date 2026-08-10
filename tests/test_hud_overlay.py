"""Tests for HUDOverlay widget."""

import pytest

from jarvis.hud_overlay import HUDOverlay
from jarvis.state import JarvisState


@pytest.fixture
def hud(qtbot):
    """Create a HUDOverlay widget."""
    widget = HUDOverlay()
    widget.resize(800, 600)
    qtbot.addWidget(widget)
    yield widget
    widget.deleteLater()


class TestHUDOverlayInit:
    def test_defaults(self, hud):
        assert hud.state == JarvisState.IDLE
        assert hud.activity == 0.0
        assert hud._assistant_name == "Jarvis"
        assert hud._palette_hue is None  # No custom palette
        assert hud._contrast_factor == 1.0
        assert hud._face_name == ""
        assert hud._profile_name == ""

    def test_particles_initialized(self, hud):
        assert len(hud._particles) == 60

    def test_bar_heights_initialized(self, hud):
        assert len(hud._bar_heights) == 40


class TestHUDOverlayState:
    def test_set_state(self, hud):
        hud.set_state(JarvisState.LISTENING)
        assert hud.state == JarvisState.LISTENING

    def test_set_state_idle(self, hud):
        hud.set_state(JarvisState.SPEAKING)
        assert hud.state == JarvisState.SPEAKING
        hud.set_state(JarvisState.IDLE)
        assert hud.state == JarvisState.IDLE

    def test_set_state_emits_wave_rings(self, hud):
        hud.set_state(JarvisState.SPEAKING)
        assert len(hud._wave_rings) == 3

    def test_set_state_no_duplicate_wave_rings(self, hud):
        hud.set_state(JarvisState.SPEAKING)
        initial_count = len(hud._wave_rings)
        hud.set_state(JarvisState.SPEAKING)
        assert len(hud._wave_rings) == initial_count + 3


class TestHUDOverlayActivity:
    def test_set_activity(self, hud):
        hud.set_activity(0.5)
        assert hud.activity == pytest.approx(0.5)

    def test_set_activity_clamped(self, hud):
        hud.set_activity(-1.0)
        assert hud.activity == 0.0
        hud.set_activity(2.0)
        assert hud.activity == 1.0

    def test_tick_smooths_activity(self, hud):
        hud._state = JarvisState.SPEAKING
        hud._activity = 0.0
        hud._tick()
        # Activity should move toward target (1.0) but not jump all the way
        assert hud._activity > 0.0
        assert hud._activity < 1.0


class TestHUDOverlayVoiceLevel:
    def test_set_voice_level(self, hud):
        hud.set_voice_level(0.5)
        assert hud._voice_level == pytest.approx(0.5)

    def test_set_voice_level_clamped(self, hud):
        hud.set_voice_level(-1.0)
        assert hud._voice_level == 0.0
        hud.set_voice_level(2.0)
        assert hud._voice_level == 1.0

    def test_voice_level_timer_set(self, hud):
        hud.set_voice_level(0.5)
        assert hud._voice_level_timer == 30


class TestHUDOverlayPalette:
    def test_set_palette_hue(self, hud):
        hud.set_palette_hue(45)
        assert hud._palette_hue == 45

    def test_clear_palette_hue(self, hud):
        hud.set_palette_hue(45)
        hud.clear_palette_hue()
        assert hud._palette_hue is None


class TestHUDOverlayContrast:
    def test_set_contrast_factor(self, hud):
        hud.set_contrast_factor(1.5)
        assert hud._contrast_factor == 1.5

    def test_set_contrast_factor_clamped(self, hud):
        hud.set_contrast_factor(-1.0)
        assert hud._contrast_factor == 0.0


class TestHUDOverlayAssistantName:
    def test_set_assistant_name(self, hud):
        hud.set_assistant_name("Hermes")
        assert hud._assistant_name == "Hermes"


class TestHUDOverlayFace:
    def test_set_face_detected(self, hud):
        hud.set_face_detected("Alice", 85.5)
        assert hud._face_name == "Alice"
        assert hud._face_confidence == 85.5

    def test_clear_face(self, hud):
        hud.set_face_detected("Bob", 90.0)
        hud.clear_face()
        assert hud._face_name == ""
        assert hud._face_confidence == 0.0

    def test_face_overlay_fades_after_timeout(self, hud):
        hud.set_face_detected("Alice", 85.0)
        # Simulate ticks to advance the face timer beyond 3.0
        hud._face_timer = 3.1
        hud.clear_face()
        assert hud._face_name == ""


class TestHUDOverlayProfile:
    def test_set_profile(self, hud):
        hud.set_profile("Alice", hue=45)
        assert hud._profile_name == "Alice"
        assert hud._profile_hue == 45

    def test_clear_profile(self, hud):
        hud.set_profile("Bob", hue=30)
        hud.clear_profile()
        assert hud._profile_name == ""
        assert hud._palette_hue is None


class TestHUDOverlayAnimation:
    def test_tick_advances_angle(self, hud):
        initial_angle = hud._angle
        hud._tick()
        assert hud._angle > initial_angle

    def test_tick_advances_pulse(self, hud):
        initial_pulse = hud._pulse
        hud._tick()
        assert hud._pulse > initial_pulse or hud._pulse < initial_pulse
        # Pulse wraps around mod (2*pi)
        assert 0 <= hud._pulse <= 6.283185307179586

    def test_tick_updates_voice_level_decay(self, hud):
        hud._voice_level = 1.0
        hud._voice_level_timer = 0
        for _ in range(10):
            hud._tick()
        # Voice level should decay toward 0
        assert hud._voice_level < 1.0
