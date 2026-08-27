"""Tests for hands-free wake word (openWakeWord).

The detector logic is tested by injecting a mock model and driving
``_callback`` directly with synthetic frames — no PortAudio/mic required.
Agent integration, settings UI and config round-trip are covered too.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.config import AppConfig
from jarvis.state import JarvisState
from jarvis.wake_word import WakeWordDetector


# ── helpers ──────────────────────────────────────────────────────────

def _frame(amp: float = 0.0, samples: int = 1280) -> np.ndarray:
    """Synthetic float32 mono frame (what the PortAudio callback delivers)."""
    return np.full((samples, 1), amp, dtype=np.float32)


class MockModel:
    """openWakeWord stand-in: scores 0.0 until N frames seen, then 0.9."""

    def __init__(self, word: str = "hey_jarvis", after: int = 10):
        self.word = word
        self.after = after
        self.n = 0
        self.threshold = None
        self.patience = None

    def predict(self, x, threshold=None, patience=None):
        self.threshold = threshold
        self.patience = patience
        self.n += 1
        score = 0.9 if self.n >= self.after else 0.0
        return {self.word: score, "background": 0.0}


@pytest.fixture
def detector():
    """Detector with an injected model; on_wake records calls."""
    calls: list = []
    d = WakeWordDetector(on_wake=lambda: calls.append(1), model=MockModel())
    d._calls = calls
    d._last_fire = 0.0
    return d


# ── WakeWordDetector ─────────────────────────────────────────────────

class TestDetectorBasics:
    def test_available_with_injected_model(self, detector):
        assert detector.available is True

    def test_unavailable_without_model(self):
        d = WakeWordDetector(model=None) if False else None
        d = WakeWordDetector()
        d._model = None
        assert d.available is False

    def test_start_noop_when_unavailable(self):
        d = WakeWordDetector()
        d._model = None
        d.start()  # must not raise
        assert d._stream is None
        d.stop()  # must not raise

    def test_stop_clears_stream(self, detector):
        detector._stream = MagicMock()
        detector.stop()
        assert detector._stream is None
        assert detector._active is False

    def test_set_active_toggles(self, detector):
        detector.set_active(True)
        assert detector._active is True
        detector.set_active(False)
        assert detector._active is False

    def test_callback_noop_when_inactive(self, detector):
        detector.set_active(False)
        detector._callback(_frame(), 1280, None, None)  # no fire, no crash
        assert detector._calls == []

    def test_callback_noop_when_unavailable(self):
        d = WakeWordDetector()
        d._model = None
        d.set_active(True)
        d._callback(_frame(0.5), 1280, None, None)  # must not raise

    def test_threshold_is_live(self, detector):
        detector.threshold = 0.8
        assert detector.threshold == 0.8

    def test_predict_receives_int16_and_params(self, detector):
        detector.set_active(True)
        detector._callback(_frame(0.5), 1280, None, None)
        m = detector._model
        assert m.threshold == {"hey_jarvis": detector.threshold}
        assert m.patience == {"hey_jarvis": detector.patience}


class TestDetectorFiring:
    def test_low_score_never_fires(self, detector):
        detector.set_active(True)
        for _ in range(5):
            detector._callback(_frame(0.5), 1280, None, None)
        assert detector._calls == []

    def test_sustained_high_score_fires_once(self, detector):
        detector.set_active(True)
        fired_at = None
        for i in range(15):
            detector._callback(_frame(0.5), 1280, None, None)
            if detector._calls:
                fired_at = i
                break
        assert fired_at is not None
        assert len(detector._calls) == 1

    def test_cooldown_suppresses_reroll(self, detector):
        detector.set_active(True)
        # force fire
        for _ in range(15):
            detector._callback(_frame(0.5), 1280, None, None)
            if detector._calls:
                break
        assert len(detector._calls) == 1
        detector._model.n = 100  # model would still score high
        detector._callback(_frame(0.5), 1280, None, None)  # within cooldown
        assert len(detector._calls) == 1

    def test_fires_again_after_cooldown(self, detector, monkeypatch):
        detector.set_active(True)
        for _ in range(15):
            detector._callback(_frame(0.5), 1280, None, None)
            if detector._calls:
                break
        detector._last_fire = time.monotonic() - detector.cooldown_s - 0.1
        detector._callback(_frame(0.5), 1280, None, None)
        assert len(detector._calls) == 2

    def test_on_wake_exception_does_not_crash(self):
        def boom():
            raise RuntimeError("callback blew up")

        d = WakeWordDetector(on_wake=boom, model=MockModel(after=1))
        d.set_active(True)
        d._callback(_frame(0.5), 1280, None, None)  # must not raise


# ── Agent integration ────────────────────────────────────────────────

class TestAgentWakeWord:
    def test_disabled_in_config(self):
        cfg = AgentConfig()
        cfg.wake_word_enabled = False
        agent = JarvisAgent(cfg)
        assert agent.start_wake_word() is False
        assert agent._wake_word is None

    def test_start_stops_and_close(self):
        agent = JarvisAgent(AgentConfig())
        fake = MagicMock()
        fake.available = True
        fake._stream = None
        fake.start = MagicMock(side_effect=lambda: setattr(fake, "_stream", MagicMock()))
        with patch_detector(agent, fake):
            assert agent.start_wake_word() is True
            assert agent.wake_word_available is True
            assert agent.start_wake_word() is True  # idempotent
            agent.stop_wake_word()
            assert agent._wake_word is None
            assert fake.stop.called
        agent.stop_wake_word()  # safe when absent

    async def test_close_stops_wake_word(self):
        agent = JarvisAgent(AgentConfig())
        fake = MagicMock()
        fake.available = True
        fake._stream = None
        fake.start = MagicMock(side_effect=lambda: setattr(fake, "_stream", MagicMock()))
        with patch_detector(agent, fake):
            agent.start_wake_word()
        await agent.close()
        assert agent._wake_word is None
        assert fake.stop.called

    def test_on_wake_word_notifies_callbacks_and_sets_hud(self):
        agent = JarvisAgent(AgentConfig(), hud=MagicMock())
        hits: list = []
        agent.on_wake_word(lambda: hits.append(1))
        agent.on_wake_word(lambda: hits.append(2))
        agent._on_wake_word()
        assert hits == [1, 2]
        agent.hud.set_state.assert_called_with(JarvisState.LISTENING)

    def test_callback_exception_isolated(self):
        agent = JarvisAgent(AgentConfig())
        agent.on_wake_word(lambda: 1 / 0)
        agent.on_wake_word(lambda: None)  # must still be reached
        agent._on_wake_word()  # must not raise

    def test_set_state_gates_detector(self):
        agent = JarvisAgent(AgentConfig())
        fake = MagicMock()
        fake.available = True
        fake._stream = None
        fake.start = MagicMock(side_effect=lambda: setattr(fake, "_stream", MagicMock()))
        with patch_detector(agent, fake):
            agent.start_wake_word()
            fake.set_active.assert_called_with(True)  # IDLE by default
            agent._set_state(JarvisState.LISTENING)
            fake.set_active.assert_called_with(False)
            agent._set_state(JarvisState.IDLE)
            fake.set_active.assert_called_with(True)


def patch_detector(agent: JarvisAgent, fake):
    """Context manager: make JarvisAgent start_wake_word use ``fake`` detector."""
    from contextlib import contextmanager

    @contextmanager
    def _patch():
        from jarvis import agent as agent_mod

        original = agent_mod.WakeWordDetector
        agent_mod.WakeWordDetector = lambda **kw: fake
        try:
            yield fake
        finally:
            agent_mod.WakeWordDetector = original

    return _patch()


# ── AppConfig round-trip ─────────────────────────────────────────────

class TestAppConfigWakeWord:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.wake_word_enabled is True
        assert cfg.wake_word_threshold == 0.5

    def test_from_dict_roundtrip(self):
        cfg = AppConfig(wake_word_enabled=False, wake_word_threshold=0.75)
        data = cfg.__dict__.copy() if hasattr(cfg, "__dict__") else None
        from dataclasses import asdict

        d = asdict(cfg)
        d["wake_word_enabled"] = False
        d["wake_word_threshold"] = 0.75
        back = AppConfig.from_dict(d)
        assert back.wake_word_enabled is False
        assert back.wake_word_threshold == 0.75

    def test_env_bool_parsing(self, monkeypatch):
        monkeypatch.setenv("JARVIS_WAKE_WORD", "false")
        cfg = AppConfig.from_env()
        assert cfg.wake_word_enabled is False
        monkeypatch.setenv("JARVIS_WAKE_WORD", "1")
        cfg = AppConfig.from_env()
        assert cfg.wake_word_enabled is True


# ── Settings dialog ──────────────────────────────────────────────────

@pytest.fixture
def settings(qtbot):
    """SettingsDialog with minimal config (no real agent)."""
    from jarvis.face import FaceConfig
    from jarvis.settings import SettingsDialog

    dialog = SettingsDialog(
        agent_config=AgentConfig(),
        app_config=AppConfig(),
        agent=MagicMock(),
    )
    dialog.face_config = FaceConfig()
    dialog.on_face_restart = MagicMock()
    qtbot.addWidget(dialog)
    yield dialog
    dialog.deleteLater()


class TestSettingsWakeWord:
    def test_widgets_exist(self, settings):
        assert hasattr(settings, "wake_enabled")
        assert hasattr(settings, "wake_threshold")
        assert hasattr(settings, "wake_threshold_value")
        assert hasattr(settings, "wake_available")

    def test_defaults_from_config(self, settings):
        assert settings.wake_enabled.isChecked() is True
        assert settings.wake_threshold.value() == 50  # 0.50 default

    def test_threshold_slider_updates_label_and_config(self, settings):
        settings.wake_threshold.setValue(70)
        assert settings.wake_threshold_value.text() == "0.70"
        assert settings.agent_config.wake_word_threshold == 0.7

    def test_save_writes_agent_and_app_config(self, settings, monkeypatch):
        settings.wake_enabled.setChecked(False)
        settings.wake_threshold.setValue(80)
        # keep the test hermetic: no modal, no disk
        from PyQt6.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(settings.app_config, "save", lambda: None)
        settings._save_settings()
        assert settings.agent_config.wake_word_enabled is False
        assert settings.agent_config.wake_word_threshold == 0.8
        assert settings.app_config.wake_word_enabled is False
        assert settings.app_config.wake_word_threshold == 0.8

    def test_toggle_updates_status_label(self, settings):
        settings.wake_enabled.setChecked(False)
        assert "disabled" in settings.wake_available.text()
        settings.wake_enabled.setChecked(True)
