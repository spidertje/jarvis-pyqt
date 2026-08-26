"""Tests for BargeInListener (user speech detection while Jarvis is speaking).

The RMS/sustained-speech gate is tested by driving ``_callback`` directly
with synthetic audio frames — no PortAudio/device required.
"""

import numpy as np
import pytest

from jarvis.barge_in import BargeInListener


def _frame(amp: float, samples: int = 512) -> np.ndarray:
    """Synthetic float32 frame (mono) at amplitude ``amp`` (0..1)."""
    return np.full((samples, 1), amp, dtype=np.float32)


@pytest.fixture
def listener():
    """BargeInListener with no audio device; on_speech records calls.

    Thresholds are on the 16-bit RMS scale (0..32768): test frames use
    amplitude 1.0 (loud) and 0.01 (quiet ≈ 328 RMS).
    """
    calls = []
    l = BargeInListener(
        on_speech=lambda: calls.append(1),
        threshold=10000.0,
        min_speech_ms=0.0,
    )
    l._calls = calls
    return l


class TestAvailability:
    def test_unavailable_without_sounddevice(self):
        l = BargeInListener()
        l._sd = None
        assert l.available is False

    def test_start_stop_noop_when_unavailable(self, listener):
        listener._sd = None
        listener.start()  # must not raise
        assert listener._stream is None
        listener.stop()   # must not raise

    def test_set_active_toggles_flag(self, listener):
        listener.set_active(True)
        assert listener._active is True
        listener.set_active(False)
        assert listener._active is False


class TestSpeechGate:
    def test_loud_speech_fires_callback(self, listener):
        listener.set_active(True)
        listener._callback(_frame(1.0), 512, None, None)  # opens the window
        assert listener._calls == []  # first frame alone never fires
        listener._callback(_frame(1.0), 512, None, None)  # sustained → fires
        assert len(listener._calls) == 1
        # After firing, listener disables itself (no re-fire).
        assert listener._active is False

    def test_quiet_never_fires(self, listener):
        listener.set_active(True)
        for _ in range(3):
            listener._callback(_frame(0.01), 512, None, None)
        assert listener._calls == []

    def test_sustained_speech_required(self, listener, monkeypatch):
        """Below min_speech_ms the loud frames must not fire."""
        calls = []
        l = BargeInListener(
            on_speech=lambda: calls.append(1),
            threshold=10000.0,
            min_speech_ms=300.0,
        )
        clock = {"t": 0.0}
        monkeypatch.setattr("time.monotonic", lambda: clock["t"])
        l.set_active(True)
        # First loud frame just opens the window — not yet sustained.
        clock["t"] = 0.0
        l._callback(_frame(1.0), 512, None, None)
        assert calls == []
        # Quiet frame resets the window.
        l._callback(_frame(0.0), 512, None, None)
        assert calls == []
        # Loud again, then 300ms later loud again → sustained → fires.
        clock["t"] = 0.0
        l._callback(_frame(1.0), 512, None, None)
        clock["t"] = 0.35
        l._callback(_frame(1.0), 512, None, None)
        assert calls == [1]

    def test_inactive_listens_nothing(self, listener):
        listener.set_active(False)
        listener._callback(_frame(1.0), 512, None, None)
        listener._callback(_frame(1.0), 512, None, None)
        assert listener._calls == []

    def test_callback_exception_is_swallowed(self, listener):
        listener.on_speech = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        listener.set_active(True)
        listener._callback(_frame(1.0), 512, None, None)  # opens the window
        listener._callback(_frame(1.0), 512, None, None)  # fires → exception swallowed
        assert listener._active is False
