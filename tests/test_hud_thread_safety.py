"""HUD thread-safety: agent must never call Qt widgets off the GUI thread.

Root cause (audit B1): wake-word and barge-in callbacks fire on the PortAudio
audio thread and used to call hud.set_state()/set_voice_level() directly.
The fix routes every agent→HUD mutation through a dispatcher bound by the
GUI (a queued pyqtSignal). These tests prove:

1. _hud() routes through the bound dispatcher (any thread → GUI thread).
2. _on_wake_word() invoked from a non-GUI thread pushes HUD work to the
   dispatcher instead of touching the HUD inline.
3. Without a dispatcher (pure-async tests/headless), _hud() stays inline.
"""
import threading
import time
from types import SimpleNamespace

import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.state import JarvisState


def make_agent(hud=None):
    agent = JarvisAgent(AgentConfig(), hud=hud)
    if hud is not None and hasattr(hud, "calls"):
        hud.calls.clear()  # constructor sets assistant name; ignore it
    return agent


class RecordingHUD:
    """Records the thread each setter ran on."""

    def __init__(self):
        self.calls = []  # (method, thread_name)

    def _rec(self, name):
        self.calls.append((name, threading.current_thread().name))

    def set_state(self, state): self._rec(f"set_state:{state}")
    def set_voice_level(self, v): self._rec(f"voice:{v}")
    def set_emotion(self, e): self._rec(f"emotion:{e}")
    def set_assistant_name(self, n): self._rec(f"name:{n}")
    def clear_profile(self): self._rec("clear_profile")
    def set_profile(self, *a): self._rec("set_profile")
    def set_palette_hue(self, h): self._rec("hue")
    def clear_palette_hue(self): self._rec("clear_hue")


class TestDispatcherRouting:
    def test_hud_routes_through_dispatcher(self):
        agent = make_agent()
        dispatched = []
        agent.bind_hud_dispatcher(lambda fn: dispatched.append(fn))
        agent.hud = RecordingHUD()
        agent._hud(lambda: agent.hud.set_state(JarvisState.LISTENING))
        assert len(dispatched) == 1
        assert agent.hud.calls == []  # NOT executed inline
        dispatched[0]()  # simulate GUI-thread execution
        assert len(agent.hud.calls) == 1
        assert "set_state" in agent.hud.calls[0][0]

    def test_no_dispatcher_stays_inline(self):
        agent = make_agent(hud=RecordingHUD())
        agent._hud(lambda: agent.hud.set_state(JarvisState.IDLE))
        assert agent.hud.calls and "set_state" in agent.hud.calls[0][0]

    def test_no_hud_noop(self):
        agent = make_agent(hud=None)
        seen = []
        agent.bind_hud_dispatcher(lambda fn: seen.append(fn))
        agent._hud(lambda: None)
        assert seen == []


class TestAudioThreadWakeWord:
    def test_wake_word_from_audio_thread_uses_dispatcher(self):
        """Simulate the PortAudio-thread wake-word callback on a foreign thread."""
        agent = make_agent(hud=RecordingHUD())
        gui_thread = threading.current_thread().name

        queue = []
        agent.bind_hud_dispatcher(lambda fn: queue.append(fn))

        done = threading.Event()

        def audio_thread():
            agent._on_wake_word()
            done.set()

        t = threading.Thread(target=audio_thread, name="PortAudio-fake")
        t.start()
        assert done.wait(2.0), "wake word callback hung"
        t.join()

        # HUD work must be queued, never executed on the audio thread.
        assert len(queue) == 1
        assert agent.hud.calls == [], "HUD touched directly off the GUI thread!"

        # Drain as the GUI would; verify it then runs on the GUI thread.
        for fn in queue:
            fn()
        assert agent.hud.calls[0][1] == gui_thread

    def test_interrupt_voice_level_uses_dispatcher(self):
        """interrupt() runs on the barge-in audio thread — HUD reset must queue."""
        agent = make_agent(hud=RecordingHUD())
        agent.audio._sd = None  # avoid real audio stop
        queue = []
        agent.bind_hud_dispatcher(lambda fn: queue.append(fn))

        done = threading.Event()

        def audio_thread():
            agent.interrupt()
            done.set()

        threading.Thread(target=audio_thread, name="PortAudio-fake").start()
        assert done.wait(2.0)

        assert len(queue) == 1
        assert agent.hud.calls == []
        for fn in queue:
            fn()
        assert any("voice" in c[0] for c in agent.hud.calls)

    def test_voice_level_from_stt_thread_dispatched(self):
        agent = make_agent(hud=RecordingHUD())
        queue = []
        agent.bind_hud_dispatcher(lambda fn: queue.append(fn))
        agent._on_voice_level(0.5)
        assert len(queue) == 1
        agent.hud.calls == []
