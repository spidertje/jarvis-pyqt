"""Status-pill regressions: pills must reflect real availability.

The original C13 badge code read a non-existent `agent.llm_base_url`
inside try/except Exception — every pill rendered RED regardless of
service state. Guard the attribute path and the health-state plumbing.
"""
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

import main as app_main


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


class FakeCfg:
    llm_base_url = "http://llm.test/v1"
    llm_api_key = ""
    llm_model = "m"


class FakeSvcCfg:
    host = "127.0.0.1"
    port = 1
    voice = "v"


class FakeSvc:
    config = FakeSvcCfg()
    _connected = True


class FakeChat:
    last_error = None


class FakeAgent:
    config = FakeCfg()
    stt = FakeSvc()
    tts = FakeSvc()
    chat = FakeChat()


def _make_pill_target(qapp):
    """Build a bare object with just what _update_status_pills touches."""
    obj = app_main.JarvisApp.__new__(app_main.JarvisApp)
    from PyQt6.QtWidgets import QLabel

    obj.agent = FakeAgent()
    obj.status_pill = QLabel()
    obj._theme = type("T", (), {"ok": "#00ff00", "error": "#ff0000"})()
    return obj


def test_pills_do_not_crash_on_real_agent_shape(qapp):
    """The fallback path must read agent.config (not agent.llm_base_url)."""
    obj = _make_pill_target(qapp)
    app_main.JarvisApp._update_status_pills(obj)  # must not raise, no red-out
    html = obj.status_pill.text()
    assert "#00ff00" in html  # connected services render green


def test_pills_use_probed_health_state(qapp):
    obj = _make_pill_target(qapp)
    obj._service_health = (False, True, True)  # LLM down, voice up
    app_main.JarvisApp._update_status_pills(obj)
    html = obj.status_pill.text()
    assert html.index("#ff0000") < html.index("LLM")  # red dot before LLM label
    assert html.count("#00ff00") == 2  # STT + TTS green
