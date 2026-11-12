"""Secrets in the settings dialog are masked with a reveal toggle (G-D16)."""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QLineEdit

from jarvis.agent import AgentConfig
from jarvis.config import AppConfig
from jarvis.face import FaceConfig
from jarvis.settings import SettingsDialog


@pytest.fixture
def dialog(qtbot):
    ac = AgentConfig()
    ac.llm_api_key = "sk-secret-123"
    ac.profile_db_password = "dbpass"
    agent = MagicMock()
    agent.hud = None
    agent.tts = MagicMock()
    agent.tts.voice = "en_US-lessac-medium"
    agent.profiles = MagicMock()
    appc = AppConfig()
    appc.db_password = "dbpass"
    d = SettingsDialog(agent_config=ac, app_config=appc, agent=agent)
    d.face_config = FaceConfig()
    d.on_face_restart = MagicMock()
    qtbot.addWidget(d)
    yield d
    d.deleteLater()


def _masked_fields(d):
    return [f for f in d.findChildren(QLineEdit)
            if f.echoMode() == QLineEdit.EchoMode.Password]


def test_api_key_masked_by_default(dialog):
    masked = _masked_fields(dialog)
    # DB password + LLM API key both masked.
    assert len(masked) >= 2
    vals = [f.text() for f in masked]
    assert "sk-secret-123" in vals
    assert "dbpass" in vals


def test_reveal_toggle_flips_echo_mode(dialog):
    field = next(
        f for f in dialog.findChildren(QLineEdit)
        if f.text() == "sk-secret-123"
    )
    acts = field.actions()
    assert acts, "no reveal action on the API key field"
    acts[0].trigger()
    assert field.echoMode() == QLineEdit.EchoMode.Normal
    acts[0].trigger()
    assert field.echoMode() == QLineEdit.EchoMode.Password
