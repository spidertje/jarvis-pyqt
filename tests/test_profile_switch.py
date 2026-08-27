"""Tests for profile switching (Phase 5).

Covers: switch_profile applies prompt/name/history AND accent hue to the HUD,
fires the profile-change callback; clear_profile restores the HUD; the
Settings Profiles tab Activate/Reset controls drive the agent.
"""

from unittest.mock import MagicMock

import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.profile import PALETTE_HUES, Profile


def _profile(name="Alice", hue=45, prompt="You are Alice.", assistant="Alice"):
    return Profile(
        name=name,
        assistant_name=assistant,
        system_prompt=prompt,
        chat_history=[{"role": "user", "content": "hi"}],
        accent_hue=hue,
        enabled=True,
    )


class TestSwitchProfileHud:
    def test_switch_applies_hue_and_name_to_hud(self):
        agent = JarvisAgent(AgentConfig())
        hud = MagicMock()
        agent.hud = hud
        p = _profile()
        agent.profiles = MagicMock()
        agent.profiles.switch.return_value = True
        agent.profiles.get.return_value = p

        assert agent.switch_profile("Alice") is True
        assert agent._active_profile is p
        assert agent.get_system_prompt() == "You are Alice."
        assert agent.config.assistant_name == "Alice"
        assert agent._messages == [{"role": "user", "content": "hi"}]
        hud.set_assistant_name.assert_called_with("Alice")
        hud.set_profile.assert_called_with("Alice", 45)

    def test_switch_unknown_profile_fails(self):
        agent = JarvisAgent(AgentConfig())
        agent.hud = MagicMock()
        agent.profiles = MagicMock()
        agent.profiles.switch.return_value = False
        assert agent.switch_profile("Nobody") is False
        assert agent._active_profile is None

    def test_switch_fires_profile_callback(self):
        agent = JarvisAgent(AgentConfig())
        agent.hud = MagicMock()
        p = _profile()
        agent.profiles = MagicMock()
        agent.profiles.switch.return_value = True
        agent.profiles.get.return_value = p

        seen = []
        agent.on_profile_changed(lambda prof: seen.append(prof))
        assert agent.switch_profile("Alice") is True
        assert seen == [p]

    def test_callback_exception_isolated(self):
        agent = JarvisAgent(AgentConfig())
        agent.hud = MagicMock()
        p = _profile()
        agent.profiles = MagicMock()
        agent.profiles.switch.return_value = True
        agent.profiles.get.return_value = p
        agent.on_profile_changed(lambda _x: 1 / 0)  # must not propagate
        agent.on_profile_changed(lambda _x: None)
        assert agent.switch_profile("Alice") is True

    def test_clear_profile_restores_hud(self):
        agent = JarvisAgent(AgentConfig())
        hud = MagicMock()
        agent.hud = hud
        # default palette index → a known hue
        agent.config.palette_index = 1  # copper → 30
        p = _profile()
        agent.profiles = MagicMock()
        agent._active_profile = p
        agent._system_prompt = "custom"

        agent.clear_profile()
        assert agent._active_profile is None
        assert agent._messages == []
        # HUD restored: profile cleared + default palette applied
        hud.clear_profile.assert_called_once()
        hud.set_palette_hue.assert_called_with(PALETTE_HUES[1])
        hud.set_assistant_name.assert_called()

    def test_clear_profile_fires_callback_with_none(self):
        agent = JarvisAgent(AgentConfig())
        agent.hud = MagicMock()
        agent.profiles = MagicMock()
        seen = []
        agent.on_profile_changed(lambda prof: seen.append(prof))
        agent.clear_profile()
        assert seen == [None]


class TestPaletteConstant:
    def test_palette_has_ten_entries(self):
        assert len(PALETTE_HUES) == 10

    def test_default_is_cyan(self):
        assert PALETTE_HUES[0] == 182


class TestSettingsProfilesTab:
    @pytest.fixture
    def dialog(self, qtbot):
        from jarvis.face import FaceConfig
        from jarvis.settings import SettingsDialog

        agent = MagicMock()
        agent.profiles = MagicMock()
        agent.profiles.list_names.return_value = ["Alice", "Bob"]
        agent.profiles.get.return_value = _profile("Alice")
        agent._active_profile = None
        agent.switch_profile = MagicMock(return_value=True)
        agent.clear_profile = MagicMock()
        agent.on_profile_changed = MagicMock()

        dlg = SettingsDialog(agent_config=AgentConfig(), app_config=None, agent=agent)
        dlg.face_config = FaceConfig()
        dlg.on_face_restart = MagicMock()
        qtbot.addWidget(dlg)
        yield dlg, agent
        dlg.deleteLater()

    def test_activate_calls_switch(self, dialog):
        dlg, agent = dialog
        dlg.profile_combo.setCurrentText("Alice")
        from PyQt6.QtWidgets import QMessageBox

        with _no_messagebox():
            dlg._activate_profile()
        agent.switch_profile.assert_called_once_with("Alice")

    def test_reset_calls_clear(self, dialog):
        dlg, agent = dialog
        with _no_messagebox():
            dlg._reset_profile()
        agent.clear_profile.assert_called_once()

    def test_status_reflects_active(self, dialog):
        dlg, agent = dialog
        agent._active_profile = _profile("Bob")
        dlg._update_profile_status()
        assert "Bob" in dlg.profile_status.text()

    def test_status_default_when_none(self, dialog):
        dlg, agent = dialog
        agent._active_profile = None
        dlg._update_profile_status()
        assert "default" in dlg.profile_status.text().lower()


def _no_messagebox():
    """Context manager that silences QMessageBox during a test action."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        from PyQt6.QtWidgets import QMessageBox

        orig = (
            QMessageBox.information,
            QMessageBox.warning,
            QMessageBox.critical,
        )
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        QMessageBox.critical = staticmethod(lambda *a, **k: None)
        try:
            yield
        finally:
            QMessageBox.information, QMessageBox.warning, QMessageBox.critical = orig

    return _ctx()
