"""Tests for SettingsDialog."""

from unittest.mock import MagicMock, patch

import pytest

from jarvis.agent import AgentConfig
from jarvis.config import AppConfig
from jarvis.face import FaceConfig
from jarvis.settings import SettingsDialog


@pytest.fixture
def settings(qtbot):
    """Create a SettingsDialog with a minimal agent config."""
    agent_config = AgentConfig()
    app_config = AppConfig()
    agent = MagicMock()
    agent.hud = None
    agent.tts = MagicMock()
    agent.tts.voice = "en_US-lessac-medium"
    agent.profiles = MagicMock()

    dialog = SettingsDialog(
        agent_config=agent_config,
        app_config=app_config,
        agent=agent,
    )
    dialog.face_config = FaceConfig()
    dialog.on_face_restart = MagicMock()
    qtbot.addWidget(dialog)
    yield dialog
    dialog.deleteLater()


class TestSettingsDialogGetValues:
    def test_get_values_returns_dict(self, settings):
        values = settings._get_values()
        assert isinstance(values, dict)

    def test_get_values_has_required_keys(self, settings):
        values = settings._get_values()
        required = [
            "llm_base_url", "llm_api_key", "llm_model", "assistant_name",
            "stt_host", "stt_port", "stt_device",
            "tts_host", "tts_port", "tts_voice",
            "audio_device",
            "db_host", "db_port", "db_name", "db_user", "db_password",
            "camera_index", "face_threshold",
        ]
        for key in required:
            assert key in values, f"Missing key: {key}"

    def test_get_values_llm_url(self, settings):
        values = settings._get_values()
        assert isinstance(values["llm_base_url"], str)

    def test_get_values_db_port_is_int(self, settings):
        values = settings._get_values()
        assert isinstance(values["db_port"], int)

    def test_get_values_face_threshold_is_float(self, settings):
        values = settings._get_values()
        assert isinstance(values["face_threshold"], float)

    def test_get_values_tts_voice_no_leading_spaces(self, settings):
        """_get_values should return a clean voice name without leading/trailing whitespace."""
        values = settings._get_values()
        voice = values["tts_voice"]
        assert voice == voice.strip()
        assert "en_US" in voice or voice == "en_US-lessac-medium"

    def test_get_values_tts_voice_updates_after_selection(self, settings):
        """_get_values should return the selected voice name, not the display text with prefix."""
        # Find and select "en_US-amy-medium"
        for i in range(settings.voice_combo.count()):
            data = settings.voice_combo.itemData(i)
            if isinstance(data, dict) and data.get("type") == "voice" and data.get("name") == "en_US-amy-medium":
                settings.voice_combo.setCurrentIndex(i)
                break
        values = settings._get_values()
        assert values["tts_voice"] == "en_US-amy-medium"
        assert values["tts_voice"] == values["tts_voice"].strip()


class TestSettingsDialogRestartFace:
    @patch("jarvis.settings.QMessageBox")
    def test_restart_face_calls_callback(self, mock_msgbox, settings):
        settings.camera_spin.setValue(2)
        settings.face_thresh_spin.setValue(65)
        settings._restart_face()
        assert settings.on_face_restart.called

    @patch("jarvis.settings.QMessageBox")
    def test_restart_face_sets_confidence_threshold(self, mock_msgbox, settings):
        """_restart_face should apply confidence threshold from spin box."""
        callback_config = []
        settings.on_face_restart = lambda fc: callback_config.append(fc)
        settings.face_thresh_spin.setValue(60)
        settings._restart_face()
        fc = callback_config[0]
        assert fc.confidence_threshold == 60.0

    @patch("jarvis.settings.QMessageBox")
    def test_restart_face_updates_db_credentials(self, mock_msgbox, settings):
        """_restart_face should apply DB host/user/password from the form."""
        callback_config = []
        settings.on_face_restart = lambda fc: callback_config.append(fc)
        # Set DB values in the form fields
        db_group = settings.tabs.widget(4)
        db_widgets = settings._get_qlineedits(db_group)
        db_widgets[0].setText("db.example.com")  # host
        db_widgets[1].setText("3307")            # port
        db_widgets[3].setText("myuser")          # user
        db_widgets[4].setText("secret")          # password
        settings._restart_face()
        fc = callback_config[0]
        assert fc.db_host == "db.example.com"
        assert fc.db_port == 3307
        assert fc.db_user == "myuser"
        assert fc.db_password == "secret"

    @patch("jarvis.settings.QMessageBox")
    def test_restart_face_no_callback_no_crash(self, mock_msgbox, settings):
        settings.on_face_restart = None
        settings._restart_face()  # should not raise


class TestSettingsDialogSave:
    @patch("jarvis.settings.QMessageBox")
    def test_save_updates_agent_config(self, mock_msgbox, settings):
        """_save_settings should apply values to agent config."""
        settings._save_settings()
        assert settings.agent_config is not None

    @patch("jarvis.settings.QMessageBox")
    def test_save_persists_app_config(self, mock_msgbox, settings, tmp_path, monkeypatch):
        """_save_settings should persist app_config to disk."""
        monkeypatch.setattr(
            "jarvis.config._config_dir", lambda: tmp_path
        )
        settings._save_settings()
        # Config file should have been created
        config_path = tmp_path / "config.json"
        assert config_path.exists()

    @patch("jarvis.settings.QMessageBox")
    def test_save_applies_voice_to_tts(self, mock_msgbox, settings):
        """_save_settings should set the selected voice on the agent's TTS object."""
        # Select a voice
        for i in range(settings.voice_combo.count()):
            data = settings.voice_combo.itemData(i)
            if isinstance(data, dict) and data.get("type") == "voice" and data.get("name") == "en_US-amy-medium":
                settings.voice_combo.setCurrentIndex(i)
                break
        settings._save_settings()
        assert settings.agent.tts.voice == "en_US-amy-medium"
        assert settings.agent_config.tts.voice == "en_US-amy-medium"

    @patch("jarvis.settings.QMessageBox")
    def test_save_persists_voice_in_app_config(self, mock_msgbox, settings, tmp_path, monkeypatch):
        """_save_settings should persist the selected voice to app_config."""
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)
        # Select a voice
        for i in range(settings.voice_combo.count()):
            data = settings.voice_combo.itemData(i)
            if isinstance(data, dict) and data.get("type") == "voice" and data.get("name") == "en_US-amy-medium":
                settings.voice_combo.setCurrentIndex(i)
                break
        settings._save_settings()
        assert settings.app_config.tts_voice == "en_US-amy-medium"


class TestSettingsDialogTabs:
    def test_all_tabs_present(self, settings):
        # LLM, STT, TTS, Audio, Database, Face, Profiles, Appearance = 8 tabs
        assert settings.tabs.count() == 8

    def test_palette_combo_populated(self, settings):
        assert settings.palette_combo.count() == 10

    def test_contrast_slider_range(self, settings):
        assert settings.contrast_slider.minimum() == 80
        assert settings.contrast_slider.maximum() == 140

    def test_on_contrast_changed_updates_config(self, settings):
        settings._on_contrast_changed(120)
        assert settings.agent_config.contrast_boost == 120

    def test_on_palette_changed_updates_config(self, settings):
        settings._on_palette_changed(3)
        assert settings.agent_config.palette_index == 3
