"""Tests for the AppConfig configuration module."""

import json

import pytest

from jarvis.config import AppConfig


@pytest.fixture
def mock_keychain(monkeypatch):
    """Mock macOS Keychain and .env file to avoid touching real stores."""
    # Clean up any env vars that load_dotenv might have set from prior tests
    for key in ["JARVIS_DB_PASSWORD", "JARVIS_LLM_API_KEY",
                "JARVIS_DB_HOST", "JARVIS_DB_USER", "JARVIS_DB_NAME"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("jarvis.config._keychain_store_secret", lambda s, v: False)
    monkeypatch.setattr("jarvis.config._keychain_update_secret", lambda s, v: None)
    monkeypatch.setattr("jarvis.config._keychain_retrieve_secret", lambda s: None)
    monkeypatch.setattr("jarvis.config._save_secrets_env_file", lambda cfg: None)


class TestAppConfigDefaults:
    def test_defaults(self):
        """AppConfig should have sensible default values."""
        cfg = AppConfig()
        assert cfg.llm_base_url == "http://192.168.55.179:8642/v1"
        assert cfg.llm_model == "auto"
        assert cfg.tts_voice == "en_US-lessac-medium"
        assert cfg.db_name == "jarvis"
        assert cfg.assistant_name == "Jarvis"
        assert cfg.silence_timeout == 2.0
        assert cfg.stt_sensitivity == 100.0
        assert cfg.palette_index == 0
        assert cfg.contrast_boost == 100


class TestAppConfigEnv:
    def test_from_env_reads_env_vars(self, monkeypatch):
        """from_env should pick up JARVIS_* environment variables."""
        monkeypatch.setenv("JARVIS_LLM_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("JARVIS_LLM_API_KEY", "secret123")
        monkeypatch.setenv("JARVIS_STT_PORT", "9999")
        monkeypatch.setenv("JARVIS_TTS_VOICE", "en_US-amy-low")
        monkeypatch.setenv("JARVIS_DB_USER", "testuser")

        cfg = AppConfig.from_env()
        assert cfg.llm_base_url == "http://localhost:8080/v1"
        assert cfg.llm_api_key == "secret123"
        assert cfg.stt_port == 9999
        assert cfg.tts_voice == "en_US-amy-low"
        assert cfg.db_user == "testuser"

    def test_from_env_falls_back_to_defaults(self, monkeypatch):
        """from_env should use defaults when env vars are not set."""
        # Clear any env vars that might be set
        for key in [
            "JARVIS_LLM_URL",
            "JARVIS_LLM_BASE_URL",
            "JARVIS_LLM_API_KEY",
            "JARVIS_LLM_MODEL",
            "JARVIS_STT_HOST",
            "JARVIS_STT_PORT",
            "JARVIS_TTS_HOST",
            "JARVIS_TTS_PORT",
            "JARVIS_TTS_VOICE",
            "JARVIS_DB_HOST",
            "JARVIS_DB_PORT",
            "JARVIS_DB_NAME",
            "JARVIS_DB_USER",
            "JARVIS_DB_PASSWORD",
            "JARVIS_STT_SENSITIVITY",
            "JARVIS_SILENCE_TIMEOUT",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg = AppConfig.from_env()
        assert cfg.llm_base_url == AppConfig.llm_base_url
        assert cfg.tts_voice == "en_US-lessac-medium"
        assert cfg.db_name == "jarvis"

    def test_from_env_reads_sensitivity(self, monkeypatch):
        """from_env should pick up JARVIS_STT_SENSITIVITY."""
        monkeypatch.setenv("JARVIS_STT_SENSITIVITY", "50")
        cfg = AppConfig.from_env()
        assert cfg.stt_sensitivity == 50.0


class TestAppConfigPersist:
    def test_save_and_load(self, tmp_path, monkeypatch, mock_keychain):
        """save() then load() should round-trip non-secret config values."""
        # Override config dir to temp
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)

        cfg = AppConfig()
        cfg.llm_api_key = "test_key"
        cfg.db_password = "secret_db_pass"
        cfg.tts_voice = "en_US-test-voice"
        cfg.palette_index = 3
        cfg.stt_sensitivity = 50.0
        cfg.save()

        loaded = AppConfig.load()
        assert loaded.tts_voice == "en_US-test-voice"
        assert loaded.palette_index == 3
        assert loaded.stt_sensitivity == 50.0
        # Secrets must NOT be persisted to disk
        assert loaded.llm_api_key == ""
        assert loaded.db_password == ""

    def test_save_redacts_secrets_on_disk(self, tmp_path, monkeypatch, mock_keychain):
        """Secrets must be redacted in the persisted config file."""
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)

        cfg = AppConfig()
        cfg.llm_api_key = "my-secret-key"
        cfg.db_password = "my-secret-pass"
        cfg.save()

        with open(tmp_path / "config.json") as f:
            saved = json.load(f)
        assert saved["llm_api_key"] == "***REDACTED***"
        assert saved["db_password"] == "***REDACTED***"
        assert saved["tts_voice"] == "en_US-lessac-medium"

    def test_load_nonexistent_file(self, tmp_path, monkeypatch, mock_keychain):
        """load() should work when no config file exists."""
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)
        cfg = AppConfig.load()
        assert cfg.llm_model == "auto"

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict should ignore keys that don't match dataclass fields."""
        data = {
            "llm_model": "gemma",
            "unknown_key": "should be ignored",
            "palette_index": 5,
        }
        cfg = AppConfig.from_dict(data)
        assert cfg.llm_model == "gemma"
        assert cfg.palette_index == 5

    def test_save_stores_secrets_in_keychain(self, tmp_path, monkeypatch):
        """save() should store secrets in Keychain when non-empty."""
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)
        # Prevent .env file creation to isolate Keychain behavior
        monkeypatch.setattr("jarvis.config._save_secrets_env_file", lambda cfg: None)
        stored = {}

        def fake_update(service, value):
            stored[service] = value

        monkeypatch.setattr("jarvis.config._keychain_update_secret", fake_update)
        monkeypatch.setattr("jarvis.config._keychain_retrieve_secret", lambda s: None)

        cfg = AppConfig()
        cfg.db_password = "my-secret-pass"
        cfg.llm_api_key = "my-api-key"
        cfg.save()

        assert stored == {"db_password": "my-secret-pass", "llm_api_key": "my-api-key"}

    def test_load_restores_secrets_from_keychain(self, tmp_path, monkeypatch):
        """load() should restore secrets from Keychain when file has redacted values."""
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)
        # Prevent .env file creation to isolate Keychain behavior
        monkeypatch.setattr("jarvis.config._save_secrets_env_file", lambda cfg: None)
        # Clean up env vars
        for key in ["JARVIS_DB_PASSWORD", "JARVIS_LLM_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        # Write a config file with redacted passwords
        cfg = AppConfig()
        cfg.db_host = "192.168.55.41"
        cfg.db_user = "root"
        cfg.db_password = "my-secret-pass"
        cfg.llm_api_key = "my-api-key"
        cfg.save()

        # Now mock Keychain to return the secrets
        keychain_secrets = {
            "db_password": "my-secret-pass",
            "llm_api_key": "my-api-key",
        }
        monkeypatch.setattr(
            "jarvis.config._keychain_retrieve_secret",
            lambda s: keychain_secrets.get(s),
        )

        loaded = AppConfig.load()
        assert loaded.db_password == "my-secret-pass"
        assert loaded.llm_api_key == "my-api-key"
        assert loaded.db_host == "192.168.55.41"
        assert loaded.db_user == "root"

    def test_load_falls_back_when_keychain_empty(self, tmp_path, monkeypatch, mock_keychain):
        """load() should fall back to env/default when Keychain is empty."""
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)

        cfg = AppConfig()
        cfg.db_password = "original"
        cfg.llm_api_key = "original-key"
        cfg.save()

        loaded = AppConfig.load()
        assert loaded.db_password == ""

    def test_load_keychain_wins_over_stale_env(self, tmp_path, monkeypatch):
        """Keychain secrets should override stale .env values on load."""
        monkeypatch.setattr("jarvis.config._config_dir", lambda: tmp_path)
        # Clean env
        for key in ["JARVIS_DB_PASSWORD", "JARVIS_LLM_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        # Write a .env with a stale password
        env_file = tmp_path / ".env"
        env_file.write_text('JARVIS_DB_PASSWORD="stale-pass"\nJARVIS_DB_HOST="192.168.55.41"\n')

        # Mock Keychain to return the NEW password
        monkeypatch.setattr(
            "jarvis.config._keychain_retrieve_secret",
            lambda s: {"db_password": "new-pass", "llm_api_key": "new-key"}.get(s),
        )

        loaded = AppConfig.load()
        assert loaded.db_password == "new-pass"
        assert loaded.db_host == "192.168.55.41"
