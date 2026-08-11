"""
Jarvis Configuration — Environment + persistent settings.

Loads configuration from:
1. System/user environment variables (highest priority)
2. .env file (if present)
3. Persistent user config file (~/.config/jarvis-pyqt/config.json)
4. Built-in defaults (lowest priority)
"""

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_KEYCHAIN_ACCOUNT = "jarvis-pyqt"
_KEYCHAIN_SERVICES = {
    "db_password": "JarvisDBPassword",
    "llm_api_key": "JarvisLLMAPIKey",
}


def _keychain_store_secret(service: str, value: str) -> bool:
    """Store a secret value in the macOS Keychain. Returns True on success."""
    svc = _KEYCHAIN_SERVICES.get(service, service)
    try:
        subprocess.run(
            [
                "security", "add-generic-password",
                "-a", _KEYCHAIN_ACCOUNT,
                "-s", svc,
                "-w", value,
                "-A",  # allow access by any app, no prompt
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except subprocess.CalledProcessError as e:
        if e.returncode != 45:  # 45 = item already exists
            logger.warning(f"Keychain store failed (code {e.returncode}): {e.stderr}")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Keychain store unavailable: {e}")
        return False


def _keychain_update_secret(service: str, value: str) -> None:
    """Store or update a secret value in the macOS Keychain."""
    if _keychain_store_secret(service, value):
        return
    # Item already exists — delete and retry once
    svc = _KEYCHAIN_SERVICES.get(service, service)
    try:
        subprocess.run(
            [
                "security", "delete-generic-password",
                "-a", _KEYCHAIN_ACCOUNT,
                "-s", svc,
            ],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _keychain_store_secret(service, value)


# Values that must never be written to keychain/.env — placeholder junk from
# template experiments that silently breaks auth. See jarvis-pyqt skill.
_PLACEHOLDER_SECRETS = {
    "my-api-key",
    "my-secret-pass",
    "changeme",
    "change-me",
    "password",
    "secret",
    "***",
    "***REDACTED***",
    "",
}


def _is_placeholder(value: str | None) -> bool:
    """True if a secret looks like a placeholder rather than a real credential."""
    if value is None:
        return True
    v = value.strip().lower()
    if v in _PLACEHOLDER_SECRETS:
        return True
    # Suspiciously short values (real gateway keys are 40+ chars, db pw 10+)
    if len(value.strip()) < 8:
        return True
    return False


def _keychain_retrieve_secret(service: str) -> str | None:
    """Retrieve a secret value from the macOS Keychain.

    Returns None for placeholder values — they must never shadow real config.
    """
    svc = _KEYCHAIN_SERVICES.get(service, service)
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-a", _KEYCHAIN_ACCOUNT,
                "-s", svc,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        pw = result.stdout.strip()
        if pw and not _is_placeholder(pw):
            logger.debug(f"Retrieved {service} from macOS Keychain")
            return pw
        if pw:
            logger.warning(
                f"Ignoring placeholder {service} in macOS Keychain "
                f"(env/.env wins)"
            )
        return None
    except subprocess.CalledProcessError:
        logger.debug(f"No {service} found in macOS Keychain")
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Keychain unavailable: {e}")
        return None


def _config_dir() -> Path:
    """Get the platform-appropriate config directory."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "jarvis-pyqt"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", "")) / "jarvis-pyqt"
    return Path(os.path.expanduser("~")) / ".config" / "jarvis-pyqt"


def _config_file_path() -> Path:
    """Get the path to the persistent config file."""
    return _config_dir() / "config.json"


@dataclass
class AppConfig:
    """Persistent user configuration (saved to disk)."""

    # LLM
    llm_base_url: str = "http://192.168.55.179:8642/v1"
    llm_api_key: str = ""
    llm_model: str = "auto"

    # STT (Wyoming)
    stt_host: str = "192.168.55.41"
    stt_port: int = 10300

    # TTS (Wyoming)
    tts_host: str = "192.168.55.41"
    tts_port: int = 10200
    tts_voice: str = "en_US-lessac-medium"

    # Audio
    audio_output_device: int = -1

    # Database
    db_host: str = "192.168.55.41"
    db_port: int = 3306
    db_name: str = "jarvis"
    db_user: str = "root"
    db_password: str = ""

    # Face recognition
    camera_index: int = 0
    face_confidence_threshold: float = 0.70

    # Appearance
    palette_index: int = 0
    contrast_boost: int = 100

    # Assistant
    assistant_name: str = "Jarvis"

    # STT
    silence_timeout: float = 2.0
    stt_sensitivity: float = 100.0  # RMS threshold (lower = more sensitive)

    def to_env_dict(self) -> dict:
        """Return config as a dict with env var keys (for debug/logging)."""
        return {
            "JARVIS_LLM_URL": self.llm_base_url,
            "JARVIS_LLM_API_KEY": "***" if self.llm_api_key else "",
            "JARVIS_LLM_MODEL": self.llm_model,
            "JARVIS_STT_HOST": self.stt_host,
            "JARVIS_STT_PORT": str(self.stt_port),
            "JARVIS_TTS_HOST": self.tts_host,
            "JARVIS_TTS_PORT": str(self.tts_port),
            "JARVIS_TTS_VOICE": self.tts_voice,
            "JARVIS_DB_HOST": self.db_host,
            "JARVIS_DB_PORT": str(self.db_port),
            "JARVIS_DB_NAME": self.db_name,
            "JARVIS_DB_USER": self.db_user,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create an AppConfig from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create an AppConfig from environment variables / .env file."""
        return cls(
            llm_base_url=os.environ.get("JARVIS_LLM_URL", "")
            or os.environ.get("JARVIS_LLM_BASE_URL", "")
            or cls.llm_base_url,
            llm_api_key=os.environ.get("JARVIS_LLM_API_KEY", "") or cls.llm_api_key,
            llm_model=os.environ.get("JARVIS_LLM_MODEL", "") or cls.llm_model,
            stt_host=os.environ.get("JARVIS_STT_HOST", "") or cls.stt_host,
            stt_port=_env_int("JARVIS_STT_PORT", cls.stt_port),
            tts_host=os.environ.get("JARVIS_TTS_HOST", "") or cls.tts_host,
            tts_port=_env_int("JARVIS_TTS_PORT", cls.tts_port),
            tts_voice=os.environ.get("JARVIS_TTS_VOICE", "") or cls.tts_voice,
            audio_output_device=_env_int("JARVIS_AUDIO_DEVICE", cls.audio_output_device),
            db_host=os.environ.get("JARVIS_DB_HOST", "") or cls.db_host,
            db_port=_env_int("JARVIS_DB_PORT", cls.db_port),
            db_name=os.environ.get("JARVIS_DB_NAME", "") or cls.db_name,
            db_user=os.environ.get("JARVIS_DB_USER", "") or cls.db_user,
            db_password=os.environ.get("JARVIS_DB_PASSWORD", "") or cls.db_password,
            camera_index=_env_int("JARVIS_CAMERA_INDEX", cls.camera_index),
            face_confidence_threshold=_env_float(
                "JARVIS_FACE_CONFIDENCE", cls.face_confidence_threshold
            ),
            palette_index=_env_int("JARVIS_PALETTE_INDEX", cls.palette_index),
            contrast_boost=_env_int("JARVIS_CONTRAST_BOOST", cls.contrast_boost),
            assistant_name=os.environ.get("JARVIS_ASSISTANT_NAME", "") or cls.assistant_name,
            silence_timeout=_env_float("JARVIS_SILENCE_TIMEOUT", cls.silence_timeout),
            stt_sensitivity=_env_float("JARVIS_STT_SENSITIVITY", cls.stt_sensitivity),
        )

    # Fields that contain secrets and must never be persisted to disk
    _SECRET_FIELDS = {"llm_api_key", "db_password"}

    @classmethod
    def load(cls) -> "AppConfig":
        """Load config: env vars take precedence, fall back to file, then defaults.

        Secrets (db_password) are redacted in the config file.  When the file
        value is the redaction placeholder, try the macOS Keychain before
        falling back to env vars or defaults.  Keychain always wins for secrets.
        """
        cfg = cls.from_env()
        # Also load secrets from .env in config dir (fallback for Keychain failures)
        env_path = _config_dir() / ".env"
        if env_path.exists():
            try:
                load_dotenv(env_path, override=False)
                cfg = cls.from_env()  # Re-read with .env secrets loaded
            except Exception:
                pass
        config_file = _config_file_path()
        if config_file.exists():
            try:
                with open(config_file) as f:
                    saved = json.load(f)
                saved_cfg = cls.from_dict(saved)
                # Clear redacted secret fields so env/defaults fill them
                for secret in cls._SECRET_FIELDS:
                    if getattr(saved_cfg, secret) == "***REDACTED***":
                        setattr(saved_cfg, secret, "")
                merged = _merge_configs(cfg, saved_cfg)
                cfg = merged
                logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        else:
            logger.info("No config file found, using defaults + env vars")
        # Keychain is a FALLBACK only — it must never override explicitly-set
        # env/.env values (a stale keychain entry used to shadow the real key,
        # causing recurring 401s). Only fill secrets that are still empty.
        for secret in cls._SECRET_FIELDS:
            current = getattr(cfg, secret)
            if not current or _is_placeholder(current):
                keychain_val = _keychain_retrieve_secret(secret)
                if keychain_val and not _is_placeholder(keychain_val):
                    setattr(cfg, secret, keychain_val)
        return cfg

    def save(self) -> None:
        """Persist config to disk (secrets redacted, password stored in Keychain)."""
        config_file = _config_file_path()
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self)
            for secret in self._SECRET_FIELDS:
                data[secret] = "***REDACTED***"
            # Write with restricted permissions (owner read/write only)
            fd = os.open(config_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            # Store secrets in macOS Keychain (only real values — never
            # placeholders, otherwise a stale memory value poisons the chain)
            for secret in self._SECRET_FIELDS:
                val = getattr(self, secret)
                if val and not _is_placeholder(val):
                    _keychain_update_secret(secret, val)
            # Also write secrets to .env file as a fallback for auto-login
            _save_secrets_env_file(self)
            logger.info(f"Config saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")


_ENV_SECRET_KEYS = {
    "db_password": "JARVIS_DB_PASSWORD",
    "llm_api_key": "JARVIS_LLM_API_KEY",
    "db_host": "JARVIS_DB_HOST",
    "db_user": "JARVIS_DB_USER",
    "db_name": "JARVIS_DB_NAME",
}


def _save_secrets_env_file(cfg: "AppConfig") -> None:
    """Write non-empty, real secrets to .env file so they survive Keychain failures.

    Merges with the existing file: keeps lines for keys not being updated, so
    a save with an empty in-memory secret never truncates away stored values.
    """
    env_path = _config_dir() / ".env"
    try:
        # Read existing lines (preserve unrelated keys)
        existing: dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip().strip('"').strip("'")

        # Overlay current non-placeholder values
        for field, env_key in _ENV_SECRET_KEYS.items():
            val = getattr(cfg, field, None)
            if val and not _is_placeholder(val):
                existing[env_key] = val

        lines = [f'{k}="{v}"' for k, v in existing.items()]
        env_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.debug(f"Secrets written to {env_path}")
    except Exception as e:
        logger.debug(f"Failed to write secrets env file: {e}")


def _env_int(key: str, default: int) -> int:
    """Read an integer env var with a fallback."""
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Read a float env var with a fallback."""
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def fields(dataclass_cls):
    """Wrapper for dataclasses.fields to avoid import confusion."""
    from dataclasses import fields as dc_fields

    return dc_fields(dataclass_cls)


def _merge_configs(env_cfg: AppConfig, file_cfg: AppConfig) -> AppConfig:
    """Merge two configs, with env values taking precedence for non-empty/non-default values."""
    env_dict = asdict(env_cfg)
    file_dict = asdict(file_cfg)

    # Default config for comparison
    default = asdict(AppConfig())

    merged: dict = {}
    for key in env_dict:
        env_val = env_dict[key]
        file_val = file_dict.get(key)

        # If env value differs from default, it was explicitly set — use it
        if env_val != default.get(key, ...):
            merged[key] = env_val
        elif file_val is not None and str(file_val).strip() != "":
            # Env wasn't explicitly set, use file value
            merged[key] = file_val
        else:
            merged[key] = env_val

    return AppConfig.from_dict(merged)
