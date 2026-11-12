"""Jarvis Desktop - PyQt6 Arc Reactor HUD with face recognition and TTS/STT.

This package uses PEP 562 lazy attribute loading: submodules (and their heavy
PyQt6 / opencv / openwakeword dependencies) are imported only when a name is
first accessed. Importing ``jarvis`` — or lightweight submodules like
``jarvis.state`` or ``jarvis.emotion`` — stays cheap and works headless
(CI boxes without libEGL/GUI libraries).
"""

import importlib

__version__ = "0.6.0"

# name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    "JarvisState": ".state",
    "HUDOverlay": ".hud_overlay",
    "SmokeFaceHUD": ".smoke_face",
    "EMOTION_PROMPT": ".emotion",
    "EmotionTagFilter": ".emotion",
    "classify_emotion": ".emotion",
    "parse_emotion": ".emotion",
    "ChatClient": ".chat",
    "ChatConfig": ".chat",
    "WhisperSTT": ".stt",
    "PiperTTS": ".tts",
    "AudioPlayer": ".audio_player",
    "AudioConfig": ".audio_player",
    "JarvisAgent": ".agent",
    "AgentConfig": ".agent",
    "FaceRecognizer": ".face",
    "FaceConfig": ".face",
    "Profile": ".profile",
    "ProfileManager": ".profile",
    "SettingsDialog": ".settings",
    "AppConfig": ".config",
}

__all__ = list(_EXPORTS)

# Guarantee module attributes for lightweight, GUI/hardware-free submodules so
# string-path tools (monkeypatch("jarvis.config.x"), patch("jarvis.agent.y"))
# resolve regardless of import order. The GUI/heavy modules (hud_overlay,
# settings, face, stt, ...) stay lazy and resolve via __getattr__/sys.modules
# once imported normally.
import importlib as _il

for _sub in ("state", "emotion", "config", "chat", "streaming"):
    _il.import_module(f".{_sub}", __name__)
del _sub


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        # Allow submodule access (e.g. `import jarvis.agent`, jarvis.agent.x)
        # without eager-importing anything.
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache: subsequent accesses skip __getattr__
    return value


def __dir__():
    return sorted(set(list(globals()) + __all__))
