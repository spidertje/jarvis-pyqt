"""Jarvis Desktop - PyQt6 Arc Reactor HUD with face recognition and TTS/STT."""

__version__ = "0.2.0"

from .state import JarvisState
from .hud_overlay import HUDOverlay
from .chat import ChatClient, ChatConfig
from .stt import WhisperSTT
from .tts import PiperTTS
from .audio_player import AudioPlayer, AudioConfig
from .agent import JarvisAgent, AgentConfig

__all__ = [
    "JarvisState", "HUDOverlay",
    "ChatClient", "ChatConfig",
    "WhisperSTT", "PiperTTS",
    "AudioPlayer", "AudioConfig",
    "JarvisAgent", "AgentConfig",
]
