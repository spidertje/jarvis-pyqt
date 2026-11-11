"""Jarvis Desktop - PyQt6 Arc Reactor HUD with face recognition and TTS/STT."""

__version__ = "0.6.0"

from .agent import AgentConfig, JarvisAgent
from .audio_player import AudioConfig, AudioPlayer
from .chat import ChatClient, ChatConfig
from .config import AppConfig
from .face import FaceConfig, FaceRecognizer
from .hud_overlay import HUDOverlay
from .smoke_face import SmokeFaceHUD
from .emotion import EMOTION_PROMPT, EmotionTagFilter, classify_emotion, parse_emotion
from .profile import Profile, ProfileManager
from .settings import SettingsDialog
from .state import JarvisState
from .stt import WhisperSTT
from .tts import PiperTTS

__all__ = [
    "JarvisState",
    "HUDOverlay",
    "SmokeFaceHUD",
    "EMOTION_PROMPT",
    "EmotionTagFilter",
    "classify_emotion",
    "parse_emotion",
    "ChatClient",
    "ChatConfig",
    "WhisperSTT",
    "PiperTTS",
    "AudioPlayer",
    "AudioConfig",
    "JarvisAgent",
    "AgentConfig",
    "FaceRecognizer",
    "FaceConfig",
    "Profile",
    "ProfileManager",
    "SettingsDialog",
    "AppConfig",
]
