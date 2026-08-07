"""
Jarvis Agent — orchestrates STT → chat → TTS flow.

State machine:
    IDLE → LISTENING (user speaking)
    LISTENING → THINKING (chat in progress)
    THINKING → SPEAKING (TTS playing)
    SPEAKING → IDLE (done speaking)
"""

import asyncio
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict

from .state import JarvisState
from .chat import ChatClient, ChatConfig
from .stt import WhisperSTT, WyomingConfig as STTConfig
from .tts import PiperTTS, WyomingConfig as TTSConfig
from .audio_player import AudioPlayer, AudioConfig
from .profile import ProfileManager, Profile

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Full agent configuration."""
    # LLM
    chat: ChatConfig = field(default_factory=ChatConfig)
    # STT
    stt: STTConfig = field(default_factory=STTConfig)
    # TTS
    tts: TTSConfig = field(default_factory=TTSConfig)
    # Audio
    audio: AudioConfig = field(default_factory=AudioConfig)
    # Profile manager — read from env vars only, no hardcoded fallbacks
    profile_db_host: Optional[str] = None
    profile_db_port: Optional[int] = None
    profile_db_user: Optional[str] = None
    profile_db_password: Optional[str] = None
    profile_db_name: Optional[str] = None
    # LLM API — read from env vars only, no hardcoded fallbacks
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    # STT silence timeout
    silence_timeout: float = 2.0
    palette_index: Optional[int] = None  # index into appearance palette list
    # Default system prompt (used when no profile is active)
    # Loaded from SOUL.md if present, otherwise fallback
    default_system_prompt: str = ""
    # Contrast boost multiplier for HUD saturation (100 = normal)
    contrast_boost: int = 100
    # TTS voice (Piper voice name)
    tts_voice: str = "en_US-lessac-medium"
    # Assistant name (extracted from SOUL.md or set via settings)
    assistant_name: str = "Jarvis"


class JarvisAgent:
    """
    Full Jarvis agent: STT → Chat → TTS.

    Drives the HUD state machine and manages the conversation flow.
    Supports profile switching based on face recognition.
    """

    _DEFAULT_LLM_URL = "http://192.168.55.179:8642/v1"

    def __init__(self, config: Optional[AgentConfig] = None, hud=None):
        self.config = config or AgentConfig()
        self.state = JarvisState.IDLE
        self.hud = hud
        self._state_callbacks: List[Callable] = []

        # Resolve None config values from env vars
        if self.config.profile_db_host is None:
            self.config.profile_db_host = os.environ.get("JARVIS_DB_HOST")
        if self.config.profile_db_port is None:
            self.config.profile_db_port = int(os.environ.get("JARVIS_DB_PORT", "3306"))
        if self.config.profile_db_user is None:
            self.config.profile_db_user = os.environ.get("JARVIS_DB_USER")
        if self.config.profile_db_password is None:
            self.config.profile_db_password = os.environ.get("JARVIS_DB_PASSWORD")
        if self.config.profile_db_name is None:
            self.config.profile_db_name = os.environ.get("JARVIS_DB_NAME")
        self.config.llm_base_url = self.config.llm_base_url or os.environ.get("JARVIS_LLM_URL") or os.environ.get("JARVIS_LLM_BASE_URL") or self._DEFAULT_LLM_URL
        self.config.llm_api_key = self.config.llm_api_key or os.environ.get("JARVIS_LLM_API_KEY")
        self.config.llm_model = self.config.llm_model or os.environ.get("JARVIS_LLM_MODEL")

        # Sub-services
        chat_cfg = ChatConfig()
        if self.config.llm_base_url:
            chat_cfg.base_url = self.config.llm_base_url
        if self.config.llm_api_key:
            chat_cfg.api_key = self.config.llm_api_key
        if self.config.llm_model:
            chat_cfg.model = self.config.llm_model
        # Copy all LLM config fields that ChatConfig supports
        chat_cfg.temperature = self.config.chat.temperature
        chat_cfg.max_tokens = self.config.chat.max_tokens
        # Don't set system_prompt in ChatConfig - we pass it per-call via messages
        chat_cfg.system_prompt = ""
        chat_cfg.timeout = self.config.chat.timeout
        self.chat = ChatClient(chat_cfg)
        # STT
        stt_cfg = self.config.stt
        stt_cfg.host = os.environ.get("JARVIS_STT_HOST") or stt_cfg.host
        stt_cfg.port = int(os.environ.get("JARVIS_STT_PORT") or stt_cfg.port)
        if stt_cfg.device is not None and stt_cfg.device < 0:
            stt_cfg.device = None
        self.stt = WhisperSTT(stt_cfg)
        if hasattr(self.config, 'tts_voice') and self.config.tts_voice:
            self.config.tts.voice = self.config.tts_voice
        self.tts = PiperTTS(self.config.tts)
        self.audio = AudioPlayer(self.config.audio)

        # Profile manager
        self.profiles = ProfileManager(
            db_host=self.config.profile_db_host,
            db_port=self.config.profile_db_port,
            db_user=self.config.profile_db_user,
            db_password=self.config.profile_db_password,
            db_name=self.config.profile_db_name,
        )

        # Load profiles
        self.profiles.load_all()

        # Import assistant name from MariaDB (default profile row)
        db_name = self.profiles.get_default_assistant_name()
        if db_name:
            self.config.assistant_name = db_name
            logger.info(f"Assistant name loaded from DB: {db_name}")

        # HUD - update with assistant name from DB
        self.hud.set_assistant_name(self.config.assistant_name)

        # Load default profile's system prompt at startup
        default_profile = self.profiles.get_default()
        if default_profile:
            self._system_prompt = default_profile.system_prompt
        else:
            self._system_prompt = self.config.default_system_prompt

        # Active profile
        self._active_profile: Optional[Profile] = None

        # Conversation history (per profile)
        self._messages: List[Dict[str, str]] = []

    def on_state_change(self, callback: Callable):
        """Register a callback for state changes."""
        self._state_callbacks.append(callback)

    def _set_state(self, new_state: JarvisState):
        """Set state and notify callbacks."""
        if self.state != new_state:
            logger.info(f"State: {self.state.label} → {new_state.label}")
            self.state = new_state
            for cb in self._state_callbacks:
                cb(new_state)

    def get_system_prompt(self) -> str:
        """Get the current system prompt (profile-specific or default)."""
        return self._system_prompt

    def switch_profile(self, name: str) -> bool:
        """
        Switch to a profile by name.

        Loads the profile's system prompt, chat history, and assistant name.
        """
        if not self.profiles.switch(name):
            return False

        self._active_profile = self.profiles.get(name)
        if self._active_profile:
            self._system_prompt = self._active_profile.system_prompt
            self.config.assistant_name = self._active_profile.assistant_name
            # Update HUD with new assistant name
            if self.hud:
                self.hud.set_assistant_name(self.config.assistant_name)
            self._messages = list(self._active_profile.chat_history)
            logger.info(
                f"Profile switched to: {name} "
                f"(assistant: {self.config.assistant_name}, "
                f"prompt: {len(self._system_prompt)} chars, "
                f"history: {len(self._messages)} messages)"
            )
        return True

    def clear_profile(self):
        """Clear active profile (return to default)."""
        self.profiles.clear()
        self._active_profile = None
        self._system_prompt = self.config.default_system_prompt or (
            "You are Jarvis, a helpful AI assistant."
        )
        self._messages = []

    def _on_voice_level(self, level: float):
        """Update HUD voice bars from microphone amplitude (called during STT listen)."""
        if self.hud:
            self.hud.set_voice_level(level)

    async def _run_voice(self):
        """Run a single voice cycle: listen → think → speak."""
        # Listen
        self._set_state(JarvisState.LISTENING)
        logger.info("STT: Starting listen cycle")
        text = await self.stt.listen(timeout=8.0, silence_threshold=100, on_voice_level=self._on_voice_level)
        logger.info(f"STT: Listen returned: {text!r}")
        if not text:
            if self.hud:
                self.hud.set_voice_level(0.0)
            self._set_state(JarvisState.IDLE)
            return

        # Think (send to LLM)
        self._set_state(JarvisState.THINKING)
        self._messages.append({"role": "user", "content": text})
        reply = await self.chat.chat(self._messages, system_prompt=self._system_prompt)
        if not reply:
            self._set_state(JarvisState.IDLE)
            return
        self._messages.append({"role": "assistant", "content": reply})

        # Save history to active profile
        if self._active_profile:
            self._active_profile.chat_history = list(self._messages)
            self.profiles.save(self._active_profile)

        # Speak
        self._set_state(JarvisState.SPEAKING)
        audio = await self.tts.speak(reply)
        if audio:
            self.audio.play(audio)
        self._set_state(JarvisState.IDLE)

    async def chat_text(self, user_text: str,
                        system_prompt: Optional[str] = None) -> str:
        """
        Direct text chat (no voice). Returns the assistant's response text.
        """
        self._set_state(JarvisState.THINKING)
        prompt = system_prompt or self._system_prompt
        self._messages.append({"role": "user", "content": user_text})
        reply = await self.chat.chat(self._messages, system_prompt=prompt)
        if reply:
            self._messages.append({"role": "assistant", "content": reply})
        self._set_state(JarvisState.IDLE)
        return reply or ""

    async def chat_text_and_speak(self, user_text: str,
                                  system_prompt: Optional[str] = None) -> str:
        """
        Text chat with TTS response.
        """
        prompt = system_prompt or self._system_prompt
        text = await self.chat_text(user_text, system_prompt=prompt)
        if text:
            self._set_state(JarvisState.SPEAKING)
            audio = await self.tts.speak(text)
            if audio:
                self.audio.play(audio)
            self._set_state(JarvisState.IDLE)
        return text or ""

    async def connect_all(self) -> bool:
        """Connect to all services. Returns True if all connected."""
        stt_ok = await self.stt.connect()
        tts_ok = await self.tts.connect()
        if not stt_ok:
            logger.warning("STT not connected — voice input disabled")
        if not tts_ok:
            logger.warning("TTS not connected — voice output disabled")
        return stt_ok and tts_ok

    async def close(self):
        """Close all connections."""
        await self.chat.close()
        await self.stt.disconnect()
        await self.tts.disconnect()
        self.audio.stop()
        self.profiles.close()
