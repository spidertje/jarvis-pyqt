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
    # Profile manager
    profile_db_host: str = "192.168.55.41"
    profile_db_port: int = 3306
    profile_db_user: str = "root"
    profile_db_password: str = "rocklobster"
    profile_db_name: str = "jarvis"
    # STT silence timeout
    silence_timeout: float = 2.0
    # Default system prompt (used when no profile is active)
    default_system_prompt: str = "You are Jarvis, a helpful AI assistant."


class JarvisAgent:
    """
    Full Jarvis agent: STT → Chat → TTS.

    Drives the HUD state machine and manages the conversation flow.
    Supports profile switching based on face recognition.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.state = JarvisState.IDLE
        self._state_callbacks: List[Callable] = []

        # Sub-services
        self.chat = ChatClient(self.config.chat)
        self.stt = WhisperSTT(self.config.stt)
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

        # Active profile
        self._active_profile: Optional[Profile] = None

        # Current system prompt (may be overridden by profile)
        self._system_prompt: str = self.config.default_system_prompt

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

        Loads the profile's system prompt and chat history.
        """
        if not self.profiles.switch(name):
            return False

        self._active_profile = self.profiles.get(name)
        if self._active_profile:
            self._system_prompt = self._active_profile.system_prompt
            self._messages = list(self._active_profile.chat_history)
            logger.info(
                f"Profile switched to: {name} "
                f"(prompt: {len(self._system_prompt)} chars, "
                f"history: {len(self._messages)} messages)"
            )
        return True

    def clear_profile(self):
        """Clear active profile (return to default)."""
        self.profiles.clear()
        self._active_profile = None
        self._system_prompt = self.config.default_system_prompt
        self._messages = []
        logger.info("Profile cleared, back to default")

    async def _run_voice(self):
        """Run voice loop: listen → think → speak."""
        while True:
            # Listen
            self._set_state(JarvisState.LISTENING)
            text = await self.stt.listen()
            if not text:
                self._set_state(JarvisState.IDLE)
                continue

            # Think (send to LLM)
            self._set_state(JarvisState.THINKING)
            self._messages.append({"role": "user", "content": text})
            reply = await self.chat.chat(self._messages, system_prompt=self._system_prompt)
            if not reply:
                self._set_state(JarvisState.IDLE)
                continue
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
