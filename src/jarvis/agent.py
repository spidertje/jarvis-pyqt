"""
Jarvis Agent — orchestrates STT → chat → TTS flow.

State machine:
    IDLE → LISTENING (user speaking)
    LISTENING → THINKING (chat in progress)
    THINKING → SPEAKING (TTS playing)
    SPEAKING → IDLE (done speaking)
"""

import asyncio
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
    # STT silence timeout (seconds of silence to end listening)
    silence_timeout: float = 2.0
    # System prompt for the LLM
    system_prompt: str = "You are Jarvis, a helpful AI assistant. Respond concisely."


class JarvisAgent:
    """
    Full Jarvis agent: STT → Chat → TTS.

    Drives the HUD state machine and manages the conversation flow.
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

        # Conversation history
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
            reply = await self.chat.chat(self._messages)
            if not reply:
                self._set_state(JarvisState.IDLE)
                continue
            self._messages.append({"role": "assistant", "content": reply})

            # Speak
            self._set_state(JarvisState.SPEAKING)
            audio = await self.tts.speak(reply)
            if audio:
                self.audio.play(audio)
            self._set_state(JarvisState.IDLE)

    async def chat_text(self, user_text: str) -> str:
        """
        Direct text chat (no voice). Returns the assistant's response text.
        """
        self._set_state(JarvisState.THINKING)
        self._messages.append({"role": "user", "content": user_text})
        reply = await self.chat.chat(self._messages)
        if reply:
            self._messages.append({"role": "assistant", "content": reply})
        self._set_state(JarvisState.IDLE)
        return reply or ""

    async def chat_text_and_speak(self, user_text: str) -> str:
        """
        Text chat with TTS response.
        """
        text = await self.chat_text(user_text)
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
