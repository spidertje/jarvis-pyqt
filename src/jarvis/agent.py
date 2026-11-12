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
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field


from .audio_player import AudioConfig, AudioPlayer
from .barge_in import BargeInListener
from .chat import ChatClient, ChatConfig
from .profile import PALETTE_HUES, Profile, ProfileManager
from .state import JarvisState
from .streaming import SentenceBuffer, split_sentences
from .emotion import EMOTION_PROMPT, EmotionTagFilter, classify_emotion, parse_emotion
from .stt import WhisperSTT
from .stt import WyomingConfig as STTConfig
from .tts import PiperTTS
from .tts import WyomingConfig as TTSConfig
from .wake_word import WakeWordDetector

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
    # HUD frontend ('arc' | 'smoke')
    frontend: str = "arc"
    # Profile manager — read from env vars only, no hardcoded fallbacks
    profile_db_host: str | None = None
    profile_db_port: int | None = None
    profile_db_user: str | None = None
    profile_db_password: str | None = None
    profile_db_name: str | None = None
    # LLM API — read from env vars only, no hardcoded fallbacks
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    # STT silence timeout
    silence_timeout: float = 2.0
    # STT detection sensitivity (RMS amplitude threshold — lower = more sensitive)
    silence_threshold: float = 100.0
    # Barge-in sensitivity (16-bit RMS scale; higher = less sensitive)
    barge_in_threshold: float = 300.0
    barge_in_min_speech_ms: float = 250.0
    # Wake word (openWakeWord) — say this to start a conversation hands-free
    wake_word_enabled: bool = True
    wake_word: str = "hey_jarvis"
    wake_word_threshold: float = 0.5
    wake_word_patience: int = 2
    wake_word_cooldown_s: float = 1.5
    # Voice-cycle hard timeout (seconds). If the entire listen→think→speak cycle
    # hangs for this long (network hang, service wedged), the cycle is cancelled
    # and an error is surfaced. 0 disables.
    voice_cycle_timeout: float = 90.0
    # Conversation history cap (messages, user+assistant counted together).
    # Oldest turns are dropped so the context can never grow past the model's
    # limit and the persisted JSON column stays bounded. 0 disables trimming.
    max_history_messages: int = 40
    palette_index: int | None = None  # index into appearance palette list
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

    def __init__(self, config: AgentConfig | None = None, hud=None):
        self.config = config or AgentConfig()
        self.state = JarvisState.IDLE
        self.hud = hud
        self._state_callbacks: list[Callable] = []

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
        self.config.llm_base_url = (
            self.config.llm_base_url
            or os.environ.get("JARVIS_LLM_URL")
            or os.environ.get("JARVIS_LLM_BASE_URL")
            or self._DEFAULT_LLM_URL
        )
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
        if hasattr(self.config, "tts_voice") and self.config.tts_voice:
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
        if self.hud:
            self.hud.set_assistant_name(self.config.assistant_name)

        # Load default profile's system prompt at startup
        default_profile = self.profiles.get_default()
        if default_profile:
            self._system_prompt = default_profile.system_prompt
        else:
            self._system_prompt = self.config.default_system_prompt

        # Active profile
        self._active_profile: Profile | None = None
        self._profile_callbacks: list[Callable] = []

        # HUD thread-safety: dispatcher bound by the GUI (main.py) that queues
        # a callable onto the Qt GUI thread (pyqtSignal emit). Without it, HUD
        # calls run on the caller's thread — fine from the GUI thread, but NOT
        # when fired from PortAudio audio callbacks (wake word / barge-in).
        self._hud_dispatcher: Callable[[Callable], None] | None = None

        # Conversation history (per profile)
        self._messages: list[dict[str, str]] = []

        # HUD panel callbacks (fired on the event-loop thread — the UI must
        # bridge them to Qt widgets via signals, as with state changes).
        self._user_text_callbacks: list[Callable[[str], None]] = []
        self._reply_text_callbacks: list[Callable[[str], None]] = []
        self._tool_call_callbacks: list[Callable[[str, str], None]] = []

        # Barge-in state
        self._barge_in: BargeInListener | None = None
        self._stop_evt = threading.Event()
        self._producer_task: asyncio.Task | None = None

        # Wake word (hands-free) — started on demand, active only while IDLE
        self._wake_word: WakeWordDetector | None = None
        self._wake_word_callbacks: list[Callable] = []

        # Service-failure reporting: callbacks receive a human-readable string
        # ("LLM endpoint unreachable: http://…") — the UI shows it verbatim.
        self._error_callbacks: list[Callable[[str], None]] = []

    def _trim_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop oldest turns beyond ``config.max_history_messages``.

        Keeps whole user→assistant pairs at the boundary where possible (never
        starts the window with an orphaned assistant reply). 0 = no cap.
        """
        cap = int(getattr(self.config, "max_history_messages", 0) or 0)
        if cap <= 0 or len(messages) <= cap:
            return list(messages)
        trimmed = list(messages[-cap:])
        while trimmed and trimmed[0].get("role") == "assistant":
            trimmed.pop(0)
        return trimmed

    def _append_message(self, role: str, content: str) -> None:
        """Append to history and keep it within the cap in-place."""
        self._messages.append({"role": role, "content": content})
        cap = int(getattr(self.config, "max_history_messages", 0) or 0)
        if cap > 0 and len(self._messages) > cap:
            excess = len(self._messages) - cap
            del self._messages[:excess]
            while self._messages and self._messages[0].get("role") == "assistant":
                self._messages.pop(0)

    # ── Error reporting ──────────────────────────────────────────────

    def on_error(self, callback: Callable[[str], None]) -> None:
        """Register a callback fired with a human-readable service error."""
        self._error_callbacks.append(callback)

    def _report_error(self, message: str) -> None:
        """Surface a service failure to the UI (and log it)."""
        logger.error(f"SERVICE ERROR: {message}")
        for cb in self._error_callbacks:
            try:
                cb(message)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error callback failed: {e}")

    # ── HUD thread-safety ────────────────────────────────────────────

    def bind_hud_dispatcher(self, dispatcher: Callable[[Callable], None]) -> None:
        """Bind a function that schedules a callable on the Qt GUI thread.

        ``dispatcher(fn)`` is invoked with a zero-arg callable; it must run
        ``fn()`` on the GUI thread (e.g. via a queued pyqtSignal). Safe to
        call from any thread.
        """
        self._hud_dispatcher = dispatcher

    def _hud(self, fn: Callable[[], None]) -> None:
        """Apply a HUD mutation on the GUI thread, whatever thread we're on."""
        if not self.hud:
            return
        if self._hud_dispatcher is not None:
            self._hud_dispatcher(fn)
        else:
            fn()

    def on_state_change(self, callback: Callable):
        """Register a callback for state changes."""
        self._state_callbacks.append(callback)

    # ── HUD panel callbacks ──────────────────────────────────────────

    def on_user_text(self, callback: Callable[[str], None]) -> None:
        """Register a callback fired with the user's utterance (per turn)."""
        self._user_text_callbacks.append(callback)

    def on_reply_text(self, callback: Callable[[str], None]) -> None:
        """Register a callback fired with each streamed reply chunk."""
        self._reply_text_callbacks.append(callback)

    def on_tool_call(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback fired for in-band LLM tool calls.

        ``callback(key, description)`` — ``key`` is the stable tool-call id,
        ``description`` is the running "name(args)" snapshot.
        """
        self._tool_call_callbacks.append(callback)

    def _emit_user_text(self, text: str) -> None:
        for cb in self._user_text_callbacks:
            try:
                cb(text)
            except Exception as e:  # noqa: BLE001
                logger.error(f"User-text callback error: {e}")

    def _emit_reply_text(self, chunk: str) -> None:
        for cb in self._reply_text_callbacks:
            try:
                cb(chunk)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Reply-text callback error: {e}")

    def _emit_tool_call(self, key: str, description: str) -> None:
        for cb in self._tool_call_callbacks:
            try:
                cb(key, description)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Tool-call callback error: {e}")

    def _set_state(self, new_state: JarvisState):
        """Set state and notify callbacks."""
        if self.state != new_state:
            logger.info(f"State: {self.state.label} → {new_state.label}")
            self.state = new_state
            # Wake word only makes sense while idle — silence it otherwise so
            # it doesn't fire on its own voice or mid-conversation.
            if self._wake_word is not None:
                self._wake_word.set_active(new_state == JarvisState.IDLE)
            for cb in self._state_callbacks:
                cb(new_state)

    def _supports_emotion(self) -> bool:
        """True when the attached frontend understands emotion tags (smoke face)."""
        return self.hud is not None and hasattr(self.hud, "set_emotion")

    def _chat_system_prompt(self) -> str:
        """System prompt plus the emotion-tag instruction for smoke frontend."""
        base = self._system_prompt
        if self._supports_emotion():
            return base + EMOTION_PROMPT
        return base

    def _apply_emotion(self, name: str | None):
        if name and self._supports_emotion():
            def _apply():
                try:
                    self.hud.set_emotion(name)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"set_emotion failed: {e}")
            self._hud(_apply)

    def get_system_prompt(self):
        """Get the current system prompt (profile-specific or default)."""
        return self._system_prompt

    def switch_profile(self, name: str) -> bool:
        """
        Switch to a profile by name.

        Loads the profile's system prompt, chat history, assistant name and
        accent color, and notifies profile-change listeners (e.g. the HUD /
        settings UI).
        """
        if not self.profiles.switch(name):
            return False

        self._active_profile = self.profiles.get(name)
        if self._active_profile:
            self._system_prompt = self._active_profile.system_prompt
            self.config.assistant_name = self._active_profile.assistant_name
            # Update HUD with new assistant name + accent hue
            if self.hud:
                self.hud.set_assistant_name(self.config.assistant_name)
                self.hud.set_profile(name, self._active_profile.accent_hue)
            self._messages = self._trim_history(self._active_profile.chat_history)
            logger.info(
                f"Profile switched to: {name} "
                f"(assistant: {self.config.assistant_name}, "
                f"prompt: {len(self._system_prompt)} chars, "
                f"history: {len(self._messages)} messages, "
                f"hue: {self._active_profile.accent_hue})"
            )
            for cb in self._profile_callbacks:
                try:
                    cb(self._active_profile)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Profile callback error: {e}")
        return True

    def on_profile_changed(self, callback: Callable) -> None:
        """Register a callback for profile switches/clears.

        ``callback`` receives the active :class:`Profile`, or ``None`` when
        the profile is cleared (back to defaults).
        """
        self._profile_callbacks.append(callback)

    def clear_profile(self):
        """Clear active profile (return to defaults)."""
        self.profiles.clear()
        self._active_profile = None
        self._system_prompt = self.config.default_system_prompt or (
            "You are Jarvis, a helpful AI assistant."
        )
        self._messages = []
        # Restore HUD: default assistant name + palette from appearance settings
        if self.hud:
            self.hud.set_assistant_name(self.config.assistant_name)
            self.hud.clear_profile()
            idx = self.config.palette_index
            if idx is not None and 0 <= idx < len(PALETTE_HUES):
                self.hud.set_palette_hue(PALETTE_HUES[idx])
            else:
                self.hud.clear_palette_hue()
        for cb in self._profile_callbacks:
            try:
                cb(None)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Profile callback error: {e}")

    def _on_voice_level(self, level: float):
        """Update HUD voice bars from microphone amplitude (called during STT listen)."""
        self._hud(lambda: self.hud.set_voice_level(level))

    # ── Barge-in ─────────────────────────────────────────────────────

    def _ensure_barge_in(self) -> BargeInListener | None:
        """Create (once) and start the barge-in mic monitor. None if unavailable."""
        if self._barge_in is None:
            self._barge_in = BargeInListener(
                on_speech=self.interrupt,
                threshold=self.config.barge_in_threshold,
                min_speech_ms=self.config.barge_in_min_speech_ms,
                device=self.config.stt.device,
            )
            if not self._barge_in.available:
                return None
            self._barge_in.start()
        return self._barge_in

    # ── Wake word (hands-free) ──────────────────────────────────────

    @property
    def wake_word_available(self) -> bool:
        """True once a wake word detector has been started successfully."""
        return self._wake_word is not None and self._wake_word.available

    def start_wake_word(self) -> bool:
        """Start listening for the wake word. Idempotent.

        Returns True if the detector is live, False if the ML/audio stack
        is unavailable (wake word silently disabled — push-to-talk still works).
        """
        if not self.config.wake_word_enabled:
            return False
        if self._wake_word is not None:
            if self._wake_word.available and self._wake_word._stream is None:
                self._wake_word.start()
            return self._wake_word.available
        detector = WakeWordDetector(
            on_wake=self._on_wake_word,
            word=self.config.wake_word,
            threshold=self.config.wake_word_threshold,
            patience=self.config.wake_word_patience,
            cooldown_s=self.config.wake_word_cooldown_s,
            device=self.config.stt.device,
        )
        if not detector.available:
            logger.info("Wake word unavailable (no openWakeWord model) — using push-to-talk")
            self._wake_word = None
            return False
        detector.start()
        detector.set_active(self.state == JarvisState.IDLE)
        self._wake_word = detector
        return detector._stream is not None

    def stop_wake_word(self) -> None:
        """Stop the wake word detector (safe to call repeatedly)."""
        if self._wake_word is not None:
            self._wake_word.stop()
            self._wake_word = None

    def on_wake_word(self, callback: Callable) -> None:
        """Register a callback fired when the wake word is detected (thread-safe).

        ``callback`` is invoked with no arguments from the audio thread. Use it
        to schedule the voice cycle on the event loop (see main.py).
        """
        self._wake_word_callbacks.append(callback)

    def _on_wake_word(self) -> None:
        """Wake word fired (audio thread) — notify registered callbacks."""
        logger.info("Wake word fired — starting voice cycle")
        self._hud(lambda: self.hud.set_state(JarvisState.LISTENING))
        for cb in self._wake_word_callbacks:
            try:
                cb()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Wake word fire callback error: {e}")

    @property
    def stop_requested(self) -> bool:
        """True if the current reply was interrupted (barge-in or STOP)."""
        return self._stop_evt.is_set()

    def interrupt(self) -> None:
        """Stop the current reply (playback + stream) immediately.

        Thread-safe — callable from the barge-in audio thread, a UI STOP
        button, or the event loop. Idempotent.
        """
        if self._stop_evt.is_set():
            return
        self._stop_evt.set()
        logger.info("Interrupt — stopping speech and in-flight stream")
        try:
            self.audio.stop()
        except Exception as e:
            logger.warning(f"audio.stop() failed: {e}")
        # Task.cancel() is thread-safe; the producer task exits on the next
        # await (the SSE read or a queue put).
        task = self._producer_task
        if task is not None and not task.done():
            task.cancel()
        if self.hud:
            self._hud(lambda: self.hud.set_voice_level(0.0))

    # ── Voice cycle (streaming) ───────────────────────────────────────

    async def run_voice(self) -> None:
        """Public entry: one voice cycle with a hard overall timeout.

        If any step hangs (network stall to STT/LLM/TTS), the whole cycle is
        cancelled after ``config.voice_cycle_timeout`` seconds and a
        human-readable error is reported instead of the app sitting in
        LISTENING/THINKING forever.
        """
        timeout = float(getattr(self.config, "voice_cycle_timeout", 0) or 0)
        try:
            if timeout > 0:
                await asyncio.wait_for(self._run_voice_inner(), timeout=timeout)
            else:
                await self._run_voice_inner()
        except asyncio.TimeoutError:
            self.interrupt()
            self._report_error(
                f"Voice cycle timed out after {timeout:.0f}s — "
                "check STT/LLM/TTS services (Settings)"
            )
            self._set_state(JarvisState.IDLE)

    async def _run_voice(self):
        """Alias kept for callers/tests; use run_voice() for timeout safety."""
        return await self._run_voice_inner()

    async def _run_voice_inner(self):
        """Run a single voice cycle: listen → think (stream) → speak.

        The reply is spoken sentence-by-sentence as the LLM streams it, so the
        first sentence starts while the rest is still generating. Barge-in
        (user speech during SPEAKING) or :meth:`interrupt` stops playback and
        the in-flight stream.
        """
        # Listen
        self._set_state(JarvisState.LISTENING)
        logger.info("STT: Starting listen cycle")
        text = await self.stt.listen(
            timeout=self.config.silence_timeout * 4,
            silence_threshold=self.config.silence_threshold,
            on_voice_level=self._on_voice_level,
        )
        logger.info(f"STT: Listen returned: {text!r}")
        if not text:
            if self.hud:
                self._hud(lambda: self.hud.set_voice_level(0.0))
            if not self.stt._connected:
                host = getattr(self.stt.config, "host", "?")
                port = getattr(self.stt.config, "port", "?")
                self._report_error(f"STT unreachable — no speech recognized (server {host}:{port})")
            self._set_state(JarvisState.IDLE)
            return

        # Think — stream tokens, speak each complete sentence as it arrives.
        self._set_state(JarvisState.THINKING)
        self._append_message("user", text)
        self._stop_evt.clear()
        self._emit_user_text(text)

        queue: asyncio.Queue = asyncio.Queue()
        buffer = SentenceBuffer()
        emo_filter = EmotionTagFilter()
        spoken: list[str] = []
        streamed_any = False

        def on_token(token: str) -> None:
            """Stream callback (event-loop thread): strip emo tag → sentences."""
            nonlocal streamed_any
            if self._stop_evt.is_set():
                return
            streamed_any = True
            clean = emo_filter.feed(token)
            if clean:
                if emo_filter.emotion and not producer._emo_applied:
                    producer._emo_applied = True
                    self._apply_emotion(emo_filter.emotion)
                self._emit_reply_text(clean)
                for sentence in buffer.feed(clean):
                    queue.put_nowait(sentence)

        async def producer() -> None:
            """Stream the LLM reply; queue sentences; always end with sentinel."""
            try:
                reply = await self.chat.chat(
                    self._messages,
                    stream=True,
                    system_prompt=self._chat_system_prompt(),
                    on_token=on_token,
                    on_tool_call=lambda key, desc: self._emit_tool_call(key, desc),
                )
                if self._stop_evt.is_set():
                    return
                if streamed_any:
                    if not producer._emo_applied and reply:
                        producer._emo_applied = True
                        self._apply_emotion(classify_emotion(reply))
                    # Streamed tokens — sentences already queued via on_token.
                    # Release any buffered emotion-tag residue, then the tail.
                    rest = emo_filter.flush()
                    if rest:
                        self._emit_reply_text(rest)
                        for sentence in buffer.feed(rest):
                            queue.put_nowait(sentence)
                    tail = buffer.flush()
                    if tail:
                        queue.put_nowait(tail)
                elif reply:
                    # Endpoint ignored `stream` — one-shot fallback: strip tag, chunk.
                    emo_tag, reply = parse_emotion(reply)
                    self._apply_emotion(emo_tag or classify_emotion(reply))
                    for s in split_sentences(reply):
                        if self._stop_evt.is_set():
                            break
                        queue.put_nowait(s)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                self._report_error(self.chat.last_error or f"LLM request failed: {e}")
            finally:
                queue.put_nowait(None)

        async def speaker() -> None:
            """Pop sentences; synthesize (async) and play (worker thread)."""
            listener = self._ensure_barge_in()
            while True:
                sentence = await queue.get()
                if sentence is None or self._stop_evt.is_set():
                    break
                if not spoken:
                    self._set_state(JarvisState.SPEAKING)
                    if listener is not None:
                        listener.set_active(True)
                try:
                    audio = await self.tts.speak(sentence)
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    host = getattr(self.tts.config, "host", "?")
                    port = getattr(self.tts.config, "port", "?")
                    self._report_error(f"TTS failed: {e} (server {host}:{port})")
                    break
                if audio and not self._stop_evt.is_set():
                    await asyncio.to_thread(self.audio.play, audio)
                spoken.append(sentence)
            if listener is not None:
                listener.set_active(False)

        producer._emo_applied = False
        self._producer_task = producer_task = asyncio.create_task(producer())
        speaker_task = asyncio.create_task(speaker())
        try:
            await asyncio.gather(producer_task, speaker_task, return_exceptions=True)
        finally:
            self._producer_task = None
            if self.hud:
                self._hud(lambda: self.hud.set_voice_level(0.0))

        interrupted = self.stop_requested

        # Persist conversation (spoken portion if interrupted).
        reply = " ".join(spoken).strip()
        if not interrupted and reply:
            self._append_message("assistant", reply)
        if self._active_profile and (reply or not interrupted):
            self._active_profile.chat_history = list(self._messages)
            if reply:
                self.profiles.save(self._active_profile)

        if interrupted:
            logger.info("Voice cycle interrupted (barge-in/STOP)")
        self._set_state(JarvisState.IDLE)

    async def chat_text(self, user_text: str, system_prompt: str | None = None) -> str:
        """
        Direct text chat (no voice). Returns the assistant's response text.
        """
        self._set_state(JarvisState.THINKING)
        prompt = system_prompt or self._system_prompt
        self._append_message("user", user_text)
        reply = await self.chat.chat(self._messages, system_prompt=prompt)
        if reply:
            self._append_message("assistant", reply)
        self._set_state(JarvisState.IDLE)
        return reply or ""

    async def chat_text_and_speak(self, user_text: str, system_prompt: str | None = None) -> str:
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
        if self._barge_in is not None:
            self._barge_in.stop()
            self._barge_in = None
        self.stop_wake_word()
        await self.chat.close()
        await self.stt.disconnect()
        await self.tts.disconnect()
        self.audio.stop()
        self.profiles.close()
