"""
Jarvis TTS — Piper text-to-speech via Wyoming protocol.

Wyoming protocol: JSON events (newline-delimited) + raw PCM audio payload.
Piper speaks English with a neural voice.

Protocol:
1. Send: {"type": "synthesize", "data": {"text": "hello world"}}
2. Receive: {"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}
3. Receive: {"type": "audio-chunk", "data": {"timestamp": 123456}} + PCM payload
4. Receive: {"type": "audio-stop", "data": {"timestamp": 123456}}
"""

import asyncio
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WyomingConfig:
    """Wyoming protocol server configuration."""

    host: str = "192.168.55.41"
    port: int = 10200
    sample_rate: int = 16000
    width: int = 2  # 16-bit
    channels: int = 1
    voice: str = "en_US-lessac-medium"  # Piper voice model name


class PiperTTS:
    """Piper TTS client via Wyoming protocol.

    Maintains a persistent connection to the Wyoming server.
    Reconnects automatically on connection errors.
    """

    def __init__(self, config: WyomingConfig | None = None):
        self.config = config or WyomingConfig()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self.voice: str = (
            config.voice if config and hasattr(config, "voice") else "en_US-lessac-medium"
        )
        self._prev_voice: str = self.voice

    async def connect(self, timeout: float = 5.0) -> bool:
        """Connect to Piper via Wyoming protocol."""
        if self._connected and self._writer is not None:
            return True

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.config.host,
                    self.config.port,
                ),
                timeout=timeout,
            )
            self._connected = True
            self._prev_voice = self.voice
            logger.info(
                f"Connected to Piper TTS at {self.config.host}:"
                f"{self.config.port} (voice={self.voice})"
            )
            return True
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to Piper at {self.config.host}:{self.config.port}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Piper TTS: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Piper."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False
        self._reader = None
        self._writer = None

    async def reconnect(self, timeout: float = 5.0) -> bool:
        """Force a fresh connection (useful after voice change or error)."""
        await self.disconnect()
        return await self.connect(timeout)

    def _send_event(self, event_type: str, data: dict = None):
        """Send a Wyoming protocol event."""
        event = {"type": event_type}
        if data:
            event["data"] = data
        self._writer.write(json.dumps(event).encode() + b"\n")

    async def _read_event(self) -> dict:
        """Read a Wyoming protocol event (JSON line)."""
        line = await self._reader.readline()
        if not line:
            return {}
        return json.loads(line.decode("utf-8"))

    async def _ensure_connected(self) -> bool:
        """Ensure we have a valid connection. Reconnect if needed."""
        if not self._connected or self._writer is None or self._writer.is_closing():
            logger.info("TTS: connection lost, attempting reconnect...")
            return await self.connect()
        return True

    def set_voice(self, voice: str) -> bool:
        """
        Set the TTS voice for subsequent speak() calls.

        Returns True if the voice changed and a reconnect may be needed.
        """
        if voice != self.voice:
            self.voice = voice
            return True
        return False

    async def speak(self, text: str) -> bytes | None:
        """
        Synthesize text to audio via Wyoming protocol.

        Maintains a persistent connection. Reconnects automatically
        if the connection was lost.

        Returns raw PCM audio bytes or None on failure.
        """
        # Check if voice changed — Piper requires reconnect for voice change
        voice_changed = self.voice != self._prev_voice
        if voice_changed:
            logger.info(f"TTS voice changed: {self._prev_voice} → {self.voice}")
            await self.reconnect()
        elif not await self._ensure_connected():
            return None

        try:
            # Send synthesize request
            self._send_event("synthesize", {"text": text, "voice": self.voice})
            await self._writer.drain()

            # Read audio-start
            event = await self._read_event()
            if event.get("type") != "audio-start":
                logger.error(f"Expected audio-start, got: {event}")
                return None

            # Read additional data from audio-start
            data_length = event.get("data_length", 0)
            if data_length > 0:
                await self._reader.readexactly(data_length)

            # Read audio chunks with payload
            audio_chunks = []
            while True:
                event = await self._read_event()

                # Read additional data
                data_length = event.get("data_length", 0)
                if data_length > 0:
                    await self._reader.readexactly(data_length)

                # Read payload (binary PCM)
                payload_length = event.get("payload_length", 0)
                if payload_length > 0:
                    payload = await self._reader.readexactly(payload_length)
                    audio_chunks.append(payload)

                if event.get("type") == "audio-stop":
                    break

            return b"".join(audio_chunks) if audio_chunks else None

        except (ConnectionError, asyncio.ConnectionError, asyncio.IncompleteReadError) as e:
            logger.warning(f"TTS connection error: {e} — will reconnect on next call")
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            return None
