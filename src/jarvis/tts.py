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
from typing import Optional

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
    """Piper TTS client via Wyoming protocol."""

    def __init__(self, config: Optional[WyomingConfig] = None):
        self.config = config or WyomingConfig()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self.voice: str = config.voice if config and hasattr(config, 'voice') else "en_US-lessac-medium"

    async def connect(self, timeout: float = 5.0) -> bool:
        """Connect to Piper via Wyoming protocol."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.config.host,
                    self.config.port
                ),
                timeout=timeout,
            )
            self._connected = True
            logger.info(f"Connected to Piper TTS at {self.config.host}:{self.config.port}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to Piper at {self.config.host}:{self.config.port}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Piper TTS: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Piper."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False

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

    async def speak(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to audio via Wyoming protocol.
        
        Returns raw PCM audio bytes or None on failure.
        """
        # Connect for each utterance so voice parameter takes effect
        if not await self.connect():
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

        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            return None
        finally:
            # Disconnect after each utterance so next speak() can use a different voice
            await self.disconnect()
