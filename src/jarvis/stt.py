"""
Jarvis STT — Whisper speech-to-text via Wyoming protocol.

Wyoming protocol: JSON events (newline-delimited) + raw PCM audio payload.
Faster-whisper transcribes audio to text.

Protocol:
1. Send: {"type": "transcribe", "data": {}}
2. Send audio chunks: {"type": "audio-chunk", ...} + PCM payload
3. Receive: {"type": "transcript", "data": {"text": "..."}}
4. Receive: {"type": "transcript-stop", "data": {}}
"""

import asyncio
import json
import struct
import logging
from dataclasses import dataclass
from typing import Optional, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class WyomingConfig:
    """Wyoming protocol server configuration."""
    host: str = "192.168.55.41"
    port: int = 10300
    sample_rate: int = 16000
    width: int = 2  # 16-bit
    channels: int = 1


class WhisperSTT:
    """Faster-whisper STT client via Wyoming protocol."""

    def __init__(self, config: Optional[WyomingConfig] = None):
        self.config = config or WyomingConfig()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to faster-whisper via Wyoming protocol."""
        try:
            self._reader, self._writer = await asyncio.open_connection(
                self.config.host,
                self.config.port
            )
            self._connected = True
            logger.info(f"Connected to faster-whisper STT at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to faster-whisper: {e}")
            return False

    async def disconnect(self):
        """Disconnect from faster-whisper."""
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

    async def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio via Wyoming protocol.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit, 16kHz, mono)
            
        Returns:
            Transcribed text or None on failure.
        """
        if not self._connected:
            if not await self.connect():
                return None

        try:
            # Send transcribe request
            self._send_event("transcribe", {})
            await self._writer.drain()

            # Send audio in chunks
            chunk_size = 3200  # ~100ms at 16kHz 16-bit mono
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                event = {
                    "type": "audio-chunk",
                    "data": {
                        "rate": self.config.sample_rate,
                        "width": self.config.width,
                        "channels": self.config.channels,
                    },
                }
                self._writer.write(json.dumps(event).encode() + b"\n")
                self._writer.write(chunk)
                await self._writer.drain()

            # Read transcript
            text = None
            while True:
                event = await self._read_event()
                
                # Read additional data
                data_length = event.get("data_length", 0)
                if data_length > 0:
                    await self._reader.readexactly(data_length)
                
                # Read payload
                payload_length = event.get("payload_length", 0)
                if payload_length > 0:
                    payload = await self._reader.readexactly(payload_length)
                    if event.get("type") == "transcript":
                        text = payload.decode("utf-8")

                if event.get("type") in ("transcript", "transcript-stop"):
                    break

            return text

        except Exception as e:
            logger.error(f"Whisper STT error: {e}")
            self._connected = False
            return None

    async def listen_stream(self) -> AsyncIterator[bytes]:
        """
        Stream audio from microphone via Wyoming protocol.
        
        Yields raw PCM audio chunks.
        """
        if not self._connected:
            if not await self.connect():
                return

        try:
            # Send START command
            start_cmd = {"start": {}}
            self._writer.write(json.dumps(start_cmd).encode() + b"\n")
            await self._writer.drain()

            # Stream audio data
            while True:
                try:
                    length_data = await self._reader.readexactly(8)
                    if len(length_data) < 8:
                        break
                    
                    length = struct.unpack("<Q", length_data)[0]
                    if length == 0:
                        break
                    
                    audio_data = await self._reader.readexactly(length)
                    yield audio_data

                except asyncio.IncompleteReadError:
                    break

            # Stop command
            stop_cmd = {"stop": {}}
            self._writer.write(json.dumps(stop_cmd).encode() + b"\n")
            await self._writer.drain()

        except Exception as e:
            logger.error(f"Whisper STT stream error: {e}")
            self._connected = False
