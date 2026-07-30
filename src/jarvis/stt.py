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
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

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

    async def connect(self, timeout: float = 5.0) -> bool:
        """Connect to faster-whisper via Wyoming protocol."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.config.host,
                    self.config.port
                ),
                timeout=timeout,
            )
            self._connected = True
            logger.info(f"Connected to faster-whisper STT at {self.config.host}:{self.config.port}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to faster-whisper at {self.config.host}:{self.config.port}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to faster-whisper: {e}")
            return False

    async def disconnect(self):
        """Disconnect from faster-whisper."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False

    def _is_silence(self, audio_data: bytes, threshold: float = 500) -> bool:
        """Check if audio data is silence (RMS amplitude below threshold)."""
        if len(audio_data) == 0:
            return True
        samples = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
        return rms < threshold

    async def listen(self, timeout: float = 5.0, silence_threshold: float = 500) -> Optional[str]:
        """
        Record audio from microphone and transcribe it.

        Records for up to `timeout` seconds, or stops on silence detection.

        Args:
            timeout: Max recording time in seconds
            silence_threshold: RMS amplitude threshold for silence detection

        Returns:
            Transcribed text or None on failure.
        """
        if not self._connected:
            if not await self.connect():
                return None

        # Capture microphone audio using sounddevice
        sd = self._get_sd()
        if sd is None:
            logger.warning("sounddevice not available for recording")
            return None

        # Record chunks, accumulating until silence or timeout
        chunk_duration = 0.5  # seconds
        chunk_size = int(self.config.sample_rate * chunk_duration)
        accumulated = b""
        silence_chunks = 0
        max_silence_chunks = int(timeout / chunk_duration)

        logger.info("Listening...")

        try:
            while True:
                # Record a chunk from microphone
                chunk = sd.rec(
                    chunk_size,
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype="int16",
                    block=True,
                )
                if chunk is None:
                    break

                # Convert numpy array to bytes
                chunk_bytes = chunk.tobytes()
                accumulated += chunk_bytes

                # Check for silence
                if self._is_silence(chunk_bytes, silence_threshold):
                    silence_chunks += 1
                    if silence_chunks >= max_silence_chunks:
                        logger.info(f"Silence detected after {len(accumulated) / self.config.width / self.config.sample_rate:.1f}s")
                        break
                else:
                    silence_chunks = 0

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

        if not accumulated:
            logger.info("No audio captured")
            return None

        # Transcribe the accumulated audio
        logger.info(f"Transcribing {len(accumulated) / self.config.width / self.config.sample_rate:.1f}s of audio...")
        text = await self.transcribe(accumulated)
        return text

    def _get_sd(self):
        """Get sounddevice module."""
        try:
            import sounddevice as sd
            return sd
        except ImportError:
            return None

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

