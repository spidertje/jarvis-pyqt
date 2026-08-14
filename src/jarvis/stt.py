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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WyomingConfig:
    """Wyoming protocol server configuration."""

    host: str = "192.168.55.41"
    port: int = 10300
    sample_rate: int = 16000
    width: int = 2  # 16-bit
    channels: int = 1
    device: int | None = None  # -1 means default, None means let sounddevice decide


class WhisperSTT:
    """Faster-whisper STT client via Wyoming protocol."""

    def __init__(self, config: WyomingConfig | None = None):
        self.config = config or WyomingConfig()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    async def connect(self, timeout: float = 5.0) -> bool:
        """Connect to faster-whisper via Wyoming protocol."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.host, self.config.port),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Connection timeout to faster-whisper at {self.config.host}:{self.config.port}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to connect to faster-whisper: {e}")
            return False

        self._reader = reader
        self._writer = writer

        # Wyoming protocol: send hello and wait for server hello
        self._send_event("hello", {"protocol_version": 0})
        await writer.drain()

        # Read server hello
        try:
            event = await asyncio.wait_for(self._read_event(), timeout=5.0)
            if event and event.get("type") == "welcome":
                logger.info(
                    f"Wyoming server welcomed — "
                    f"features: {event.get('data', {}).get('features', [])}"
                )
            elif event:
                logger.info(f"Wyoming server response: {event.get('type')}")
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            logger.warning("No Wyoming hello response from server — continuing anyway")

        self._connected = True
        logger.info(f"Connected to faster-whisper STT at {self.config.host}:{self.config.port}")
        return True

    async def disconnect(self):
        """Disconnect from faster-whisper."""
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
        """Force a fresh connection (useful after errors)."""
        await self.disconnect()
        return await self.connect(timeout)

    async def _ensure_connected(self) -> bool:
        """Ensure we have a valid connection. Reconnect if needed."""
        if not self._connected or self._writer is None or self._writer.is_closing():
            logger.info("STT: connection lost, attempting reconnect...")
            return await self.connect()
        return True

    def _is_silence(self, audio_data: bytes, threshold: float = 500) -> bool:
        """Check if audio data is silence (RMS amplitude below threshold)."""
        if len(audio_data) == 0:
            return True
        samples = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
        return rms < threshold

    async def listen(
        self,
        timeout: float = 5.0,
        silence_threshold: float = 100.0,
        on_voice_level: Callable[[float], None] | None = None,
    ) -> str | None:
        """
        Record audio from microphone and transcribe it.

        Records for up to `timeout` seconds, or stops on silence detection.

        Args:
             timeout: Max recording time in seconds
            silence_threshold: RMS amplitude threshold for silence detection
            on_voice_level: Optional callback(float) for mic amplitude (0–1)

        Returns:
            Transcribed text or None on failure.
        """
        if not await self._ensure_connected():
            logger.info("STT: Connecting...")
            if not await self.connect():
                return None

        # Capture microphone audio using sounddevice
        sd = self._get_sd()
        if sd is None:
            logger.warning("sounddevice not available for recording")
            return None

        # Auto-detect best input device if not explicitly configured
        if self.config.device is None:
            self.config.device = self._find_input_device(sd)

        # Record chunks, accumulating until silence or timeout
        chunk_duration = 0.5  # seconds
        chunk_size = int(self.config.sample_rate * chunk_duration)
        accumulated = b""
        silence_chunks = 0
        max_silence_chunks = int(timeout / chunk_duration)
        speech_detected = False

        logger.info(
            f"STT: Listening... (device={self.config.device}, "
            f"sample_rate={self.config.sample_rate})"
        )

        try:
            while True:
                # Record a chunk from microphone (offload blocking call to thread)
                rec_kwargs = {
                    "samplerate": self.config.sample_rate,
                    "channels": self.config.channels,
                    "dtype": "int16",
                    "blocking": True,
                }
                if self.config.device is not None:
                    rec_kwargs["device"] = self.config.device
                chunk = await asyncio.to_thread(sd.rec, chunk_size, **rec_kwargs)
                if chunk is None:
                    break

                # Convert numpy array to bytes
                chunk_bytes = chunk.tobytes()
                accumulated += chunk_bytes

                # Report voice level to callback (normalized 0–1)
                if on_voice_level is not None:
                    samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    level = min(1.0, rms / 30.0)  # Webcam mic: 30+ RMS = full scale
                    on_voice_level(level)

                # Check for silence
                if self._is_silence(chunk_bytes, silence_threshold):
                    silence_chunks += 1
                    if silence_chunks >= max_silence_chunks:
                        logger.info(
                            f"Silence detected after "
                            f"{len(accumulated) / self.config.width / self.config.sample_rate:.1f}s"
                        )
                        break
                else:
                    silence_chunks = 0
                    speech_detected = True

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

        if not accumulated:
            logger.info("No audio captured")
            return None

        # If we never detected speech, treat as no speech
        if not speech_detected:
            logger.info("No speech detected")
            return None

        # Transcribe the accumulated audio
        logger.info(
            f"Transcribing "
            f"{len(accumulated) / self.config.width / self.config.sample_rate:.1f}s "
            "of audio..."
        )
        text = await self.transcribe(accumulated)
        return text

    def _get_sd(self):
        """Get sounddevice module."""
        try:
            import sounddevice as sd

            return sd
        except ImportError:
            return None

    def _find_input_device(self, sd) -> int | None:
        """Find the best input device by testing each one for audio levels."""
        try:
            devices = sd.query_devices()
        except Exception:
            return None

        best_device = None
        best_rms = 0.0
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] < 1:
                continue
            try:
                test_rec = sd.rec(
                    int(self.config.sample_rate * 0.3),
                    samplerate=self.config.sample_rate,
                    channels=1,
                    device=i,
                    dtype="int16",
                    blocking=True,
                )
                samples = np.frombuffer(test_rec.tobytes(), dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                if rms > best_rms:
                    best_rms = rms
                    best_device = i
            except Exception:
                continue

        if best_device is not None:
            logger.info(f"Auto-selected input device {best_device} (RMS={best_rms:.2f})")
        return best_device

    def _send_event(self, event_type: str, data: dict | None = None, payload: bytes | None = None):
        """Send a Wyoming protocol event (JSON header + optional data/payload)."""
        if self._writer is None:
            return
        event: dict[str, Any] = {"type": event_type, "version": "1.10.0"}
        data_bytes = b""
        if data:
            data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
            event["data_length"] = len(data_bytes)
        else:
            event.pop("data", None)

        if payload:
            event["payload_length"] = len(payload)

        header = json.dumps(event, ensure_ascii=False).encode("utf-8")
        self._writer.write(header + b"\n")
        if data_bytes:
            self._writer.write(data_bytes)
        if payload:
            self._writer.write(payload)

    async def _read_event(self) -> dict:
        """Read a Wyoming protocol event (JSON header + optional data/payload)."""
        if self._reader is None:
            return {}
        line = await self._reader.readline()
        if not line:
            return {}
        event = json.loads(line.decode("utf-8"))

        data_length = event.pop("data_length", 0)
        if data_length > 0:
            data = await self._reader.readexactly(data_length)
            event["data"] = json.loads(data.decode("utf-8"))

        payload_length = event.pop("payload_length", 0)
        if payload_length > 0:
            event["payload"] = await self._reader.readexactly(payload_length)

        return event

    async def transcribe(self, audio_data: bytes) -> str | None:
        """
        Transcribe audio via Wyoming protocol.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, 16kHz, mono)

        Returns:
            Transcribed text or None on failure.
        """
        if not await self._ensure_connected():
            if not await self.connect():
                return None

        try:
            return await self._do_transcribe(audio_data)

        except (ConnectionError, asyncio.IncompleteReadError) as e:
            logger.warning(f"STT connection error: {e} — will reconnect on next call")
            self._connected = False
            # Try reconnecting and retrying once
            logger.info("Attempting STT reconnection...")
            if await self.connect():
                try:
                    return await self._do_transcribe(audio_data)
                except Exception as e2:
                    logger.error(f"STT retry also failed: {e2}")
            return None
        except Exception as e:
            logger.error(f"Whisper STT error: {e}")
            return None

    async def _do_transcribe(self, audio_data: bytes) -> str | None:
        """Send audio data and read back the transcript."""
        if self._writer is None or self._reader is None:
            return None
        writer = self._writer
        # Send transcribe request with audio format metadata
        self._send_event(
            "transcribe",
            {
                "rate": self.config.sample_rate,
                "width": self.config.width,
                "channels": self.config.channels,
            },
        )
        await writer.drain()

        # Send audio chunks with format metadata in data + raw audio as payload
        chunk_size = 3200  # ~100ms at 16kHz 16-bit mono
        timestamp = 0
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            self._send_event(
                "audio-chunk",
                {
                    "rate": self.config.sample_rate,
                    "width": self.config.width,
                    "channels": self.config.channels,
                    "timestamp": timestamp,
                },
                payload=chunk,
            )
            timestamp += len(chunk) // (self.config.width * self.config.channels)
            await writer.drain()

        # Send audio-stop to signal end of stream
        self._send_event("audio-stop", {"timestamp": timestamp})
        await writer.drain()

        # Read transcript
        text = None
        while True:
            event = await self._read_event()
            if not event:
                break

            if event.get("type") == "transcript":
                data = event.get("data", {})
                text = data.get("text", "")
                break

            if event.get("type") == "transcript-stop":
                break

        return text
