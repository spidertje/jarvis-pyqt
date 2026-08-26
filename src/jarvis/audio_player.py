"""
Jarvis Audio Player — plays raw PCM audio via sounddevice.

PCM format: 16-bit, 16kHz, mono (matches Piper TTS output).
Uses sounddevice + numpy for playback.
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio playback configuration."""

    sample_rate: int = 16000
    channels: int = 1
    width: int = 2  # 16-bit
    device: int | None = None  # None = default device


class AudioPlayer:
    """Plays raw PCM audio via sounddevice.

    Playback is interruptible: :meth:`play` writes audio in small chunks and
    checks the internal stop flag between chunks, so :meth:`stop` cuts output
    within one chunk (<= ~160ms) instead of after the whole clip.

    Thread-safe: playback may run on a worker thread while :meth:`stop` is
    called from another (e.g. the barge-in audio thread). A lock guards the
    stream lifecycle so the two can't corrupt each other.
    """

    # ~160ms of 16-bit samples at 16kHz — the max interruption latency.
    _CHUNK_SAMPLES = 160 * 16000 // 1000

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._stream = None
        self._playing = False
        self._sd = None
        self._stop_flag = False
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        """Initialize sounddevice."""
        try:
            import sounddevice as sd

            self._sd = sd
            devices = sd.query_devices()
            # Handle -1 as default device (None)
            dev_index = self.config.device if self.config.device is not None and self.config.device >= 0 else 0
            dev_name = devices[dev_index]["name"]
            logger.info(f"Audio device: {dev_name}")
        except ImportError:
            logger.warning("sounddevice not installed — audio playback disabled")
        except Exception as e:
            logger.warning(f"sounddevice init failed: {e} — audio playback disabled")

    def play(self, pcm_data: bytes):
        """
        Play raw PCM audio (interruptible).

        Writes in ~160ms chunks, checking the stop flag between chunks so
        :meth:`stop` takes effect promptly.

        Args:
            pcm_data: Raw PCM bytes (16-bit, mono, 16kHz)
        """
        if self._sd is None:
            logger.warning("Audio not available — sounddevice not installed")
            return

        if len(pcm_data) == 0:
            return

        # Convert bytes to numpy int16 array
        width = self.config.width
        channels = self.config.channels

        if width == 2:
            samples = np.frombuffer(pcm_data, dtype=np.int16)
        elif width == 1:
            samples = np.frombuffer(pcm_data, dtype=np.uint8).astype(np.int16)
        else:
            samples = np.frombuffer(pcm_data, dtype=np.int32)

        # Reshape to (samples, channels)
        if channels > 1:
            samples = samples.reshape(-1, channels)

        # Convert to float32 in range [-1.0, 1.0]
        max_val = 32767 if width <= 2 else 2147483647
        audio_float = samples.astype(np.float32) / max_val

        # Handle device: -1 means default, pass None to sounddevice
        device = (
            self.config.device
            if self.config.device is not None and self.config.device >= 0
            else None
        )

        self._stop_flag = False
        try:
            with self._lock:
                self._stream = self._sd.OutputStream(
                    samplerate=self.config.sample_rate,
                    channels=channels,
                    dtype="float32",
                    device=device,
                )
                self._stream.start()
            self._playing = True

            # Write in chunks so stop() can interrupt between chunks.
            chunk = self._CHUNK_SAMPLES
            for i in range(0, len(audio_float), chunk):
                if self._stop_flag:
                    logger.info("Audio playback interrupted by stop()")
                    break
                self._stream.write(audio_float[i : i + chunk])

            self._cleanup_stream()

        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            self._playing = False
            self._cleanup_stream()

    def _cleanup_stream(self):
        """Tear down the active stream (thread-safe)."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._playing = False

    def stop(self):
        """Stop current playback (interrupts chunked writes promptly)."""
        self._stop_flag = True
        self._cleanup_stream()

    @property
    def is_playing(self) -> bool:
        return self._playing
