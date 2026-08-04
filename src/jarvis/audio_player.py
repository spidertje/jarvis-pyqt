"""
Jarvis Audio Player — plays raw PCM audio via sounddevice.

PCM format: 16-bit, 16kHz, mono (matches Piper TTS output).
Uses sounddevice + numpy for playback.
"""

import logging
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio playback configuration."""
    sample_rate: int = 16000
    channels: int = 1
    width: int = 2  # 16-bit
    device: Optional[int] = None  # None = default device


class AudioPlayer:
    """Plays raw PCM audio via sounddevice."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._stream = None
        self._playing = False
        self._sd = None
        self._init()

    def _init(self):
        """Initialize sounddevice."""
        try:
            import sounddevice as sd
            self._sd = sd
            devices = sd.query_devices()
            dev_index = self.config.device if self.config.device is not None else 0
            dev_name = devices[dev_index]["name"]
            logger.info(f"Audio device: {dev_name}")
        except ImportError:
            logger.warning("sounddevice not installed — audio playback disabled")
        except Exception as e:
            logger.warning(f"sounddevice init failed: {e} — audio playback disabled")

    def play(self, pcm_data: bytes):
        """
        Play raw PCM audio.

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
        n_samples = len(pcm_data) // width

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
        device = self.config.device if self.config.device is not None and self.config.device >= 0 else None

        try:
            self._stream = self._sd.OutputStream(
                samplerate=self.config.sample_rate,
                channels=channels,
                dtype="float32",
                device=device,
            )
            self._stream.start()
            self._playing = True

            self._stream.write(audio_float)

            # Wait for playback to complete
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._playing = False

        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            self._playing = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def stop(self):
        """Stop current playback."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing
