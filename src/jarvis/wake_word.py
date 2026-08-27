"""Wake-word detection — hear the user say the wake word (e.g. "Hey Jarvis").

Runs a continuous microphone monitor (sounddevice InputStream callback) and
feeds each 80 ms frame (1280 samples @ 16 kHz, int16) to an openWakeWord
ONNX model. When the target word's score crosses the threshold with the
configured patience (consecutive frames above threshold), ``on_wake`` fires
once. A cooldown prevents immediate re-triggering on the tail of the same
utterance.

The detector degrades gracefully: if ``openwakeword``/``onnxruntime`` are
unavailable or the model fails to load, :attr:`available` is False and the
methods are no-ops. A custom model object (anything with a ``.models`` dict
and ``.predict(frame, threshold=, patience=)``) may be injected for testing.

Note on latency: a single frame inference is ~5 ms on CPU, comfortably
within the ~12 ms real-time budget for 80 ms frames.
"""

import logging
import time

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects a spoken wake word via the microphone."""

    def __init__(
        self,
        on_wake=None,
        word: str = "hey_jarvis",
        threshold: float = 0.5,
        patience: int = 2,
        cooldown_s: float = 1.5,
        sample_rate: int = 16000,
        blocksize: int = 1280,
        device: int | None = None,
        model=None,
    ):
        self.on_wake = on_wake
        self.word = word
        self.threshold = threshold
        self.patience = patience
        self.cooldown_s = cooldown_s
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device = device
        self._active = False
        self._last_fire = 0.0
        self._sd = None
        self._stream = None
        self._model = model

        if self._model is None:
            self._model = self._load_model()

    # ── model ─────────────────────────────────────────────────────

    @staticmethod
    def _load_model():
        """Load the openWakeWord model (all pretrained words, clean keys)."""
        try:
            from openwakeword.model import Model

            m = Model()
            return m
        except Exception as e:  # noqa: BLE001 - any ML-stack failure
            logger.warning(f"openWakeWord unavailable — wake word disabled: {e}")
            return None

    @property
    def available(self) -> bool:
        """True if a model is loaded and the audio stack is present."""
        return self._model is not None

    # ── lifecycle ─────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """Enable/disable detection (call when entering/leaving IDLE)."""
        self._active = active

    def start(self) -> None:
        """Start the input stream (no-op if already started or unavailable)."""
        if not self.available or self._stream is not None:
            return
        try:
            import sounddevice as sd

            self._sd = sd
        except Exception as e:  # noqa: BLE001
            logger.warning(f"sounddevice unavailable — wake word disabled: {e}")
            return
        try:
            kwargs = {
                "samplerate": self.sample_rate,
                "channels": 1,
                "dtype": "float32",
                "blocksize": self.blocksize,
                "callback": self._callback,
            }
            if self.device is not None:
                kwargs["device"] = self.device
            self._stream = self._sd.InputStream(**kwargs)
            self._stream.start()
            logger.info(f"Wake word detector started (word={self.word!r})")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not start wake word detector: {e}")
            self._stream = None

    def stop(self) -> None:
        """Stop and close the input stream (safe to call repeatedly)."""
        self._active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ── detection ─────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        """PortAudio callback — runs on the audio thread. Must stay fast."""
        if not self._active or not self.on_wake or not self.available:
            return
        now = time.monotonic()
        if now - self._last_fire < self.cooldown_s:
            return
        try:
            import numpy as np

            # float32 [-1,1] → int16, as openWakeWord expects.
            samples = np.clip(indata[:, 0] * 32767.0, -32768, 32767).astype(np.int16)
            scores = self._model.predict(
                samples,
                threshold={self.word: self.threshold},
                patience={self.word: self.patience},
            )
            # With patience applied the score is 0.0 unless the word was
            # sustained above threshold for `patience` consecutive frames.
            score = float(scores.get(self.word, 0.0))
            if score > 0.0:
                self._last_fire = now
                logger.info(f"Wake word detected: {self.word!r} (score={score:.3f})")
                try:
                    self.on_wake()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Wake word callback error: {e}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wake word frame error: {e}")
