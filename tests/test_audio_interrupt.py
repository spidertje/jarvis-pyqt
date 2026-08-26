"""Tests for interruptible AudioPlayer playback (barge-in support)."""

import threading
import time
from unittest.mock import MagicMock

from jarvis.audio_player import AudioPlayer


def _pcm(samples: int) -> bytes:
    """Silent 16-bit mono PCM of ``samples`` samples."""
    return b"\x00\x00" * samples


class TestInterruption:
    def test_stop_interrupts_long_playback(self):
        """stop() must cut playback far earlier than the full clip duration."""
        player = AudioPlayer()
        player._sd = MagicMock()
        mock_stream = MagicMock()
        player._sd.OutputStream.return_value = mock_stream

        # ~10 seconds of audio at 16kHz; each chunk write sleeps 8ms →
        # uninterrupted playback would take ~0.5s (63 chunks).
        pcm = _pcm(160000)
        writes = []

        def slow_write(data):
            time.sleep(0.008)
            writes.append(len(data))

        mock_stream.write.side_effect = slow_write

        result = {}

        def play_thread():
            start = time.monotonic()
            player.play(pcm)
            result["elapsed"] = time.monotonic() - start
            result["writes"] = len(writes)
            result["playing"] = player.is_playing

        t = threading.Thread(target=play_thread)
        t.start()
        time.sleep(0.15)  # let playback begin
        player.stop()
        t.join(timeout=3.0)

        assert not t.is_alive(), "playback did not stop"
        assert result["elapsed"] < 0.5, f"playback ran too long: {result['elapsed']:.2f}s"
        assert result["writes"] < 63, "too many chunks written — stop() not honoured"
        assert result["playing"] is False
        mock_stream.close.assert_called()

    def test_stop_is_idempotent(self):
        player = AudioPlayer()
        player._sd = None
        player.stop()
        player.stop()
        assert player._playing is False

    def test_play_respects_stop_flag_set_during_stream_setup(self):
        player = AudioPlayer()
        player._sd = MagicMock()
        player._stream = MagicMock()
        player._stop_flag = True
        pcm = _pcm(4000)
        player.play(pcm)  # must return without writing
        assert player.is_playing is False
