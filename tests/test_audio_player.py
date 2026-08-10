"""Tests for AudioPlayer and AudioConfig."""

from unittest.mock import MagicMock

from jarvis.audio_player import AudioConfig, AudioPlayer


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.width == 2
        assert cfg.device is None

    def test_custom_values(self):
        cfg = AudioConfig(
            sample_rate=44100,
            channels=2,
            width=4,
            device=5,
        )
        assert cfg.sample_rate == 44100
        assert cfg.channels == 2
        assert cfg.width == 4
        assert cfg.device == 5


class TestAudioPlayerInit:
    def test_init_without_sounddevice(self):
        """AudioPlayer should work even if sounddevice is not available."""
        player = AudioPlayer()
        assert player.config is not None
        assert player._playing is False


class TestAudioPlayerPlay:
    def test_play_empty_data(self):
        """play() should return early for empty PCM data."""
        player = AudioPlayer()
        player.play(b"")  # should not raise
        assert player._playing is False

    def test_play_without_sounddevice(self):
        """play() should gracefully handle missing sounddevice."""
        player = AudioPlayer()
        player._sd = None
        test_data = b"\x00\x01\x00\x02"
        player.play(test_data)  # should log warning, not crash
        assert player._playing is False


class TestAudioPlayerStop:
    def test_stop_without_stream(self):
        """stop() should not raise if no stream is active."""
        player = AudioPlayer()
        player._stream = None
        player.stop()
        assert player._playing is False

    def test_stop_with_stream(self):
        """stop() should close the stream if active."""
        player = AudioPlayer()
        mock_stream = MagicMock()
        player._stream = mock_stream
        player.stop()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert player._stream is None
        assert player._playing is False


class TestAudioPlayerProperties:
    def test_is_playing_false(self):
        player = AudioPlayer()
        assert player.is_playing is False

    def test_is_playing_true(self):
        player = AudioPlayer()
        player._playing = True
        assert player.is_playing is True
