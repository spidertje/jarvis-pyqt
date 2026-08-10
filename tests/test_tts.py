"""Tests for PiperTTS connection management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.tts import PiperTTS, WyomingConfig


class TestPiperTTS:
    def test_voice_property_syncs_with_config(self):
        """self.voice should default from config.voice."""
        cfg = WyomingConfig(voice="en_US-amy-low")
        tts = PiperTTS(cfg)
        assert tts.voice == "en_US-amy-low"
        assert tts._prev_voice == "en_US-amy-low"

    def test_voice_change_tracked(self):
        """set_voice should track whether the voice changed."""
        tts = PiperTTS(WyomingConfig(voice="en_US-lessac-medium"))
        assert not tts.set_voice("en_US-lessac-medium")  # no change
        assert tts.set_voice("en_US-amy-low")  # changed
        assert tts.voice == "en_US-amy-low"

    @pytest.mark.asyncio
    async def test_ensure_connected_with_existing_connection(self):
        """_ensure_connected should return True if already connected."""
        tts = PiperTTS()
        tts._connected = True
        tts._writer = MagicMock()
        tts._writer.is_closing = MagicMock(return_value=False)
        # Should not call connect()
        result = await tts._ensure_connected()
        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects_when_needed(self):
        """_ensure_connected should reconnect if connection is stale."""
        tts = PiperTTS()
        tts._connected = False
        with patch.object(tts, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            result = await tts._ensure_connected()
            assert result is True
            mock_connect.assert_awaited_once()
