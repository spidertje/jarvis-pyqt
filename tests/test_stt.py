"""Tests for WhisperSTT connection management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.stt import WhisperSTT, WyomingConfig


class TestWhisperSTT:
    def test_defaults(self):
        """WyomingConfig should have standard Whisper defaults."""
        cfg = WyomingConfig()
        assert cfg.host == "192.168.55.41"
        assert cfg.port == 10300
        assert cfg.sample_rate == 16000
        assert cfg.device is None

    @pytest.mark.asyncio
    async def test_ensure_connected_with_active_connection(self):
        """_ensure_connected should return True when already connected."""
        stt = WhisperSTT()
        stt._connected = True
        stt._writer = MagicMock()
        stt._writer.is_closing = MagicMock(return_value=False)
        result = await stt._ensure_connected()
        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects_when_stale(self):
        """_ensure_connected should reconnect when connection is None."""
        stt = WhisperSTT()
        stt._connected = False
        stt._writer = None
        with patch.object(stt, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            result = await stt._ensure_connected()
            assert result is True
            mock_connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects_when_closing(self):
        """_ensure_connected should reconnect when writer is closing."""
        stt = WhisperSTT()
        stt._connected = True
        stt._writer = MagicMock()
        stt._writer.is_closing = MagicMock(return_value=True)
        with patch.object(stt, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            result = await stt._ensure_connected()
            assert result is True
            mock_connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        """disconnect should clear _connected, _reader, _writer."""
        stt = WhisperSTT()
        stt._connected = True
        stt._writer = MagicMock()
        stt._writer.wait_closed = AsyncMock()
        stt._reader = MagicMock()
        await stt.disconnect()
        assert stt._connected is False
        assert stt._reader is None
        assert stt._writer is None

    @pytest.mark.asyncio
    async def test_do_transcribe_returns_none_when_not_connected(self):
        """_do_transcribe must bail safely when writer/reader are None."""
        stt = WhisperSTT()
        stt._writer = None
        stt._reader = None
        result = await stt._do_transcribe(b"\x00\x00")
        assert result is None

    @pytest.mark.asyncio
    async def test_transcribe_returns_none_when_connection_fails(self):
        """transcribe should return None (not raise) if it cannot connect."""
        stt = WhisperSTT()
        stt._writer = None
        stt._reader = None
        with patch.object(stt, "connect", new_callable=AsyncMock, return_value=False):
            result = await stt.transcribe(b"\x00\x00")
        assert result is None
