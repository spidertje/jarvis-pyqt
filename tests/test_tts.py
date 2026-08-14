"""Tests for PiperTTS connection management."""

import json
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

    @pytest.mark.asyncio
    async def test_speak_returns_none_when_reconnect_fails(self):
        """speak() must bail out (not crash) if a voice-change reconnect fails."""
        tts = PiperTTS(WyomingConfig(voice="en_US-lessac-medium"))
        tts._connected = True  # so voice_changed triggers the reconnect branch
        tts._prev_voice = "en_US-lessac-medium"
        tts.set_voice("en_US-amy-low")  # changes voice -> reconnect path
        # reconnect() -> disconnect() + connect() ; stub connect to fail
        with patch.object(tts, "connect", new_callable=AsyncMock, return_value=False):
            result = await tts.speak("hello")
        assert result is None
        # voice should remain marked as prev even though reconnect failed,
        # so a subsequent call would attempt reconnect again rather than no-op
        assert tts._writer is None or not tts._connected

    @pytest.mark.asyncio
    async def test_speak_sends_voice_as_object_with_name(self):
        """synthesize event must send voice as {"name": "..."} per Wyoming spec."""
        tts = PiperTTS(WyomingConfig(voice="en_US-amy-medium"))
        tts._connected = True
        tts._writer = MagicMock()
        tts._writer.is_closing = MagicMock(return_value=False)
        tts._writer.drain = AsyncMock()
        tts._writer.write = MagicMock()

        # Mock reader for audio-start, audio-chunk (with payload), audio-stop
        reader = MagicMock()
        events = [
            {"type": "audio-start"},
            {"type": "audio-chunk", "payload_length": 0},
            {"type": "audio-stop"},
        ]
        reader.readline = AsyncMock(
            side_effect=[json.dumps(e).encode() + b"\n" for e in events]
        )
        tts._reader = reader

        with patch.object(tts, "_ensure_connected", AsyncMock(return_value=True)):
            await tts.speak("hello world")

        # Inspect the bytes written for the synthesize event
        written = b"".join(call.args[0] for call in tts._writer.write.call_args_list)
        assert b'"synthesize"' in written
        event_json = json.loads(written.decode().strip())
        assert event_json["type"] == "synthesize"
        assert event_json["data"]["text"] == "hello world"
        assert event_json["data"]["voice"] == {"name": "en_US-amy-medium"}

    @pytest.mark.asyncio
    async def test_speak_omits_voice_when_empty(self):
        """When voice is empty, no voice key in synthesize data."""
        tts = PiperTTS()
        tts.voice = ""  # no voice
        tts._prev_voice = ""  # avoid reconnect path
        tts._connected = True
        tts._writer = MagicMock()
        tts._writer.is_closing = MagicMock(return_value=False)
        tts._writer.drain = AsyncMock()

        reader = MagicMock()
        events = [
            {"type": "audio-start"},
            {"type": "audio-stop"},
        ]
        reader.readline = AsyncMock(
            side_effect=[json.dumps(e).encode() + b"\n" for e in events]
        )
        tts._reader = reader

        with patch.object(tts, "_ensure_connected", AsyncMock(return_value=True)):
            await tts.speak("hello")

        written = b"".join(call.args[0] for call in tts._writer.write.call_args_list)
        event_json = json.loads(written.decode().strip())
        assert "voice" not in event_json["data"]
