"""Tests for JarvisAgent and AgentConfig."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.state import JarvisState


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.silence_timeout == 2.0
        assert cfg.silence_threshold == 100.0
        assert cfg.assistant_name == "Jarvis"
        assert cfg.tts_voice == "en_US-lessac-medium"
        assert cfg.contrast_boost == 100
        assert cfg.palette_index is None

    def test_custom_values(self):
        cfg = AgentConfig(
            silence_timeout=5.0,
            silence_threshold=50.0,
            assistant_name="Hermes",
        )
        assert cfg.silence_timeout == 5.0
        assert cfg.silence_threshold == 50.0
        assert cfg.assistant_name == "Hermes"


class TestJarvisAgent:
    def test_init_state_is_idle(self):
        """JarvisAgent should start in IDLE state."""
        agent = JarvisAgent()
        assert agent.state == JarvisState.IDLE

    def test_init_with_hud(self):
        """JarvisAgent should store hud reference."""
        mock_hud = MagicMock()
        agent = JarvisAgent(AgentConfig(), hud=mock_hud)
        assert agent.hud is mock_hud

    def test_on_state_change_registers_callback(self):
        """on_state_change should register a callback."""
        agent = JarvisAgent()
        callback = MagicMock()
        agent.on_state_change(callback)
        agent._set_state(JarvisState.LISTENING)
        callback.assert_called_once_with(JarvisState.LISTENING)

    def test_set_state_same_no_callback(self):
        """_set_state should not call callbacks if state is unchanged."""
        agent = JarvisAgent()
        callback = MagicMock()
        agent.on_state_change(callback)
        agent._set_state(JarvisState.IDLE)  # already IDLE
        callback.assert_not_called()

    def test_set_state_different_calls_callback(self):
        """_set_state should call callbacks when state changes."""
        agent = JarvisAgent()
        callback1 = MagicMock()
        callback2 = MagicMock()
        agent.on_state_change(callback1)
        agent.on_state_change(callback2)
        agent._set_state(JarvisState.SPEAKING)
        callback1.assert_called_once_with(JarvisState.SPEAKING)
        callback2.assert_called_once_with(JarvisState.SPEAKING)

    def test_get_system_prompt_default(self):
        """get_system_prompt should return default or profile-specific prompt."""
        agent = JarvisAgent(AgentConfig(default_system_prompt="You are test"))
        agent._system_prompt = "test prompt"
        assert agent.get_system_prompt() == "test prompt"

    def test_switch_profile_not_found(self):
        """switch_profile should return False for unknown profile."""
        agent = JarvisAgent()
        result = agent.switch_profile("nonexistent")
        assert result is False

    def test_clear_profile(self):
        """clear_profile should reset to default state."""
        agent = JarvisAgent()
        agent._active_profile = MagicMock()
        agent._system_prompt = "custom"
        agent.clear_profile()
        assert agent._active_profile is None
        assert agent._messages == []


class TestJarvisAgentAsync:
    @pytest.mark.asyncio
    async def test_chat_text_returns_reply(self):
        """chat_text should return LLM reply."""
        agent = JarvisAgent()
        agent.chat = MagicMock()
        agent.chat.chat = AsyncMock(return_value="Hello there!")
        reply = await agent.chat_text("Hi")
        assert reply == "Hello there!"

    @pytest.mark.asyncio
    async def test_chat_text_no_reply(self):
        """chat_text should return empty string on None reply."""
        agent = JarvisAgent()
        agent.chat = MagicMock()
        agent.chat.chat = AsyncMock(return_value=None)
        reply = await agent.chat_text("Hi")
        assert reply == ""

    @pytest.mark.asyncio
    async def test_close(self):
        """close should call close on all services."""
        agent = JarvisAgent()
        agent.chat = MagicMock()
        agent.chat.close = AsyncMock()
        agent.stt = MagicMock()
        agent.stt.disconnect = AsyncMock()
        agent.tts = MagicMock()
        agent.tts.disconnect = AsyncMock()
        agent.audio = MagicMock()
        agent.profiles = MagicMock()
        await agent.close()
        agent.chat.close.assert_awaited_once()
        agent.stt.disconnect.assert_awaited_once()
        agent.tts.disconnect.assert_awaited_once()
        agent.audio.stop.assert_called_once()
        agent.profiles.close.assert_called_once()
