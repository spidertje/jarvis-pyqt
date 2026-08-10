"""Tests for ChatConfig, ChatClient, and AgentConfig."""

import pytest

from jarvis.agent import AgentConfig
from jarvis.chat import ChatClient, ChatConfig


class TestAgentConfig:
    def test_defaults(self):
        """AgentConfig should have sensible defaults."""
        cfg = AgentConfig()
        assert cfg.silence_timeout == 2.0
        assert cfg.silence_threshold == 100.0
        assert cfg.assistant_name == "Jarvis"
        assert cfg.tts_voice == "en_US-lessac-medium"

    def test_custom_silence_threshold(self):
        """AgentConfig should accept custom silence_threshold."""
        cfg = AgentConfig(silence_threshold=50.0, silence_timeout=3.0)
        assert cfg.silence_threshold == 50.0
        assert cfg.silence_timeout == 3.0


class TestChatConfig:
    def test_defaults(self):
        """ChatConfig should have sensible defaults."""
        cfg = ChatConfig()
        assert cfg.base_url is None
        assert cfg.api_key is None
        assert cfg.model == "auto"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1024
        assert cfg.system_prompt.startswith("You are Jarvis")
        assert cfg.timeout == 120

    def test_custom_values(self):
        """ChatConfig should accept custom values."""
        cfg = ChatConfig(
            base_url="http://localhost:8080/v1",
            api_key="secret",
            model="gemma-2b",
            temperature=0.3,
            max_tokens=512,
        )
        assert cfg.base_url == "http://localhost:8080/v1"
        assert cfg.api_key == "secret"
        assert cfg.model == "gemma-2b"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 512


class TestChatClient:
    @pytest.fixture
    def client(self):
        return ChatClient(ChatConfig(base_url="http://localhost:8080/v1"))

    def test_initial_session_is_none(self, client):
        """A new ChatClient should have no active session."""
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_without_session(self, client):
        """close() should not raise if no session exists."""
        await client.close()  # should not raise
        assert client._session is None
