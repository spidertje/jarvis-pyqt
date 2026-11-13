"""Tests for the face-recognition greeting ('Hello <name>!')."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.state import JarvisState


def _make_agent():
    agent = JarvisAgent(AgentConfig())
    agent.hud = None
    agent.tts = MagicMock()
    agent.tts.speak = AsyncMock(return_value=b"\x00\x00" * 10)
    agent.audio = MagicMock()
    agent.audio.play = MagicMock()
    agent._active_profile = None
    agent._barge_in = None
    agent.profiles = MagicMock()
    return agent


@pytest.mark.asyncio
async def test_greeting_speaks_and_returns_idle():
    agent = _make_agent()
    assert agent.state == JarvisState.IDLE
    ok = await agent.greet("Filip")
    assert ok is True
    agent.tts.speak.assert_awaited_once_with("Hello Filip!")
    agent.audio.play.assert_called_once()
    assert agent.state == JarvisState.IDLE  # SPEAKING was transient


@pytest.mark.asyncio
async def test_greeting_emits_to_panel_callbacks():
    agent = _make_agent()
    got = []
    agent.on_reply_text(got.append)
    await agent.greet("Markus")
    assert got == ["Hello Markus!"]


@pytest.mark.asyncio
async def test_greeting_disabled_by_config():
    agent = _make_agent()
    agent.config.greeting_enabled = False
    assert await agent.greet("Filip") is False
    agent.tts.speak.assert_not_awaited()


@pytest.mark.asyncio
async def test_greeting_skipped_when_busy():
    """Never talk over a running voice cycle / active mic."""
    agent = _make_agent()
    agent.state = JarvisState.LISTENING
    assert await agent.greet("Filip") is False
    agent.tts.speak.assert_not_awaited()


@pytest.mark.asyncio
async def test_greeting_tts_failure_recovers_to_idle():
    agent = _make_agent()
    agent.tts.speak = AsyncMock(side_effect=ConnectionError("piper down"))
    assert await agent.greet("Filip") is False
    assert agent.state == JarvisState.IDLE


@pytest.mark.asyncio
async def test_greeting_empty_name_noop():
    agent = _make_agent()
    assert await agent.greet("") is False
    agent.tts.speak.assert_not_awaited()
