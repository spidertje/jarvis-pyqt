"""Tests for agent HUD-panel callbacks (user text / reply stream / tool activity).

Drives JarvisAgent with mocked STT/chat/TTS/audio (same pattern as
test_agent_streaming) and asserts:
- the user utterance, each reply token, and in-band tool calls reach
  registered callbacks.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.agent import AgentConfig, JarvisAgent


def _make_agent(reply_tokens=("Sure!", " Here's", " the", " plan", ".")):
    agent = JarvisAgent(AgentConfig())
    agent.stt = MagicMock()
    agent.stt.listen = AsyncMock(return_value="Tell me a short story.")

    async def fake_chat(messages, stream=False, system_prompt=None, on_token=None, on_tool_call=None):
        if on_token:
            for tok in reply_tokens:
                on_token(tok)
                await asyncio.sleep(0.001)
        return "".join(reply_tokens)

    agent.chat = MagicMock()
    agent.chat.chat = fake_chat
    agent.tts = MagicMock()
    agent.tts.speak = AsyncMock(return_value=b"\x00\x00" * 10)
    agent.audio = MagicMock()
    agent.audio.stop = MagicMock()
    agent.audio.play = MagicMock()
    agent.hud = None
    agent._active_profile = None
    agent._barge_in = None
    agent.profiles = MagicMock()
    return agent


class TestPanelCallbacks:
    @pytest.mark.asyncio
    async def test_user_text_callback_fires_with_utterance(self):
        agent = _make_agent()
        seen = []
        agent.on_user_text(seen.append)
        await agent._run_voice()
        assert seen == ["Tell me a short story."]

    @pytest.mark.asyncio
    async def test_reply_text_callback_fires_per_token(self):
        tokens = ["One", ".", " Two", "."]
        agent = _make_agent(tuple(tokens))
        seen = []
        agent.on_reply_text(seen.append)
        await agent._run_voice()
        assert "".join(seen) == "".join(tokens)
        assert len(seen) == len(tokens)

    @pytest.mark.asyncio
    async def test_tool_call_callback_fires(self):
        agent = _make_agent()
        calls = []

        async def fake_chat(messages, stream=False, system_prompt=None, on_token=None, on_tool_call=None):
            on_tool_call("call_1", "▸ get_weather(city=Rīga)")
            if on_token:
                on_token("Done.")
            return "Done."

        agent.chat.chat = fake_chat
        agent.on_tool_call(lambda k, d: calls.append((k, d)))
        await agent._run_voice()
        assert calls == [("call_1", "▸ get_weather(city=Rīga)")]

    @pytest.mark.asyncio
    async def test_no_callbacks_registered_is_safe(self):
        agent = _make_agent()
        # No callbacks at all — the run must still complete cleanly.
        await agent._run_voice()
        assert agent.audio.play.call_args_list
