"""Tests for agent HUD-panel callbacks and the local approval gate.

Drives JarvisAgent with mocked STT/chat/TTS/audio (same pattern as
test_agent_streaming) and asserts:
- the user utterance, each reply token, and in-band tool calls reach
  registered callbacks;
- request_approval blocks until resolve_approval, denies on timeout,
  and rejects concurrent requests.
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


class TestApprovalGate:
    @pytest.mark.asyncio
    async def test_allow_resolves_pending_request(self):
        agent = _make_agent()
        fut = asyncio.create_task(agent.request_approval("send_email", timeout=5.0))
        await asyncio.sleep(0.05)  # let it block
        threading.Timer(0.05, lambda: agent.resolve_approval(True)).start()
        result = await fut
        assert result is True
        assert agent._approval_evt is None  # gate released

    @pytest.mark.asyncio
    async def test_deny_resolves_pending_request(self):
        agent = _make_agent()
        fut = asyncio.create_task(agent.request_approval("send_email", timeout=5.0))
        await asyncio.sleep(0.05)
        agent.resolve_approval(False)
        assert await fut is False

    @pytest.mark.asyncio
    async def test_timeout_defaults_to_deny(self):
        agent = _make_agent()
        result = await agent.request_approval("open_url", timeout=0.15)
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_request_rejected(self):
        agent = _make_agent()
        fut = asyncio.create_task(agent.request_approval("first", timeout=5.0))
        await asyncio.sleep(0.05)
        # A second request while one is pending must not deadlock — it denies.
        assert await agent.request_approval("second", timeout=0.1) is False
        agent.resolve_approval(True)
        assert await fut is True

    def test_resolve_without_request_is_noop(self):
        agent = _make_agent()
        agent.resolve_approval(True)  # must not raise
        assert agent._approval_evt is None
