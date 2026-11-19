"""Agent-side chat logging + text-path mirroring (chat_messages rows).

Guards the 2026-11-19 fix: the text chat path used to emit nothing to the
panel/chat-log callbacks and persisted nothing; greetings vanished from the
chat window on the IDLE fold because they never reached the DB.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.agent import AgentConfig, JarvisAgent


def _make_agent(reply_tokens=("Sure!", " Here!")):
    agent = JarvisAgent(AgentConfig())
    agent.stt = MagicMock()
    agent.stt.listen = AsyncMock(return_value="weather please")

    async def fake_chat(messages, stream=False, system_prompt=None, on_token=None, on_tool_call=None):
        if on_token:
            for tok in reply_tokens:
                on_token(tok)
                await asyncio.sleep(0.001)
        return "".join(reply_tokens)

    agent.chat = MagicMock()
    agent.chat.chat = fake_chat
    agent.tts = MagicMock()
    agent.tts.speak = AsyncMock(return_value=b"\x00" * 4)
    agent.audio = MagicMock()
    agent.audio.stop = MagicMock()
    agent.audio.play = MagicMock()
    agent.hud = None
    agent._barge_in = None
    agent.profiles = MagicMock()
    agent._active_profile = MagicMock()
    agent._active_profile.name = "markus"
    agent._active_profile.chat_history = []
    return agent


def _logged(agent):
    return [(c.args[1], c.args[2]) for c in agent.profiles.append_chat_log.call_args_list]


class TestChatLogPersistence:
    @pytest.mark.asyncio
    async def test_text_chat_logs_user_and_assistant_rows(self):
        agent = _make_agent()
        await agent.chat_text("open the pod bay doors")
        assert _logged(agent) == [
            ("user", "open the pod bay doors"),
            ("assistant", "Sure! Here!"),
        ]

    @pytest.mark.asyncio
    async def test_voice_cycle_logs_user_and_assistant_rows(self):
        agent = _make_agent()
        await agent._run_voice()
        kinds = [r for r, _ in _logged(agent)]
        assert kinds == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_greeting_logs_row_but_not_llm_messages(self):
        agent = _make_agent()
        assert await agent.greet("markus") is True
        assert ("assistant", "Hello markus!") in _logged(agent)
        # greetings must never enter the LLM context
        assert agent._messages == []

    @pytest.mark.asyncio
    async def test_no_profile_means_no_chat_log_writes(self):
        agent = _make_agent()
        agent._active_profile = None
        await agent.chat_text("hello")
        agent.profiles.append_chat_log.assert_not_called()


class TestTextPathPanelCallbacks:
    @pytest.mark.asyncio
    async def test_text_chat_emits_user_and_reply_text(self):
        agent = _make_agent()
        users, replies = [], []
        agent.on_user_text(users.append)
        agent.on_reply_text(replies.append)
        await agent.chat_text("hello there")
        assert users == ["hello there"]
        assert replies == ["Sure! Here!"]

    @pytest.mark.asyncio
    async def test_failed_llm_emits_user_only(self):
        """ChatClient.chat returns None on failure (never raises) — the
        text path must not log an assistant row in that case."""
        agent = _make_agent()

        async def unreachable(messages, stream=False, system_prompt=None,
                              on_token=None, on_tool_call=None):
            return None

        agent.chat.chat = unreachable
        agent.chat.last_error = "LLM down"
        users, replies = [], []
        agent.on_user_text(users.append)
        agent.on_reply_text(replies.append)
        out = await agent.chat_text("hi")
        assert out == ""
        assert users == ["hi"]
        assert replies == []
        assert [r for r, _ in _logged(agent)] == ["user"]
