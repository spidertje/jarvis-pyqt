"""Voice-cycle timeout + error surfacing (audit B2 / U2).

- run_voice() must cancel a hung cycle after voice_cycle_timeout and report a
  human-readable error, returning to IDLE (never wedged in LISTENING).
- ChatClient.last_error classifies connection failures with the endpoint.
- agent.on_error callbacks receive service-failure messages.
"""
import asyncio

import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.chat import ChatClient, ChatConfig
from jarvis.state import JarvisState


class HangingSTT:
    """stt.listen() that never returns — simulates a network hang."""
    config = type("C", (), {"host": "10.9.9.9", "port": 10300})()
    _connected = True

    async def listen(self, **kwargs):
        await asyncio.sleep(3600)
        return None


def make_agent_with_timeout(timeout=0.3):
    cfg = AgentConfig()
    cfg.voice_cycle_timeout = timeout
    agent = JarvisAgent(cfg)
    agent.stt = HangingSTT()
    return agent


class TestVoiceCycleTimeout:
    @pytest.mark.asyncio
    async def test_hung_cycle_cancelled_and_reported(self):
        agent = make_agent_with_timeout(timeout=0.3)
        errors = []
        agent.on_error(errors.append)

        # Must complete in ~timeout, not 3600s.
        await asyncio.wait_for(agent.run_voice(), timeout=5.0)

        assert errors, "no error reported for hung voice cycle"
        assert "timed out" in errors[0].lower()
        assert agent.state == JarvisState.IDLE, "stuck in LISTENING after timeout"

    @pytest.mark.asyncio
    async def test_zero_timeout_disables_wrapper(self):
        cfg = AgentConfig()
        cfg.voice_cycle_timeout = 0
        agent = JarvisAgent(cfg)

        called = []

        async def fake_inner():
            called.append(True)

        agent._run_voice_inner = fake_inner
        await agent.run_voice()
        assert called


class TestChatErrorClassification:
    def test_no_base_url_sets_actionable_error(self):
        client = ChatClient(ChatConfig(base_url=None))

        async def _go():
            try:
                return await client.chat([{"role": "user", "content": "hi"}])
            finally:
                await client.close()

        assert asyncio.run(_go()) is None
        assert client.last_error is not None
        assert "base_url" in client.last_error

    def test_unreachable_endpoint_error_names_host(self):
        client = ChatClient(ChatConfig(base_url="http://127.0.0.1:1", timeout=2))

        async def _go():
            try:
                return await client.chat([{"role": "user", "content": "hi"}])
            finally:
                await client.close()

        assert asyncio.run(_go()) is None
        assert client.last_error is not None
        assert "127.0.0.1:1" in client.last_error
        assert "unreachable" in client.last_error.lower()


class TestOnErrorPlumbing:
    def test_report_error_reaches_callbacks(self):
        agent = JarvisAgent(AgentConfig())
        seen = []
        agent.on_error(seen.append)
        agent._report_error("boom")
        assert seen == ["boom"]

    def test_callback_exception_does_not_break(self):
        agent = JarvisAgent(AgentConfig())
        agent.on_error(lambda m: (_ for _ in ()).throw(RuntimeError("bad cb")))
        agent._report_error("boom")  # must not raise
