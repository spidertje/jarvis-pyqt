"""Integration tests for the streaming voice cycle (producer → speaker → barge-in).

Drives JarvisAgent._run_voice with a mocked STT, a chat client that streams
tokens, a TTS that returns fake PCM, and a recording audio player — asserting
the reply is spoken sentence-by-sentence and that barge-in stops it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.state import JarvisState


def _make_agent(reply_tokens, speak_delay=0.0):
    """Build an agent whose chat streams ``reply_tokens`` and TTS returns PCM."""
    agent = JarvisAgent(AgentConfig())

    # STT returns a canned transcript.
    agent.stt = MagicMock()
    agent.stt.listen = AsyncMock(return_value="Tell me a short story.")

    # Chat streams tokens through on_token (simulating a real SSE stream).
    async def fake_chat(messages, stream=False, system_prompt=None, on_token=None):
        if on_token:
            for tok in reply_tokens:
                on_token(tok)
                await asyncio.sleep(0.002)
        return "".join(reply_tokens)

    agent.chat = MagicMock()
    agent.chat.chat = fake_chat

    # TTS returns one fake PCM block per sentence (16-bit silence), recording
    # each text so tests can assert per-sentence synthesis order.
    async def _fake_speak(text):
        if speak_delay:
            await asyncio.sleep(speak_delay)
        return b"\x00\x00" * 100

    agent.tts = MagicMock()
    agent.tts.speak = AsyncMock(side_effect=_fake_speak)

    # Audio player records each played sentence's length.
    agent.audio = MagicMock()
    agent.audio.stop = MagicMock()
    agent.audio.play = MagicMock()
    agent.audio.is_playing = False

    # No HUD, no barge-in device, no profile.
    agent.hud = None
    agent._active_profile = None
    agent._barge_in = None
    agent.profiles = MagicMock()
    return agent


class TestStreamingVoiceCycle:
    @pytest.mark.asyncio
    async def test_reply_spoken_sentence_by_sentence(self):
        agent = _make_agent(["Sure!", " Here's", " the", " plan", ".", " step", " one", "."])
        await agent._run_voice()

        # Sentences: "Sure!", "Here's the plan.", "step one." (tail flush).
        played = len(agent.audio.play.call_args_list)
        assert played == 3
        # TTS was called per sentence, in order.
        tts_texts = [c.args[0] for c in agent.tts.speak.call_args_list]
        assert tts_texts == ["Sure!", "Here's the plan.", "step one."]

    @pytest.mark.asyncio
    async def test_states_progress_through_speaking(self):
        seen = []
        agent = _make_agent(["Hi.", " Bye."])
        agent.on_state_change(lambda s: seen.append(s))
        await agent._run_voice()
        assert JarvisState.LISTENING in seen
        assert JarvisState.SPEAKING in seen
        assert seen[-1] == JarvisState.IDLE

    @pytest.mark.asyncio
    async def test_interrupt_stops_speaking(self):
        """interrupt() mid-stream must stop playback and end the cycle."""
        agent = _make_agent(
            ["One.", " Two.", " Three.", " Four.", " Five."],
            speak_delay=0.04,
        )

        # Interrupt deterministically the moment the first sentence starts
        # playing (no wall-clock races): playback runs in a worker thread,
        # interrupt() is thread-safe by design.
        def play_and_interrupt(audio):
            agent.interrupt()

        agent.audio.play = MagicMock(side_effect=play_and_interrupt)

        await agent._run_voice()

        # Exactly the first sentence was played; the remaining four were cut.
        played = len(agent.audio.play.call_args_list)
        assert played == 1, f"expected exactly 1 sentence played, got {played}"
        agent.audio.stop.assert_called()

    @pytest.mark.asyncio
    async def test_interrupt_is_idempotent(self):
        agent = _make_agent(["Hi."])
        agent.interrupt()
        agent.interrupt()  # second call must not raise
        assert agent.stop_requested is True

    @pytest.mark.asyncio
    async def test_no_stream_falls_back_to_nonstream(self):
        """If on_token is never called, the one-shot reply is still spoken."""
        agent = _make_agent([])  # no tokens

        async def fake_chat(messages, stream=False, system_prompt=None, on_token=None):
            return "This whole reply is one shot. No streaming."

        agent.chat.chat = fake_chat
        await agent._run_voice()

        played = [c.args[0] for c in agent.audio.play.call_args_list]
        assert len(played) >= 1

    @pytest.mark.asyncio
    async def test_close_stops_barge_in(self):
        agent = _make_agent(["Hi."])
        agent.chat.close = AsyncMock()
        agent.stt.disconnect = AsyncMock()
        agent.tts.disconnect = AsyncMock()
        barge = MagicMock()
        agent._barge_in = barge
        await agent.close()
        barge.stop.assert_called_once()
        assert agent._barge_in is None
