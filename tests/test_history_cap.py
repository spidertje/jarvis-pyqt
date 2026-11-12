"""Conversation-history cap (audit B4): unbounded _messages + DB JSON growth."""
from jarvis.agent import AgentConfig, JarvisAgent


def make(cap):
    cfg = AgentConfig()
    cfg.max_history_messages = cap
    return JarvisAgent(cfg)


class TestTrimHelpers:
    def test_append_respects_cap(self):
        agent = make(cap=6)
        for i in range(10):
            agent._append_message("user", f"u{i}")
            agent._append_message("assistant", f"a{i}")
        assert len(agent._messages) <= 6
        # newest retained, window starts with a user message (no orphan reply)
        assert agent._messages[-1]["content"] == "a9"
        assert agent._messages[0]["role"] == "user"

    def test_cap_zero_disables(self):
        agent = make(cap=0)
        for i in range(50):
            agent._append_message("user", f"u{i}")
        assert len(agent._messages) == 50

    def test_trim_history_pure(self):
        agent = make(cap=4)
        msgs = [{"role": "user", "content": str(i)} for i in range(10)]
        out = agent._trim_history(msgs)
        assert len(out) == 4
        assert out[-1]["content"] == "9"

    def test_trim_skips_orphan_assistant_start(self):
        agent = make(cap=4)
        msgs = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
            {"role": "assistant", "content": "5"},  # dup assistant tail
        ]
        out = agent._trim_history(msgs)
        # window [-4:] starts with assistant -> dropped to next user turn
        assert out[0]["role"] == "user"


class TestPersistedHistoryStaysBounded:
    def test_switch_profile_loads_trimmed(self):
        from jarvis.profile import Profile
        agent = make(cap=4)
        # Pre-poison the DB row's history: 20 messages
        big = []
        for i in range(10):
            big.append({"role": "user", "content": f"u{i}"})
            big.append({"role": "assistant", "content": f"a{i}"})
        prof = Profile(name="p1", system_prompt="sp", assistant_name="Jarvis",
                       chat_history=big)
        agent.profiles._profiles["p1"] = prof
        assert agent.switch_profile("p1")
        assert len(agent._messages) <= 4
        assert agent._messages[-1]["content"] == "a9"
