"""Tests for the live HUD panels (transcript / reply / activity + approval card)."""

import pytest

from jarvis.hud_panels import ACTIVITY_LINES, ApprovalCard, HUDPanels, REPLY_VISIBLE_CHARS


@pytest.fixture
def panels(qtbot):
    w = HUDPanels()
    qtbot.addWidget(w)
    yield w
    w.deleteLater()


@pytest.fixture
def card(qtbot):
    w = ApprovalCard()
    qtbot.addWidget(w)
    yield w
    w.deleteLater()


class TestPanelsLifecycle:
    def test_hidden_by_default(self, panels):
        assert not panels.isVisible()

    def test_show_and_hide_turn(self, panels):
        panels.show_turn()
        assert panels.isVisible()
        panels.hide_turn()
        assert not panels.isVisible()

    def test_clear_turn_resets_all(self, panels):
        panels.set_user_text("hello")
        panels.append_reply("a reply")
        panels.set_activity("k1", "▸ tool()")
        panels.clear_turn()
        assert panels.user_text == ""
        assert panels.reply_text == ""
        assert panels.activity_lines == []
        assert panels.user_label.text() == ""
        assert panels.reply_label.text() == ""
        assert panels.activity_label.text() == ""


class TestUserText:
    def test_set_user_text(self, panels):
        panels.set_user_text("  turn off the lights  ")
        assert panels.user_text == "turn off the lights"
        assert panels.user_label.text() == "turn off the lights"
        assert panels.isVisible()


class TestReplyText:
    def test_append_accumulates(self, panels):
        panels.append_reply("Hello ")
        panels.append_reply("world")
        assert panels.reply_text == "Hello world"
        assert panels.reply_label.text() == "Hello world"

    def test_long_reply_shows_tail(self, panels):
        text = "word " * 60  # well past REPLY_VISIBLE_CHARS
        panels.append_reply(text)
        shown = panels.reply_label.text()
        assert shown.startswith("… ")
        # The last word must be visible even though the label is truncated.
        assert text.rstrip().split()[-1] in shown
        assert len(shown) < REPLY_VISIBLE_CHARS + 20
        # Full text is still retained internally.
        assert panels.reply_text == text

    def test_empty_append_is_harmless(self, panels):
        panels.append_reply("")
        assert panels.reply_text == ""


class TestActivityFeed:
    def test_set_activity_adds_line(self, panels):
        panels.set_activity("call_1", "▸ get_weather(city=Rīga)")
        assert panels.activity_lines == ["▸ get_weather(city=Rīga)"]

    def test_same_key_replaces_not_duplicates(self, panels):
        """Streaming tool calls re-send an accumulated snapshot per fragment."""
        panels.set_activity("call_1", "▸ get_weather(…)")
        panels.set_activity("call_1", "▸ get_weather(city=Rīga)")
        assert panels.activity_lines == ["▸ get_weather(city=Rīga)"]

    def test_multiple_keys_keep_order(self, panels):
        panels.set_activity("a", "▸ one()")
        panels.set_activity("b", "▸ two()")
        panels.set_activity("c", "▸ three()")
        assert panels.activity_lines == ["▸ one()", "▸ two()", "▸ three()"]

    def test_capped_at_activity_lines(self, panels):
        for i in range(ACTIVITY_LINES + 3):
            panels.set_activity(f"k{i}", f"▸ t{i}()")
        lines = panels.activity_lines
        assert len(lines) == ACTIVITY_LINES
        assert lines[-1] == f"▸ t{ACTIVITY_LINES + 2}()"
        # Oldest lines were dropped.
        assert "▸ t0()" not in lines

    def test_label_shows_latest_two(self, panels):
        panels.set_activity("a", "▸ one()")
        panels.set_activity("b", "▸ two()")
        panels.set_activity("c", "▸ three()")
        assert panels.activity_label.text() == "▸ two()\n▸ three()"


class TestApprovalCard:
    def test_starts_hidden(self, card):
        assert not card.isVisible()

    def test_show_action_populates_title(self, card):
        card.show_action("send_email", "to: bob@example.com")
        assert "send_email" in card.title_label.text()
        assert card.detail_label.text() == "to: bob@example.com"
        assert card.isVisible()

    def test_allow_emits_true(self, card):
        results = []
        card.decided.connect(results.append)
        card.allow_btn.clicked.emit()
        assert results == [True]
        assert not card.isVisible()

    def test_deny_emits_false(self, card):
        results = []
        card.decided.connect(results.append)
        card.deny_btn.clicked.emit()
        assert results == [False]
