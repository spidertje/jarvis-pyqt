"""Tests for the live scrolling transcript panel (HUDPanels)."""

import pytest

from jarvis.hud_panels import ACTIVITY_LINES, MAX_TURNS, REPLY_VISIBLE_CHARS, HUDPanels


@pytest.fixture
def panels(qtbot):
    w = HUDPanels()
    qtbot.addWidget(w)
    yield w
    w.deleteLater()


class TestLifecycle:
    def test_hidden_by_default(self, panels):
        assert not panels.isVisible()

    def test_show_turn_shows(self, panels):
        panels.show_turn()
        assert panels.isVisible()

    def test_content_shows_panel(self, panels):
        panels.set_user_text("hello")
        assert panels.isVisible()

    def test_hide_turn_keeps_history_visible(self, panels):
        panels.set_user_text("hello")
        panels.append_reply("hi there")
        panels.hide_turn()
        assert panels.isVisible()  # history remains readable

    def test_hide_turn_empty_hides(self, panels):
        panels.set_user_text("x")
        panels.clear_turn()
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
        assert "hello" not in panels.html


class TestUserText:
    def test_set_user_text(self, panels):
        panels.set_user_text("  turn off the lights  ")
        assert panels.user_text == "turn off the lights"
        assert "turn off the lights" in panels.html
        assert panels.isVisible()


class TestReplyStream:
    def test_append_accumulates(self, panels):
        panels.append_reply("Hello ")
        panels.append_reply("world")
        assert panels.reply_text == "Hello world"
        assert "Hello world" in panels.html

    def test_long_reply_capped_internally(self, panels):
        text = "word " * 400  # well past REPLY_VISIBLE_CHARS
        panels.append_reply(text)
        assert len(panels.reply_text) <= REPLY_VISIBLE_CHARS
        # The newest content survives.
        assert panels.reply_text.rstrip().endswith("word")

    def test_empty_append_is_harmless(self, panels):
        panels.append_reply("")
        assert panels.reply_text == ""


class TestHistory:
    def test_turn_rolls_over_on_new_user_text(self, panels):
        panels.set_user_text("first")
        panels.append_reply("r1")
        panels.set_user_text("second")
        assert panels.user_text == "second"
        assert panels.reply_text == ""
        # both turns still visible
        assert "first" in panels.html and "second" in panels.html

    def test_begin_turn_commits(self, panels):
        panels.set_user_text("a")
        panels.append_reply("b")
        panels.begin_turn()
        assert panels.user_text == ""
        assert "a" in panels.html and "b" in panels.html

    def test_history_capped(self, panels):
        for i in range(MAX_TURNS + 4):
            panels.set_user_text(f"uniquetag{i}")
            panels.append_reply(f"a{i}")
        # newest kept, oldest dropped
        assert f"uniquetag{MAX_TURNS + 3}" in panels.html
        assert "uniquetag0<" not in panels.html and "uniquetag0 " not in panels.html
        assert "uniquetag0\"" not in panels.html


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
        assert "▸ t0()" not in lines

    def test_activity_rendered_in_html(self, panels):
        panels.set_activity("a", "▸ one()")
        assert "▸ one()" in panels.html


class TestMarkdownLite:
    def test_bold_italic_code(self):
        from jarvis.hud_panels import md_lite

        h = md_lite("**bold** *it* `code`")
        assert "<b>bold</b>" in h
        assert "<i>it</i>" in h
        assert "<code>code</code>" in h

    def test_fenced_block(self):
        from jarvis.hud_panels import md_lite

        assert "<pre>" in md_lite("```python\nprint(1)\n```")

    def test_html_injected_is_escaped(self):
        from jarvis.hud_panels import md_lite

        h = md_lite("<script>alert(1)</script>")
        assert "<script>" not in h
        assert "&lt;script&gt;" in h

    def test_panel_shapes_rendered_text(self, panels):
        panels.append_reply("**bold** `code`")
        # QTextBrowser normalizes tags but the content must be present.
        assert "bold" in panels.html and "code" in panels.html
        doc = panels.view.document().find("bold")
        assert doc.charFormat().fontWeight() >= 75  # bold applied
