"""ChatLogWindow: two-sided conversation view + toolbar integration."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture()
def log_window(qtbot):
    from jarvis.chat_log import ChatLogWindow

    win = ChatLogWindow()
    qtbot.addWidget(win)
    return win


def test_starts_empty(log_window):
    assert "No messages yet" in log_window.view.toPlainText() or log_window.view.toHtml()


def test_set_history_renders_both_sides(log_window):
    log_window.set_history([
        {"role": "user", "content": "hello jarvis"},
        {"role": "assistant", "content": "hello **human**"},
    ])
    html = log_window.view.toHtml()
    assert "hello" in html
    assert "human" in html
    # markdown-lite made the reply bold (Qt re-serializes <b> as font-weight)
    assert "<b>" in html or "font-weight:700" in html.replace(" ", "")
    # both speaker labels present
    assert "You" in html and "Jarvis" in html


def test_live_turn_streaming(log_window):
    log_window.set_history([{"role": "user", "content": "old"},])
    log_window.set_live(user_text="what time is it")
    assert "what time is it" in log_window.view.toHtml()
    log_window.set_live(reply_text="it is")
    log_window.set_live(reply_text="it is noon")
    html = log_window.view.toHtml()
    assert "noon" in html
    log_window.clear_live()
    assert "noon" not in log_window.view.toHtml()


def test_send_requested_signal(log_window, qtbot):
    with qtbot.waitSignal(log_window.send_requested, timeout=1000) as blocker:
        log_window.input.setText("hi there")
        log_window.send_btn.click()
    assert blocker.args == ["hi there"]
    assert log_window.input.text() == ""


def test_send_empty_ignored(log_window):
    calls = []
    log_window.send_requested.connect(calls.append)
    log_window.send_btn.click()
    assert calls == []


def test_closed_signal_on_close(log_window, qtbot):
    with qtbot.waitSignal(log_window.closed, timeout=1000):
        log_window.close()


def test_theme_switch_keeps_text(log_window):
    from jarvis import theme
    log_window.set_history([{"role": "user", "content": "keepme"}])
    t = theme.themed(theme.Theme(accent="#ff0000"))
    log_window.set_theme(t)
    assert "keepme" in log_window.view.toHtml()
