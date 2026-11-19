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


def test_send_row_removed(log_window):
    """History window is view/search only — typing lives in the toolbar box."""
    assert not hasattr(log_window, "input")
    assert not hasattr(log_window, "send_btn")
    assert not hasattr(log_window, "send_requested")
    log_window.show()
    assert log_window.search.isVisible()
    log_window.close()


def test_search_signal_and_results(log_window, qtbot):
    log_window.set_history([{"role": "user", "content": "alpha"},
                            {"role": "assistant", "content": "beta"}])
    with qtbot.waitSignal(log_window.search_requested, timeout=1000) as blocker:
        log_window.search.setText("alpha")
        log_window._on_search()
    assert blocker.args == ["alpha"]
    # host wires show_results; simulate what main.py does
    log_window.show_results("alpha", [{"role": "user", "content": "alpha"}])
    html = log_window.view.toHtml()
    assert "alpha" in html and "beta" not in html
    assert "Search:" in log_window.title_label.text()


def test_empty_search_emits_cleared(log_window, qtbot):
    log_window.show_results("q", [{"role": "user", "content": "q row"}])
    with qtbot.waitSignal(log_window.search_cleared, timeout=1000):
        log_window.search.setText("")
        log_window._on_search()


def test_set_history_exits_search_mode(log_window):
    log_window.show_results("q", [{"role": "user", "content": "q"}])
    log_window.set_history([{"role": "assistant", "content": "normal"}])
    assert log_window.title_label.text() == "Chat"
    assert "normal" in log_window.view.toHtml()


def test_created_at_shown_in_bubbles(log_window):
    import datetime
    log_window.set_history([{
        "role": "user", "content": "remember me",
        "created_at": datetime.datetime(2026, 11, 19, 9, 30),
    }])
    html = log_window.view.toHtml()
    assert "remember me" in html
    expected = (datetime.datetime(2026, 11, 19, 9, 30, tzinfo=datetime.timezone.utc)
                .astimezone().strftime("%Y-%m-%d %H:%M"))
    assert expected in html  # UTC row rendered in local tz

def test_closed_signal_on_close(log_window, qtbot):
    with qtbot.waitSignal(log_window.closed, timeout=1000):
        log_window.close()


def test_theme_switch_keeps_text(log_window):
    from jarvis import theme
    log_window.set_history([{"role": "user", "content": "keepme"}])
    t = theme.themed(theme.Theme(accent="#ff0000"))
    log_window.set_theme(t)
    assert "keepme" in log_window.view.toHtml()
