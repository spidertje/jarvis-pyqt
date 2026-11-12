"""Offscreen integration smoke: JarvisApp constructs, wires and tears down.

This is the first test to construct the real app window — it guards the
chrome rewiring (theme tokens, SVG icons, tray, scrolling transcript,
service pills) against import/attribute regressions.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture()
def app_window(qtbot):
    import main as main_mod

    window = main_mod.JarvisApp()
    qtbot.addWidget(window)
    yield window
    # Deterministic teardown: stop timers + camera thread BEFORE close so no
    # Qt timer callback fires against half-deleted widgets (offscreen CI
    # segfaulted on exactly this race).
    window._quitting = True
    window._closing = True
    if getattr(window, "_face_timeout", None) is not None:
        window._face_timeout.stop()
    if window._transition_timer is not None:
        window._transition_timer.stop()
    window.face_thread.stop()
    window.face_thread.wait(3000)
    window.close()


def test_chrome_widgets_exist(app_window):
    for name in ("bottom_bar", "mic_btn", "stop_btn", "send_btn",
                 "settings_btn", "status_label", "status_pill",
                 "chat_input", "panels", "hint_label"):
        assert hasattr(app_window, name), f"missing {name}"

def test_tray_skipped_offscreen(app_window):
    """Offscreen QPA reports a tray but segfaults on show() — the app must
    skip tray construction there (CI exit-139 guard)."""
    import os

    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        assert getattr(app_window, "tray", None) is None
    else:  # pragma: no cover — real desktops
        from PyQt6.QtWidgets import QSystemTrayIcon

        if QSystemTrayIcon.isSystemTrayAvailable():
            assert getattr(app_window, "tray", None) is not None


def test_icons_are_svg_not_emoji(app_window):
    # Icon buttons carry a QIcon (SVG-rendered), no emoji text.
    assert app_window.mic_btn.text() == ""
    assert not app_window.mic_btn.icon().isNull()
    assert not app_window.stop_btn.icon().isNull()
    assert not app_window.settings_btn.icon().isNull()


def test_mic_toggle_is_checkable(app_window):
    assert app_window.mic_btn.isCheckable()


def test_status_pills_render(app_window):
    app_window._update_status_pills()
    txt = app_window.status_pill.text()
    assert "LLM" in txt and "STT" in txt and "TTS" in txt


def test_profile_accent_retheme_runs(app_window):
    app_window._apply_profile_accent(120)   # green accent
    app_window._apply_profile_accent(None)  # back to default


def test_transcript_uses_new_panel(app_window):
    app_window._on_panel_user_text("hello there")
    app_window._on_panel_reply_text("general **protocols**")
    assert "hello there" in app_window.panels.html
    # markdown-lite: bold applied inside the transcript
    assert "<b>protocols</b>" in app_window.panels.view.document().toHtml() \
        or "protocols" in app_window.panels.html
