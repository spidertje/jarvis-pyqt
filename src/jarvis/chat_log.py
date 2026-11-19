"""Chat log — a normal top-level window showing the full conversation.

The HUD transcript (``hud_panels.HUDPanels``) is a short scroll of recent
turns pinned to the overlay; this window is the roomy reading view: every
committed turn of the active profile (user AND assistant sides, from the
agent's persisted history) plus the in-flight turn streamed live.

Threading note (same rule as hud_panels): main.py drives this only from the
Qt GUI thread — never call it from the asyncio loop thread.

Integration points in main.py:
  * toolbar button (bottom bar) toggles the window and re-seeds history
  * _on_panel_user_text / _on_panel_reply_text mirror live turns here
  * state changes to LISTENING/IDLE re-sync from ``agent._messages``
"""

from __future__ import annotations

import html as _html

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from jarvis.hud_panels import md_lite


class ChatLogWindow(QWidget):
    """Scrollable two-sided conversation view with a history search box.

    Typing to Jarvis lives in the toolbar input only; this window shows
    the full DB-backed transcript and searches it.
    """

    #: Emitted when the window is closed via its own close button, so the
    #: toolbar toggle can uncheck itself.
    closed = pyqtSignal()
    #: Search over the persisted DB history (main.py runs the LIKE query).
    search_requested = pyqtSignal(str)
    #: Empty submit / Escape leaves search mode and re-seeds the normal view.
    search_cleared = pyqtSignal()

    def __init__(self, theme=None, parent=None):
        super().__init__(parent)
        from jarvis import theme as _theme

        self._theme = theme or _theme.DEFAULT
        self.setObjectName("chatLog")
        self.setWindowTitle("Chat")
        self.resize(560, 640)

        # Committed turns: list of (role, text). Live turn kept separately so
        # a streamed reply never duplicates against a mid-write history.
        # Committed turns: list of (role, text, when|None). Live turn kept
        # separately so a streamed reply never duplicates against a mid-write
        # history.
        self._history: list[tuple[str, str, object]] = []
        self._live_user = ""
        self._live_reply = ""
        self._assistant_name = "Jarvis"
        self._search_query: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.title_label = QLabel("Chat")
        self.title_label.setObjectName("chatLogTitle")
        layout.addWidget(self.title_label)

        # Search persisted history (chat_messages table). Enter runs the
        # query; an empty submit or Escape drops back to the normal view.
        self.search = QLineEdit()
        self.search.setObjectName("chatLogSearch")
        self.search.setPlaceholderText("Search history\u2026 (Enter)")
        self.search.setAccessibleName("Search chat history")
        self.search.returnPressed.connect(self._on_search)
        layout.addWidget(self.search)

        self.view = QTextBrowser(self)
        self.view.setObjectName("chatLogView")
        self.view.setOpenLinks(False)
        f = QFont()
        f.setFamilies(["Inter", "SF Pro Text", "Segoe UI", "Cantarell", "Sans Serif"])
        f.setPointSize(11)
        self.view.setFont(f)
        layout.addWidget(self.view, stretch=1)

        self._apply_qss()
        self._render()

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)

    def _on_search(self) -> None:
        text = self.search.text().strip()
        if not text:
            self.search_cleared.emit()
        else:
            self.search_requested.emit(text)

    # ── Public API (GUI thread only) ─────────────────────────────────
    def set_assistant_name(self, name: str) -> None:
        self._assistant_name = name or "Jarvis"
        self._render()

    def set_history(self, messages: list[dict]) -> None:
        """Replace committed turns ([{role, content[, created_at]}...]).

        Also clears search mode — main.py calls this when re-seeding.
        """
        self._search_query = None
        self._history = [
            (m.get("role", ""), m.get("content", ""), m.get("created_at"))
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        self.title_label.setText("Chat")
        self._render()

    def show_results(self, query: str, rows: list[dict]) -> None:
        """Render search results; the title reflects the active query."""
        self._search_query = query
        self._live_user = ""
        self._live_reply = ""
        self._history = [
            (m.get("role", ""), m.get("content", ""), m.get("created_at"))
            for m in rows
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        self.title_label.setText(f"Search: {query!r} \u2014 {len(self._history)} result(s)")
        self._render()

    def set_live(self, user_text: str | None = None, reply_text: str | None = None) -> None:
        """Update the in-flight turn (the streaming user/reply pair)."""
        if user_text is not None:
            self._live_user = user_text
        if reply_text is not None:
            self._live_reply = reply_text
        self._render()

    def clear_live(self) -> None:
        self._live_user = ""
        self._live_reply = ""
        self._render()

    # ── Rendering ────────────────────────────────────────────────────
    def _bubble(self, role: str, text: str, when=None) -> str:
        t = self._theme
        name = "You" if role == "user" else self._assistant_name
        if when is not None:
            try:
                # DB timestamps are UTC-naive (session pinned to +00:00);
                # render in the viewer's local timezone.
                from datetime import timezone as _tz
                if when.tzinfo is None:
                    when = when.replace(tzinfo=_tz.utc)
                name = f"{name}  {_html.escape(when.astimezone().strftime('%Y-%m-%d %H:%M'))}"
            except AttributeError:
                pass
        if role == "user":
            color, bg, align = t.text_user, "rgba(0, 255, 160, 18)", "right"
        else:
            color, bg, align = t.text, "rgba(255, 255, 255, 12)", "left"
        return (
            f"<div style='margin:6px 0;'>"
            f"<div style='color:{t.text_dim};font-size:10px;text-align:{align};'>{_html.escape(name)}</div>"
            f"<div style='color:{color};background:{bg};border-radius:10px;padding:6px 10px;"
            f"text-align:{align};'>{md_lite(text)}</div></div>"
        )

    def _render(self) -> None:
        parts = [self._bubble(r, c, w) for (r, c, w) in self._history]
        if self._live_user:
            parts.append(self._bubble("user", self._live_user))
        if self._live_reply:
            parts.append(self._bubble("assistant", self._live_reply))
        self.view.setHtml("".join(parts) or "<div style='color:#888'>No messages yet.</div>")
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Theming ──────────────────────────────────────────────────────
    def set_theme(self, theme) -> None:
        self._theme = theme
        self._apply_qss()
        self._render()

    def _apply_qss(self) -> None:
        t = self._theme
        self.setStyleSheet(f"""
        QWidget#chatLog {{
            background: rgba(10, 14, 20, 245);
            color: {t.text};
        }}
        QLabel#chatLogTitle {{
            color: {t.accent}; font-size: 14px; font-weight: bold;
        }}
        QTextBrowser#chatLogView {{
            background: rgba(6, 10, 16, 220);
            border: 1px solid {t.border};
            border-radius: {t.radius}px;
            padding: 8px;
        }}
        QLineEdit#chatLogSearch {{
            background: {t.surface_soft};
            border: 1px solid {t.border};
            border-radius: {max(6, t.radius - 4)}px;
            padding: 5px 10px;
            color: {t.text_dim};
        }}
        QLineEdit#chatLogSearch:focus {{ border-color: {t.accent}; color: {t.text}; }}
        """)
