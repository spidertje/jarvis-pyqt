"""
Jarvis HUD Panels — live scrolling chat transcript.

A transcript you can read *back through*: one user bubble + one assistant
bubble per turn, streaming replies update the last bubble in place, tool
activity renders as dim monospace lines, and a markdown-lite pass
(bold/italic/inline code/fenced blocks) makes LLM output readable.
History persists across turns (max ``MAX_TURNS``); the panel hides only
when empty.

Threading note: the agent runs on a background asyncio loop, so main.py
bridges agent callbacks into these widgets via PyQt signals (the same
pattern used for ``status_label``). The widgets themselves are plain Qt —
never call them from the loop thread directly.
"""

from __future__ import annotations

import html as _html
import re
from collections import OrderedDict

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QTextBrowser, QVBoxLayout

# Per-bubble tail cap for in-flight streaming replies.
REPLY_VISIBLE_CHARS = 800
# How many activity-feed lines we keep per turn (latest wins).
ACTIVITY_LINES = 3
# Turns kept in the scrolling history.
MAX_TURNS = 8


def md_lite(text: str) -> str:
    """Escape HTML, then apply a tiny markdown subset (bold/italic/code)."""
    t = _html.escape(text)
    t = re.sub(r"```(?:\w*\n)?(.*?)```", r"<pre>\1</pre>", t, flags=re.S)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", t)
    return t.replace("\n", "<br>")


class HUDPanels(QFrame):
    """Scrolling conversation transcript shown above the bottom bar."""

    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        from jarvis import theme as _theme

        self._theme = theme or _theme.DEFAULT
        self.setObjectName("panels")
        self.setFixedHeight(150)
        self.setStyleSheet(_theme.panels_qss(self._theme))

        # Turn state (source of truth; the view renders from these).
        self._user_text = ""
        self._reply_text = ""
        self._activity: "OrderedDict[str, str]" = OrderedDict()
        self._turns: list[tuple[str, str, list[str]]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(0)

        self.view = QTextBrowser(self)
        self.view.setOpenLinks(False)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
        f = QFont()
        f.setFamilies(["Inter", "SF Pro Text", "Segoe UI", "Cantarell", "Sans Serif"])
        f.setPointSize(10)
        self.view.setFont(f)
        layout.addWidget(self.view)

        self.hide()

    # ── Turn lifecycle ───────────────────────────────────────────────
    def begin_turn(self) -> None:
        """Commit any in-flight turn and start a fresh (empty) one."""
        self._commit_current()
        self._render()

    def show_turn(self) -> None:
        self.show()

    def hide_turn(self) -> None:
        """End the current turn. Transcript stays visible while it has history."""
        self._commit_current()
        self._render()
        if not self._turns:
            self.hide()

    def clear_turn(self) -> None:
        """Full reset including history (used on hard state resets)."""
        self._user_text = ""
        self._reply_text = ""
        self._activity.clear()
        self._turns.clear()
        self._render()
        self.hide()

    def _commit_current(self) -> None:
        if self._user_text or self._reply_text or self._activity:
            self._turns.append((self._user_text, self._reply_text, self._activity_snapshot()))
            if len(self._turns) > MAX_TURNS:
                self._turns.pop(0)
        self._user_text = ""
        self._reply_text = ""
        self._activity.clear()

    # ── Content ──────────────────────────────────────────────────────
    def set_user_text(self, text: str) -> None:
        """Set the user's utterance for the current turn."""
        text = text.strip()
        if (self._reply_text or self._activity) and (self._user_text or text):
            # New utterance while a reply is in flight — roll the turn over.
            self._commit_current()
        self._user_text = text
        self._render()
        self.show()

    @property
    def user_text(self) -> str:
        return self._user_text

    def append_reply(self, chunk: str) -> None:
        """Append streamed reply text to the current turn's assistant bubble."""
        if not chunk:
            return
        self._reply_text += chunk
        if len(self._reply_text) > REPLY_VISIBLE_CHARS:
            self._reply_text = self._reply_text[-REPLY_VISIBLE_CHARS:]
        self._render()
        self.show()

    @property
    def reply_text(self) -> str:
        return self._reply_text

    def set_activity(self, key: str, line: str) -> None:
        """Upsert a tool-activity line for ``key`` (e.g. the tool-call id).

        Streaming tool calls arrive in fragments (name, then argument
        chunks); each fragment re-sends the accumulated snapshot, so the
        same key *replaces* its line instead of appending duplicates.
        """
        self._activity[key] = line
        self._activity.move_to_end(key)
        self._render()
        self.show()

    @property
    def activity_lines(self) -> list[str]:
        return self._activity_snapshot()

    def _activity_snapshot(self) -> list[str]:
        return list(self._activity.values())[-ACTIVITY_LINES:]

    # ── Rendering ────────────────────────────────────────────────────
    def _bubble(self, kind: str, text: str) -> str:
        t = self._theme
        if kind == "user":
            color, bg = t.text_user, "rgba(0, 255, 160, 16)"
        elif kind == "reply":
            color, bg = t.text, "rgba(255, 255, 255, 10)"
        else:
            color, bg = t.text_activity, "transparent"
        if kind == "activity":
            body = f"<pre style='margin:0'>{_html.escape(text)}</pre>"
        else:
            body = md_lite(text)
        return (
            f"<div style='color:{color};background:{bg};"
            f"border-radius:8px;padding:4px 8px;margin:2px 0;'>{body}</div>"
        )

    def _turn_html(self, user: str, reply: str, activity: list[str]) -> str:
        out = []
        if user:
            out.append(self._bubble("user", user))
        if reply:
            out.append(self._bubble("reply", reply))
        if activity:
            out.append(self._bubble("activity", "\n".join(activity)))
        return "".join(out)

    def _render(self) -> None:
        parts = [self._turn_html(u, r, a) for (u, r, a) in self._turns]
        parts.append(self._turn_html(self._user_text, self._reply_text, self._activity_snapshot()))
        self.view.setHtml("".join(parts) or "&nbsp;")
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # Exposed for tests / theming refresh.
    @property
    def html(self) -> str:
        return self.view.toHtml()

    def set_theme(self, theme) -> None:
        from jarvis import theme as _theme

        self._theme = theme
        self.setStyleSheet(_theme.panels_qss(theme))
        self._render()
