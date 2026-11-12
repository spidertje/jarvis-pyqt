"""
Jarvis HUD Panels — live transcript, streaming reply, tool-activity feed,


These widgets make the HUD *informational*: instead of only showing a state
word, you see what you said, what Jarvis is saying (as it streams in), and
any tool-call activity the LLM surfaces in-band.

Threading note: the agent runs on a background asyncio loop, so main.py
bridges agent callbacks into these widgets via PyQt signals (the same
pattern used for ``status_label``). The widgets themselves are plain Qt —
never call them from the loop thread directly.
"""

from collections import OrderedDict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# How much of the reply we keep visible (tail of the stream).
REPLY_VISIBLE_CHARS = 240
# How many activity-feed lines we keep (latest wins).
ACTIVITY_LINES = 3


class HUDPanels(QWidget):
    """Transcript / reply / activity panel, shown above the bottom bar.

    Hidden by default. ``show_turn()`` reveals it for the duration of a
    conversation turn; ``clear_turn()`` resets all lines for the next one.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(96)
        self.setStyleSheet(
            """
            QWidget {
                background: rgba(0, 0, 0, 190);
                border-top: 1px solid rgba(0, 200, 255, 100);
            }
            QLabel {
                color: white;
                background: transparent;
            }
            """
        )

        # Internal accumulators (source of truth; labels show the visible tail).
        self._user_text = ""
        self._reply_text = ""
        self._activity: "OrderedDict[str, str]" = OrderedDict()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(3)

        self.user_label = QLabel("")
        self.user_label.setStyleSheet("color: rgba(0, 255, 160, 220); font-size: 12px;")
        self.user_label.setWordWrap(True)

        self.reply_label = QLabel("")
        self.reply_label.setStyleSheet("color: rgba(255, 255, 255, 230); font-size: 13px;")
        self.reply_label.setWordWrap(True)

        self.activity_label = QLabel("")
        self.activity_label.setStyleSheet("color: rgba(255, 190, 60, 200); font-size: 11px;")

        layout.addWidget(self.user_label)
        layout.addWidget(self.reply_label)
        layout.addWidget(self.activity_label)

        self.hide()

    # ── Turn lifecycle ───────────────────────────────────────────────
    def show_turn(self) -> None:
        """Reveal the panel for the current turn."""
        self.show()

    def hide_turn(self) -> None:
        """Hide the panel (end of turn / idle)."""
        self.hide()

    def clear_turn(self) -> None:
        """Reset all panel content for the next turn."""
        self._user_text = ""
        self._reply_text = ""
        self._activity.clear()
        self.user_label.setText("")
        self.reply_label.setText("")
        self.activity_label.setText("")

    # ── Content ──────────────────────────────────────────────────────
    def set_user_text(self, text: str) -> None:
        """Set the user's utterance (top line)."""
        self._user_text = text.strip()
        self.user_label.setText(self._user_text)
        self.show()

    @property
    def user_text(self) -> str:
        return self._user_text

    def append_reply(self, chunk: str) -> None:
        """Append streamed reply text; the label shows the visible tail."""
        self._reply_text += chunk
        self._render_reply()
        self.show()

    @property
    def reply_text(self) -> str:
        return self._reply_text

    def _render_reply(self) -> None:
        if len(self._reply_text) <= REPLY_VISIBLE_CHARS:
            self.reply_label.setText(self._reply_text)
        else:
            tail = self._reply_text[-REPLY_VISIBLE_CHARS:]
            # Cut back to a word boundary so we don't show a half-word.
            cut = tail.find(" ")
            if 0 < cut < len(tail):
                tail = tail[cut + 1:]
            self.reply_label.setText("… " + tail)

    def set_activity(self, key: str, line: str) -> None:
        """Upsert a tool-activity line for ``key`` (e.g. the tool-call id).

        Streaming tool calls arrive in fragments (name, then argument
        chunks); each fragment re-sends the accumulated snapshot, so the
        same key *replaces* its line instead of appending duplicates.
        """
        self._activity[key] = line
        self._activity.move_to_end(key)
        while len(self._activity) > ACTIVITY_LINES:
            self._activity.popitem(last=False)
        self.activity_label.setText("\n".join(list(self._activity.values())[-2:]))
        self.show()

    @property
    def activity_lines(self) -> list[str]:
        return list(self._activity.values())
