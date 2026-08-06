"""
Face Recognition Screen — PyQt6 widget showing live camera feed
with face detection overlay, shown until a face is recognized.

Displays:
- Live webcam feed as background
- Green bounding boxes around detected faces (with pulsing glow)
- A vertical scanning line animation (indicates active detection)
- Status text at the bottom ("Looking for a registered face...")
- Name input overlay for face enrollment (unknown face → enter name → collect samples)
- Welcome message when a face is recognized
- Fade-to-black transition on successful recognition or timeout
"""

import math

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FaceRecScreen(QWidget):
    """Full-window widget showing live camera feed until face is recognized."""

    name_submitted = pyqtSignal(str)
    skip_requested = pyqtSignal()  # emitted when user clicks Skip

    def __init__(self, parent=None):
        super().__init__(parent)

        # Use Window (not just FramelessWindowHint) to ensure proper z-ordering
        # on macOS — FramelessWindowHint alone can place the window behind others.
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setStyleSheet("background-color: #000000;")
        self.resize(800, 600)

        # Camera feed state
        self._frame: QImage | None = None
        self._faces: list[tuple] = []

        # Recognition status
        self._face_name: str = ""
        self._face_confidence: float = 0.0
        self._recognized: bool = False

        # Animation state
        self._pulse = 0.0
        self._scan_y = 0.0  # Vertical scan line position (0–1)
        self._fade_alpha: float = 0.0  # 0 = visible, 1 = fully faded out

        # Status message
        self._status_text: str = "Initializing camera..."

        # Animation timer (60fps)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // 60)

        # ── Face enrollment UI (initially hidden) ──────────────────────
        self._reg_overlay = QFrame(self)
        self._reg_overlay.setGeometry(180, 140, 440, 320)
        self._reg_overlay.setStyleSheet(
            " background: rgba(0, 0, 0, 230);"
            " border: 2px solid rgba(0, 200, 255, 150);"
            " border-radius: 14px;"
        )

        reg_layout = QVBoxLayout(self._reg_overlay)
        reg_layout.setContentsMargins(24, 24, 24, 24)
        reg_layout.setSpacing(14)

        reg_title = QLabel("👤 New Face Detected")
        reg_title.setStyleSheet(
            "color: white; font-size: 18px; font-weight: bold;"
        )
        reg_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reg_layout.addWidget(reg_title)

        reg_prompt = QLabel("Face detected but not recognized. Please enter your name:")
        reg_prompt.setStyleSheet(
            "color: rgba(200, 200, 200, 220); font-size: 12px;"
        )
        reg_prompt.setWordWrap(True)
        reg_layout.addWidget(reg_prompt)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Enter your name...")
        self._name_input.setFixedHeight(36)
        self._name_input.setStyleSheet(
            " QLineEdit {"
            "   background: rgba(0, 0, 0, 100);"
            "   border: 1px solid rgba(0, 200, 255, 80);"
            "   border-radius: 6px;"
            "   padding: 6px 12px;"
            "   color: white;"
            "   font-size: 14px;"
            " }"
            " QLineEdit:focus {"
            "   border-color: rgba(0, 200, 255, 150);"
            " }"
        )
        self._name_input.returnPressed.connect(self._on_name_submitted)
        reg_layout.addWidget(self._name_input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        skip_btn = QPushButton("Skip")
        skip_btn.setFixedHeight(36)
        skip_btn.setStyleSheet(
            " QPushButton {"
            "   background: rgba(80, 80, 80, 60);"
            "   border: 1px solid rgba(150, 150, 150, 100);"
            "   border-radius: 6px;"
            "   color: #ccc;"
            "   font-size: 13px;"
            " }"
            " QPushButton:hover {"
            "   background: rgba(120, 120, 120, 80);"
            " }"
        )
        skip_btn.clicked.connect(self._on_name_skipped)
        btn_row.addWidget(skip_btn)

        reg_btn = QPushButton("Register")
        reg_btn.setFixedHeight(36)
        reg_btn.setStyleSheet(
            " QPushButton {"
            "   background: rgba(0, 200, 255, 80);"
            "   border: 2px solid rgba(0, 200, 255, 150);"
            "   border-radius: 6px;"
            "   color: white;"
            "   font-size: 13px;"
            "   font-weight: bold;"
            " }"
            " QPushButton:hover {"
            "   background: rgba(0, 200, 255, 150);"
            " }"
        )
        reg_btn.clicked.connect(self._on_name_submitted)
        btn_row.addWidget(reg_btn)

        reg_layout.addLayout(btn_row)

        # Progress label (hidden until collection starts)
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            "color: rgba(0, 200, 255, 200); font-size: 11px;"
        )
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reg_layout.addWidget(self._progress_label)

        self._reg_overlay.hide()

        # Enrollment progress state
        self._enrolling = False
        self._enrollment_name: str = ""
        self._enrollment_progress: tuple[int, int] = (0, 0)

    def _on_name_submitted(self):
        """Handle name input submission."""
        name = self._name_input.text().strip()
        if name:
            self._enrollment_name = name
            self._name_input.setEnabled(False)
            self._progress_label.setText("Starting face enrollment...")
            self.name_submitted.emit(name)

    def _on_name_skipped(self):
        """Handle skip button — fall back to HUD without registration."""
        self._reg_overlay.hide()
        self.skip_requested.emit()
        self.start_fade()

    def keyPressEvent(self, event):
        """Close the registration overlay on Escape."""
        if event.key() == Qt.Key.Key_Escape and self._reg_overlay.isVisible():
            self._on_name_skipped()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Prevent accidental window close — only allow hide()."""
        event.ignore()

    def _tick(self):
        """Advance animation state."""
        self._pulse = (self._pulse + 0.04) % (2 * math.pi)
        self._scan_y = (self._scan_y + 0.008) % 1.0
        if self._recognized:
            self._fade_alpha = min(1.0, self._fade_alpha + 0.025)
        self.update()

    # ── Public API ──────────────────────────────────────────────────────

    def set_frame(self, frame: QImage, faces: list[tuple]):
        """Receive a new camera frame with detected face rectangles."""
        self._frame = frame
        self._faces = faces
        if not self._recognized and not self._enrolling:
            if self._faces:
                count = len(self._faces)
                label = "face" if count == 1 else "faces"
                self._status_text = f"Looking for a registered face... ({count} {label} detected)"
            else:
                self._status_text = "Looking for a registered face..."
        self.update()

    def set_face_detected(self, name: str, confidence: float):
        """Called when a face is recognized — triggers fade-out transition."""
        self._face_name = name
        self._face_confidence = confidence
        self._recognized = True
        self._status_text = f"Welcome, {name}!"
        self._fade_alpha = 0.0
        self.update()

    def set_status(self, text: str):
        """Update the status message shown below the camera feed."""
        self._status_text = text
        self.update()

    def start_fade(self):
        """Start the fade-out animation (e.g. on timeout fallback)."""
        self._reg_overlay.hide()
        self._recognized = True
        self._fade_alpha = 0.0

    def show_name_input(self):
        """Show the name input overlay for face enrollment."""
        self._enrolling = True
        self._name_input.setText("")
        self._name_input.setEnabled(True)
        self._progress_label.setText("")
        self._reg_overlay.show()
        self._name_input.setFocus()
        self.update()

    def show_sample_progress(self, collected: int, target: int):
        """Update the sample collection progress display."""
        self._enrollment_progress = (collected, target)
        pct = int(collected / target * 100) if target > 0 else 0
        self._progress_label.setText(f"Collecting samples... {collected}/{target} ({pct}%)")
        self._status_text = f"Collecting face samples for {self._enrollment_name}... ({collected}/{target})"
        self.update()

    def clear_enrollment(self):
        """Reset enrollment state."""
        self._enrolling = False
        self._enrollment_name = ""
        self._enrollment_progress = (0, 0)
        self._reg_overlay.hide()

    @property
    def is_fade_complete(self) -> bool:
        """True when the fade-out animation is finished."""
        return self._fade_alpha >= 0.85

    @property
    def recognized(self) -> bool:
        """True if a face has been recognized (and fade-out initiated)."""
        return self._recognized

    # ── Painting ─────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # 1. Camera frame (background)
        if self._frame is not None:
            p.drawImage(self.rect(), self._frame)
        else:
            p.setBrush(QColor(0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(self.rect())

        # 2. Face detection rectangles
        for (x, y, fw, fh) in self._faces:
            self._draw_face_rect(p, x, y, fw, fh)

        # 3. Scanning line animation
        self._draw_scan_line(p, w, h)

        # 4. Bottom gradient overlay (for text readability)
        gradient = QLinearGradient(0, h - 220, 0, h)
        gradient.setColorAt(0, QColor(0, 0, 0, 0))
        gradient.setColorAt(0.3, QColor(0, 0, 0, 120))
        gradient.setColorAt(1, QColor(0, 0, 0, 220))
        p.fillRect(0, h - 220, w, 220, QBrush(gradient))

        # 5. Status text
        self._draw_status(p, w, h)

        # 6. Face name with glow (if recognized)
        if self._recognized and self._face_name:
            self._draw_face_name(p, w, h)

        # 7. Fade-out overlay
        if self._fade_alpha > 0:
            fade_alpha = int(self._fade_alpha * 255)
            p.setBrush(QColor(0, 0, 0, fade_alpha))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(self.rect())

        p.end()

    def _draw_face_rect(self, p: QPainter, x: float, y: float, fw: float, fh: float):
        """Draw a face detection rectangle with pulsing glow and corner markers."""
        # Scale from frame-space to widget-space
        if self._frame is not None and self._frame.width() > 0:
            sx = self.width() / self._frame.width()
            sy = self.height() / self._frame.height()
        else:
            sx = sy = 1.0

        x = x * sx
        y = y * sy
        fw = fw * sx
        fh = fh * sy

        x, y, fw, fh = int(x), int(y), int(fw), int(fh)

        # Pulsing glow
        glow_a = int(80 + 40 * math.sin(self._pulse * 3))
        glow_pen = QPen(QColor(0, 255, 100, glow_a), 3)
        p.setPen(glow_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(x, y, fw, fh, 8, 8)

        # Solid outline
        p.setPen(QPen(QColor(0, 255, 100, 220), 2))
        p.drawRoundedRect(x, y, fw, fh, 6, 6)

        # Corner markers
        cl = 14
        p.setPen(QPen(QColor(0, 255, 100, 200), 2))
        # Top-left
        p.drawLine(x, y, x + cl, y)
        p.drawLine(x, y, x, y + cl)
        # Top-right
        p.drawLine(x + fw, y, x + fw - cl, y)
        p.drawLine(x + fw, y, x + fw, y + cl)
        # Bottom-left
        p.drawLine(x, y + fh, x + cl, y + fh)
        p.drawLine(x, y + fh, x, y + fh - cl)
        # Bottom-right
        p.drawLine(x + fw, y + fh, x + fw - cl, y + fh)
        p.drawLine(x + fw, y + fh, x + fw, y + fh - cl)

    def _draw_scan_line(self, p: QPainter, w: int, h: int):
        """Draw a vertical scanning line that moves down the screen."""
        scan_y = int(h * self._scan_y)
        scan_a = int(120 + 60 * math.sin(self._pulse * 2))
        pen = QPen(QColor(0, 200, 255, scan_a), 2)
        p.setPen(pen)
        # Main line
        p.drawLine(0, scan_y, w, scan_y)
        # Glow above and below
        for offset in [2, 4, 6]:
            off_pen = QPen(QColor(0, 200, 255, int(scan_a * (6 - offset) / 24)), 1)
            p.setPen(off_pen)
            p.drawLine(0, scan_y - offset, w, scan_y - offset)
            p.drawLine(0, scan_y + offset, w, scan_y + offset)

    def _draw_status(self, p: QPainter, w: int, h: int):
        """Draw status text at the bottom of the screen."""
        # Title
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QColor(255, 255, 255, 230))
        title = "Face Recognition"
        tm = p.fontMetrics()
        tw = tm.horizontalAdvance(title)
        p.drawText(int((w - tw) / 2), h - 72, title)

        # Status text
        status_font = QFont()
        status_font.setPointSize(13)
        p.setFont(status_font)
        p.setPen(QColor(200, 200, 200, 220))
        sm = p.fontMetrics()
        sw = sm.horizontalAdvance(self._status_text)
        p.drawText(int((w - sw) / 2), h - 40, self._status_text)

        # Subtext
        sub = "Standing by for recognized face..." if not self._recognized else "Transitioning to Jarvis..."
        sub_font = QFont()
        sub_font.setPointSize(11)
        p.setFont(sub_font)
        p.setPen(QColor(0, 200, 255, 160))
        sm2 = p.fontMetrics()
        sw2 = sm2.horizontalAdvance(sub)
        p.drawText(int((w - sw2) / 2), h - 16, sub)

    def _draw_face_name(self, p: QPainter, w: int, h: int):
        """Draw the recognized face name with a pulsing glow effect."""
        text = f"👤 {self._face_name}"
        if self._face_confidence > 0:
            text += f"  ({self._face_confidence:.0f}% match)"

        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        p.setFont(font)
        metrics = p.fontMetrics()
        tw = metrics.horizontalAdvance(text)

        # Glow layers
        glow_a = int(60 + 40 * math.sin(self._pulse * 3))
        for offset in range(5, 0, -1):
            glow_col = QColor(0, 200, 255, int(glow_a * (offset / 5)))
            p.setPen(QPen(glow_col, offset * 2))
            p.drawText(int((w - tw) / 2), int(h / 2 + offset), text)

        # Main text
        p.setPen(QColor(255, 255, 255, 230))
        p.drawText(int((w - tw) / 2), int(h / 2), text)
