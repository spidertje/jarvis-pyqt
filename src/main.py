#!/usr/bin/env python3
"""
Jarvis Desktop — Main entry point.

Phases:
  1. Arc Reactor HUD overlay with state machine
  2. TTS/STT integration (piper + whisper via Wyoming)
  3. Chat with LLM backend + voice mode
  4. Face recognition (OpenCV LBPH + MariaDB)
  5. Profile switching + polish
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import asyncio
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QFrame, QStatusBar,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPalette

from jarvis.hud_overlay import HUDOverlay
from jarvis.state import JarvisState
from jarvis.agent import JarvisAgent, AgentConfig, ChatConfig
from jarvis.stt import WyomingConfig as STTWyomingConfig
from jarvis.tts import WyomingConfig as TTSWyomingConfig


class JarvisApp(QWidget):
    """Main Jarvis application window."""

    def __init__(self):
        super().__init__()

        # HUD overlay (background layer)
        self.hud = HUDOverlay()
        self.hud.resize(800, 600)
        self.hud.move(0, 0)

        # Agent
        self.agent = JarvisAgent(AgentConfig(
            chat=ChatConfig(
                base_url="http://192.168.55.43:3001/v1",
                api_key="freellmapi",
                model="auto",
            ),
            stt=STTWyomingConfig(host="192.168.55.41", port=10300),
            tts=TTSWyomingConfig(host="192.168.55.41", port=10200),
        ))

        # UI controls (foreground layer)
        self._setup_ui()

        # State change tracking
        self.agent.on_state_change(self._on_state_change)

        # Background tasks
        self._running = False
        self._voice_task = None
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(500)

        # Connect all services on startup
        asyncio.get_event_loop().run_until_complete(self._init_services())

    def _setup_ui(self):
        """Set up the foreground UI controls."""
        self.setWindowTitle("Jarvis")
        self.resize(800, 600)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Spacer to push controls to bottom
        layout.addStretch()

        # Bottom control bar
        bar = QFrame()
        bar.setFixedHeight(60)
        bar.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 180);
                border-top: 1px solid rgba(0, 200, 255, 100);
            }
        """)

        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(15, 5, 15, 5)

        # Microphone button
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setFixedSize(40, 40)
        self.mic_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 200, 255, 80);
                border: 2px solid rgba(0, 200, 255, 150);
                border-radius: 20px;
                color: white;
                font-size: 18px;
            }
            QPushButton:hover {
                background: rgba(0, 200, 255, 150);
            }
        """)
        self.mic_btn.clicked.connect(self._toggle_voice)
        bar_layout.addWidget(self.mic_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Status label
        self.status_label = QLabel("⏹ Standby")
        self.status_label.setStyleSheet("color: rgba(0, 200, 255, 200); font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Chat input
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 100);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 5px 10px;
                color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: rgba(0, 200, 255, 150);
            }
        """)
        self.chat_input.returnPressed.connect(self._send_chat)
        bar_layout.addWidget(self.chat_input, alignment=Qt.AlignmentFlag.AlignRight, stretch=1)

        # Send button
        self.send_btn = QPushButton("▶")
        self.send_btn.setFixedSize(40, 35)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 200, 255, 80);
                border: 2px solid rgba(0, 200, 255, 150);
                border-radius: 4px;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(0, 200, 255, 150);
            }
        """)
        self.send_btn.clicked.connect(self._send_chat)
        bar_layout.addWidget(self.send_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(bar)

    async def _init_services(self):
        """Initialize all services on startup."""
        ok = await self.agent.connect_all()
        if ok:
            self.status_label.setText("✓ Connected")
        else:
            self.status_label.setText("⚠ Partial connection")
            print("Warning: Some services not connected")

    def _on_state_change(self, state: JarvisState):
        """Update UI on state change."""
        self.status_label.setText(state.label)
        state_colors = {
            JarvisState.IDLE: "rgba(0, 200, 255, 200)",
            JarvisState.LISTENING: "rgba(0, 255, 100, 220)",
            JarvisState.THINKING: "rgba(200, 200, 0, 220)",
            JarvisState.SPEAKING: "rgba(255, 150, 0, 220)",
        }
        self.status_label.setStyleSheet(
            f"color: {state_colors.get(state, 'white')}; font-size: 14px;"
        )

    def _update_status(self):
        """Periodically update status display."""
        if self.agent.state != JarvisState.IDLE:
            return
        self.status_label.setText("⏹ Standby")

    def _toggle_voice(self):
        """Toggle voice mode (listen → chat → speak loop)."""
        if self._running:
            # Stop voice mode
            self._running = False
            if self._voice_task:
                self._voice_task.cancel()
            self._voice_task = None
            self.mic_btn.setText("🎤")
            self.status_label.setText("⏹ Standby")
        else:
            # Start voice mode
            self._running = True
            self.mic_btn.setText("⏹")
            self.status_label.setText("Listening...")
            self._voice_task = asyncio.ensure_future(self._voice_loop())

    async def _voice_loop(self):
        """Main voice loop."""
        while self._running:
            await self.agent._run_voice()

    async def _send_chat(self):
        """Send a text message and get response (with optional TTS)."""
        text = self.chat_input.text().strip()
        if not text:
            return

        self.chat_input.clear()
        self.status_label.setText("Thinking...")

        # Chat and optionally speak
        reply = await self.agent.chat_text_and_speak(text)
        if reply:
            self.status_label.setText("✓")
        else:
            self.status_label.setText("✗ Error")

    def closeEvent(self, event):
        """Clean shutdown."""
        self._running = False
        if self._voice_task:
            self._voice_task.cancel()
        try:
            asyncio.get_event_loop().run_until_complete(self.agent.close())
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Jarvis")
    app.setOrganizationName("Jarvis")

    # Dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(200, 200, 200))
    palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 180))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(10, 10, 10))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 180, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(200, 200, 200))
    palette.setColor(QPalette.ColorRole.Button, QColor(20, 20, 30))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(200, 200, 200))
    app.setPalette(palette)

    window = JarvisApp()
    window.show()

    print("Jarvis running. Ctrl+C to exit.")
    ret = app.exec()

    return ret


if __name__ == "__main__":
    sys.exit(main())
