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
    QLineEdit, QPushButton, QLabel, QFrame,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPalette

from jarvis.hud_overlay import HUDOverlay
from jarvis.state import JarvisState
from jarvis.agent import JarvisAgent, AgentConfig, ChatConfig
from jarvis.stt import WyomingConfig as STTWyomingConfig
from jarvis.tts import WyomingConfig as TTSWyomingConfig
from jarvis.face import FaceConfig
from jarvis.profile import ProfileManager, Profile
from jarvis.settings import SettingsDialog

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMenu, QAction
import cv2
import logging
import asyncio

logger = logging.getLogger(__name__)


class EventLoopThread(QThread):
    """Run an asyncio event loop in a background thread."""

    def __init__(self):
        super().__init__()
        self._loop = None
        self._stop = False

    def run(self):
        """Start asyncio event loop in this thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        logger.info("Async event loop started in background thread")
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def stop(self):
        """Stop the event loop."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.wait()

    @property
    def loop(self):
        return self._loop


class FaceRecThread(QThread):
    """Face detection thread — runs OpenCV LBPH recognition loop."""

    face_detected = pyqtSignal(str, float)  # name, confidence

    def __init__(self, config: FaceConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.recognizer = None
        self.stop_flag = False
        self.camera = None

    def run(self):
        """Run face detection loop."""
        from jarvis.face import FaceRecognizer

        # Load models from DB
        self.recognizer = FaceRecognizer(self.config)
        count = self.recognizer.load_models()
        if count == 0:
            logger.info("No face models loaded — face recognition disabled")

        # Open camera
        self.camera = cv2.VideoCapture(self.config.camera_index)
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera {self.config.camera_index}")
            return

        logger.info(f"Face detection started on camera {self.config.camera_index}")

        while not self.stop_flag:
            ret, frame = self.camera.read()
            if not ret:
                continue

            name = self.recognizer.recognize(frame)
            if name:
                # Get confidence for overlay
                faces = self.recognizer.detect_faces(frame)
                conf = 0.0
                if faces:
                    x, y, w, h = faces[0]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi, self.config.resize_dims)
                    if name in self.recognizer._models:
                        model = self.recognizer._models[name]
                        try:
                            _, conf = model.predict(roi)
                        except Exception:
                            pass

                self.face_detected.emit(name, conf)

            self.msleep(100)  # 10fps detection loop

        self.camera.release()
        self.recognizer.close()

    def stop(self):
        """Stop the face detection thread."""
        self.stop_flag = True
        self.wait()


class JarvisApp(QWidget):
    """Main Jarvis application window."""

    def __init__(self):
        super().__init__()

        # HUD overlay (background layer)
        self.hud = HUDOverlay()
        self.hud.resize(800, 600)
        self.hud.move(0, 0)

        # Agent
        # Build agent config (env vars override defaults)
        agent_config = AgentConfig()
        self.agent = JarvisAgent(agent_config)

        # Start async event loop in background thread
        self.event_loop_thread = EventLoopThread()
        self.event_loop_thread.start()

        # Face recognition thread
        face_config = FaceConfig(camera_index=0)
        self.face_thread = FaceRecThread(face_config)
        self.face_thread.face_detected.connect(self._on_face_detected)

        # UI controls (foreground layer)
        self._setup_ui()

        # State change tracking
        self.agent.on_state_change(self._on_state_change)

        # Background tasks
        self._running = False
        self._voice_task = None
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

        self._last_status_state = None

        # Start face detection thread
        self.face_thread.start()

        # Connect all services on startup (async via background loop)
        if self.event_loop_thread.loop:
            asyncio.ensure_future(self._init_services(), loop=self.event_loop_thread.loop)
        else:
            print("Warning: async event loop not available")

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

        # Settings button
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(35, 35)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background: rgba(80, 80, 80, 60);
                border: 1px solid rgba(150, 150, 150, 100);
                border-radius: 18px;
                color: #ccc;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(120, 120, 120, 80);
                border-color: rgba(200, 200, 200, 150);
            }
        """)
        self.settings_btn.clicked.connect(self._open_settings)
        bar_layout.addWidget(self.settings_btn, alignment=Qt.AlignmentFlag.AlignRight)

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
        """Update status only when state changes."""
        if self._last_status_state == self.agent.state:
            return
        self._last_status_state = self.agent.state
        if self.agent.state == JarvisState.IDLE:
            self.status_label.setText("⏹ Standby")

    def _on_face_detected(self, name: str, confidence: float):
        """Handle face detection — switch to that profile, update HUD."""
        # Switch profile if we have one for this person
        profile = self.agent.profiles.get(name)
        if profile:
            if self.agent.switch_profile(name):
                # Update HUD with profile name + accent color
                self.hud.set_profile(profile.name, profile.accent_hue)
                logger.info(f"Profile switched to: {profile.name}")
            else:
                self.hud.set_face_detected(name, confidence)
                logger.warning(f"Could not switch to profile: {name}")
        else:
            # No profile for this person — just show face overlay
            self.hud.set_face_detected(name, confidence)
            logger.info(f"Face detected but no profile: {name}")

    def _open_settings(self):
        """Open the settings/preferences dialog."""
        dialog = SettingsDialog(agent_config=self.agent.config, parent=self)
        dialog.exec()

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
            if self.event_loop_thread.loop:
                self._voice_task = asyncio.ensure_future(self._voice_loop(), loop=self.event_loop_thread.loop)
            else:
                print("Error: async event loop not available")

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

        # Stop face detection thread
        if hasattr(self, 'face_thread'):
            self.face_thread.stop()

        # Stop event loop thread
        if hasattr(self, 'event_loop_thread'):
            self.event_loop_thread.stop()

        # Close all services
        try:
            if self.event_loop_thread.loop:
                asyncio.get_event_loop().run_until_complete(self.agent.close())
            else:
                # Fallback: run sync close
                import asyncio as _ai
                _ai.run(self.agent.close())
        except Exception:
            pass

        super().closeEvent(event)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

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
