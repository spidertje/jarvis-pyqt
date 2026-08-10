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

import asyncio
import logging
import os
import sys
import threading
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.config import AppConfig
from jarvis.face import FaceConfig
from jarvis.face_screen import FaceRecScreen
from jarvis.hud_overlay import HUDOverlay
from jarvis.settings import SettingsDialog
from jarvis.state import JarvisState

logger = logging.getLogger(__name__)


class EventLoopThread(QThread):
    """Run an asyncio event loop in a background thread."""

    def __init__(self):
        super().__init__()
        self._loop = None
        self._stop = False
        self._loop_ready = threading.Event()

    def run(self):
        """Start asyncio event loop in this thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()  # Signal that the loop is ready
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
    frame_captured = pyqtSignal(QImage, list)  # frame (RGB), face rectangles
    unknown_face_detected = pyqtSignal()  # face seen but not recognized
    face_registered = pyqtSignal(str)  # name
    samples_progress = pyqtSignal(int, int)  # collected, target
    enrollment_error = pyqtSignal(str)  # error message

    def __init__(self, config: FaceConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.recognizer = None
        self.stop_flag = False
        self.camera = None

        # Face enrollment state
        self._collecting = False
        self._collecting_name = ""
        self._collected_samples = 0
        self._last_unknown_face_time: float = 0.0
        self._last_enrollment_error_time: float = 0.0

    def _numpy_to_qimage(self, frame: np.ndarray) -> QImage:
        """Convert a BGR numpy frame to an RGB QImage."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg.copy()

    def start_collection(self, name: str):
        """Begin collecting face samples for a new person."""
        self._collecting = True
        self._collecting_name = name
        self._collected_samples = 0

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

            # Detect faces (single pass — used for both display and recognition)
            faces = self.recognizer.detect_faces(frame)

            # Emit frame for display
            qimg = self._numpy_to_qimage(frame)
            self.frame_captured.emit(qimg, faces)

            if self._collecting:
                # ── Face enrollment mode ──────────────────────────────
                if faces:
                    added = self.recognizer.add_face(self._collecting_name, frame)
                    if not added:
                        now = time.time()
                        if now - self._last_enrollment_error_time >= 5.0:
                            self._last_enrollment_error_time = now
                            db_err = getattr(self.recognizer, "_last_db_error", None)
                            detail = f" ({db_err})" if db_err else ""
                            self.enrollment_error.emit(
                                f"Could not store face sample for "
                                f"'{self._collecting_name}'{detail}. "
                                f"Open Settings → Database to verify DB host, "
                                f"user, and password."
                            )
                    else:
                        self._last_enrollment_error_time = 0.0
                        self._collected_samples += 1
                        self.samples_progress.emit(
                            self._collected_samples, self.config.min_faces_to_add
                        )
                if self._collecting and self._collected_samples >= self.config.min_faces_to_add:
                    # Train and store the new model
                    if self.recognizer.train_new_face(self._collecting_name):
                        self._collecting = False
                        self._collected_samples = 0
                        self.face_registered.emit(self._collecting_name)
                    else:
                        logger.error(f"Failed to train model for {self._collecting_name}")
                        self._collecting = False
                        self._collected_samples = 0
            else:
                # ── Normal recognition mode ──────────────────────────
                if faces and self.recognizer._models:
                    result = self.recognizer.recognize(frame)
                    if result:
                        name, raw_conf = result
                        # LBPH distance (lower = better); convert to percentage
                        confidence_pct = max(0.0, 100.0 - raw_conf)
                        self.face_detected.emit(name, confidence_pct)
                    else:
                        # Face detected but not recognized — prompt for enrollment
                        now = time.time()
                        if now - self._last_unknown_face_time >= self.config.debounce_seconds:
                            self._last_unknown_face_time = now
                            self.unknown_face_detected.emit()
                elif faces and not self.recognizer._models:
                    # No models loaded at all — prompt for enrollment
                    now = time.time()
                    if now - self._last_unknown_face_time >= self.config.debounce_seconds:
                        self._last_unknown_face_time = now
                        self.unknown_face_detected.emit()

            self.msleep(100)  # 10fps detection loop

        self.camera.release()
        self.recognizer.close()

    def stop(self):
        """Stop the face detection thread."""
        self.stop_flag = True
        self.wait()
        # Clean up resources (safe even if camera/open failed)
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
        if self.recognizer is not None:
            try:
                self.recognizer.close()
            except Exception:
                pass


class JarvisApp(QWidget):
    """Main Jarvis application window."""

    _status_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # HUD overlay (background layer — visible underneath face screen)
        self.hud = HUDOverlay()
        self.hud.resize(800, 600)
        self.hud.move(0, 0)
        self.hud.show()

        # Face recognition screen (shown on startup, fades out when face is recognized)
        self.face_screen = FaceRecScreen()
        self.face_screen.resize(800, 600)
        self.face_screen.move(0, 0)

        # Agent
        # Load persistent config (env vars + .env + config.json)
        self.app_config = AppConfig.load()

        # Build agent config from AppConfig
        agent_config = AgentConfig()
        # LLM
        agent_config.llm_base_url = self.app_config.llm_base_url
        agent_config.llm_api_key = self.app_config.llm_api_key
        agent_config.llm_model = self.app_config.llm_model
        agent_config.chat.model = self.app_config.llm_model
        # STT
        agent_config.stt.host = self.app_config.stt_host
        agent_config.stt.port = self.app_config.stt_port
        # TTS
        agent_config.tts.host = self.app_config.tts_host
        agent_config.tts.port = self.app_config.tts_port
        agent_config.tts_voice = self.app_config.tts_voice
        # Audio
        agent_config.audio.device = self.app_config.audio_output_device
        # DB
        agent_config.profile_db_host = self.app_config.db_host
        agent_config.profile_db_port = self.app_config.db_port
        agent_config.profile_db_name = self.app_config.db_name
        agent_config.profile_db_user = self.app_config.db_user
        agent_config.profile_db_password = self.app_config.db_password
        # Appearance
        agent_config.palette_index = self.app_config.palette_index
        agent_config.contrast_boost = self.app_config.contrast_boost
        # Assistant name
        agent_config.assistant_name = self.app_config.assistant_name
        # STT
        agent_config.silence_timeout = self.app_config.silence_timeout
        agent_config.silence_threshold = self.app_config.stt_sensitivity

        self.agent = JarvisAgent(agent_config, hud=self.hud)
        # Apply palette hue from agent config if available
        if hasattr(agent_config, "palette_index") and agent_config.palette_index is not None:
            # Map index to hue (same as in settings)
            palette_hues = [182, 30, 120, 250, 200, 0, 80, 220, 40, 60]
            idx = agent_config.palette_index
            if 0 <= idx < len(palette_hues):
                self.hud.set_palette_hue(palette_hues[idx])

        # Apply contrast factor to HUD
        self.hud.set_contrast_factor(agent_config.contrast_boost / 100.0)

        # Start async event loop in background thread
        self.event_loop_thread = EventLoopThread()
        self.event_loop_thread.start()

        # Face recognition thread
        face_config = FaceConfig(
            camera_index=self.app_config.camera_index,
            db_host=self.app_config.db_host,
            db_port=self.app_config.db_port,
            db_user=self.app_config.db_user,
            db_password=self.app_config.db_password,
            db_name=self.app_config.db_name,
            confidence_threshold=self.app_config.face_confidence_threshold * 100,
        )
        self.face_thread = FaceRecThread(face_config)
        self.face_thread.face_detected.connect(self._on_face_detected)
        self.face_thread.frame_captured.connect(self.face_screen.set_frame)
        self.face_thread.unknown_face_detected.connect(self._on_unknown_face)
        self.face_thread.face_registered.connect(self._on_face_registered)
        self.face_thread.samples_progress.connect(self._on_samples_progress)
        self.face_thread.enrollment_error.connect(self._on_enrollment_error)

        # Face recognition screen management
        self._face_rec_active = True
        self._enrollment_prompted = False  # Track if name dialog already shown
        self._transition_timer = None
        self._enrollment_delay_timer: QTimer | None = None
        self.face_screen.show()

        # Timeout: if no face recognized within 10 seconds, fall back to HUD
        self._face_timeout = QTimer(self)
        self._face_timeout.setSingleShot(True)
        self._face_timeout.timeout.connect(self._on_face_timeout)
        self._face_timeout.start(10000)

        # UI controls (foreground layer)
        self._setup_ui()

        # Signal: async status updates → status label
        self._status_updated.connect(self.status_label.setText)
        # Signal: user entered a name on the face screen
        self.face_screen.name_submitted.connect(self._on_name_submitted)
        # Signal: user skipped face enrollment
        self.face_screen.skip_requested.connect(self._on_face_screen_skip)

        # State change tracking
        self.agent.on_state_change(self._on_state_change)

        # Background tasks
        self._running = False
        self._voice_task = None

        # Start face detection thread
        self.face_thread.start()

        # Connect all services on startup (async via background loop)
        self.event_loop_thread._loop_ready.wait(timeout=10.0)  # Wait for loop to start
        if self.event_loop_thread.loop:
            loop = self.event_loop_thread.loop
            asyncio.set_event_loop(loop)
            loop.call_soon_threadsafe(loop.create_task, self._init_services())
        else:
            print("Warning: async event loop not available")

    def _setup_ui(self):
        """Set up the foreground UI controls."""
        self.setWindowTitle(getattr(self.agent.config, "assistant_name", "Jarvis"))
        self.resize(800, 600)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Spacer to push controls to bottom
        layout.addStretch()

        # Bottom control bar
        self.bottom_bar = QFrame()
        self.bottom_bar.setFixedHeight(60)
        self.bottom_bar.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 180);
                border-top: 1px solid rgba(0, 200, 255, 100);
            }
        """)

        bar_layout = QHBoxLayout(self.bottom_bar)
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
        self.chat_input.returnPressed.connect(self._on_send_clicked)
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
        self.send_btn.clicked.connect(self._on_send_clicked)
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

        layout.addWidget(self.bottom_bar)

        # Initially hide the bottom bar (shown after face recognition)
        self.bottom_bar.hide()

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
        self.hud.set_state(state)
        state_colors = {
            JarvisState.IDLE: ("⏹ Standby", "rgba(0, 200, 255, 200)"),
            JarvisState.LISTENING: ("🎙 Listening...", "rgba(0, 255, 100, 220)"),
            JarvisState.THINKING: ("🤔 Thinking...", "rgba(200, 200, 0, 220)"),
            JarvisState.SPEAKING: ("🔊 Speaking...", "rgba(255, 150, 0, 220)"),
        }
        text, color = state_colors.get(state, ("?", "white"))
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 14px;")

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

        # If still in the face recognition splash phase, trigger transition
        if self._face_rec_active:
            self._face_rec_active = False
            self._face_timeout.stop()
            if self._enrollment_delay_timer:
                self._enrollment_delay_timer.stop()
            self.face_screen.set_face_detected(name, confidence)
            self._start_transition_check()

    def _on_face_timeout(self):
        """Fallback: no face recognized within timeout — show HUD anyway."""
        if not self._face_rec_active:
            return
        logger.info("Face recognition timeout — falling back to HUD")
        self._face_rec_active = False
        if self._enrollment_delay_timer:
            self._enrollment_delay_timer.stop()
        self.face_screen.set_status("No registered face found. Loading Jarvis...")
        self.face_screen.start_fade()
        self._start_transition_check()

    def _on_face_screen_skip(self):
        """User clicked Skip on the name input — transition to HUD."""
        if self._face_rec_active:
            self._face_rec_active = False
            self._face_timeout.stop()
        if self._enrollment_delay_timer:
            self._enrollment_delay_timer.stop()
        self._start_transition_check()

    def _start_transition_check(self):
        """Start a timer to poll the face screen for fade completion."""
        if self._transition_timer is not None:
            self._transition_timer.stop()
        self._transition_timer = QTimer(self)
        self._transition_timer.timeout.connect(self._check_transition)
        self._transition_timer.start(50)

    def _check_transition(self):
        """Check if the face screen fade-out is complete."""
        if self.face_screen.is_fade_complete:
            if self._transition_timer is not None:
                self._transition_timer.stop()
                self._transition_timer = None
            self.face_thread.stop()  # Stop camera + sampling loop
            self.face_screen.hide()
            self.bottom_bar.show()
            logger.info("Face recognition complete, showing HUD")

    # ── Face enrollment handlers ──────────────────────────────────────

    def _on_unknown_face(self):
        """An unknown face was detected — wait briefly before prompting for enrollment."""
        if not self._face_rec_active:
            return  # Already past the splash screen
        if self._enrollment_prompted:
            return  # Name dialog already shown — don't reset it
        # Debounce: only trigger the delay once per unknown-face burst
        if self._enrollment_delay_timer and self._enrollment_delay_timer.isActive():
            return
        logger.info("Unknown face detected — will prompt for name in 3s")
        self._face_timeout.stop()  # Give user unlimited time to enter a name

        self._enrollment_delay_timer = QTimer(self)
        self._enrollment_delay_timer.setSingleShot(True)
        self._enrollment_delay_timer.timeout.connect(self._show_name_input)
        self._enrollment_delay_timer.start(3000)

    def _show_name_input(self):
        """Actually display the name input overlay after the delay."""
        if not self._face_rec_active:
            return
        self._enrollment_prompted = True
        self.face_screen.show_name_input()

    def _on_name_submitted(self, name: str):
        """User submitted a name for face enrollment."""
        logger.info(f"Starting face enrollment for: {name}")
        self.face_screen.show_sample_progress(0, self.face_thread.config.min_faces_to_add)
        self.face_thread.start_collection(name)

    def _on_samples_progress(self, collected: int, target: int):
        """Update sample collection progress on the face screen."""
        self.face_screen.show_sample_progress(collected, target)

    def _on_enrollment_error(self, msg: str):
        """Show a DB or sample-collection error on the face screen."""
        logger.error(msg)
        self.face_screen.set_status(msg)

    def _on_face_registered(self, name: str):
        """Face model trained and stored — new face is now recognizable."""
        logger.info(f"Face registered for: {name}")
        self._enrollment_prompted = False  # Reset for next enrollment
        self.face_screen.clear_enrollment()
        self.face_screen.set_status(f"Model trained for {name}! Recognizing...")
        # The face thread will now recognize this face and emit face_detected,
        # which triggers the normal transition to the HUD.

    def _open_settings(self):
        """Open the settings/preferences dialog."""
        dialog = SettingsDialog(
            agent_config=self.agent.config,
            app_config=self.app_config,
            agent=self.agent,
            parent=self,
        )
        dialog.face_config = self._build_face_config()
        dialog.on_face_restart = self._restart_face_thread
        dialog.exec()

        # After dialog closes, persist any changes from SettingsDialog
        # that modified app_config (e.g., appearance, silence timeout)
        self._save_app_config()

    def _save_app_config(self):
        """Sync in-memory app_config with agent config, then persist to disk."""
        pw_len = len(self.agent.config.profile_db_password or "")
        logger.debug(f"_save_app_config: db_password length={pw_len}")
        # Sync settings back to AppConfig
        self.app_config.llm_base_url = self.agent.config.llm_base_url
        self.app_config.llm_api_key = self.agent.config.llm_api_key
        self.app_config.llm_model = self.agent.config.llm_model
        self.app_config.stt_host = self.agent.config.stt.host
        self.app_config.stt_port = self.agent.config.stt.port
        self.app_config.tts_host = self.agent.config.tts.host
        self.app_config.tts_port = self.agent.config.tts.port
        self.app_config.tts_voice = self.agent.config.tts_voice
        self.app_config.db_host = self.agent.config.profile_db_host
        self.app_config.db_port = self.agent.config.profile_db_port
        self.app_config.db_name = self.agent.config.profile_db_name
        self.app_config.db_user = self.agent.config.profile_db_user
        self.app_config.db_password = self.agent.config.profile_db_password
        self.app_config.assistant_name = self.agent.config.assistant_name
        self.app_config.silence_timeout = self.agent.config.silence_timeout
        self.app_config.stt_sensitivity = self.agent.config.silence_threshold
        self.app_config.palette_index = self.agent.config.palette_index
        self.app_config.contrast_boost = self.agent.config.contrast_boost

        # Save to config file
        try:
            self.app_config.save()
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def _build_face_config(self) -> FaceConfig:
        """Build a FaceConfig from current face_thread config, preserving DB settings."""
        cfg = FaceConfig()
        if hasattr(self, "face_thread") and hasattr(self.face_thread, "config"):
            existing = self.face_thread.config
            cfg.camera_index = existing.camera_index
            cfg.db_host = existing.db_host
            cfg.db_port = existing.db_port
            cfg.db_user = existing.db_user
            cfg.db_password = existing.db_password
            cfg.db_name = existing.db_name
            cfg.confidence_threshold = existing.confidence_threshold
        return cfg

    def _restart_face_thread(self, new_config: FaceConfig):
        """Stop old face thread and start new one with updated config."""
        logger.info(f"Restarting face thread with camera={new_config.camera_index}")
        # Stop current thread
        if hasattr(self, "face_thread"):
            self.face_thread.stop()
        # Create and start new thread
        self.face_thread = FaceRecThread(new_config)
        self.face_thread.face_detected.connect(self._on_face_detected)
        self.face_thread.frame_captured.connect(self.face_screen.set_frame)
        self.face_thread.unknown_face_detected.connect(self._on_unknown_face)
        self.face_thread.face_registered.connect(self._on_face_registered)
        self.face_thread.samples_progress.connect(self._on_samples_progress)
        self.face_thread.enrollment_error.connect(self._on_enrollment_error)
        self.face_thread.start()
        logger.info("Face thread restarted")

    def _toggle_voice(self):
        """Toggle voice mode (listen → chat → speak loop)."""
        if self._running:
            # Stop voice mode
            self._running = False
            if self._voice_task:
                self._voice_task.cancel()
            self._voice_task = None
            self.mic_btn.setText("🎤")
            self.hud.set_state(JarvisState.IDLE)
        else:
            # Start voice mode
            self._running = True
            self.mic_btn.setText("⏹")
            self.hud.set_state(JarvisState.LISTENING)
            logger.info("Voice mode started — beginning listen loop")
            if self.event_loop_thread.loop:
                loop = self.event_loop_thread.loop
                loop.call_soon_threadsafe(loop.create_task, self._voice_loop())
            else:
                print("Error: async event loop not available")

    async def _voice_loop(self):
        """Main voice loop."""
        while self._running:
            try:
                await self.agent._run_voice()
            except Exception as e:
                import traceback

                logger.error(f"Voice loop error: {e}")
                logger.error(traceback.format_exc())
            await asyncio.sleep(0.1)  # Brief pause between cycles

    def _on_send_clicked(self):
        """Sync wrapper — dispatches async _send_chat on the background event loop."""
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self._status_updated.emit("Thinking...")
        if self.event_loop_thread and self.event_loop_thread.loop:
            asyncio.run_coroutine_threadsafe(self._send_chat(text), self.event_loop_thread.loop)

    async def _send_chat(self, text: str):
        """Send a text message and get response (with optional TTS)."""
        reply = await self.agent.chat_text_and_speak(text)
        self._status_updated.emit("✓" if reply else "✗ Error")

    def closeEvent(self, event):
        """Clean shutdown."""
        self._running = False
        if self._voice_task:
            self._voice_task.cancel()

        # Stop face detection thread
        if hasattr(self, "face_thread"):
            self.face_thread.stop()

        # Hide face screen
        if hasattr(self, "face_screen"):
            self.face_screen.hide()

        # Stop transition timer
        if self._transition_timer is not None:
            self._transition_timer.stop()

        # Reset enrollment state
        self._enrollment_prompted = False

        # Close all async services — must happen BEFORE stopping the event loop thread
        try:
            if self.event_loop_thread.loop and not self.event_loop_thread.loop.is_closed():
                fut = asyncio.run_coroutine_threadsafe(
                    self.agent.close(), self.event_loop_thread.loop
                )
                fut.result(timeout=5)
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")

        # Stop event loop thread (last — all async work is done now)
        if hasattr(self, "event_loop_thread"):
            self.event_loop_thread.stop()

        super().closeEvent(event)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
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
