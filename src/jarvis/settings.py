"""
Jarvis Settings — Preferences dialog for API endpoints and DB config.
"""

import os
import urllib.request
import json
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QSpinBox, QPushButton,
    QGroupBox, QMessageBox, QFrame, QTabWidget, QWidget,
    QComboBox, QSlider
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor

from jarvis.face import FaceConfig


def _fetch_piper_voices(host: str = "127.0.0.1", port: int = 10200) -> list[str]:
    """Return a list of available Piper voice names from the local Wyoming TTS server.
    Falls back to scanning the local voice directory if the server is unreachable."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/models", timeout=2) as resp:
            data = json.load(resp)
            voices = sorted([m["name"] for m in data])
            if voices:
                return voices
    except Exception:
        pass
    voice_dir = os.path.expanduser("~/.local/share/piper/voices")
    if os.path.isdir(voice_dir):
        voices = sorted([d for d in os.listdir(voice_dir)
                           if os.path.isdir(os.path.join(voice_dir, d))])
        if voices:
            return voices
    # final fallback: return a known default so the combo is never empty
    return ["en_US-lessac-medium"]


class SettingsDialog(QDialog):
    """Settings/preferences dialog for Jarvis configuration."""

    def __init__(self, agent_config=None, agent=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Jarvis — Settings")
        self.setMinimumSize(500, 450)
        self.setModal(True)
        self.agent_config = agent_config
        self.agent = agent
        # Face tab callback — caller sets these before showing
        self.face_config: Optional[FaceConfig] = None
        self.on_face_restart: Optional[Callable] = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("⚙ Jarvis Settings")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: rgba(0, 200, 255, 220); padding: 5px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tab widget for sections
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(0, 200, 255, 60);
                border-radius: 6px;
                background: rgba(10, 10, 20, 0.95);
            }
            QTabBar::tab {
                background: rgba(0, 30, 60, 0.8);
                border: 1px solid rgba(0, 200, 255, 40);
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                color: rgba(0, 200, 255, 180);
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: rgba(0, 200, 255, 30);
                border-bottom: 2px solid rgba(0, 200, 255, 180);
                color: rgba(0, 255, 255, 255);
            }
            QTabBar::tab:hover {
                background: rgba(0, 200, 255, 15);
            }
        """)

        # Build tabs
        self.tabs.addTab(self._build_llm_tab(), "🤖 LLM")
        self.tabs.addTab(self._build_stt_tab(), "🎙 STT")
        self.tabs.addTab(self._build_tts_tab(), "🔊 TTS")
        self.tabs.addTab(self._build_audio_tab(), "🔈 Audio")
        self.tabs.addTab(self._build_db_tab(), "🗄 Database")
        self.tabs.addTab(self._build_face_tab(), "👁 Face")
        self.tabs.addTab(self._build_appearance_tab(), "🎨 Appearance")

        layout.addWidget(self.tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.test_btn = QPushButton("🔌 Test Connection")
        self.test_btn.setFixedHeight(36)
        self.test_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 120, 180, 70);
                border: 1px solid rgba(0, 200, 255, 120);
                border-radius: 4px;
                color: white;
                font-size: 13px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background: rgba(0, 150, 220, 100);
            }
        """)
        self.test_btn.clicked.connect(self._test_connection)
        btn_layout.addWidget(self.test_btn)

        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setFixedWidth(90)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: rgba(80, 80, 80, 60);
                border: 1px solid rgba(150, 150, 150, 100);
                border-radius: 4px;
                color: #ccc;
                font-size: 13px;
            }
            QPushButton:hover {
                background: rgba(100, 100, 100, 80);
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setFixedWidth(90)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 180, 120, 70);
                border: 1px solid rgba(0, 220, 150, 120);
                border-radius: 4px;
                color: white;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(0, 200, 140, 100);
            }
        """)
        self.save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def _make_group(self, title, icon=""):
        """Create a styled group box."""
        group = QGroupBox(f"{icon} {title}" if icon else title)
        group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(0, 200, 255, 60);
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                font-size: 13px;
                color: rgba(0, 200, 255, 200);
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)
        return group

    def _make_field(self, label, placeholder="", default=""):
        """Create a label + line edit pair."""
        lbl = QLabel(label)
        lbl.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        if default:
            field.setText(str(default))
        field.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: rgba(0, 220, 255, 160);
            }
            QLineEdit:disabled {
                color: rgba(150, 150, 150, 100);
            }
        """)
        return lbl, field

    def _build_llm_tab(self):
        """LLM configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("LLM Configuration", "🤖")
        layout = QVBoxLayout(group)

        fields = [
            ("API Base URL:", config.llm_base_url, "http://host:port/v1"),
            ("API Key:", config.llm_api_key, "..."),
            ("Model:", config.llm_model, "auto"),
        ]

        for label, default, placeholder in fields:
            lbl, field = self._make_field(label, placeholder, default)
            field.setPlaceholderText(placeholder)
            layout.addWidget(lbl)
            layout.addWidget(field)

        layout.addStretch()
        return group

    def _build_stt_tab(self):
        """STT (Whisper) configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("Speech-to-Text (Whisper)", "🎙")
        layout = QVBoxLayout(group)

        # Host
        lbl, host_field = self._make_field(
            "Server Host:",
            config.stt.host if config.stt else "192.168.55.41",
            "192.168.55.41"
        )
        layout.addWidget(lbl)
        layout.addWidget(host_field)

        # Port
        self.stt_port_spin = QSpinBox()
        self.stt_port_spin.setRange(1, 65535)
        self.stt_port_spin.setValue(config.stt.port if config.stt else 10300)
        self.stt_port_spin.setFixedHeight(30)
        self.stt_port_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)
        port_label = QLabel("Server Port:")
        port_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(port_label)
        layout.addWidget(self.stt_port_spin)

        # Silence timeout
        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(0, 30)
        self.silence_spin.setValue(int(getattr(config, 'silence_timeout', 2.0)))
        self.silence_spin.setSuffix(" sec")
        self.silence_spin.setFixedHeight(30)
        self.silence_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)

        sil_label = QLabel("Silence Detection Timeout:")
        sil_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(sil_label)
        layout.addWidget(self.silence_spin)

        layout.addStretch()
        return group

    def _build_tts_tab(self):
        """TTS (Piper) configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("Text-to-Speech (Piper)", "🔊")
        layout = QVBoxLayout(group)

        # Host
        lbl, host_field = self._make_field(
            "Server Host:",
            config.tts.host if config.tts else "192.168.55.41",
            "192.168.55.41"
        )
        layout.addWidget(lbl)
        layout.addWidget(host_field)

        # Port
        self.tts_port_spin = QSpinBox()
        self.tts_port_spin.setRange(1, 65535)
        self.tts_port_spin.setValue(config.tts.port if config.tts else 10200)
        self.tts_port_spin.setFixedHeight(30)
        self.tts_port_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)
        port_label = QLabel("Server Port:")
        port_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(port_label)
        layout.addWidget(self.tts_port_spin)

        # Voice model
        self.voice_field = QLineEdit()
        self.voice_field.setText("en_US-lessac-medium")
        self.voice_field.setPlaceholderText("en_US-lessac-medium")
        self.voice_field.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: rgba(0, 220, 255, 160);
            }
        """)
        voice_label = QLabel("Voice Model:")
        voice_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(voice_label)
        layout.addWidget(self.voice_field)

        # Piper voice selector
        voice_label = QLabel("Piper Voice:")
        voice_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(voice_label)

        self.piper_voice_combo = QComboBox()
        self.piper_voice_combo.setMinimumWidth(200)
        self.piper_voice_combo.setStyleSheet("""
            QComboBox {
                background: rgba(0,0,0,60);
                border: 1px solid rgba(0,200,255,80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px;
                border-left-width: 1px; border-left-color: rgba(0,200,255,80);
                border-left-style: solid; border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
            QComboBox QAbstractItemView {
                background: rgba(0,0,0,80);
                selection-background-color: rgba(0,200,255,150);
            }
        """)
        # Populate
        for v in _fetch_piper_voices():
            self.piper_voice_combo.addItem(v)
        # Set current value from config (if present)
        if self.agent_config and hasattr(self.agent_config, "tts_voice"):
            idx = self.piper_voice_combo.findText(self.agent_config.tts_voice)
            if idx >= 0:
                self.piper_voice_combo.setCurrentIndex(idx)
        layout.addWidget(self.piper_voice_combo)

        layout.addStretch()
        return group

    def _build_audio_tab(self):
        """Audio output configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("Audio Output", "🔈")
        layout = QVBoxLayout(group)

        # Output device
        self.audio_device_spin = QSpinBox()
        default_dev = config.audio.device if config.audio else None
        self.audio_device_spin.setValue(default_dev if default_dev is not None else -1)
        self.audio_device_spin.setSuffix(" (default if -1)")
        self.audio_device_spin.setFixedHeight(30)
        self.audio_device_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)

        dev_label = QLabel("Output Device Index:")
        dev_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(dev_label)
        layout.addWidget(self.audio_device_spin)

        layout.addStretch()
        return group

    def _build_db_tab(self):
        """Database configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("MariaDB Connection", "🗄")
        layout = QVBoxLayout(group)

        fields = [
            ("Host:", config.profile_db_host, "192.168.55.41"),
            ("Port:", str(config.profile_db_port), "3306"),
            ("Database:", config.profile_db_name, "jarvis"),
            ("User:", config.profile_db_user, "alex"),
            ("Password:", "", "••••••••"),
        ]

        self.password_field = QLineEdit()
        self.password_field.setText(config.profile_db_password)
        self.password_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_field.setPlaceholderText("••••••••")
        self.password_field.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: rgba(0, 220, 255, 160);
            }
        """)

        for i, (label, default, placeholder) in enumerate(fields):
            lbl, field = self._make_field(label, placeholder, default)
            field.setPlaceholderText(placeholder)
            layout.addWidget(lbl)
            layout.addWidget(field)
            if i == len(fields) - 1:  # Last field is password
                self.password_field = field

        layout.addStretch()
        return group

    def _build_face_tab(self):
        """Face recognition configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("Face Recognition", "👁")
        layout = QVBoxLayout(group)

        # Read face config from face_config passed by caller
        fc = self.face_config
        if fc is None:
            fc = FaceConfig()

        # Camera index
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(fc.camera_index)
        self.camera_spin.setSuffix(" (device index)")
        self.camera_spin.setFixedHeight(30)
        self.camera_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)

        cam_label = QLabel("Camera Device Index:")
        cam_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(cam_label)
        layout.addWidget(self.camera_spin)

        # Confidence threshold
        self.face_thresh_spin = QSpinBox()
        self.face_thresh_spin.setRange(0, 200)
        self.face_thresh_spin.setValue(int(fc.confidence_threshold))
        self.face_thresh_spin.setSuffix(" (lower = stricter)")
        self.face_thresh_spin.setFixedHeight(30)
        self.face_thresh_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)

        thresh_label = QLabel("Confidence Threshold:")
        thresh_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(thresh_label)
        layout.addWidget(self.face_thresh_spin)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: rgba(0, 200, 255, 40);")
        layout.addWidget(sep)

        # Restart button
        self.restart_btn = QPushButton("🔄 Restart Face Thread")
        self.restart_btn.setFixedHeight(36)
        self.restart_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 100, 180, 70);
                border: 1px solid rgba(0, 180, 255, 120);
                border-radius: 4px;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 0 12px;
            }
            QPushButton:hover {
                background: rgba(0, 130, 220, 100);
            }
        """)
        self.restart_btn.clicked.connect(self._restart_face)
        layout.addWidget(self.restart_btn)

        layout.addStretch()
        return group

    def _get_values(self):
        """Extract all form values into a config dict.
        
        Uses stored references to form fields rather than re-scanning layouts.
        """
        config = self.agent_config or {}
        if not config:
            return {}

        # LLM: fields[0]=base_url, [1]=api_key, [2]=model
        llm_group = self.tabs.widget(0)
        llm_widgets = self._get_qlineedits(llm_group)
        llm_base_url = llm_widgets[0].text() if len(llm_widgets) > 0 else config.get("llm_base_url", "")
        llm_api_key = llm_widgets[1].text() if len(llm_widgets) > 1 else config.get("llm_api_key", "")
        llm_model = llm_widgets[2].text() if len(llm_widgets) > 2 else config.get("llm_model", "auto")

        # STT: host (widget[0]), port (self.stt_port_spin)
        stt_group = self.tabs.widget(1)
        stt_widgets = self._get_qlineedits(stt_group)
        stt_host = stt_widgets[0].text() if len(stt_widgets) > 0 else ""
        stt_port = self.stt_port_spin.value() if hasattr(self, 'stt_port_spin') else 10300

        # TTS: host (widget[0]), port (self.tts_port_spin), voice (self.voice_field)
        tts_group = self.tabs.widget(2)
        tts_widgets = self._get_qlineedits(tts_group)
        tts_host = tts_widgets[0].text() if len(tts_widgets) > 0 else ""
        tts_port = self.tts_port_spin.value() if hasattr(self, 'tts_port_spin') else 10200
        tts_voice = self.voice_field.text().strip()
        if hasattr(self, 'piper_voice_combo'):
            combo_text = self.piper_voice_combo.currentText().strip()
            if combo_text and not tts_voice:
                tts_voice = combo_text
        if not tts_voice:
            tts_voice = "en_US-lessac-medium"

        # Audio: output_device
        audio_device = self.audio_device_spin.value() if hasattr(self, 'audio_device_spin') else -1

        # DB: host, port, name, user, password
        db_group = self.tabs.widget(4)
        db_widgets = self._get_qlineedits(db_group)
        db_host = db_widgets[0].text() if len(db_widgets) > 0 else ""
        db_port_str = db_widgets[1].text() if len(db_widgets) > 1 else "3306"
        db_name = db_widgets[2].text() if len(db_widgets) > 2 else ""
        db_user = db_widgets[3].text() if len(db_widgets) > 3 else ""
        db_password = db_widgets[4].text() if len(db_widgets) > 4 else ""

        try:
            db_port = int(db_port_str) if db_port_str.isdigit() else 3306
        except ValueError:
            db_port = 3306

        # Face: camera_index, threshold
        camera_index = self.camera_spin.value() if hasattr(self, 'camera_spin') else 0
        face_threshold = self.face_thresh_spin.value() if hasattr(self, 'face_thresh_spin') else 75

        return {
            "llm_base_url": llm_base_url,
            "llm_api_key": llm_api_key,
            "llm_model": llm_model,
            "stt_host": stt_host,
            "stt_port": stt_port,
            "tts_host": tts_host,
            "tts_port": tts_port,
            "tts_voice": tts_voice,
            "audio_device": audio_device,
            "db_host": db_host,
            "db_port": db_port,
            "db_name": db_name,
            "db_user": db_user,
            "db_password": db_password,
            "camera_index": camera_index,
            "face_threshold": face_threshold / 100.0,
        }

    def _get_qlineedits(self, parent):
        """Recursively find all QLineEdit widgets in a widget tree."""
        result = []
        for child in parent.findChildren(QLineEdit):
            result.append(child)
        return result

    def _restart_face(self):
        """Build FaceConfig from form and call restart callback."""
        if not self.on_face_restart or not self.face_config:
            return

        fc = self.face_config
        fc.camera_index = self.camera_spin.value()
        fc.confidence_threshold = float(self.face_thresh_spin.value())

        self.on_face_restart(fc)
        QMessageBox.information(
            self,
            "Face Thread Restarted",
            f"Face detection restarted with:\n"
            f"Camera index: {fc.camera_index}\n"
            f"Confidence threshold: {fc.confidence_threshold}"
        )

    def _test_connection(self):
        """Test database connection with current settings."""
        values = self._get_values()

        # Only test DB since LLM/STT/TTS are async
        host = values.get("db_host", "localhost")
        port = values.get("db_port", 3306)
        user = values.get("db_user", "root")
        password = values.get("db_password", "")
        dbname = values.get("db_name", "test")

        self.test_btn.setEnabled(False)
        self.test_btn.setText("⏳ Testing...")

        try:
            import pymysql
            conn = pymysql.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname,
                cursorclass=pymysql.cursors.DictCursor,
            )
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION()")
                version = cur.fetchone()["VERSION()"]

            QMessageBox.information(
                self,
                "Connection Successful",
                f"Connected to MariaDB {version}\n\n"
                f"Host: {host}:{port}\n"
                f"Database: {dbname}\n"
                f"User: {user}"
            )
            self.test_btn.setText("✅ Connected")
        except pymysql.Error as e:
            QMessageBox.critical(
                self,
                "Connection Failed",
                f"Error: {e}\n\n"
                f"Check:\n"
                f"• Host: {host}:{port}\n"
                f"• User: {user}\n"
                f"• Password\n"
                f"• Database: {dbname}"
            )
            self.test_btn.setText("❌ Failed")
        finally:
            self.test_btn.setEnabled(True)

    def _build_appearance_tab(self):
        """Appearance configuration tab for color schemes."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("Appearance & Themes", "🎨")
        layout = QVBoxLayout(group)

        # Palette selector
        lbl = QLabel("Color Palette:")
        lbl.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(lbl)

        self.palette_combo = QComboBox()
        self.palette_combo.setFixedHeight(30)
        self.palette_combo.setStyleSheet("""
            QComboBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: rgba(0, 200, 255, 80);
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox QAbstractItemView {
                background: rgba(0, 0, 0, 80);
                selection-background-color: rgba(0, 200, 255, 150);
            }
        """)
        # Define 10 high-contrast hues (HSL hue values 0-360)
        self._palette_hues = [182, 30, 120, 250, 200, 0, 80, 220, 40, 60]  # cyan, copper, emerald, violet, matrix, red, green, blue, yellow, orange
        self._palette_names = ["Cyan/Teal", "Copper/Amber", "Emerald/Teal", "Violet/Purple", "Matrix Green", "Red", "Green", "Blue", "Yellow", "Orange"]
        for name in self._palette_names:
            self.palette_combo.addItem(name)
        # Set current index from config if available
        if hasattr(config, 'palette_index') and config.palette_index is not None:
            idx = config.palette_index
            if 0 <= idx < len(self._palette_names):
                self.palette_combo.setCurrentIndex(idx)
        else:
            self.palette_combo.setCurrentIndex(0)  # default to first
        # Connect signal to update HUD and config
        self.palette_combo.currentIndexChanged.connect(self._on_palette_changed)
        layout.addWidget(self.palette_combo)

        # Contrast boost slider
        layout.addSpacing(12)
        contrast_label = QLabel("Contrast Boost:")
        contrast_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(contrast_label)

        contrast_row = QHBoxLayout()
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(80, 140)  # 80% to 140% of base saturation
        self.contrast_slider.setValue(100)
        self.contrast_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(0, 200, 255, 40);
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: rgba(0, 200, 255, 180);
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 200, 255, 100);
                border-radius: 2px;
            }
        """)
        # Read current contrast from config
        current_contrast = getattr(self.agent_config, "contrast_boost", 100) if self.agent_config else 100
        self.contrast_slider.setValue(int(current_contrast))
        self.contrast_value_label = QLabel(f"{int(current_contrast)}%")
        self.contrast_value_label.setStyleSheet("color: rgba(200, 220, 240, 200); font-size: 12px; min-width: 35px;")
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        contrast_row.addWidget(self.contrast_slider, stretch=1)
        contrast_row.addWidget(self.contrast_value_label)
        layout.addLayout(contrast_row)

        # Contrast boost slider
        layout.addSpacing(12)
        contrast_label = QLabel("Contrast Boost:")
        contrast_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(contrast_label)

        contrast_row = QHBoxLayout()
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(80, 140)  # 80% to 140% of base saturation
        self.contrast_slider.setValue(100)
        self.contrast_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(0, 200, 255, 40);
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: rgba(0, 200, 255, 180);
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 200, 255, 100);
                border-radius: 2px;
            }
        """)
        # Read current contrast from config
        current_contrast = getattr(self.agent_config, 'contrast_boost', 100) if self.agent_config else 100
        self.contrast_slider.setValue(int(current_contrast))
        self.contrast_value_label = QLabel(f"{int(current_contrast)}%")
        self.contrast_value_label.setStyleSheet("color: rgba(200, 220, 240, 200); font-size: 12px; min-width: 35px;")
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        contrast_row.addWidget(self.contrast_slider, stretch=1)
        contrast_row.addWidget(self.contrast_value_label)
        layout.addLayout(contrast_row)

        layout.addStretch()
        return group

    def _on_palette_changed(self, index: int):
        """Called when the user selects a different palette in the appearance tab."""
        hue = self._palette_hues[index] if hasattr(self, '_palette_hues') and index < len(self._palette_hues) else 0
        # Update HUD overlay if it exists
        if hasattr(self, 'agent') and hasattr(self.agent, 'hud') and self.agent.hud:
            self.agent.hud.set_palette_hue(hue)
        # Store the index in agent config for persistence
        if self.agent_config:
            self.agent_config.palette_index = index

    def _on_contrast_changed(self, value: int):
        """Called when the contrast boost slider is moved."""
        if hasattr(self, "contrast_value_label"):
            self.contrast_value_label.setText(f"{value}%")
        if self.agent_config:
            self.agent_config.contrast_boost = value
        # Apply live to HUD
        if self.agent is not None and hasattr(self.agent, "hud") and self.agent.hud:
            self.agent.hud.set_contrast_factor(value / 100.0)

    def _on_contrast_changed(self, value: int):
        """Called when the contrast boost slider is moved."""
        if hasattr(self, 'contrast_value_label'):
            self.contrast_value_label.setText(f"{value}%")
        if self.agent_config:
            self.agent_config.contrast_boost = value
        # Apply live to HUD
        if self.agent is not None and hasattr(self.agent, 'hud') and self.agent.hud:
            self.agent.hud.set_contrast_factor(value / 100.0)

    def _save_settings(self):
        """Save settings and close dialog."""
        values = self._get_values()

        if not values:
            QMessageBox.warning(self, "Error", "No configuration available to save.")
            return

        # Apply settings to agent config
        if self.agent_config:
            # LLM
            if values.get("llm_base_url"):
                self.agent_config.llm_base_url = values["llm_base_url"]
            if values.get("llm_api_key"):
                self.agent_config.llm_api_key = values["llm_api_key"]
            if values.get("llm_model"):
                self.agent_config.llm_model = values["llm_model"]

            # STT
            stt_config = self.agent_config.stt
            if stt_config:
                if values.get("stt_host"):
                    stt_config.host = values["stt_host"]
                stt_config.port = values.get("stt_port", stt_config.port)

            # TTS
            tts_config = self.agent_config.tts
            if tts_config:
                if values.get("tts_host"):
                    tts_config.host = values["tts_host"]
                tts_config.port = values.get("tts_port", tts_config.port)
                if values.get("tts_voice"):
                    tts_config.voice = values["tts_voice"]

            # Update the agent's TTS object immediately so the next speak uses the new voice
            if self.agent and hasattr(self.agent, 'tts'):
                self.agent.tts.voice = values.get("tts_voice", self.agent.tts.voice)

            # Audio
            audio_config = self.agent_config.audio
            if audio_config:
                audio_config.device = values.get("audio_device", audio_config.device)

            # Silence timeout
            stt_config = self.agent_config.stt
            if stt_config and hasattr(self, 'silence_spin'):
                self.agent_config.silence_timeout = float(self.silence_spin.value())

            # DB
            self.agent_config.profile_db_host = values.get("db_host", self.agent_config.profile_db_host)
            self.agent_config.profile_db_port = values.get("db_port", self.agent_config.profile_db_port)
            self.agent_config.profile_db_name = values.get("db_name", self.agent_config.profile_db_name)
            self.agent_config.profile_db_user = values.get("db_user", self.agent_config.profile_db_user)
            self.agent_config.profile_db_password = values.get("db_password", self.agent_config.profile_db_password)

            # TTS voice model (stored as custom attr since WyomingConfig doesn't have it)
            if hasattr(self, 'voice_field') and self.agent_config:
                self.agent_config._tts_voice = self.voice_field.text()
            # Also persist the Piper voice combo selection at the top level for clarity
            if hasattr(self, 'piper_voice_combo') and self.agent_config:
                self.agent_config.tts_voice = self.piper_voice_combo.currentText()

            # Face — just update the face_config object (caller handles restart)
            if self.face_config:
                self.face_config.camera_index = self.camera_spin.value()
                self.face_config.confidence_threshold = float(self.face_thresh_spin.value())

            # Palette and contrast
            if hasattr(self, 'palette_combo'):
                self.agent_config.palette_index = self.palette_combo.currentIndex()
            if hasattr(self, 'contrast_slider'):
                self.agent_config.contrast_boost = self.contrast_slider.value()

        QMessageBox.information(
            self,
            "Settings Saved",
            "Settings have been saved.\n\n"
            "Some changes require restarting the application to take effect."
        )
        self.accept()