"""
Jarvis Settings - Preferences dialog for API endpoints and DB config.
"""

import logging
from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.face import FaceConfig
from jarvis.profile import PALETTE_HUES

logger = logging.getLogger(__name__)

class SettingsDialog(QDialog):
    """Settings/preferences dialog for Jarvis configuration."""

    def __init__(self, agent_config=None, app_config=None, agent=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Jarvis — Settings")
        self.setMinimumSize(500, 450)
        self.setModal(True)
        # Restore standard system window decorations (title bar, close, etc.)
        # since parent JarvisApp is frameless
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.agent_config = agent_config
        self.app_config = app_config
        self.agent = agent
        # Face tab callback — caller sets these before showing
        self.face_config: FaceConfig | None = None
        self.on_face_restart: Callable | None = None

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
        self.tabs.addTab(self._build_profiles_tab(), "👤 Profiles")
        self.tabs.addTab(self._build_appearance_tab(), "🎨 Appearance")

        # Reflect the live wake-word detector state (if an agent is attached)
        self._update_wake_status()

        # Reflect the active profile (and keep it live if face recognition
        # switches profiles while this dialog is open).
        if hasattr(self, "profile_status"):
            self._update_profile_status()
        if self.agent is not None and hasattr(self.agent, "on_profile_changed"):
            try:
                self.agent.on_profile_changed(lambda _p: self._update_profile_status())
            except Exception:  # noqa: BLE001
                pass

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

        # Assistant name
        self.assistant_name_field = QLineEdit()
        self.assistant_name_field.setText(getattr(config, "assistant_name", "Jarvis"))
        self.assistant_name_field.setPlaceholderText("Jarvis")
        self.assistant_name_field.setStyleSheet("""
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
        assistant_name_label = QLabel("Assistant Name:")
        assistant_name_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(assistant_name_label)
        layout.addWidget(self.assistant_name_field)

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
            "Server Host:", config.stt.host if config.stt else "192.168.55.41", "192.168.55.41"
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

        # Input device
        self.stt_device_spin = QSpinBox()
        self.stt_device_spin.setRange(-1, 65535)
        default_stt_dev = config.stt.device if config.stt else None
        self.stt_device_spin.setValue(default_stt_dev if default_stt_dev is not None else -1)
        self.stt_device_spin.setSuffix(" (default if -1)")
        self.stt_device_spin.setFixedHeight(30)
        self.stt_device_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)
        dev_label = QLabel("Input Device Index:")
        dev_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(dev_label)
        layout.addWidget(self.stt_device_spin)

        # Silence timeout
        self.silence_spin = QDoubleSpinBox()
        self.silence_spin.setRange(0.1, 30.0)
        self.silence_spin.setSingleStep(0.1)
        self.silence_spin.setValue(float(getattr(config, "silence_timeout", 2.0)))
        self.silence_spin.setSuffix(" sec")
        self.silence_spin.setFixedHeight(30)
        self.silence_spin.setStyleSheet("""
            QDoubleSpinBox {
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

        # STT Sensitivity (RMS threshold — lower = more sensitive)
        layout.addSpacing(8)
        sens_label = QLabel("Silence Sensitivity (RMS threshold):")
        sens_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(sens_label)

        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(10.0, 1000.0)
        self.sensitivity_spin.setSingleStep(5.0)
        default_sensitivity = getattr(config, "silence_threshold", 100.0)
        self.sensitivity_spin.setValue(float(default_sensitivity))
        self.sensitivity_spin.setSuffix(" (lower = more sensitive)")
        self.sensitivity_spin.setFixedHeight(30)
        self.sensitivity_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
        """)
        layout.addWidget(self.sensitivity_spin)

        # ── Wake Word (hands-free) ────────────────────────────────────
        layout.addSpacing(16)
        wake_group = self._make_group("Wake Word (hands-free)", "🎤")
        wake_layout = QVBoxLayout(wake_group)

        self.wake_enabled = QCheckBox('Say the wake word to start a conversation (no click needed)')
        self.wake_enabled.setChecked(bool(getattr(config, "wake_word_enabled", True)))
        self.wake_enabled.setStyleSheet("color: rgba(210, 230, 250, 230); font-size: 12px;")
        self.wake_enabled.toggled.connect(
            lambda _c: (self._update_wake_status(), self.wake_threshold.setEnabled(_c))
        )
        wake_layout.addWidget(self.wake_enabled)

        self.wake_available = QLabel("Status: …")
        self.wake_available.setStyleSheet("color: rgba(160, 190, 220, 200); font-size: 11px;")
        self.wake_available.setWordWrap(True)
        wake_layout.addWidget(self.wake_available)

        wake_thresh_row = QHBoxLayout()
        wake_thresh_lbl = QLabel("Sensitivity threshold (lower = triggers more easily):")
        wake_thresh_lbl.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        wake_thresh_row.addWidget(wake_thresh_lbl)
        self.wake_threshold = QSlider(Qt.Orientation.Horizontal)
        self.wake_threshold.setRange(30, 95)          # 0.30 – 0.95
        self.wake_threshold.setSingleStep(5)
        default_wake_thresh = int(round(getattr(config, "wake_word_threshold", 0.5) * 100))
        self.wake_threshold.setValue(max(30, min(95, default_wake_thresh)))
        self.wake_threshold.valueChanged.connect(self._update_wake_threshold)
        wake_thresh_row.addWidget(self.wake_threshold, stretch=1)
        self.wake_threshold_value = QLabel("0.50")
        self.wake_threshold_value.setStyleSheet("color: rgba(0, 220, 255, 230); font-size: 12px; min-width: 35px;")
        wake_thresh_row.addWidget(self.wake_threshold_value)
        wake_layout.addLayout(wake_thresh_row)

        layout.addWidget(wake_group)

        layout.addStretch()
        return group

    def _build_tts_tab(self):
        """TTS (Piper) configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("Text-to-Speech (Piper)", "���������🔊")
        layout = QVBoxLayout(group)

        # Host
        lbl, host_field = self._make_field(
            "Server Host:", config.tts.host if config.tts else "192.168.55.41", "192.168.55.41"
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

        # Voice selector dropdown
        voice_layout = QVBoxLayout()
        voice_label = QLabel("Voice Model:")
        voice_label.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        voice_layout.addWidget(voice_label)

        self.voice_combo = QComboBox()
        self.voice_combo.setFixedHeight(30)
        self.voice_combo.setStyleSheet("""
            QComboBox {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; }
            QComboBox QAbstractItemView {
                background: rgba(0, 0, 0, 80);
                selection-background-color: rgba(0, 200, 255, 150);
            }
        """)
        voice_layout.addWidget(self.voice_combo)
        layout.addLayout(voice_layout)

        # Populate voice combobox with common voices
        self._populate_voice_combo_simple()

        layout.addStretch()
        return group

    def _populate_voice_combo_simple(self):
        """Populate voice combo box with common voices."""
        # Clear existing items
        self.voice_combo.clear()
        
        # Add common voices grouped by language for better UX
        voice_groups = {
            "English (US)": [
                "en_US-lessac-low",
                "en_US-lessac-medium", 
                "en_US-lessac-high",
                "en_US-amy-low",
                "en_US-amy-medium",
                "en_US-joe-medium",
                "en_US-john-medium",
                "en_US-bryce-medium",
                "en_US-kathleen-low",
                "en_US-kristin-medium",
                "en_US-kusal-medium",
                "en_US-l2arctic-medium",
                "en_US-ryan-medium",
                "en_US-ryan-high",
                "en_US-ryan-low",
                "en_US-hfc_male-medium",
                "en_US-hfc_female-medium"
            ],
            "English (GB)": [
                "en_GB-alan-low",
                "en_GB-alan-medium",
                "en_GB-alba-medium",
                "en_GB-aru-medium",
                "en_GB-cori-high",
                "en_GB-cori-medium",
                "en_GB-jenny_dioco-medium",
                "en_GB-northern_english_male-medium",
                "en_GB-semaine-medium",
                "en_GB-southern_english_female-low",
                "en_GB-vctk-medium",
                "en_GB-southern_english_male-medium"
            ],
            "French": [
                "fr_FR-mls-medium",
                "fr_FR-tom-medium",
                "fr_FR-upmc-medium",
                "fr_FR-gilles-low",
                "fr_FR-siwis-medium"
            ],
            "German": [
                "de_DE-karlsson-medium",
                "de_DE-kessler-medium",
                "de_DE-eva_k-medium",
                "de_DE-kerstin-medium",
                "de_DE-david-medium"
            ],
            "Spanish": [
                "es_ES-sharvard-medium",
                "es_ES-mls_9972-medium",
                "es_ES-mls-medium",
                "es_ES-mls_10246-medium",
                "es_ES-davefx-medium",
                "es_ES-carlfm-medium",
                "es_ES-mls_1840-medium"
            ],
            "Dutch": [
                "nl_BE-nathalie-medium",
                "nl_NL-alex-medium",
                "nl_NL-mls-medium",
                "nl_NL-ronnie-medium",
                "nl_NL-pim-medium"
            ],
            "Italian": [
                "it_IT-paola-medium",
                "it_IT-silva-medium",
                "it_IT-riccardo-medium",
                "it_IT-serena-medium"
            ],
            "Portuguese": [
                "pt_BR-faber-medium",
                "pt_BR-cadu-medium",
                "pt_BR-edresson-low",
                "pt_PT-tugão-medium"
            ],
            "Russian": [
                "ru_RU-irina-medium",
                "ru_RU-dmitri-medium",
                "ru_RU-ruslan-medium",
                "ru_RU-klava-medium"
            ],
            "Ukrainian": [
                "uk_UA-lada-medium",
                "uk_UA-oleksa-medium",
                "uk_UA-mykhailo-medium",
                "uk_UA-viktoriia-medium"
            ],
            "Chinese": [
                "zh_CN-huayan-medium",
                "zh_CN-chaowen-medium",
                "zh_CN-xiao_ya-medium",
                "zh_CN-xiao_xiong-medium"
            ],
            "Japanese": [
                "ja_JP-nanako-medium",
                "ja_JP-nanako-high",
                "ja_JP-nanako-low"
            ],
            "Polish": [
                "pl_PL-mc_speech-medium",
                "pl_PL-gosia-medium",
                "pl_PL-darkman-medium",
                "pl_PL-bass-medium"
            ]
        }
        
        # Add voice groups as sections
        for language, voices in voice_groups.items():
            # Add language header
            self.voice_combo.addItem(f"--- {language} ---")
            self.voice_combo.setItemData(self.voice_combo.count() - 1, 
                                       {"type": "header", "language": language})
            
            # Add voices in this language
            for voice in voices:
                self.voice_combo.addItem(f"  {voice}")
                self.voice_combo.setItemData(self.voice_combo.count() - 1, 
                                           {"type": "voice", "name": voice})
        
        # Set current voice
        current_voice = getattr(self.agent_config, "tts_voice", "en_US-lessac-medium") if self.agent_config else "en_US-lessac-medium"
        index = self.voice_combo.findText(current_voice)
        if index < 0:
            # Try with prefix (voices are indented with 2 spaces)
            index = self.voice_combo.findText(f"  {current_voice}")
        if index >= 0:
            self.voice_combo.setCurrentIndex(index)
            
        # Connect signal
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)

    def _on_voice_changed(self, index: int):
        """Called when voice selection changes."""
        if index < 0:
            return
        data = self.voice_combo.itemData(index)
        if isinstance(data, dict) and data.get("type") == "voice":
            voice_name = data.get("name")
            if voice_name and self.agent_config:
                self.agent_config.tts_voice = voice_name
                # Update the agent's TTS object immediately
                if self.agent and hasattr(self.agent, "tts"):
                    self.agent.tts.voice = voice_name
                logger.info(f"TTS voice changed to: {voice_name}")

    def _build_audio_tab(self):
        """Audio output configuration tab."""
        config = self.agent_config
        if not config:
            return QWidget()

        group = self._make_group("Audio Output", "🔈")
        layout = QVBoxLayout(group)

        # Output device
        self.audio_device_spin = QSpinBox()
        self.audio_device_spin.setRange(-1, 65535)
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
        group = self._make_group("MariaDB Connection", "🗄")
        layout = QVBoxLayout(group)

        # Prefer app_config values (with proper defaults) over agent_config
        # which may have None values resolved only at runtime
        db_host = ""
        db_port = "3306"
        db_name = "jarvis"
        db_user = "root"
        db_password = ""
        if self.app_config:
            db_host = self.app_config.db_host or ""
            db_port = str(self.app_config.db_port)
            db_name = self.app_config.db_name or ""
            db_user = self.app_config.db_user or ""
            db_password = self.app_config.db_password or ""
        elif self.agent_config:
            db_host = self.agent_config.profile_db_host or ""
            db_port = str(self.agent_config.profile_db_port or 3306)
            db_name = self.agent_config.profile_db_name or ""
            db_user = self.agent_config.profile_db_user or ""
            db_password = self.agent_config.profile_db_password or ""

        fields = [
            ("Host:", db_host, "192.168.55.41"),
            ("Port:", db_port, "3306"),
            ("Database:", db_name, "jarvis"),
            ("User:", db_user, "root"),
            ("Password:", db_password, ""),
        ]

        for i, (label, default, placeholder) in enumerate(fields):
            lbl, field = self._make_field(label, placeholder, default)
            field.setPlaceholderText(placeholder)
            layout.addWidget(lbl)
            layout.addWidget(field)
            if i == len(fields) - 1:  # Last field is password
                self.password_field = field
                self.password_field.setEchoMode(QLineEdit.EchoMode.Password)

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
        llm_base_url = (
            llm_widgets[0].text() if len(llm_widgets) > 0 else config.get("llm_base_url", "")
        )
        llm_api_key = (
            llm_widgets[1].text() if len(llm_widgets) > 1 else config.get("llm_api_key", "")
        )
        llm_model = (
            llm_widgets[2].text() if len(llm_widgets) > 2 else config.get("llm_model", "auto")
        )
        # Assistant name is the 4th field (index 3)
        assistant_name = (
            llm_widgets[3].text()
            if len(llm_widgets) > 3
            else config.get("assistant_name", "Jarvis")
        )

        # STT: host (widget[0]), port (self.stt_port_spin), input device (self.stt_device_spin)
        stt_group = self.tabs.widget(1)
        stt_widgets = self._get_qlineedits(stt_group)
        stt_host = stt_widgets[0].text() if len(stt_widgets) > 0 else ""
        stt_port = self.stt_port_spin.value() if hasattr(self, "stt_port_spin") else 10300
        stt_device = self.stt_device_spin.value() if hasattr(self, "stt_device_spin") else -1

        # TTS: host (widget[0]), port (self.tts_port_spin), voice (self.voice_combo)
        tts_group = self.tabs.widget(2)
        tts_widgets = self._get_qlineedits(tts_group)
        tts_host = tts_widgets[0].text() if len(tts_widgets) > 0 else ""
        tts_port = self.tts_port_spin.value() if hasattr(self, "tts_port_spin") else 10200
        tts_voice = ""
        if hasattr(self, "voice_combo"):
            data = self.voice_combo.itemData(self.voice_combo.currentIndex())
            if isinstance(data, dict) and data.get("type") == "voice":
                tts_voice = data.get("name")
        if not tts_voice:
            tts_voice = "en_US-lessac-medium"

        # Audio: output_device
        audio_device = self.audio_device_spin.value() if hasattr(self, "audio_device_spin") else -1

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
        camera_index = self.camera_spin.value() if hasattr(self, "camera_spin") else 0
        face_threshold = self.face_thresh_spin.value() if hasattr(self, "face_thresh_spin") else 75

        return {
            "llm_base_url": llm_base_url,
            "llm_api_key": llm_api_key,
            "llm_model": llm_model,
            "assistant_name": assistant_name,
            "stt_host": stt_host,
            "stt_port": stt_port,
            "stt_device": stt_device,
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
        """Build FaceConfig from form values and call restart callback."""
        if not self.on_face_restart or not self.face_config:
            return

        values = self._get_values()
        fc = self.face_config
        fc.camera_index = self.camera_spin.value()
        fc.confidence_threshold = float(self.face_thresh_spin.value())
        fc.db_host = values["db_host"] or fc.db_host
        fc.db_port = values["db_port"]
        fc.db_name = values["db_name"] or fc.db_name
        fc.db_user = values["db_user"] or fc.db_user
        fc.db_password = values["db_password"]

        self.on_face_restart(fc)
        QMessageBox.information(
            self,
            "Face Thread Restarted",
            f"Face detection restarted with:\n"
            f"Camera index: {fc.camera_index}\n"
            f"Confidence threshold: {fc.confidence_threshold}\n"
            f"DB host: {fc.db_host}:{fc.db_port}\n"
            f"DB user: {fc.db_user}",
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
                f"User: {user}",
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
                f"• Database: {dbname}",
            )
            self.test_btn.setText("❌ Failed")
        finally:
            self.test_btn.setEnabled(True)

    def _build_profiles_tab(self):
        """Profiles tab — view/edit face↔profile links and switch active profile."""
        group = self._make_group("Profiles & Face Links", "👤")
        layout = QVBoxLayout(group)

        # Profile selector
        lbl = QLabel("Active Profile: (also switched automatically by face recognition)")
        lbl.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(lbl)

        self.profile_combo = QComboBox()
        self.profile_combo.setFixedHeight(30)
        self.profile_combo.setStyleSheet("""
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
            }
            QComboBox QAbstractItemView {
                background: rgba(0, 0, 0, 80);
                selection-background-color: rgba(0, 200, 255, 150);
            }
        """)
        # Populate from agent's profile manager
        self._profile_names = []
        if self.agent and hasattr(self.agent, "profiles"):
            self._profile_names = self.agent.profiles.list_names()
            for name in self._profile_names:
                self.profile_combo.addItem(name)
        layout.addWidget(self.profile_combo)

        # Link label + edit (face_name column)
        layout.addSpacing(8)
        link_lbl = QLabel("Linked Face (face_name):")
        link_lbl.setStyleSheet("color: rgba(180, 200, 220, 200); font-size: 12px;")
        layout.addWidget(link_lbl)

        self.profile_face_link = QLineEdit()
        self.profile_face_link.setPlaceholderText("Face model name that activates this profile")
        self.profile_face_link.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 60);
                border: 1px solid rgba(0, 200, 255, 80);
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
                font-size: 13px;
            }
        """)
        # Load face_name for the currently selected profile
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        layout.addWidget(self.profile_face_link)

        # Active-profile status
        self.profile_status = QLabel("Active: default (no profile)")
        self.profile_status.setStyleSheet("color: rgba(0, 220, 255, 220); font-size: 12px; font-weight: bold;")
        layout.addWidget(self.profile_status)

        # Activate / Reset controls (manual switching — face recognition also
        # switches automatically)
        btn_row = QHBoxLayout()
        self.profile_activate_btn = QPushButton("✅ Activate Selected")
        self.profile_activate_btn.setFixedHeight(34)
        self.profile_activate_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 180, 120, 70);
                border: 1px solid rgba(0, 220, 150, 120);
                border-radius: 4px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background: rgba(0, 200, 140, 100); }
        """)
        self.profile_activate_btn.clicked.connect(self._activate_profile)
        btn_row.addWidget(self.profile_activate_btn)

        self.profile_reset_btn = QPushButton("↩ Reset to Default")
        self.profile_reset_btn.setFixedHeight(34)
        self.profile_reset_btn.setStyleSheet("""
            QPushButton {
                background: rgba(80, 80, 80, 60);
                border: 1px solid rgba(150, 150, 150, 100);
                border-radius: 4px;
                color: #ccc;
                font-size: 12px;
            }
            QPushButton:hover { background: rgba(100, 100, 100, 80); }
        """)
        self.profile_reset_btn.clicked.connect(self._reset_profile)
        btn_row.addWidget(self.profile_reset_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        # Load face names initially after construction
        if self._profile_names:
            self._on_profile_selected(0)
        return group

    def _current_active_profile_name(self) -> str | None:
        """Name of the active profile, if any."""
        if self.agent is not None and getattr(self.agent, "_active_profile", None):
            return self.agent._active_profile.name
        return None

    def _update_profile_status(self) -> None:
        """Reflect the active profile in the Profiles tab status label."""
        if not hasattr(self, "profile_status"):
            return
        try:
            from PyQt6 import sip

            if sip.isdeleted(self.profile_status):
                return
        except Exception:  # noqa: BLE001
            return
        name = self._current_active_profile_name()
        if name:
            self.profile_status.setText(f"Active: {name}")
        else:
            self.profile_status.setText("Active: default (no profile)")

    def _activate_profile(self) -> None:
        """Activate the selected profile (immediate, applies prompt/voice/theme)."""
        if not (self.agent and hasattr(self, "profile_combo") and self.profile_combo.count()):
            QMessageBox.warning(self, "No Profiles", "No profiles are available.\n"
                                   "Check the Database tab connection first.")
            return
        name = self.profile_combo.currentText()
        if self.agent.switch_profile(name):
            self._update_profile_status()
            QMessageBox.information(self, "Profile Activated",
                                    f"Switched to profile: {name}\n"
                                    "System prompt, assistant name, chat history and "
                                    "accent color now apply.")
        else:
            QMessageBox.warning(self, "Switch Failed",
                                f"Could not activate profile '{name}'.")

    def _reset_profile(self) -> None:
        """Clear the active profile back to defaults."""
        if self.agent is None:
            QMessageBox.warning(self, "No Agent", "No agent attached.")
            return
        self.agent.clear_profile()
        self._update_profile_status()

    def _on_profile_selected(self, index: int):
        """Populate face-name link field when a profile is selected."""
        if not (self.agent and hasattr(self.agent, "profiles") and self.profile_combo.count()):
            return
        name = self.profile_combo.currentText()
        profile = self.agent.profiles.get(name)
        if profile:
            self.profile_face_link.setText(profile.face_name or "")

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
        self._palette_hues = list(PALETTE_HUES)
        self._palette_names = [
            "Cyan/Teal",
            "Copper/Amber",
            "Emerald/Teal",
            "Violet/Purple",
            "Matrix Green",
            "Red",
            "Green",
            "Blue",
            "Yellow",
            "Orange",
        ]
        for name in self._palette_names:
            self.palette_combo.addItem(name)
        # Set current index from config if available
        if hasattr(config, "palette_index") and config.palette_index is not None:
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
        current_contrast = (
            getattr(self.agent_config, "contrast_boost", 100) if self.agent_config else 100
        )
        self.contrast_slider.setValue(int(current_contrast))
        self.contrast_value_label = QLabel(f"{int(current_contrast)}%")
        self.contrast_value_label.setStyleSheet(
            "color: rgba(200, 220, 240, 200); font-size: 12px; min-width: 35px;"
        )
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        contrast_row.addWidget(self.contrast_slider, stretch=1)
        contrast_row.addWidget(self.contrast_value_label)
        layout.addLayout(contrast_row)

        layout.addStretch()
        return group

    def _on_palette_changed(self, index: int):
        """Called when the user selects a different palette in the appearance tab."""
        hue = (
            self._palette_hues[index]
            if hasattr(self, "_palette_hues") and index < len(self._palette_hues)
            else 0
        )
        # Update HUD overlay if it exists
        if hasattr(self, "agent") and hasattr(self.agent, "hud") and self.agent.hud:
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

    def _update_wake_threshold(self, value: int):
        """Update the wake-word threshold label and apply it live to the agent."""
        if hasattr(self, "wake_threshold_value"):
            self.wake_threshold_value.setText(f"{value / 100.0:.2f}")
        if self.agent_config is not None:
            self.agent_config.wake_word_threshold = value / 100.0
        # Apply live so the detector picks up the new threshold immediately
        if self.agent is not None and getattr(self.agent, "_wake_word", None):
            self.agent._wake_word.threshold = value / 100.0

    def _update_wake_status(self):
        """Show the live wake-word detector state in the STT tab's status label."""
        if not hasattr(self, "wake_available"):
            return
        if self.agent is None:
            self.wake_available.setText(
                "Status: no agent attached — enable in the running app"
            )
            return
        if getattr(self, "wake_enabled", None) is not None and not self.wake_enabled.isChecked():
            self.wake_available.setText("Status: disabled (see checkbox above)")
            return
        if self.agent.wake_word_available:
            self.wake_available.setText("Status: ✓ detector loaded and ready")
        else:
            self.wake_available.setText(
                "Status: ✗ not available — push-to-talk (mic button) still works"
            )

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
            # Assistant name
            if values.get("assistant_name"):
                self.agent_config.assistant_name = values["assistant_name"]
                # Persist to MariaDB so it survives restarts
                if self.agent and hasattr(self.agent, "profiles"):
                    self.agent.profiles.set_default_assistant_name(values["assistant_name"])
                # Update HUD immediately
                if self.agent and getattr(self.agent, "hud", None):
                    self.agent.hud.set_assistant_name(values["assistant_name"])

            # STT
            stt_config = self.agent_config.stt
            if stt_config:
                if values.get("stt_host"):
                    stt_config.host = values["stt_host"]
                stt_config.port = values.get("stt_port", stt_config.port)
                if "stt_device" in values:
                    stt_config.device = values["stt_device"]

            # TTS
            tts_config = self.agent_config.tts
            if tts_config:
                if values.get("tts_host"):
                    tts_config.host = values["tts_host"]
                tts_config.port = values.get("tts_port", tts_config.port)
                if values.get("tts_voice"):
                    tts_config.voice = values["tts_voice"]

            # Update the agent's TTS object immediately so the next speak uses the new voice
            if self.agent and hasattr(self.agent, "tts"):
                self.agent.tts.voice = values.get("tts_voice", self.agent.tts.voice)

            # Audio
            audio_config = self.agent_config.audio
            if audio_config:
                audio_config.device = values.get("audio_device", audio_config.device)

            # Silence timeout
            stt_config = self.agent_config.stt
            if stt_config and hasattr(self, "silence_spin"):
                self.agent_config.silence_timeout = float(self.silence_spin.value())

            # STT sensitivity (RMS threshold)
            if hasattr(self, "sensitivity_spin"):
                self.agent_config.silence_threshold = float(self.sensitivity_spin.value())

            # Wake word (hands-free)
            if hasattr(self, "wake_enabled"):
                self.agent_config.wake_word_enabled = bool(self.wake_enabled.isChecked())
            if hasattr(self, "wake_threshold"):
                self.agent_config.wake_word_threshold = self.wake_threshold.value() / 100.0

            # DB — only update non-empty values from the form (don't
            # overwrite with empty strings from blank QLineEdit fields)
            db_host = values.get("db_host", "")
            db_name = values.get("db_name", "")
            db_user = values.get("db_user", "")
            db_password = values.get("db_password", "")
            if db_host:
                self.agent_config.profile_db_host = db_host
            if db_name:
                self.agent_config.profile_db_name = db_name
            if db_user:
                self.agent_config.profile_db_user = db_user
            if db_password:
                self.agent_config.profile_db_password = db_password

            # TTS voice model (stored as custom attr since WyomingConfig doesn't have it)
            if hasattr(self, "voice_field") and self.agent_config:
                self.agent_config._tts_voice = self.voice_field.text()

            # Face — just update the face_config object (caller handles restart)
            if self.face_config:
                self.face_config.camera_index = self.camera_spin.value()
                self.face_config.confidence_threshold = float(self.face_thresh_spin.value())

            # Palette and contrast
            if hasattr(self, "palette_combo"):
                self.agent_config.palette_index = self.palette_combo.currentIndex()
            if hasattr(self, "contrast_slider"):
                self.agent_config.contrast_boost = self.contrast_slider.value()

            # Profile face-link: write back the edited face_name for the selected profile
            if self.agent and hasattr(self, "profile_combo") and self.profile_combo.count():
                name = self.profile_combo.currentText()
                profile = self.agent.profiles.get(name)
                if profile:
                    new_face = self.profile_face_link.text().strip() or None
                    if new_face != profile.face_name:
                        profile.face_name = new_face
                        self.agent.profiles.save(profile)
                        logger.info(
                            f"Updated face_name for {profile.name} -> {new_face}"
                        )

        # Sync changes back to AppConfig for persistence
        if self.app_config and self.agent_config:
            self.app_config.llm_base_url = self.agent_config.llm_base_url
            self.app_config.llm_api_key = self.agent_config.llm_api_key
            self.app_config.llm_model = self.agent_config.llm_model
            self.app_config.stt_host = self.agent_config.stt.host
            self.app_config.stt_port = self.agent_config.stt.port
            self.app_config.tts_host = self.agent_config.tts.host
            self.app_config.tts_port = self.agent_config.tts.port
            self.app_config.tts_voice = getattr(
                self.agent_config, "tts_voice", self.app_config.tts_voice
            )
            self.app_config.db_host = self.agent_config.profile_db_host or ""
            self.app_config.db_port = self.agent_config.profile_db_port or 3306
            self.app_config.db_name = self.agent_config.profile_db_name or ""
            self.app_config.db_user = self.agent_config.profile_db_user or ""
            self.app_config.db_password = self.agent_config.profile_db_password or ""
            self.app_config.assistant_name = self.agent_config.assistant_name
            self.app_config.silence_timeout = self.agent_config.silence_timeout
            self.app_config.stt_sensitivity = self.agent_config.silence_threshold
            self.app_config.wake_word_enabled = self.agent_config.wake_word_enabled
            self.app_config.wake_word_threshold = self.agent_config.wake_word_threshold
            self.app_config.palette_index = self.agent_config.palette_index
            self.app_config.contrast_boost = self.agent_config.contrast_boost

        # Persist to disk
        if self.app_config:
            try:
                self.app_config.save()
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save config file:\n{e}")

        QMessageBox.information(
            self,
            "Settings Saved",
            "Settings have been saved.\n\n"
            "Some changes require restarting the application to take effect.",
        )
        self.accept()
