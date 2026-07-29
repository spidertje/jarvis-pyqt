#!/usr/bin/env python3
"""
Jarvis Desktop — Main entry point.

Phase 1: Arc Reactor HUD overlay with state machine.
Phase 2: TTS/STT integration (piper + whisper).
Phase 3: Chat with Hermes backend.
Phase 4: Face recognition (OpenCV LBPH + MariaDB).
Phase 5: Profile switching + polish.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt6.QtCore import Qt

from jarvis.hud_overlay import HUDOverlay
from jarvis.state import JarvisState


def create_tray_icon():
    """Create a simple cyan circle icon for the system tray."""
    pm = QPixmap(24, 24)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(0, 200, 255))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 20, 20)
    p.end()
    return pm


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Jarvis")
    app.setOrganizationName("Jarvis")

    # Create HUD overlay
    hud = HUDOverlay()
    hud.setWindowTitle("Jarvis HUD")
    hud.show()

    print("Jarvis HUD running. Press Ctrl+C to exit.")
    print(f"State: {hud.state.label} | Activity: {hud.activity:.2f}")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
