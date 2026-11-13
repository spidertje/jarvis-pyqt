"""Inline SVG icon set (Lucide-style paths, MIT) rendered to QIcon/QPixmap.

Emoji-as-icons misalign, render inconsistently across platforms and can't
be tinted; these paths stroke in the current accent color so buttons match
the HUD. No external files needed.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QByteArray, Qt

_ICONS: dict[str, str] = {
    "mic": (
        '<path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>'
        '<path d="M19 10v1a7 7 0 0 1-14 0v-1"/>'
        '<line x1="12" y1="18" x2="12" y2="22"/>'
    ),
    "mic_off": (
        '<line x1="3" y1="3" x2="21" y2="21"/>'
        '<path d="M9 9v2a3 3 0 0 0 5.1 2.1"/>'
        '<path d="M15 9.3V5a3 3 0 0 0-5.9-.7"/>'
        '<path d="M19 10v1a7 7 0 0 1-10.6 6"/>'
        '<path d="M5 10v1a7 7 0 0 0 1 3.7"/>'
        '<line x1="12" y1="18" x2="12" y2="22"/>'
    ),
    "stop": (
        '<rect x="6" y="6" width="12" height="12" rx="2"/>'
    ),
    "send": (
        '<line x1="22" y1="2" x2="11" y2="13"/>'
        '<polygon points="22 2 15 22 11 13 2 9 22 2"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83'
        'l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0'
        'l-.01-.01a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83'
        'l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.01'
        'a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83'
        'l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0'
        'l-.01.01a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83'
        'l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.01'
        'a1.65 1.65 0 0 0-1.51 1z"/>'
    ),
    "eye": (
        '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "eye_off": (
        '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>'
        '<path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>'
        '<line x1="1" y1="1" x2="23" y2="23"/>'
    ),
    "power": (
        '<path d="M18.36 6.64a9 9 0 1 1-12.73 0"/>'
        '<line x1="12" y1="2" x2="12" y2="12"/>'
    ),
    "chat": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    ),
}


def svg(name: str, color: str = "#e6f0fa", stroke_w: float = 2.0) -> str:
    paths = _ICONS.get(name)
    if paths is None:
        raise KeyError(f"unknown icon: {name}")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke_w}" stroke-linecap="round" '
        f'stroke-linejoin="round">{paths}</svg>'
    )


def icon(name: str, color: str = "#e6f0fa", size: int = 22) -> QIcon:
    r = QSvgRenderer(QByteArray(svg(name, color).encode()))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    from PyQt6.QtGui import QPainter

    p = QPainter(pm)
    r.render(p)
    p.end()
    return QIcon(pm)


def known_icons() -> list[str]:
    return sorted(_ICONS)
