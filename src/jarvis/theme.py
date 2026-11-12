"""Central theme tokens for the Jarvis chrome.

All QSS in the app chrome is generated from these tokens so accent color,
surface treatment and state colors change in one place (audit G-B7).
The HUD canvases keep their own HSV palette (they animate hue continuously);
this module governs bars, buttons, panels, inputs and status pills.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Theme:
    # Accent (default cyan, matches the HUD's default hue)
    accent: str = "#00c8ff"
    accent_dim: str = "rgba(0, 200, 255, 80)"
    accent_line: str = "rgba(0, 200, 255, 110)"

    # Surfaces
    surface: str = "rgba(10, 14, 20, 215)"
    surface_soft: str = "rgba(10, 14, 20, 140)"
    border: str = "rgba(120, 160, 200, 40)"
    inner_highlight: str = "rgba(255, 255, 255, 14)"

    # Text
    text: str = "rgba(230, 240, 250, 235)"
    text_dim: str = "rgba(160, 180, 200, 190)"
    text_user: str = "rgba(0, 255, 160, 220)"
    text_activity: str = "rgba(255, 190, 60, 200)"

    # Status colors
    ok: str = "#3ddc84"
    warn: str = "#ffc857"
    error: str = "#ff5f56"

    # State label colors (standby / listening / thinking / speaking)
    state_idle: str = "rgba(0, 200, 255, 210)"
    state_listening: str = "rgba(61, 220, 132, 230)"
    state_thinking: str = "rgba(255, 200, 87, 230)"
    state_speaking: str = "rgba(255, 150, 60, 235)"

    radius: int = 10


def accent_for_hue(hue: int) -> tuple[str, str, str]:
    """(accent, accent_dim, accent_line) HSV->hex for a profile hue (0-359)."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360.0, 0.85, 1.0)
    hexc = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    def rgba(a: int) -> str:
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})"

    return hexc, rgba(80), rgba(110)


def themed(t: Theme) -> Theme:
    """Return a copy with accent/accent_dim/accent_line derived from `accent`."""
    a = t.accent.lstrip("#")
    if len(a) == 6:
        r, g, b = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
        return replace(
            t,
            accent_dim=f"rgba({r}, {g}, {b}, 80)",
            accent_line=f"rgba({r}, {g}, {b}, 110)",
        )
    return t


# ── Convenience: expose the builders as Theme methods too ──
def _bind_methods():
    Theme.bottom_bar_qss = lambda self: bottom_bar_qss(self)
    Theme.button_qss = lambda self, **kw: button_qss(self, **kw)
    Theme.input_qss = lambda self: input_qss(self)
    Theme.panels_qss = lambda self: panels_qss(self)
    Theme.status_qss = lambda self, color: status_qss(self, color)


_bind_methods()

DEFAULT = Theme()


# ── QSS builders ─────────────────────────────────────────────────────

def bottom_bar_qss(t: Theme) -> str:
    return f"""
    QFrame#bottomBar {{
        background: {t.surface};
        border-top: 1px solid {t.accent_line};
        border-radius: {t.radius}px;
    }}
    """


def button_qss(t: Theme, *, kind: str = "accent", size: int = 40) -> str:
    """Circular-ish tool button. kind: accent | danger | ghost."""
    r = size // 2
    if kind == "danger":
        base, hover, border = "rgba(180, 40, 40, 70)", "rgba(220, 60, 60, 130)", "rgba(255, 120, 120, 140)"
    elif kind == "ghost":
        base, hover, border = "rgba(80, 90, 100, 60)", "rgba(120, 130, 140, 80)", "rgba(150, 160, 170, 100)"
    else:
        base, hover, border = t.accent_dim, "rgba(0, 200, 255, 150)", t.accent_line
    return f"""
    QPushButton {{
        background: {base};
        border: 2px solid {border};
        border-radius: {r}px;
        padding: 0;
    }}
    QPushButton:hover {{ background: {hover}; }}
    QPushButton:pressed {{ background: {hover}; border-color: {t.accent}; }}
    QPushButton:checked {{
        background: rgba(0, 255, 140, 90);
        border-color: {t.state_listening};
    }}
    """


def input_qss(t: Theme) -> str:
    return f"""
    QLineEdit {{
        background: {t.surface_soft};
        border: 1px solid {t.border};
        border-radius: {max(6, t.radius - 4)}px;
        padding: 7px 12px;
        color: {t.text};
        selection-background-color: {t.accent};
    }}
    QLineEdit:focus {{ border-color: {t.accent}; }}
    """


def panels_qss(t: Theme) -> str:
    return f"""
    QFrame#panels {{
        background: {t.surface};
        border: 1px solid {t.border};
        border-radius: {t.radius}px;
    }}
    """


def status_qss(t: Theme, color: str) -> str:
    return f"color: {color}; font-size: 13px;"
