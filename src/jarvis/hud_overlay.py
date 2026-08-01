"""
Jarvis HUD Overlay — PyQt6 widget with Arc Reactor animation.

Frameless, translucent window that renders:
- Dark vignette backdrop
- Outer glow rings
- Arc Reactor (center) with concentric rings
- Partial arcs with rotation
- Wave rings (emit on state changes)
- Voice level bars
- Floating particles with connections
- State label
- Face recognition overlay (name + confidence)

Animated by QTimer at 60fps. Activity level drives color (cyan idle → amber speaking)
and animation speed.
"""

import math
import random
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QRadialGradient, QFont,
    QPainterPath, QPolygonF,
)

from .state import JarvisState


# ── Configuration ──────────────────────────────────────────────────────
PARTICLE_COUNT = 60
BAR_COUNT = 40
RING_RADII = [120, 200, 280]
PARTICLE_SPEED_BASE = 0.6
BAR_SMOOTHING = 0.18
WAVE_LIFETIME = 2.5
WAVE_SPEED = 0.8
CONN_DISTANCE = 100
CONN_DISTANCE_ACTIVE = 140


class HUDOverlay(QWidget):
    """Arc Reactor HUD widget — frameless, translucent, animated."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Normal window that stays on top for visibility
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("background-color: #000000;")
        self.resize(800, 600)

        # State
        self._state = JarvisState.IDLE
        self._activity = 0.0  # 0.0–1.0, smoothed
        self._angle = 0.0
        self._pulse = 0.0

        # Geometry center
        self._cx = 400
        self._cy = 300

        # Particles
        self._particles = self._init_particles()

        # Voice bars
        self._bar_heights = [2.0] * BAR_COUNT

        # Wave rings (expand outward, fade)
        self._wave_rings = []

        # Face recognition overlay
        self._face_name = ""
        self._face_confidence = 0.0
        self._face_timer = 0  # fade out timer

        # Active profile
        self._profile_name: str = ""
        self._profile_hue: int = 182  # Default cyan

        # Contrast boost factor (1.0 = no change)
        self._contrast_factor: float = 1.0

        # Timer: 60fps
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // 60)

    # ── State management ─────────────────────────────────────────────
    def set_palette_hue(self, hue: int):
        """Set a fixed hue for the HUD palette (overrides automatic color)."""
        self._palette_hue = hue
        self.update()

    def clear_palette_hue(self):
        """Clear custom palette hue and revert to automatic color selection."""
        if hasattr(self, '_palette_hue'):
            delattr(self, '_palette_hue')
            self.update()

    def set_contrast_factor(self, factor: float):
        """Set contrast factor for HUD saturation (1.0 = default)."""
        self._contrast_factor = max(0.0, factor)
        self.update()

    def set_state(self, state: JarvisState):
        """Set the current HUD state. Triggers wave rings on SPEAKING."""
        self._state = state
        if state == JarvisState.SPEAKING:
            # Emit 3 concentric wave rings
            for i in range(3):
                self._wave_rings.append({
                    "phase": i * 0.6,
                    "speed": WAVE_SPEED,
                })

    def set_activity(self, value: float):
        """Override activity level directly (e.g. from voice amplitude)."""
        self._activity = max(0.0, min(1.0, value))

    @property
    def state(self) -> JarvisState:
        return self._state

    @property
    def activity(self) -> float:
        return self._activity

    # ── Animation tick ───────────────────────────────────────────────
    def _tick(self):
        # Smooth activity toward state target
        target = self._state.target_activity
        lerp = 0.15 if self._activity < target else 0.08
        self._activity += (target - self._activity) * lerp

        # Update angle/pulse
        speed = 1.0 + self._activity * 3.0
        self._angle += 0.02 * speed
        self._pulse = (self._pulse + 0.06) % (math.pi * 2)

        # Update particles
        self._update_particles(speed)

        # Update voice bars
        self._update_bars(speed)

        # Update wave rings
        self._wave_rings = [w for w in self._wave_rings if w["phase"] < WAVE_LIFETIME]
        for w in self._wave_rings:
            w["phase"] += w["speed"] * 0.04

        self.update()

    def _update_particles(self, speed):
        for p in self._particles:
            p["x"] += p["sx"] * speed
            p["y"] += p["sy"] * speed
            # Wrap around
            if p["x"] < 0:
                p["x"] = 800
            elif p["x"] > 800:
                p["x"] = 0
            if p["y"] < 0:
                p["y"] = 600
            elif p["y"] > 600:
                p["y"] = 0

    def _update_bars(self, speed):
        for i in range(BAR_COUNT):
            target = 2.0
            if self._state == JarvisState.LISTENING:
                target = 6.0 + random.random() * 4.0 * self._activity
            elif self._state == JarvisState.SPEAKING:
                target = 10.0 + random.random() * 5.5 * self._activity
            self._bar_heights[i] += (target - self._bar_heights[i]) * BAR_SMOOTHING

    # ── Particle initialization ──────────────────────────────────────
    @staticmethod
    def _init_particles():
        particles = []
        for _ in range(PARTICLE_COUNT):
            particles.append({
                "x": random.uniform(0, 800),
                "y": random.uniform(0, 600),
                "size": random.uniform(1, 3),
                "sx": random.uniform(-PARTICLE_SPEED_BASE, PARTICLE_SPEED_BASE),
                "sy": random.uniform(-PARTICLE_SPEED_BASE, PARTICLE_SPEED_BASE),
                "op": random.uniform(0.3, 0.8),
            })
        return particles

    # ── Color helper ─────────────────────────────────────────────────
    @staticmethod
    def _color(h: int, s: int, l: int, a: int) -> QColor:
        c = QColor()
        c.setHsl(h, s, l)
        c.setAlpha(max(0, min(255, int(a))))
        return c

    # ── Paint ────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self._cx, self._cy
        act = self._activity

        # Color: profile hue overrides default cyan when active
        if hasattr(self, '_palette_hue') and self._palette_hue is not None:
            hue = self._palette_hue
            sat = 110
        elif self._profile_name:
            hue = self._profile_hue
            sat = 110
        elif act < 0.4:
            hue = 182  # cyan
        elif act > 0.7:
            hue = 45   # amber
        else:
            hue = 120  # green (transition)
        sat = int((110 + int((1 - act) * 10)) * self._contrast_factor)
        sat = min(255, max(0, sat))

        # 1. Dark vignette backdrop
        self._draw_vignette(p, cx, cy, act, hue, sat)

        # 2. Outer glow rings
        self._draw_glow_rings(p, cx, cy, act, hue, sat)

        # 3. Arc Reactor (center)
        self._draw_reactor(p, cx, cy, act, hue, sat)

        # 4. Partial arcs (rotating)
        self._draw_arcs(p, cx, cy, act, hue, sat)

        # 5. Wave rings
        self._draw_wave_rings(p, cx, cy, act, hue, sat)

        # 6. Voice bars
        self._draw_voice_bars(p, cx, cy, act, hue, sat)

        # 7. Particles
        self._draw_particles(p, act, hue, sat)

        # 8. Particle connections
        self._draw_connections(p, act, hue, sat)

        # 9. Face recognition overlay
        self._draw_face_overlay(p, cx, cy, act, hue, sat)

        # 10. State label
        self._draw_state_label(p, cx, cy, act, hue, sat)

        p.end()

    def _draw_vignette(self, p, cx, cy, act, hue, sat):
        """Dark radial gradient background for contrast."""
        vg = QRadialGradient(cx, cy, 400)
        vg.setColorAt(0, self._color(0, 0, 0, int(60 + act * 80)))
        vg.setColorAt(0.6, self._color(0, 0, 0, int(30 + act * 50)))
        vg.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), vg)

    def _draw_glow_rings(self, p, cx, cy, act, hue, sat):
        """Three concentric glow rings with subtle pulse."""
        pen = QPen(
            self._color(hue, sat, 60, int(100 + act * 100)),
            2 + act * 3,
        )
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        for r in RING_RADII:
            s = 0.92 + 0.08 * math.sin(self._pulse + r * 0.01)
            p.drawEllipse(QPointF(cx, cy), int(r * s), int(r * s))

    def _draw_reactor(self, p, cx, cy, act, hue, sat):
        """Arc Reactor core — central glow + concentric rings."""
        # Central glow
        cg = QRadialGradient(cx, cy, 35)
        cg.setColorAt(0, self._color(hue, sat, 80, int(200 + 55 * act)))
        cg.setColorAt(0.5, self._color(hue, sat, 60, int(100 + 80 * act)))
        cg.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(cg)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), 35, 35)

        # Inner rings
        for r in [22, 14, 6]:
            a = 180 + int(75 * act)
            lw = 2.5 if r == 22 else (2.0 if r == 14 else 1.5)
            pen = QPen(self._color(hue, sat, 80, a), lw)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # Speaking flash
        if self._state == JarvisState.SPEAKING:
            fa = int(100 + 155 * (0.5 + 0.5 * math.sin(self._pulse * 4)))
            fp = QPen(self._color(hue, sat, 95, fa), 3)
            p.setPen(fp)
            s = 1 + 0.08 * math.sin(self._pulse * 3)
            p.drawEllipse(QPointF(cx, cy), int(28 * s), int(28 * s))

    def _draw_arcs(self, p, cx, cy, act, hue, sat):
        """Three rotating partial arcs with glow fill."""
        span = math.pi * 1.5
        for i, radius in enumerate(RING_RADII):
            a = self._angle + i * 0.5
            lw = 3.0 + act * 4.0 + i * 1.0
            pen = QPen(
                self._color(hue, sat, 55 + i * 8, int(140 + act * 115)),
                lw,
            )
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)

            r = radius
            start_deg = int(math.degrees(a) * 16)
            span_deg = int(math.degrees(span) * 16)
            p.drawArc(
                int(cx - r), int(cy - r),
                int(r * 2), int(r * 2),
                start_deg, span_deg,
            )

            # Glow fill when active
            if act > 0.2:
                fill = self._color(hue, sat, 55, int(35 + act * 50))
                p.setBrush(fill)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPie(
                    int(cx - r), int(cy - r),
                    int(r * 2), int(r * 2),
                    start_deg, span_deg,
                )

    def _draw_wave_rings(self, p, cx, cy, act, hue, sat):
        """Expanding wave rings from center."""
        for w in self._wave_rings:
            scale = 0.5 + w["phase"] * 1.2
            wr = 70 * scale
            wa = int(150 - w["phase"] * 90)
            pen = QPen(
                self._color(hue, sat, 65, max(0, wa)),
                2.5 - w["phase"] * 0.6,
            )
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), int(wr), int(wr))

    def _draw_voice_bars(self, p, cx, cy, act, hue, sat):
        """Vertical voice level bars below the reactor."""
        bar_w, gap = 5, 3
        total_w = BAR_COUNT * (bar_w + gap)
        start_x = cx - total_w / 2
        base_y = cy + 70

        for i, h in enumerate(self._bar_heights):
            if h < 1:
                continue
            x = start_x + i * (bar_w + gap)
            ba = int(150 + 105 * act)
            bcol = self._color(
                hue, sat,
                60 + int(h / 60 * 30),
                ba,
            )
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bcol)
            p.drawRoundedRect(int(x), int(base_y - h), bar_w, int(h), 2, 2)

    def _draw_particles(self, p, act, hue, sat):
        """Floating particles with activity-modulated opacity and size."""
        part_scale = 0.5 + act * 0.5
        for pt in self._particles:
            pa = int(pt["op"] * 255 * (0.5 + act * 0.5))
            sz = pt["size"] * (1 + act * 0.5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self._color(hue, sat, 75, pa))
            p.drawEllipse(QPointF(pt["x"], pt["y"]), int(sz), int(sz))

    def _draw_connections(self, p, act, hue, sat):
        """Lines between nearby particles."""
        conn_a = int(15 + act * 30)
        dist = CONN_DISTANCE + act * (CONN_DISTANCE_ACTIVE - CONN_DISTANCE)
        pen = QPen(self._color(hue, sat, 55, conn_a), 0.8)
        p.setPen(pen)

        for i, a_p in enumerate(self._particles):
            for b_p in self._particles[i + 1:]:
                dx = a_p["x"] - b_p["x"]
                dy = a_p["y"] - b_p["y"]
                d = math.sqrt(dx * dx + dy * dy)
                if d < dist:
                    p.drawLine(
                        QPointF(a_p["x"], a_p["y"]),
                        QPointF(b_p["x"], b_p["y"]),
                    )

    def _draw_state_label(self, p, cx, cy, act, hue, sat):
        """State text below the voice bars."""
        labels = ["STANDBY", "LISTENING", "THINKING", "SPEAKING"]
        pen = QPen(self._color(hue, sat, 80, 220), 1)
        p.setPen(pen)
        fnt = p.font()
        fnt.setPointSize(11)
        fnt.setBold(True)
        p.setFont(fnt)

        # Show profile name above state label if active
        if self._profile_name:
            profile_text = f"👤 {self._profile_name}"
            profile_pen = QPen(self._color(hue, sat, 90, 200), 1)
            p.setPen(profile_pen)
            p.drawText(int(cx - 40), int(cy + 95), profile_text)

        p.drawText(int(cx - 50), int(cy + 110), labels[self._state])

    def _draw_face_overlay(self, p, cx, cy, act, hue, sat):
        """Draw face recognition overlay (top-right corner)."""
        if not self._face_name:
            return

        # Fade out after 3 seconds
        self._face_timer += 0.016  # ~60fps
        if self._face_timer > 3.0:
            self.clear_face()
            return

        # Opacity: full for 2s, fade over next 1s
        if self._face_timer < 2.0:
            alpha = 255
        else:
            alpha = max(0, int(255 * (3.0 - self._face_timer)))

        # Background pill
        text = f"👤 {self._face_name}"
        if self._face_confidence > 0:
            text += f" ({self._face_confidence:.0f}%)"

        fnt = p.font()
        fnt.setPointSize(12)
        fnt.setBold(True)
        p.setFont(fnt)

        # Measure text
        metrics = p.fontMetrics()
        text_w = metrics.horizontalAdvance(text)
        pill_h = 28
        pill_x = int(cx + 280)
        pill_y = 20
        pill_r = 14

        # Glow effect
        glow_pen = QPen(
            self._color(182, 85, 70, int(alpha * 0.4)),
            4,
        )
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(glow_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(pill_x - 8, pill_y - 4, text_w + 16, pill_h + 8, pill_r + 4, pill_r + 4)

        # Pill background
        pill_bg = self._color(0, 0, 0, int(alpha * 0.7))
        p.setBrush(pill_bg)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(pill_x, pill_y, text_w, pill_h, pill_r, pill_r)

        # Pill border
        border_color = self._color(182, 85, 60, int(alpha * 0.8))
        p.setPen(QPen(border_color, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(pill_x, pill_y, text_w, pill_h, pill_r, pill_r)

        # Text
        text_color = self._color(182, 85, 80, alpha)
        p.setPen(QPen(text_color, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawText(pill_x + 10, pill_y + pill_h - 6, text)

    def set_face_detected(self, name: str, confidence: float = 0.0):
        """Set the currently recognized face for overlay display."""
        self._face_name = name
        self._face_confidence = confidence

    def clear_face(self):
        """Clear face recognition overlay."""
        self._face_name = ""
        self._face_confidence = 0.0

    def set_profile(self, name: str, hue: int = 182):
        """Set active profile — updates name display and reactor accent."""
        self._profile_name = name
        self._profile_hue = hue

    def clear_profile(self):
        """Clear active profile."""
        self._profile_name = ""
        self._profile_hue = 182
        self._palette_hue = None  # Optional fixed hue from appearance settings

    def set_palette_hue(self, hue: int):
        """Set a fixed hue for the HUD palette (overrides automatic color)."""
        self._palette_hue = hue
        self.update()

    def clear_palette_hue(self):
        """Clear custom palette hue and revert to automatic color selection."""
        if hasattr(self, '_palette_hue'):
            delattr(self, '_palette_hue')
        self.update()
