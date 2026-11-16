"""
Solid Face HUD — emotive SOLID (non-particle) avatars, PyQt6 port of the three
validated web canvases:

  smiley  -> solid-avatar   (/opt/data/smiley-avatar/index.html)
  goddess -> goddess-avatar (/opt/data/goddess-avatar/index.html)
  bender  -> bender-style   (/opt/data/bender-avatar/index.html)

Implements the same public interface as HUDOverlay / SmokeFaceHUD so main.py and
the agent swap frontends freely:

  set_state / set_activity / set_voice_level / set_palette_hue /
  clear_palette_hue / set_contrast_factor / set_assistant_name /
  set_face_detected / clear_face / set_profile / clear_profile
  set_emotion(name) / set_level_source(cb) / emotion_changed signal

Life engine (all three, shared): spring dynamics with overshoot (not exponential
lerp), squash & stretch keyed to the emotion switch and jaw, laugh body-bob
(bounded spring), idle sway/breathing, tremble when turb>0.85, blinks,
eyebrow micro-raises, eye saccades, tear tracks while sad, per-emotion face
tone (hue/sat/lit are emotion params — the face colour itself emotes).
Solid rendering: no particles, no additive blending — pure vector fills.
"""

from __future__ import annotations

import math
import random
import time

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QComboBox, QWidget

from .state import JarvisState

# ── noise (organic idle motion) ─────────────────────────────────────────
def _mulberry32(a):
    a &= 0xFFFFFFFF
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt

_RND = _mulberry32(20261116)
_PERM = [0] * 512
def _init_perm():
    p = list(range(256))
    for i in range(255, 0, -1):
        j = int(_RND() * (i + 1))
        p[i], p[j] = p[j], p[i]
    for i in range(512):
        _PERM[i] = p[i & 255]
_init_perm()

def _grad2(h, x, y):
    return (x if h & 1 == 0 else -x) + (y if h & 2 == 0 else -y)

def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def noise2(x, y):
    X = math.floor(x) & 255
    Y = math.floor(y) & 255
    x -= math.floor(x)
    y -= math.floor(y)
    u, v = _fade(x), _fade(y)
    A = _PERM[X] + Y
    B = _PERM[X + 1] + Y
    return 1.35 * (
        (1 - v) * ((1 - u) * _grad2(_PERM[A], x, y) + u * _grad2(_PERM[B], x - 1, y))
        + v * ((1 - u) * _grad2(_PERM[A + 1], x, y - 1) + u * _grad2(_PERM[B + 1], x - 1, y - 1))
    )

def fbm(x, y):
    return 0.6 * noise2(x, y) + 0.3 * noise2(x * 2.1 + 11, y * 2.1) + 0.1 * noise2(x * 4.7, y * 4.7 + 5)

def hsla(h, s, l, a):
    """QColor from h(deg), s/l/a as 0..1 floats."""
    c = QColor.fromHslF((h % 360) / 360.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, l)))
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


# ── per-figure emotion tables (identical to the web files) ──────────────
EMOTIONS = {
    "smiley": {
        "neutral":   dict(mouthCurve=0.30, mouthWidth=0.50, mouthOpen=0.02, browAngle=-0.03, browHeight=0.02, browCurve=0.42, eyeOpen=0.60, pupilY=0.00, cheekLift=0.00, turb=0.30, hue=46, sat=0.88, lit=0.55, spd=1.0, tear=0, bounce=0, tilt=0.00, droop=0.0),
        "happy":     dict(mouthCurve=0.95, mouthWidth=0.80, mouthOpen=0.22, browAngle=-0.10, browHeight=0.08, browCurve=0.62, eyeOpen=0.30, pupilY=-0.02, cheekLift=0.40, turb=0.45, hue=48, sat=0.92, lit=0.58, spd=1.3, tear=0, bounce=0, tilt=0.02, droop=-0.02),
        "laugh":     dict(mouthCurve=1.00, mouthWidth=0.92, mouthOpen=0.85, browAngle=-0.25, browHeight=0.16, browCurve=0.66, eyeOpen=0.06, pupilY=-0.04, cheekLift=0.55, turb=0.70, hue=50, sat=0.94, lit=0.58, spd=1.6, tear=0, bounce=0, tilt=0.04, droop=0.0),
        "sad":       dict(mouthCurve=-0.80, mouthWidth=0.42, mouthOpen=0.10, browAngle=0.62, browHeight=-0.06, browCurve=0.10, eyeOpen=0.55, pupilY=0.07, cheekLift=0.00, turb=0.18, hue=214, sat=0.42, lit=0.48, spd=0.6, tear=1, bounce=0, tilt=0.05, droop=1.0),
        "angry":     dict(mouthCurve=-0.60, mouthWidth=0.62, mouthOpen=0.16, browAngle=-0.75, browHeight=-0.12, browCurve=0.05, eyeOpen=0.45, pupilY=0.00, cheekLift=0.00, turb=1.00, hue=8, sat=0.82, lit=0.48, spd=1.9, tear=0, bounce=0, tilt=-0.02, droop=0.1),
        "surprised": dict(mouthCurve=0.05, mouthWidth=0.30, mouthOpen=0.82, browAngle=0.15, browHeight=0.38, browCurve=0.45, eyeOpen=1.00, pupilY=-0.02, cheekLift=0.00, turb=0.60, hue=286, sat=0.66, lit=0.52, spd=1.4, tear=0, bounce=0, tilt=0.00, droop=-0.03),
        "fear":      dict(mouthCurve=-0.35, mouthWidth=0.40, mouthOpen=0.55, browAngle=0.55, browHeight=0.32, browCurve=0.20, eyeOpen=1.08, pupilY=-0.06, cheekLift=0.00, turb=1.10, hue=96, sat=0.30, lit=0.50, spd=2.3, tear=0, bounce=0, tilt=0.00, droop=0.15),
        "calm":      dict(mouthCurve=0.42, mouthWidth=0.48, mouthOpen=0.00, browAngle=0.02, browHeight=0.04, browCurve=0.50, eyeOpen=0.14, pupilY=0.00, cheekLift=0.12, turb=0.10, hue=178, sat=0.58, lit=0.48, spd=0.4, tear=0, bounce=0, tilt=0.02, droop=0.2),
    },
    "goddess": {
        "neutral":   dict(mouthCurve=0.30, mouthWidth=0.46, mouthOpen=0.02, browAngle=-0.03, browHeight=0.02, browCurve=0.55, eyeOpen=0.60, pupilY=0.00, cheekLift=0.00, turb=0.25, hue=26, sat=0.42, lit=0.70, spd=1.0, tear=0, bounce=0, tilt=0.00, droop=0.0),
        "happy":     dict(mouthCurve=0.95, mouthWidth=0.72, mouthOpen=0.20, browAngle=-0.10, browHeight=0.08, browCurve=0.68, eyeOpen=0.30, pupilY=-0.02, cheekLift=0.42, turb=0.40, hue=22, sat=0.50, lit=0.71, spd=1.3, tear=0, bounce=0, tilt=0.02, droop=-0.02),
        "laugh":     dict(mouthCurve=1.00, mouthWidth=0.84, mouthOpen=0.80, browAngle=-0.22, browHeight=0.15, browCurve=0.72, eyeOpen=0.06, pupilY=-0.04, cheekLift=0.55, turb=0.62, hue=20, sat=0.52, lit=0.72, spd=1.6, tear=0, bounce=0, tilt=0.04, droop=0.0),
        "sad":       dict(mouthCurve=-0.80, mouthWidth=0.40, mouthOpen=0.10, browAngle=0.62, browHeight=-0.06, browCurve=0.18, eyeOpen=0.55, pupilY=0.07, cheekLift=0.00, turb=0.16, hue=24, sat=0.24, lit=0.62, spd=0.6, tear=1, bounce=0, tilt=0.05, droop=1.0),
        "angry":     dict(mouthCurve=-0.60, mouthWidth=0.56, mouthOpen=0.14, browAngle=-0.75, browHeight=-0.12, browCurve=0.10, eyeOpen=0.45, pupilY=0.00, cheekLift=0.00, turb=0.95, hue=10, sat=0.58, lit=0.64, spd=1.8, tear=0, bounce=0, tilt=-0.02, droop=0.1),
        "surprised": dict(mouthCurve=0.05, mouthWidth=0.28, mouthOpen=0.72, browAngle=0.15, browHeight=0.38, browCurve=0.60, eyeOpen=1.00, pupilY=-0.02, cheekLift=0.00, turb=0.55, hue=26, sat=0.38, lit=0.71, spd=1.4, tear=0, bounce=0, tilt=0.00, droop=-0.03),
        "fear":      dict(mouthCurve=-0.35, mouthWidth=0.36, mouthOpen=0.46, browAngle=0.55, browHeight=0.32, browCurve=0.30, eyeOpen=1.06, pupilY=-0.06, cheekLift=0.00, turb=1.05, hue=30, sat=0.20, lit=0.66, spd=2.2, tear=0, bounce=0, tilt=0.00, droop=0.15),
        "calm":      dict(mouthCurve=0.42, mouthWidth=0.44, mouthOpen=0.00, browAngle=0.02, browHeight=0.04, browCurve=0.58, eyeOpen=0.14, pupilY=0.00, cheekLift=0.12, turb=0.08, hue=26, sat=0.40, lit=0.68, spd=0.4, tear=0, bounce=0, tilt=0.02, droop=0.15),
    },
    "bender": {
        "neutral":   dict(mouthCurve=0.10, mouthWidth=0.50, mouthOpen=0.16, browAngle=0.00, browHeight=0.02, browCurve=0.15, eyeOpen=0.85, pupilY=0.00, cheekLift=0.35, turb=0.25, hue=36, sat=0.14, lit=0.62, spd=1.0, tear=0, bounce=0, tilt=0.00, droop=0.0),
        "happy":     dict(mouthCurve=0.85, mouthWidth=0.85, mouthOpen=0.34, browAngle=-0.15, browHeight=0.08, browCurve=0.30, eyeOpen=0.45, pupilY=-0.02, cheekLift=0.75, turb=0.40, hue=48, sat=0.22, lit=0.66, spd=1.3, tear=0, bounce=0, tilt=0.02, droop=-0.05),
        "laugh":     dict(mouthCurve=1.00, mouthWidth=0.95, mouthOpen=0.72, browAngle=-0.30, browHeight=0.15, browCurve=0.35, eyeOpen=0.12, pupilY=-0.04, cheekLift=0.90, turb=0.60, hue=50, sat=0.24, lit=0.68, spd=1.6, tear=0, bounce=0, tilt=0.04, droop=-0.1),
        "sad":       dict(mouthCurve=-0.80, mouthWidth=0.40, mouthOpen=0.12, browAngle=0.65, browHeight=-0.06, browCurve=0.05, eyeOpen=0.50, pupilY=0.07, cheekLift=0.10, turb=0.15, hue=214, sat=0.18, lit=0.54, spd=0.6, tear=1, bounce=0, tilt=0.04, droop=1.0),
        "angry":     dict(mouthCurve=-0.60, mouthWidth=0.60, mouthOpen=0.20, browAngle=-0.80, browHeight=-0.12, browCurve=0.05, eyeOpen=0.72, pupilY=0.00, cheekLift=1.00, turb=0.95, hue=10, sat=0.42, lit=0.58, spd=1.8, tear=0, bounce=0, tilt=-0.02, droop=0.15),
        "surprised": dict(mouthCurve=0.00, mouthWidth=0.30, mouthOpen=0.68, browAngle=0.15, browHeight=0.36, browCurve=0.20, eyeOpen=1.00, pupilY=-0.02, cheekLift=0.55, turb=0.55, hue=200, sat=0.16, lit=0.66, spd=1.4, tear=0, bounce=0, tilt=0.00, droop=-0.25),
        "fear":      dict(mouthCurve=-0.30, mouthWidth=0.45, mouthOpen=0.52, browAngle=0.55, browHeight=0.30, browCurve=0.10, eyeOpen=1.06, pupilY=-0.06, cheekLift=0.40, turb=1.10, hue=96, sat=0.18, lit=0.60, spd=2.2, tear=0, bounce=0, tilt=0.00, droop=0.3),
        "calm":      dict(mouthCurve=0.35, mouthWidth=0.50, mouthOpen=0.10, browAngle=0.02, browHeight=0.03, browCurve=0.18, eyeOpen=0.25, pupilY=0.00, cheekLift=0.20, turb=0.08, hue=200, sat=0.15, lit=0.60, spd=0.4, tear=0, bounce=0, tilt=0.02, droop=0.15),
    },
}
EMOTION_ORDER = ["neutral", "happy", "laugh", "sad", "angry", "surprised", "fear", "calm"]
PKEYS = list(EMOTIONS["smiley"]["neutral"].keys())
STATE_EMOTION = {
    JarvisState.IDLE: "calm",
    JarvisState.LISTENING: "neutral",
    JarvisState.THINKING: "surprised",
    JarvisState.SPEAKING: "happy",
}
VISEMES = [
    (0.75, 0.55, 0.15), (0.25, 0.85, 0.30), (0.45, 0.30, 0.10),
    (0.90, 0.60, 0.20), (0.10, 0.60, 0.30),
]

W = H = 720.0
FIGURES = ("smiley", "goddess", "bender")


class SolidFaceHUD(QWidget):
    """Solid emotive avatar frontend. figure in {'smiley','goddess','bender'}."""

    emotion_changed = pyqtSignal(str)

    def __init__(self, figure: str = "smiley", parent=None):
        super().__init__(parent)
        assert figure in FIGURES, f"unknown figure {figure!r}"
        self.figure = figure

        self._state = JarvisState.IDLE
        self._activity = 0.0
        self._voice_level = 0.0
        self._assistant_name = "Jarvis"
        self._face_name = ""
        self._face_confidence = 0.0
        self._face_timer = 0
        self._profile_name = ""
        self._profile_hue = 182
        self._palette_hue: int | None = None
        self._contrast_factor = 1.0

        self._emotion_mode = "auto"
        self._auto_emotion = "calm"
        self.target = dict(EMOTIONS[figure]["calm"])
        self.P = dict(EMOTIONS[figure]["calm"])
        self.vel = {k: 0.0 for k in PKEYS}

        self._speak_level = 0.0
        self._speak_target = 0.0
        self._level_source = None
        self._vis_i = 0
        self._in_word = False
        self._word_t = 0.0

        self._t0 = time.monotonic()
        self._blink_at = self._next_blink()
        self._blink_left = 0.0
        self._micro_at = 3.0 + random.random() * 5.0
        self._micro_left = 0.0
        self._sac_at = 0.0
        self._sac_x = self._sac_y = self._sac_tx = self._sac_ty = 0.0

        self._bounce = 0.0
        self._laugh_b = 0.0
        self._laugh_v = 0.0
        self._last_sway_x = 0.0
        self._hoop_ang = 0.0
        self._hoop_vel = 0.0
        self._ant_lean = 0.0
        self._ant_vel = 0.0

        # geometry (720-space)
        self.CX, self.CY = W * 0.5, (H * 0.52 if figure == "smiley" else H * 0.50)
        if figure == "smiley":
            self.RX, self.RY = W * 0.200, W * 0.225
            self.EX, self.EY, self.EW = W * 0.084, self.CY - W * 0.060, W * 0.052
            self.MY0 = self.CY + W * 0.105
        elif figure == "goddess":
            self.RX, self.RY = W * 0.150, W * 0.205
            self.EX, self.EY, self.EW = W * 0.066, self.CY - W * 0.045, W * 0.046
            self.MY0 = self.CY + W * 0.100
        else:
            self.HW, self.HH = W * 0.148, W * 0.200
            self.ER, self.EX = W * 0.050, W * 0.060
            self.EY = self.CY - W * 0.078
            self.MY0 = self.CY + W * 0.098
        self.MY = self.MY0

        self._locks = self._build_locks() if figure == "goddess" else []

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

        self.emotion_combo = QComboBox(self)
        self.emotion_combo.addItem("Auto", "auto")
        for e in EMOTION_ORDER:
            self.emotion_combo.addItem(e.capitalize(), e)
        self.emotion_combo.setFixedHeight(26)
        self.emotion_combo.setStyleSheet(
            "QComboBox{background:rgba(0,0,0,140);color:#dfe7ef;border:1px solid "
            "rgba(170,170,150,90);border-radius:5px;padding:3px 8px;font-size:12px;}"
        )
        self.emotion_combo.currentIndexChanged.connect(self._on_combo_emotion)
        self.emotion_combo.move(10, 40)
        self.emotion_combo.setFixedWidth(110)

    # ── HUDOverlay-compatible interface ─────────────────────────────────
    def showEvent(self, event):
        self._timer.start(33)
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def set_palette_hue(self, hue: int):
        # solid avatars emote with their own per-emotion face tones; hue stored
        # but not applied (same sanctioned pattern as frontends that must keep
        # their own colour identity).
        self._palette_hue = hue

    def clear_palette_hue(self):
        self._palette_hue = None

    def set_contrast_factor(self, factor: float):
        self._contrast_factor = max(0.5, float(factor))

    def set_assistant_name(self, name: str):
        self._assistant_name = name

    def set_state(self, state: JarvisState):
        self._state = state
        if self._emotion_mode == "auto":
            emo = STATE_EMOTION.get(state, "neutral")
            if emo != self._auto_emotion:
                self._auto_emotion = emo
                self._set_target(emo)
                self.emotion_changed.emit(emo)

    def set_activity(self, value: float):
        self._activity = max(0.0, min(1.0, value))

    def set_voice_level(self, level: float):
        self._voice_level = max(0.0, min(1.0, level))

    def set_face_detected(self, name: str, confidence: float = 0.0):
        self._face_name, self._face_confidence, self._face_timer = name, confidence, 240

    def clear_face(self):
        self._face_name, self._face_timer = "", 0

    def set_profile(self, name: str, hue: int = 182):
        self._profile_name, self._profile_hue = name, hue

    def clear_profile(self):
        self._profile_name = ""
        self._profile_hue = 182

    # ── emotion-specific ────────────────────────────────────────────────
    def set_emotion(self, name: str):
        if name in EMOTIONS[self.figure]:
            self._emotion_mode = name
            self.emotion_combo.blockSignals(True)
            self.emotion_combo.setCurrentIndex(EMOTION_ORDER.index(name) + 1)
            self.emotion_combo.blockSignals(False)
            self._set_target(name)
            self.emotion_changed.emit(name)
        else:
            self._emotion_mode = "auto"
            self.emotion_combo.blockSignals(True)
            self.emotion_combo.setCurrentIndex(0)
            self.emotion_combo.blockSignals(False)
            self._set_target(self._auto_emotion)

    def set_level_source(self, cb):
        self._level_source = cb

    def _on_combo_emotion(self, idx: int):
        self.set_emotion("auto" if idx == 0 else EMOTION_ORDER[idx - 1])

    def _set_target(self, name: str):
        prev = self.target
        self.target = dict(EMOTIONS[self.figure][name])
        # squash impulse on emotion switches (anticipation dip -> springy pop)
        jump = (abs(self.target["mouthCurve"] - prev["mouthCurve"])
                + abs(self.target["eyeOpen"] - prev["eyeOpen"])
                + abs(self.target["mouthOpen"] - prev["mouthOpen"])
                + abs(self.target["browHeight"] - prev["browHeight"]) * 2)
        self.vel["bounce"] -= 0.045 * max(0.5, jump)

    # ── speech envelope (RMS-driven visemes; same contract as smoke) ───
    def _speech_tick(self, dt):
        if self._level_source is not None:
            try:
                lvl, playing = self._level_source()
            except Exception:
                lvl, playing = 0.0, False
            if playing:
                self._word_t -= dt
                if self._word_t <= 0:
                    if self._in_word:
                        self._in_word = False
                        self._word_t = random.uniform(0.06, 0.15)
                    else:
                        self._in_word = True
                        self._word_t = random.uniform(0.09, 0.31)
                        self._vis_i = random.randrange(len(VISEMES))
                if self._in_word and random.random() < 0.16:
                    self._vis_i = random.randrange(len(VISEMES))
            self._speak_target = lvl if playing and lvl > 0.02 else 0.0
        else:
            self._speak_target = self._activity * 0.5 if self._state == JarvisState.SPEAKING else 0.0
        self._speak_level += (self._speak_target - self._speak_level) * min(1.0, dt * 20.0)

    def _eff(self):
        P, mix = self.P, min(1.0, self._speak_level * 1.4)
        v = VISEMES[self._vis_i]
        e = dict(P)
        e["mouthOpen"] = P["mouthOpen"] * (1 - mix) + v[0] * mix + self._speak_level * 0.10
        e["mouthWidth"] = P["mouthWidth"] * (1 - mix) + v[1] * mix
        e["mouthCurve"] = P["mouthCurve"] * (1 - mix * 0.65) + v[2] * mix
        return e

    # ── micro-gestures ──────────────────────────────────────────────────
    def _next_blink(self):
        base = 2.0 if self.figure == "bender" else 2.3
        span = 3.6 if self.figure == "bender" else 4.2
        return time.monotonic() - self._t0 + random.uniform(base, base + span)

    def _blink(self, t_now):
        if t_now >= self._blink_at and self._blink_left <= 0:
            self._blink_left = 0.13
        if self._blink_left > 0:
            self._blink_left -= 1 / 30
            if self._blink_left < 0:
                self._blink_left = 0.0
            b = max(self._blink_left, 0.0)
            if self.figure == "bender":  # mechanical shutter: snap down, snap up
                return (0.13 - b) / 0.055 if b > 0.055 else b / 0.055
            return math.sin(math.pi * (1 - b / 0.13))
        return 0.0

    def _micro(self, t_now):
        if t_now >= self._micro_at and self._micro_left <= 0:
            self._micro_left = 0.26
        if self._micro_left > 0:
            self._micro_left -= 1 / 30
            if self._micro_left < 0:
                self._micro_left = 0.0
            return math.sin(math.pi * (1 - max(self._micro_left, 0.0) / 0.26))
        return 0.0

    def _saccade(self, t_now, dt):
        if t_now >= self._sac_at:
            self._sac_at = t_now + 0.7 + random.random() * 2.2
            self._sac_tx = (random.random() - 0.5) * 0.5
            self._sac_ty = (random.random() - 0.5) * 0.35
        rate = 0.28 if self.figure == "bender" else 0.12
        k = min(1.0, dt * rate * 30)
        self._sac_x += (self._sac_tx - self._sac_x) * k
        self._sac_y += (self._sac_ty - self._sac_y) * k

    # ── tick: springs + life physics (30 fps budget) ──────────────────
    def _tick(self):
        dt = 1 / 30
        t_now = time.monotonic() - self._t0
        self._speech_tick(dt)
        self._saccade(t_now, dt)
        # spring dynamics: overshoot = emotional pop (bounce is a normal spring
        # whose target is always 0 — the impulse is injected in _set_target)
        for k in PKEYS:
            self.vel[k] += (self.target[k] - self.P[k]) * 0.10
            self.vel[k] *= 0.80
            self.P[k] += self.vel[k]
        self.P["eyeOpen"] = max(0.0, self.P["eyeOpen"])
        self.P["mouthOpen"] = max(0.0, self.P["mouthOpen"])
        self.P["tear"] = max(0.0, min(1.0, self.P["tear"]))
        if self._face_timer > 0:
            self._face_timer -= 1

        e = self._eff()
        jaw = 0.030 if self.figure == "smiley" else (0.028 if self.figure == "goddess" else 0.016)
        self.MY = self.MY0 + e["mouthOpen"] * W * jaw

        # laugh body-bob (bounded spring — the unbounded version flew off-canvas
        # and produced a negative ellipse radius that killed the loop in JS)
        cur = self._emotion_mode if self._emotion_mode != "auto" else self._auto_emotion
        if cur == "laugh":
            if self._laugh_b >= 0 and abs(self._laugh_v) < 0.6:
                self._laugh_v = -(2.2 + abs(self.P["mouthOpen"]) * 2.0)
        elif self._laugh_b > 0:
            self._laugh_b, self._laugh_v = 0.0, 0.0
        self._laugh_v += (-self._laugh_b) * 0.09
        self._laugh_v *= 0.92
        self._laugh_b = max(-18.0, min(6.0, self._laugh_b + self._laugh_v))

        self.update()

    # ── paint ───────────────────────────────────────────────────────────
    def paintEvent(self, event):
        t_now = time.monotonic() - self._t0
        self._tnow = t_now
        e = self._eff()
        P = self.P

        pt = QPainter(self)
        pt.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale = min(self.width() / W, self.height() / H)
        pt.translate((self.width() - W * scale) / 2, (self.height() - H * scale) / 2)
        pt.scale(scale, scale)

        # background vignette
        if self.figure == "goddess":
            inner, outer = QColor(23, 18, 8), QColor(7, 5, 3)
        elif self.figure == "bender":
            inner, outer = QColor(18, 20, 26), QColor(5, 6, 8)
        else:
            inner, outer = QColor(10, 15, 22), QColor(3, 5, 10)
        bg = QRadialGradient(self.CX, self.CY, W * 0.75)
        bg.setColorAt(0.0, inner)
        bg.setColorAt(1.0, outer)
        pt.fillRect(QRectF(0, 0, W, H), QBrush(bg))

        # life: sway / breathe / tremble / squash (clamped: negative radii
        # throw/kill the paint loop)
        t = t_now
        spd = P["spd"]
        sway_x = fbm(t * 0.22, 11) * W * (0.012 if self.figure != "bender" else 0.010)
        sway_y = fbm(t * 0.26, 77) * W * (0.010 if self.figure != "bender" else 0.008)
        tilt = P["tilt"] + fbm(t * 0.15, 5) * (0.03 if self.figure != "bender" else 0.018) + e["mouthCurve"] * 0.008
        breathe = 1 + (0.006 if self.figure == "bender" else 0.012) * math.sin(t * 1.9 * spd)
        turb = P["turb"]
        trem_x = fbm(t * 18, 2) * W * 0.006 * (turb - 0.5) if turb > 0.85 else 0.0
        trem_y = fbm(t * 16, 9) * W * 0.005 * (turb - 0.5) if turb > 0.85 else 0.0
        squish = -max(-0.10, min(0.10, self.vel["mouthOpen"] * 0.5)) + e["mouthOpen"] * (0.035 if self.figure != "bender" else 0.030)
        bsquash = 0.7 if self.figure == "bender" else 0.9
        sy_squash = 1.4 if self.figure == "bender" else 1.6
        bnc = self.P["bounce"]
        sx = max(0.5, breathe * (1 - bnc * bsquash - squish * 0.6))
        sy = max(0.5, breathe * (1 + bnc * sy_squash + squish))

        # antenna / hoop pendulums (goddess hoops, bender antenna)
        head_vx = (sway_x - self._last_sway_x)
        self._last_sway_x = sway_x
        if self.figure == "goddess":
            self._hoop_vel += (-0.22 * self._hoop_ang - head_vx * 0.16) 
            self._hoop_vel *= 0.90
            self._hoop_ang = max(-0.5, min(0.5, self._hoop_ang + self._hoop_vel))
        elif self.figure == "bender":
            ant_target = P["droop"] * 0.55 + fbm(t * 0.5, 9) * 0.10 + (-0.25 if self._auto_emotion == "surprised" and self._emotion_mode == "auto" else 0)
            self._ant_vel += ((ant_target - self._ant_lean) * 0.06 - head_vx * 0.12)
            self._ant_vel *= 0.90
            self._ant_lean = max(-0.9, min(0.9, self._ant_lean + self._ant_vel))

        nod = math.sin(t * (4.6 if self.figure == "bender" else 4.2)) * W * 0.004 if self._speak_level > 0.05 else 0.0

        pt.translate(self.CX + sway_x + trem_x, self.CY + sway_y + trem_y + self._laugh_b * 6 + nod)
        pt.rotate(math.degrees(tilt))
        pt.translate(-self.CX, -self.CY)
        pt.translate(self.CX, self.CY)
        pt.scale(sx, sy)
        pt.translate(-self.CX, -self.CY)

        if self.figure == "smiley":
            self._paint_smiley(pt, e)
        elif self.figure == "goddess":
            self._paint_goddess(pt, e)
        else:
            self._paint_bender(pt, e)

        # grounding shadow
        if self.figure == "bender":
            sh_y, sh_rx, sh_ry = self.CY + self.HH * 2.6, W * 0.30, W * 0.035
        else:
            sh_y = self.CY + self.RY * (1.9 if self.figure == "goddess" else 1.45)
            sh_rx, sh_ry = self.RX * (1.4 if self.figure == "goddess" else 1.05), self.RY * (0.16 if self.figure == "goddess" else 0.14)
        gS = QRadialGradient(self.CX, sh_y, max(1.0, sh_rx))
        gS.setColorAt(0.0, hsla(30, 0.4, 0.03, 0.55))
        gS.setColorAt(1.0, hsla(30, 0.4, 0.03, 0.0))
        pt.setBrush(QBrush(gS))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawEllipse(QPointF(self.CX, sh_y), max(1.0, sh_rx), max(1.0, sh_ry))

        self._draw_labels(pt)
        pt.end()

    # ══════════════════ SMILEY ══════════════════
    def _paint_smiley(self, pt: QPainter, e):
        CX, CY, RX, RY, EX, EY, EW = self.CX, self.CY, self.RX, self.RY, self.EX, self.EY, self.EW
        hue, sat, lit = self.P["hue"], self.P["sat"], self.P["lit"]
        # head
        g = QRadialGradient(CX - RX * 0.38, CY - RY * 0.42, max(1.0, RX * 1.55))
        g.setColorAt(0.0, hsla(hue, sat, min(0.96, lit + 0.18), 1))
        g.setColorAt(0.62, hsla(hue, sat, lit, 1))
        g.setColorAt(1.0, hsla(hue, min(1, sat + 0.1), max(0.2, lit - 0.22), 1))
        pt.setBrush(QBrush(g))
        pt.setPen(QPen(hsla(hue, 0.6, 0.1, 0.35), W * 0.004))
        pt.drawEllipse(QPointF(CX, CY), RX, RY)
        # rim light arc
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(hsla(hue, 0.8, 0.96, 0.30), W * 0.0075, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        # canvas 0.72pi..1.42pi (y-flip) = Qt start 104.4deg, CCW span 126deg: upper-left rim
        pt.drawArc(QRectF(CX - RX + 4, CY - RY + 4, RX * 2 - 8, RY * 2 - 8), int(104.4 * 16), int(126 * 16))
        self._paint_cheeks(pt, e)
        self._paint_brows(pt, e)
        self._paint_eyes(pt, e, almond=True)
        self._paint_nose(pt, e)
        self._paint_mouth(pt, e)
        self._paint_tears(pt)

    # ══════════════════ shared feature painters (tone per figure) ═════════
    def _paint_cheeks(self, pt: QPainter, e):
        if self.figure == "smiley":
            CX, RX, RY, EX, EY, EW = self.CX, self.RX, self.RY, self.EX, self.EY, self.EW
        else:
            CX, RX, RY, EX, EY, EW = self.CX, self.RX, self.RY, self.EX, self.EY, self.EW
        lift = e["cheekLift"]
        for side in (-1, 1):
            bx = CX + side * RX * (0.62 if self.figure == "smiley" else 0.56)
            by = EY + RY * (0.42 if self.figure == "smiley" else 0.34) + lift * W * 0.012
            br = EW * (0.85 + lift * 0.9)
            blush = min(0.45, 0.05 + lift * 0.45 + (0.25 if (self.figure in ("goddess", "bender") and self._emotion_mode == "angry") else 0))
            ch = 4 if self.figure == "smiley" else (6 if self.figure == "goddess" else 35)
            gB = QRadialGradient(bx, by, max(1.0, br))
            gB.setColorAt(0.0, hsla(ch, 0.85 if self.figure == "smiley" else 0.7, 0.62, blush))
            gB.setColorAt(1.0, hsla(ch, 0.85, 0.62, 0.0))
            pt.setBrush(QBrush(gB))
            pt.setPen(Qt.PenStyle.NoPen)
            pt.drawEllipse(QPointF(bx, by), max(1.0, br), max(1.0, br * 0.72))
        if self.figure == "smiley" and e["cheekLift"] > 0.25:
            CX, EX, EY, EW = self.CX, self.EX, self.EY, self.EW
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(hsla(self.P["hue"], 0.7, max(0.18, self.P["lit"] - 0.30), 0.10 + e["cheekLift"] * 0.25),
                           EW * 0.10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for side in (-1, 1):
                pth = QPainterPath(QPointF(CX + side * EX - EW * 0.7, EY + EW * 0.62))
                pth.quadTo(CX + side * EX, EY + EW * (0.95 + e["cheekLift"] * 0.4), CX + side * EX + EW * 0.7, EY + EW * 0.62)
                pt.drawPath(pth)

    def _paint_brows(self, pt: QPainter, e):
        CX, EX = self.CX, self.EX
        if self.figure == "bender":
            ER = self.ER
            raise_ = self._micro(self._tnow) * 0.30
            trem = fbm(self._tnow * 22, 3) * W * 0.004 * e["turb"] if e["turb"] > 0.85 else 0.0
            brow_y = self.EY - ER * 1.45 + e["browHeight"] * W * 0.05 + raise_ * W * 0.05 + trem
            for side in (-1, 1):
                x_in, x_out = CX + side * W * 0.010, CX + side * W * 0.100
                y_in = brow_y - e["browAngle"] * W * 0.045
                y_out = brow_y + e["browAngle"] * W * 0.008
                cym = (y_in + y_out) / 2 - e["browCurve"] * W * 0.012
                pth = QPainterPath(QPointF(x_out, y_out))
                pth.quadTo((x_in + x_out) / 2, cym, x_in, y_in)
                pt.setBrush(Qt.BrushStyle.NoBrush)
                pt.setPen(QPen(hsla(self.P["hue"], min(1, self.P["sat"] + 0.08), max(0.12, self.P["lit"] - 0.38), 0.95),
                               W * 0.013, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                pt.drawPath(pth)
            return
        EW = self.EW
        raise_ = self._micro(self._tnow) * (0.35 if self.figure == "smiley" else 0.30)
        trem = fbm(self._tnow * 20, 3) * W * (0.004 if self.figure == "smiley" else 0.0035) * e["turb"] if e["turb"] > (0.9 if self.figure == "smiley" else 0.85) else 0.0
        BY = self.EY - W * (0.052 if self.figure == "smiley" else 0.056)
        bh = (e["browHeight"] + raise_) * W * 0.05
        for side in (-1, 1):
            x_out, x_in = CX + side * EX * (1.32 if self.figure == "smiley" else 1.30), CX + side * EX * (0.42 if self.figure == "smiley" else 0.36)
            y_out = BY + bh + e["browAngle"] * W * (0.014 if self.figure == "smiley" else 0.013) + trem
            y_in = BY + bh - e["browAngle"] * W * (0.042 if self.figure == "smiley" else 0.040) + trem
            cym = (y_out + y_in) / 2 - e["browCurve"] * W * (0.022 if self.figure == "smiley" else 0.026) + e["cheekLift"] * W * 0.010
            pt.setBrush(Qt.BrushStyle.NoBrush)
            if self.figure == "smiley":
                col = hsla(self.P["hue"], 0.55, max(0.12, self.P["lit"] - 0.44), 0.95)
                pt.setPen(QPen(col, W * 0.0085, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            else:
                col = hsla(32, 0.5, 0.26, 0.95)
                pt.setPen(QPen(col, W * 0.006, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pth = QPainterPath(QPointF(x_out, y_out))
            pth.quadTo((x_out + x_in) / 2, cym, x_in, y_in)
            pt.drawPath(pth)

    def _paint_eyes(self, pt: QPainter, e, almond=True):
        CX, EX, EY = self.CX, self.EX, self.EY
        blink = self._blink(self._tnow)
        for side in (-1, 1):
            open_ = max(e["eyeOpen"] * (1 - blink * 0.96), 0.0)
            cx = CX + side * EX
            cy = EY + e["pupilY"] * W * (0.4 if self.figure == "smiley" else 0.35)
            EW = self.EW
            if open_ < 0.16:
                happy_closed = e["cheekLift"] > 0.25 or self._emotion_mode == "laugh"
                y0 = cy + EW * 0.05
                if self.figure == "goddess":
                    col = hsla(32, 0.45, 0.24, 0.95)
                else:
                    col = hsla(self.P["hue"], 0.6, max(0.10, self.P["lit"] - 0.48), 0.95)
                pt.setBrush(Qt.BrushStyle.NoBrush)
                pt.setPen(QPen(col, EW * 0.13, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                pth = QPainterPath(QPointF(cx - EW * 0.82, y0))
                pth.quadTo(cx, y0 + (EW * (0.42 + e["eyeOpen"] * 0.5) if not happy_closed else -EW * (0.42 + e["eyeOpen"] * 0.5)), cx + EW * 0.82, y0)
                pt.drawPath(pth)
                continue
            eh = EW * (0.52 if self.figure == "smiley" else 0.50) * open_
            tilt = side * EW * 0.14 if self.figure == "smiley" else 0
            inn = QPointF(cx - side * EW * (0.94 if self.figure == "smiley" else 0.92), cy + EW * (0.06 if self.figure == "smiley" else 0.04) + tilt * 0.3)
            out = QPointF(cx + side * EW * (1.0 if self.figure == "smiley" else 1.02), cy - EW * (0.10 + (0.05 if self.figure == "smiley" else 0.0)) - tilt * 0.5)
            y_top, y_bot = cy - eh, cy + eh
            pt.setBrush(QColor(250, 250, 250, 242))
            pt.setPen(Qt.PenStyle.NoPen)
            almondp = QPainterPath(inn)
            almondp.quadTo(cx, y_top, out.x(), out.y())
            almondp.quadTo(cx, y_bot * 0.98 + cy * 0.02, inn.x(), inn.y())
            almondp.closeSubpath()
            pt.drawPath(almondp)
            if open_ > 0.10:
                pt.save()
                pt.setClipPath(almondp)
                ir = EW * (0.46 if self.figure == "smiley" else 0.52)
                ix = cx + self._sac_x * EW * 0.22
                iy = cy + e["pupilY"] * W * 0.35 + ir * 0.05
                if self.figure == "smiley":
                    gI = QRadialGradient(ix, iy, max(1.0, ir))
                    gI.setColorAt(0.0, hsla((self.P["hue"] + 200) % 360, 0.45, 0.38, 0.95))
                    gI.setColorAt(0.72, hsla((self.P["hue"] + 200) % 360, 0.50, 0.26, 0.95))
                    gI.setColorAt(1.0, hsla((self.P["hue"] + 200) % 360, 0.55, 0.16, 0.95))
                else:
                    gI = QRadialGradient(ix, iy, max(1.0, ir))
                    gI.setColorAt(0.0, hsla(190, 0.40, 0.38, 0.96))
                    gI.setColorAt(0.7, hsla(170, 0.35, 0.26, 0.96))
                    gI.setColorAt(1.0, hsla(160, 0.35, 0.14, 0.96))
                pt.setBrush(QBrush(gI))
                pt.drawEllipse(QPointF(ix, iy), max(1.0, ir), max(1.0, ir))
                pd = 0.30 if self._emotion_mode == "fear" else (0.24 if self._emotion_mode == "surprised" else (-0.16 if self._emotion_mode == "angry" else 0))
                pr = max(1.0, ir * (0.40 + pd))
                pt.setBrush(QColor(15, 22, 30))
                pt.drawEllipse(QPointF(ix, iy), pr, pr)
                pt.setBrush(QColor(255, 255, 255, 217))
                cl = max(1.0, ir * 0.17)
                pt.drawEllipse(QPointF(ix - ir * 0.36, iy - ir * 0.40), cl, cl)
                cl2 = max(0.7, ir * 0.07)
                pt.drawEllipse(QPointF(ix + ir * 0.28, iy + ir * 0.30), cl2, cl2)
                pt.restore()
            # lash line
            if self.figure == "goddess":
                lash = hsla(32, 0.4, 0.16, 0.95)
            else:
                lash = hsla(self.P["hue"], 0.5, max(0.08, self.P["lit"] - 0.52), 0.95)
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(lash, EW * (0.10 if self.figure == "smiley" else 0.14), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            up = QPainterPath(inn)
            up.quadTo(cx, y_top - EW * 0.02, out.x(), out.y())
            pt.drawPath(up)
            pt.setPen(QPen(lash, EW * 0.05, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pt.drawLine(out, QPointF(out.x() + side * EW * (0.16 if self.figure == "smiley" else 0.18), out.y() - EW * (0.14 if self.figure == "smiley" else 0.15)))
            pt.setPen(QPen(hsla(self.P["hue"], 0.5, max(0.12, self.P["lit"] - 0.40), 0.35), EW * 0.045))
            lo = QPainterPath(inn)
            lo.quadTo(cx, y_bot, out.x(), out.y())
            pt.drawPath(lo)

    def _paint_nose(self, pt: QPainter, e):
        CX, CY = self.CX, self.CY
        hue, sat, lit = self.P["hue"], self.P["sat"], self.P["lit"]
        if self.figure == "goddess":
            tip = CY + W * 0.058
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(hsla(hue, 0.45, max(0.14, lit - 0.30), 0.65), W * 0.0045, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pth = QPainterPath(QPointF(CX - W * 0.003, self.EY - W * 0.056 + W * 0.030))
            pth.quadTo(CX + W * 0.004, (self.EY + tip) / 2, CX, tip)
            pt.drawPath(pth)
            pt.setPen(QPen(hsla(hue, 0.30, min(0.94, lit + 0.18), 0.5), W * 0.003))
            pth2 = QPainterPath(QPointF(CX - W * 0.006, self.EY - W * 0.056 + W * 0.030))
            pth2.quadTo(CX - W * 0.001, (self.EY + tip) / 2, CX - W * 0.004, tip - W * 0.004)
            pt.drawPath(pth2)
        else:
            tip = CY + W * 0.078
            gB = QLinearGradient(CX - W * 0.02, tip, CX + W * 0.02, tip)
            gB.setColorAt(0.0, QColor(255, 255, 255, 0))
            gB.setColorAt(0.45, QColor(255, 255, 255, 42))
            gB.setColorAt(1.0, QColor(255, 255, 255, 0))
            pth = QPainterPath(QPointF(CX - W * 0.004, CY - W * 0.006))
            pth.quadTo(CX + W * 0.002, (CY + tip) / 2, CX, tip - W * 0.006)
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(QBrush(gB), W * 0.007, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pt.drawPath(pth)
        pt.setPen(Qt.PenStyle.NoPen)
        pt.setBrush(hsla(hue, 0.55, max(0.12, lit - 0.42), 0.85))
        for sgn in (-1, 1):
            pt.save()
            pt.translate(CX + sgn * W * 0.0135 if self.figure == "smiley" else CX + sgn * W * 0.0105, tip - W * 0.002)
            pt.rotate(math.degrees(0.35 * sgn))
            pt.drawEllipse(QPointF(0, 0), W * 0.0042, W * 0.0026)
            pt.restore()
        pt.setBrush(hsla(hue, 0.6, min(0.92, lit + 0.26), 0.5))
        pt.drawEllipse(QPointF(CX, tip - W * 0.008), W * 0.009 if self.figure == "smiley" else W * 0.007, W * 0.0045 if self.figure == "smiley" else W * 0.0035)

    def _paint_mouth(self, pt: QPainter, e):
        CX = self.CX
        my = self.MY
        hue = self.P["hue"]
        if self.figure == "goddess":
            half = (0.036 + 0.030 * e["mouthWidth"]) * W
            rose = hsla(355, 0.52, 0.44 if self._emotion_mode == "angry" else 0.52, 1)
        else:
            half = (0.045 + 0.036 * e["mouthWidth"]) * W
            rose = None
        lift = e["mouthCurve"] * W * (0.016 if self.figure == "smiley" else 0.015)
        open_h = e["mouthOpen"] * W * (0.14 if self.figure == "smiley" else 0.13)
        lx, ly = CX - half, my + lift
        rx = CX + half
        pt.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if open_h < W * 0.02:
            if self.figure == "goddess":
                cym = ly + e["mouthCurve"] * W * 0.048
                pth = QPainterPath(QPointF(lx, ly))
                pth.quadTo(CX - half * 0.5, ly - W * 0.009, CX - W * 0.005, ly - W * 0.007)
                pth.quadTo(CX, ly - W * 0.002, CX + W * 0.005, ly - W * 0.007)
                pth.quadTo(CX + half * 0.5, ly - W * 0.009, rx, ly)
                pth.quadTo(CX, cym + W * 0.006, lx, ly)
                pth.closeSubpath()
                pt.setBrush(rose)
                pt.setPen(Qt.PenStyle.NoPen)
                pt.drawPath(pth)
                pt.setBrush(Qt.BrushStyle.NoBrush)
                pt.setPen(QPen(hsla(355, 0.55, 0.32, 1), W * 0.0022, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                ln = QPainterPath(QPointF(lx, ly))
                ln.quadTo(CX, cym, rx, ly)
                pt.drawPath(ln)
            else:
                cym = ly + e["mouthCurve"] * W * 0.052
                pt.setBrush(Qt.BrushStyle.NoBrush)
                pt.setPen(QPen(hsla(hue, 0.6, max(0.08, self.P["lit"] - 0.52), 0.95), W * 0.0125,
                               Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                ln = QPainterPath(QPointF(lx, ly))
                ln.quadTo(CX, cym, rx, ly)
                pt.drawPath(ln)
            if e["mouthCurve"] > 0.55:
                pt.setPen(QPen(hsla(hue, 0.6, max(0.10, self.P["lit"] - 0.46), 0.5), W * 0.005,
                               Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                for side in (-1, 1):
                    tick = QPainterPath(QPointF(CX + half * side * 1.02, ly - W * 0.008))
                    tick.quadTo(CX + half * side * 1.16, ly, CX + half * side * 1.0, ly + W * 0.012)
                    pt.drawPath(tick)
            return
        ctop = my - open_h * 0.50 + e["mouthCurve"] * W * 0.010
        cbot = my + open_h * 1.10 + e["mouthCurve"] * W * 0.020
        cav = QPainterPath(QPointF(lx, ly))
        cav.quadTo(CX, ctop, rx, ly)
        cav.quadTo(CX, cbot, lx, ly)
        cav.closeSubpath()
        gC = QLinearGradient(0, my - open_h * 0.6, 0, my + open_h * 1.2)
        gC.setColorAt(0.0, QColor(10, 10, 10, 242))
        gC.setColorAt(0.65, QColor(13, 13, 13, 230))
        gC.setColorAt(1.0, QColor(10, 10, 10, 217))
        pt.setBrush(QBrush(gC))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(cav)
        if e["mouthOpen"] > 0.55 and e["mouthCurve"] > 0.6:
            pt.setBrush(hsla(350, 0.62, 0.58, 1))
            tg = QPainterPath(QPointF(CX - half * 0.62, cbot - W * 0.006))
            tg.quadTo(CX, cbot - open_h * 0.75, CX + half * 0.62, cbot - W * 0.006)
            tg.quadTo(CX, cbot + W * 0.012, CX - half * 0.62, cbot - W * 0.006)
            pt.drawPath(tg)
        frame = rose if self.figure == "goddess" else hsla(hue, 0.6, max(0.08, self.P["lit"] - 0.52), 0.9)
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(frame, W * (0.008 if self.figure == "smiley" else 0.0065), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        up = QPainterPath(QPointF(lx, ly))
        up.quadTo(CX, ctop - (0 if self.figure == "smiley" else W * 0.004), rx, ly)
        pt.drawPath(up)
        pt.setPen(QPen(rose if self.figure == "goddess" else hsla(hue, 0.7, min(0.92, self.P["lit"] + 0.22), 0.55),
                       W * (0.005 if self.figure == "smiley" else 0.0075), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        lo = QPainterPath(QPointF(lx, ly + open_h * 0.6))
        lo.quadTo(CX, cbot + W * 0.006, rx, ly + open_h * 0.6)
        pt.drawPath(lo)

    def _paint_tears(self, pt: QPainter):
        if self.P["tear"] <= 0.05:
            return
        CX, EX, EY = self.CX, self.EX, self.EY
        EW = self.EW
        for side in (-1, 1):
            ph = (self._tnow * 0.00055 + (0.5 if side > 0 else 0)) % 1
            tx = CX + side * (EX + EW * 0.95)
            ty = EY + EW * 0.7 + ph * W * 0.15
            a = math.sin(math.pi * ph) * self.P["tear"]
            if a <= 0.02:
                continue
            dr = W * 0.009
            pth = QPainterPath(QPointF(tx, ty - dr * 1.6))
            pth.quadTo(tx + dr, ty - dr * 0.2, tx + dr, ty + dr * 0.35)
            pt.setBrush(hsla(205, 0.9, 0.78, 0.85 * a))
            pt.setPen(Qt.PenStyle.NoPen)
            arc = QPainterPath(QPointF(tx + dr, ty + dr * 0.35))
            pt.drawPath(pth)
            pt.drawEllipse(QPointF(tx, ty + dr * 0.35), dr, dr)
            pt.setBrush(hsla(205, 0.95, 0.92, 0.7 * a))
            cl = max(0.7, dr * 0.22)
            pt.drawEllipse(QPointF(tx - dr * 0.3, ty + dr * 0.15), cl, cl)

    # ══════════════════ GODDESS ══════════════════
    def _build_locks(self):
        r = _mulberry32(777)
        CX, CY, RX, RY = self.CX, self.CY, self.RX, self.RY
        locks = []
        for i in range(30):
            side = -1 if i % 2 == 0 else 1
            a = math.pi * (1.02 + r() * 0.94)
            sx = CX + RX * 1.04 * math.cos(a) * 1.18
            sy = CY + RY * 0.98 * math.sin(a)
            fall_y = CY + RY * (1.15 + r() * 0.75)
            flare = RX * (1.25 + r() * 0.55)
            locks.append([sx, sy, CX + side * flare, fall_y, 0.8 + r() * 1.2, 3 + r() * 4, side, 0])
        for i in range(8):
            side = -1 if i % 2 == 0 else 1
            a = math.pi * (1.15 + r() * 0.7)
            sx = CX + RX * 1.0 * math.cos(a) * 1.12
            sy = CY + RY * 0.95 * math.sin(a)
            locks.append([sx, sy, CX + side * RX * (0.95 + r() * 0.12), CY + RY * (1.3 + r() * 0.3),
                          1.4 + r(), 5 + r() * 3, side, 1])
        return locks

    def _lock_pts(self, L, i):
        CX = self.CX
        mx, my = (L[0] + L[2]) / 2, (L[1] + L[3]) / 2
        cxq, cyq = mx + (mx - CX) * 0.5, my * 1.03
        pts = []
        tn = self._tnow
        for sN in range(13):
            u, bu = sN / 12, 1 - sN / 12
            x = bu * bu * L[0] + 2 * bu * u * cxq + u * u * L[2]
            y = bu * bu * L[1] + 2 * bu * u * cyq + u * u * L[3]
            x += math.sin(tn * 1.3 * L[4] + u * L[5] + L[6] * 2) * W * 0.005 * (1 + u * 2.2)
            y += math.cos(tn * 1.1 * L[4] + u * L[5]) * W * 0.002
            pts.append(QPointF(x, y))
        return pts

    def _draw_lock(self, pt: QPainter, L, i, hi):
        pts = self._lock_pts(L, i)
        for seg in range(3):
            a0, a1 = seg * 4, min(12, seg * 4 + 4)
            t = (seg + 0.5) / 3
            if hi:
                col = hsla(32, 0.60, 0.72, 0.30 * (1 - t * 0.5))
            else:
                col = hsla(32, 0.48, 0.30 + t * 0.08, 0.95)
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(col, W * 0.011 * (1 - t * 0.5), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pth = QPainterPath(pts[a0])
            for sN in range(a0 + 1, a1 + 1):
                pth.lineTo(pts[sN])
            pt.drawPath(pth)
        tip = pts[12]
        pt.setPen(QPen(hsla(32, 0.60, 0.74, 0.5) if hi else hsla(32, 0.48, 0.40, 0.95), W * 0.0045))
        r0 = W * 0.016 * (1 + 0.15 * math.sin(self._tnow * 2 + L[5]))
        pth = QPainterPath()
        first = True
        a = 0.0
        while a <= 2.2 * math.pi * 2:
            rr = r0 * (1 - a / (2.2 * math.pi * 2 + 0.001))
            px = tip.x() + math.cos(a * L[6] + L[6]) * rr
            py = tip.y() + math.sin(a * L[6] + L[6]) * rr
            if first:
                pth.moveTo(px, py)
                first = False
            else:
                pth.lineTo(px, py)
            a += 0.3
        pt.drawPath(pth)

    def _face_path_goddess(self):
        CX, CY, RX, RY = self.CX, self.CY, self.RX, self.RY
        pth = QPainterPath(QPointF(CX, CY - RY))
        pth.cubicTo(CX + RX * 0.82, CY - RY * 0.96, CX + RX, CY - RY * 0.42, CX + RX * 0.94, CY + RY * 0.18)
        pth.cubicTo(CX + RX * 0.86, CY + RY * 0.66, CX + RX * 0.5, CY + RY * 0.96, CX, CY + RY)
        pth.cubicTo(CX - RX * 0.5, CY + RY * 0.96, CX - RX * 0.86, CY + RY * 0.66, CX - RX * 0.94, CY + RY * 0.18)
        pth.cubicTo(CX - RX, CY - RY * 0.42, CX - RX * 0.82, CY - RY * 0.96, CX, CY - RY)
        pth.closeSubpath()
        return pth

    def _paint_goddess(self, pt: QPainter, e):
        CX, CY, RX, RY = self.CX, self.CY, self.RX, self.RY
        hue, sat, lit = self.P["hue"], self.P["sat"], self.P["lit"]
        # hair mass
        g = QLinearGradient(CX, CY - RY * 1.7, CX, CY + RY * 1.6)
        g.setColorAt(0.0, hsla(32, 0.54, 0.40, 1))
        g.setColorAt(0.55, hsla(32, 0.48, 0.30, 1))
        g.setColorAt(1.0, hsla(32, 0.40, 0.22, 1))
        mass = QPainterPath(QPointF(CX - RX * 1.52, CY + RY * 1.45))
        mass.cubicTo(CX - RX * 1.75, CY + RY * 0.3, CX - RX * 1.5, CY - RY * 1.28, CX - RX * 0.62, CY - RY * 1.5)
        mass.quadTo(CX, CY - RY * 1.66, CX + RX * 0.62, CY - RY * 1.5)
        mass.cubicTo(CX + RX * 1.5, CY - RY * 1.28, CX + RX * 1.75, CY + RY * 0.3, CX + RX * 1.52, CY + RY * 1.45)
        mass.quadTo(CX, CY + RY * 1.16, CX - RX * 1.52, CY + RY * 1.45)
        mass.closeSubpath()
        pt.setBrush(QBrush(g))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(mass)
        for i in range(14):
            a = math.pi * (1.02 + (i / 13) * 0.96)
            bx = CX + RX * 1.5 * math.cos(a) * 1.06
            by = CY + RY * 1.32 * math.sin(a) + math.sin(self._tnow * 1.6 + i) * W * 0.003
            pt.setBrush(hsla(32, 0.48, 0.30 + (0.04 if i % 2 else -0.02), 1))
            br = W * (0.026 + 0.006 * math.sin(i * 2.4))
            pt.drawEllipse(QPointF(bx, by), max(1.0, br), max(1.0, br))
        # neck + chiton + necklace
        self._draw_goddess_body(pt, e)
        # back locks
        for i, L in enumerate(self._locks):
            if L[7] == 0:
                self._draw_lock(pt, L, i, False)
        # face
        fp = self._face_path_goddess()
        gF = QRadialGradient(CX - RX * 0.4, CY - RY * 0.5, max(1.0, RX * 1.9))
        gF.setColorAt(0.0, hsla(hue, max(0.08, sat - 0.06), min(0.92, lit + 0.09), 1))
        gF.setColorAt(0.6, hsla(hue, sat, lit, 1))
        gF.setColorAt(1.0, hsla(hue, min(1, sat + 0.06), max(0.18, lit - 0.20), 1))
        pt.setBrush(QBrush(gF))
        pt.setPen(QPen(hsla(hue, 0.4, 0.20, 0.35), W * 0.003))
        pt.drawPath(fp)
        gSh = QLinearGradient(CX - RX, CY, CX + RX, CY)
        gSh.setColorAt(0.0, hsla(hue, sat, max(0.14, lit - 0.24), 0.5))
        gSh.setColorAt(0.25, QColor(0, 0, 0, 0))
        gSh.setColorAt(0.75, QColor(0, 0, 0, 0))
        gSh.setColorAt(1.0, hsla(hue, sat, max(0.14, lit - 0.24), 0.65))
        pt.setBrush(QBrush(gSh))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(fp)
        self._paint_cheeks(pt, e)
        self._paint_brows(pt, e)
        self._paint_eyes(pt, e)
        self._paint_nose(pt, e)
        self._paint_mouth(pt, e)
        # hoops
        for side in (-1, 1):
            ax = CX + side * RX * 0.86
            ay = CY + RY * 0.52
            ang = self._hoop_ang * side
            bx = ax + math.sin(ang) * W * 0.055
            by = ay + math.cos(ang) * W * 0.055
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(QColor(217, 178, 95), W * 0.004))
            hr = W * 0.012
            pt.drawEllipse(QPointF(bx, by + W * 0.012), hr, hr)
            pt.setBrush(QColor(217, 178, 95))
            br = W * 0.0035
            pt.drawEllipse(QPointF(ax, ay), br, br)
        # front locks + highlights
        for i, L in enumerate(self._locks):
            if L[7] == 1:
                self._draw_lock(pt, L, i, False)
        for i in range(0, len(self._locks), 3):
            self._draw_lock(pt, self._locks[i], i, True)
        # stephane
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(QColor(217, 178, 95), W * 0.008, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        band = QRectF(CX - RX * 1.12, CY - RY * 0.28 - RY * 1.12, RX * 2.24, RY * 2.24)
        # canvas angles PI*1.13..PI*1.87 (crown arc) invert to Qt's 0.13..0.87 *pi
        pt.drawArc(band, int(0.13 * 180 * 16), int(0.74 * 180 * 16))
        for side in (-1, 1):
            for k in range(4):
                a = math.pi * (1.5 + side * (0.16 + k * 0.17))
                lx = CX + RX * 1.12 * math.cos(a)
                ly = CY - RY * 0.28 + RY * 1.12 * math.sin(a)
                pt.save()
                pt.translate(lx, ly)
                pt.rotate(math.degrees(a + math.pi / 2 + side * 0.3))
                pt.setBrush(hsla(120, 0.26, 0.42, 1))
                pt.setPen(Qt.PenStyle.NoPen)
                lw = W * 0.012
                pt.drawEllipse(QPointF(0, 0), lw, max(1.0, W * 0.004))
                pt.restore()
        pt.setBrush(QColor(217, 178, 95))
        mr = W * 0.008
        pt.drawEllipse(QPointF(CX, CY - RY * 1.40), mr, mr)
        self._paint_tears(pt)

    def _draw_goddess_body(self, pt: QPainter, e):
        CX, CY, RX, RY = self.CX, self.CY, self.RX, self.RY
        hue, sat, lit = self.P["hue"], self.P["sat"], self.P["lit"]
        nw = RX * 0.42
        gN = QLinearGradient(CX - nw, 0, CX + nw, 0)
        gN.setColorAt(0.0, hsla(hue, sat, max(0.16, lit - 0.16), 1))
        gN.setColorAt(0.45, hsla(hue, sat, lit, 1))
        gN.setColorAt(1.0, hsla(hue, sat, max(0.16, lit - 0.20), 1))
        neck = QPainterPath(QPointF(CX - nw, CY + RY * 0.62))
        neck.quadTo(CX - nw * 1.1, CY + RY * 1.02, CX - nw * 1.5, CY + RY * 1.22)
        neck.lineTo(CX + nw * 1.5, CY + RY * 1.22)
        neck.quadTo(CX + nw * 1.1, CY + RY * 1.02, CX + nw, CY + RY * 0.62)
        neck.closeSubpath()
        pt.setBrush(QBrush(gN))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(neck)
        gC = QLinearGradient(0, CY + RY * 1.1, 0, H)
        gC.setColorAt(0.0, hsla(42, 0.30, 0.90, 1))
        gC.setColorAt(0.5, hsla(40, 0.24, 0.80, 1))
        gC.setColorAt(1.0, hsla(38, 0.20, 0.66, 1))
        ch = QPainterPath(QPointF(CX - nw * 1.5, CY + RY * 1.2))
        ch.quadTo(CX - RX * 1.9, CY + RY * 1.3, CX - RX * 2.4, H)
        ch.lineTo(CX + RX * 2.4, H)
        ch.quadTo(CX + RX * 1.9, CY + RY * 1.3, CX + nw * 1.5, CY + RY * 1.2)
        ch.quadTo(CX, CY + RY * 1.42, CX - nw * 1.5, CY + RY * 1.2)
        ch.closeSubpath()
        pt.setBrush(QBrush(gC))
        pt.drawPath(ch)
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(hsla(38, 0.25, 0.45, 0.35), W * 0.0035))
        for fx in (-1.5, -0.75, 0.75, 1.5):
            fold = QPainterPath(QPointF(CX + RX * fx, CY + RY * 1.35))
            fold.quadTo(CX + RX * fx * 1.35, (CY + RY * 1.35 + H) / 2, CX + RX * fx * 1.7, H)
            pt.drawPath(fold)
        pt.setPen(QPen(QColor(217, 178, 95), W * 0.005))
        trim = QPainterPath(QPointF(CX - nw * 1.55, CY + RY * 1.19))
        trim.quadTo(CX, CY + RY * 1.44, CX + nw * 1.55, CY + RY * 1.19)
        pt.drawPath(trim)
        nsway = math.sin(self._tnow * 2) * W * 0.002 + self._speak_level * W * 0.001
        chain = QPainterPath(QPointF(CX - nw * 1.15, CY + RY * 0.98))
        chain.quadTo(CX + nsway, CY + RY * 1.14, CX + nw * 1.15, CY + RY * 0.98)
        pt.drawPath(chain)
        px_, py_ = CX + nsway, CY + RY * 1.16
        pend = QPainterPath(QPointF(px_, py_ - W * 0.008))
        pend.quadTo(px_ + W * 0.006, py_, px_ + W * 0.005, py_ + W * 0.006)
        pt.setBrush(QColor(217, 178, 95))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(pend)
        dr = W * 0.005
        pt.drawEllipse(QPointF(px_, py_ + W * 0.006), dr, dr)

    # ══════════════════ BENDER ══════════════════
    def _rr(self, x, y, w, h, r):
        pth = QPainterPath(QPointF(x + r, y))
        pth.arcTo(QRectF(x + w - 2 * r, y, 2 * r, 2 * r), 90, -90)
        pth.arcTo(QRectF(x + w - 2 * r, y + h - 2 * r, 2 * r, 2 * r), 0, -90)
        pth.arcTo(QRectF(x, y + h - 2 * r, 2 * r, 2 * r), -90, -90)
        pth.arcTo(QRectF(x, y, 2 * r, 2 * r), 180, -90)
        pth.closeSubpath()
        return pth

    def _glow_col(self, l):
        return hsla(self.P["hue"], min(0.85, self.P["sat"] + 0.48), l, 1)

    def _paint_bender(self, pt: QPainter, e):
        CX, CY, HW, HH = self.CX, self.CY, self.HW, self.HH
        hue, sat, lit = self.P["hue"], self.P["sat"], self.P["lit"]

        def metal_g(x0, y0, x1, y1):
            g = QLinearGradient(x0, y0, x1, y1)
            g.setColorAt(0.0, hsla(hue, min(1, sat + 0.04), min(0.88, lit + 0.16), 1))
            g.setColorAt(0.45, hsla(hue, sat, lit, 1))
            g.setColorAt(1.0, hsla(hue, min(1, sat + 0.06), max(0.16, lit - 0.22), 1))
            return g

        # antenna (drawn first: pole base tucks under the crown bump)
        base_x, base_y = CX, CY - HH
        lean = self._ant_lean * 0.6
        ln = W * (0.095 - e["droop"] * 0.028)
        tip_x = base_x + math.sin(lean) * ln
        tip_y = base_y - ln * math.cos(lean)
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(hsla(40, 0.10, 0.46, 1), W * 0.008, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        pt.drawLine(QPointF(base_x, base_y), QPointF(tip_x, tip_y))
        pt.setPen(QPen(hsla(40, 0.12, 0.62, 1), W * 0.003))
        pt.drawLine(QPointF(base_x - 2, base_y), QPointF(tip_x - 2, tip_y))
        g = QRadialGradient(tip_x, tip_y, max(1.0, W * 0.017))
        g.setColorAt(0.0, self._glow_col(0.85))
        g.setColorAt(0.7, self._glow_col(0.60))
        g.setColorAt(1.0, hsla(hue, min(1, sat + 0.20), 0.30, 1))
        pt.setBrush(QBrush(g))
        pt.setPen(Qt.PenStyle.NoPen)
        br = W * 0.017
        pt.drawEllipse(QPointF(tip_x, tip_y), br, br)

        # neck piston
        pt.setBrush(QBrush(metal_g(CX - W * 0.04, 0, CX + W * 0.04, 0)))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(self._rr(CX - W * 0.034, CY + HH * 0.72, W * 0.068, W * 0.10, W * 0.014))
        pt.setPen(QPen(hsla(40, 0.10, 0.18, 0.5), W * 0.003))
        for dy in (0.78, 0.84, 0.90):
            pt.drawLine(QPointF(CX - W * 0.034, CY + HH * dy), QPointF(CX + W * 0.034, CY + HH * dy))
        # shoulders
        sh_y = CY + HH + W * 0.055
        pt.setBrush(QBrush(metal_g(CX - W * 0.24, sh_y, CX + W * 0.24, H)))
        pt.setPen(QPen(hsla(40, 0.12, 0.14, 0.55), W * 0.004))
        pt.drawPath(self._rr(CX - W * 0.235, sh_y, W * 0.47, H - sh_y + 40, W * 0.07))
        for side in (-1, 1):
            cx2 = CX + side * W * 0.235
            gC = QRadialGradient(cx2, sh_y + W * 0.035, max(1.0, W * 0.045))
            gC.setColorAt(0.0, hsla(hue, min(1, sat + 0.04), min(0.84, lit + 0.12), 1))
            gC.setColorAt(1.0, hsla(hue, min(1, sat + 0.06), max(0.18, lit - 0.20), 1))
            pt.setBrush(QBrush(gC))
            pt.setPen(Qt.PenStyle.NoPen)
            rr_ = W * 0.042
            pt.drawEllipse(QPointF(cx2, sh_y + W * 0.035), rr_, rr_)
        # rivets
        pt.setBrush(hsla(hue, 0.10, max(0.22, lit - 0.26), 1))
        rriv = W * 0.005
        for i in range(-4, 5):
            pt.drawEllipse(QPointF(CX + i * W * 0.045, sh_y + W * 0.022), rriv, rriv)
        # chest emotion screen
        sc_w, sc_h = W * 0.15, W * 0.062
        sc_x, sc_y = CX - sc_w / 2, sh_y + W * 0.05
        pt.setBrush(QColor(18, 22, 18))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(self._rr(sc_x, sc_y, sc_w, sc_h, W * 0.008))
        gSc = QRadialGradient(CX, sc_y + sc_h / 2, max(1.0, sc_w * 0.6))
        gSc.setColorAt(0.0, hsla(hue, min(0.90, sat + 0.50), 0.55, 0.28))
        gSc.setColorAt(1.0, hsla(hue, min(0.90, sat + 0.50), 0.55, 0.0))
        pt.setBrush(QBrush(gSc))
        pt.drawPath(self._rr(sc_x, sc_y, sc_w, sc_h, W * 0.008))
        s_cx, s_cy = CX, sc_y + sc_h * 0.5
        pt.setBrush(self._glow_col(0.72))
        dot = max(1.0, W * 0.004 * (0.5 + 0.6 * min(1.0, e["eyeOpen"])))
        pt.drawEllipse(QPointF(s_cx - W * 0.026, s_cy - W * 0.006 * (1 - e["eyeOpen"] * 0.3)), dot, dot)
        pt.drawEllipse(QPointF(s_cx + W * 0.026, s_cy - W * 0.006 * (1 - e["eyeOpen"] * 0.3)), dot, dot)
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(self._glow_col(0.72), W * 0.0035, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        mini = QPainterPath(QPointF(s_cx - W * 0.024, s_cy + W * 0.012))
        mini.quadTo(s_cx, s_cy + W * 0.012 + e["mouthCurve"] * W * 0.016, s_cx + W * 0.024, s_cy + W * 0.012)
        pt.drawPath(mini)
        pt.setPen(QPen(hsla(40, 0.10, 0.20, 0.6), W * 0.003))
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.drawPath(self._rr(sc_x, sc_y, sc_w, sc_h, W * 0.008))

        # head shell
        pt.setBrush(QBrush(metal_g(CX - HW, CY - HH, CX + HW, CY + HH)))
        pt.setPen(QPen(hsla(40, 0.12, 0.12, 0.6), W * 0.004))
        head = self._rr(CX - HW, CY - HH, HW * 2, HH * 2, W * 0.045)
        pt.drawPath(head)
        pt.save()
        pt.setClipPath(head)
        gH = QLinearGradient(CX - HW, CY - HH, CX - HW * 0.2, CY + HH)
        gH.setColorAt(0.0, QColor(255, 255, 255, 56))
        gH.setColorAt(0.28, QColor(255, 255, 255, 0))
        pt.fillRect(QRectF(CX - HW, CY - HH, HW * 2, HH * 2), QBrush(gH))
        pt.restore()
        # crown bump
        pt.setBrush(hsla(hue, sat, max(0.16, lit - 0.08), 1))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawChord(QRectF(CX - W * 0.028, CY - HH - W * 0.028 + 2, W * 0.056, W * 0.056), 0, 180 * 16)
        # ear bolts
        for side in (-1, 1):
            bx = CX + side * HW
            by = CY + HH * 0.08
            gB = QRadialGradient(bx, by, max(1.0, W * 0.021))
            gB.setColorAt(0.0, hsla(hue, 0.12, min(0.80, lit + 0.08), 1))
            gB.setColorAt(1.0, hsla(hue, 0.14, max(0.16, lit - 0.28), 1))
            pt.setBrush(QBrush(gB))
            pt.setPen(QPen(hsla(40, 0.12, 0.12, 0.5), W * 0.0028))
            rb = W * 0.021
            pt.drawEllipse(QPointF(bx, by), rb, rb)
            pt.setPen(QPen(hsla(40, 0.12, 0.14, 0.6), W * 0.004))
            pt.drawLine(QPointF(bx - W * 0.009, by), QPointF(bx + W * 0.009, by))

        # eyes: glowing diodes with lid plates
        blink = self._blink(self._tnow)
        ER = self.ER
        for side in (-1, 1):
            ecx = CX + side * self.EX
            ecy = self.EY + e["pupilY"] * W * 0.25
            gl = min(1.0, e["cheekLift"])
            if gl > 0.06:
                gG = QRadialGradient(ecx, ecy, max(1.0, ER * 2.1))
                gG.setColorAt(0.0, hsla(hue, min(0.90, sat + 0.55), 0.75, 0.22 * gl))
                gG.setColorAt(1.0, hsla(hue, min(0.90, sat + 0.55), 0.75, 0.0))
                pt.setBrush(QBrush(gG))
                pt.setPen(Qt.PenStyle.NoPen)
                pt.drawEllipse(QPointF(ecx, ecy), ER * 2.1, ER * 2.1)
            pt.setBrush(QColor(26, 23, 17))
            ring = ER * 1.12
            pt.drawEllipse(QPointF(ecx, ecy), ring, ring)
            open_ = max(0.02, e["eyeOpen"] * (1 - blink))
            pt.save()
            clip = QPainterPath()
            clip.addEllipse(QPointF(ecx, ecy), ER, ER)
            rectp = QPainterPath()
            rectp.addRect(QRectF(ecx - ER * 1.1, ecy + ER - 2 * ER * open_, ER * 2.2, ER * 2.2))
            clip = clip.intersected(rectp)
            pt.setClipPath(clip)
            gE = QRadialGradient(ecx, ecy, max(1.0, ER * 1.15))
            gE.setColorAt(0.0, self._glow_col(0.95))
            gE.setColorAt(0.55, self._glow_col(0.80))
            gE.setColorAt(1.0, self._glow_col(0.52))
            pt.setBrush(QBrush(gE))
            pt.setPen(Qt.PenStyle.NoPen)
            pt.drawRect(QRectF(ecx - ER, ecy - ER, ER * 2, ER * 2))
            pd = 0.34 if self._emotion_mode == "fear" else (0.26 if self._emotion_mode == "surprised" else (-0.20 if self._emotion_mode == "angry" else 0))
            pr = max(1.5, ER * (0.34 + pd))
            jit = math.sin(self._tnow * 50) * ER * 0.10 if self._emotion_mode == "fear" else 0.0
            pt.setBrush(hsla(0, 0.85, 0.45, 1) if self._emotion_mode == "angry" else QColor(16, 17, 13))
            pt.drawEllipse(QPointF(ecx + self._sac_x * ER * 0.3 + jit, ecy + self._sac_y * ER * 0.25), pr, pr)
            pt.setBrush(QColor(255, 255, 255, 204))
            cl = max(1.0, ER * 0.14)
            pt.drawEllipse(QPointF(ecx - ER * 0.4, ecy - ER * 0.45), cl, cl)
            pt.restore()
            if open_ < 0.985:
                lid_y = ecy + ER - 2 * ER * open_
                pt.setBrush(QBrush(metal_g(ecx - ER, lid_y - ER * 0.3, ecx + ER, lid_y + ER * 0.2)))
                pt.setPen(Qt.PenStyle.NoPen)
                pt.drawRect(QRectF(ecx - ER * 1.12, ecy - ER * 1.12, ER * 2.24, max(0.0, lid_y - (ecy - ER * 1.12))))
                pt.setPen(QPen(hsla(40, 0.12, 0.14, 0.6), W * 0.003))
                pt.drawLine(QPointF(ecx - ER * 1.12, lid_y), QPointF(ecx + ER * 1.12, lid_y))
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(hsla(40, 0.14, 0.10, 0.7), W * 0.0055))
            ring = ER * 1.10
            pt.drawEllipse(QPointF(ecx, ecy), ring, ring)

        # unibrow plate
        raise_ = self._micro(self._tnow) * 0.30
        trem = fbm(self._tnow * 22, 3) * W * 0.004 * e["turb"] if e["turb"] > 0.85 else 0.0
        brow_y = self.EY - ER * 1.45 + e["browHeight"] * W * 0.05 + raise_ * W * 0.05 + trem
        for side in (-1, 1):
            x_in, x_out = CX + side * W * 0.010, CX + side * W * 0.100
            y_in = brow_y - e["browAngle"] * W * 0.045
            y_out = brow_y + e["browAngle"] * W * 0.008
            cym = (y_in + y_out) / 2 - e["browCurve"] * W * 0.012
            pt.setBrush(Qt.BrushStyle.NoBrush)
            pt.setPen(QPen(hsla(hue, min(1, sat + 0.08), max(0.12, lit - 0.38), 1), W * 0.013,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pth = QPainterPath(QPointF(x_out, y_out))
            pth.quadTo((x_in + x_out) / 2, cym, x_in, y_in)
            pt.drawPath(pth)

        # grill mouth
        half = (0.034 + 0.030 * e["mouthWidth"]) * W
        open_h = e["mouthOpen"] * W * 0.11
        top_h = max(W * 0.004, open_h * 0.5)
        bot_h = max(W * 0.004, open_h * 0.5)
        my = self.MY
        curve = e["mouthCurve"] * W * 0.030
        def grill():
            pth = QPainterPath(QPointF(CX - half, my - top_h))
            pth.quadTo(CX, my - top_h - curve * 2, CX + half, my - top_h)
            pth.quadTo(CX, my + bot_h - curve * 2, CX + half, my + bot_h)
            pth.quadTo(CX, my + bot_h - curve * 2, CX - half, my + bot_h)
            pth.quadTo(CX, my - top_h - curve * 2, CX - half, my - top_h)
            pth.closeSubpath()
            return pth
        pt.setBrush(QColor(12, 11, 9, int(140 + 100 * min(1.0, e["mouthOpen"]))))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(grill())
        pt.save()
        pt.setClipPath(grill())
        jit = math.sin(self._tnow * 55) * W * 0.004 if self._emotion_mode == "fear" else 0.0
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(QBrush(metal_g(CX - half, my - open_h, CX + half, my + open_h)), W * 0.0045))
        for i in range(11):
            x = CX - half + (i + 0.5) * (half * 2 / 11) + jit
            pt.drawLine(QPointF(x, my - W * 0.06), QPointF(x, my + W * 0.06))
        pt.restore()
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.setPen(QPen(hsla(hue, min(1, sat + 0.08), max(0.14, lit - 0.30), 1), W * 0.006))
        pt.drawPath(grill())
        pt.setPen(QPen(hsla(hue, min(1, sat + 0.04), min(0.86, lit + 0.14), 1), W * 0.0022))
        pt.drawPath(grill())

        # oil tears
        if self.P["tear"] > 0.05:
            for side in (-1, 1):
                ph = (self._tnow * 0.0005 + (0.5 if side > 0 else 0)) % 1
                tx = CX + side * (self.EX + ER * 0.85)
                ty = self.EY + ER * 0.9 + ph * W * 0.16
                a = math.sin(math.pi * ph) * self.P["tear"]
                if a <= 0.02:
                    continue
                dr = W * 0.0075
                pt.setBrush(hsla(35, 0.25, 0.06, 0.9 * a))
                pt.setPen(Qt.PenStyle.NoPen)
                pth = QPainterPath(QPointF(tx, ty - dr * 1.6))
                pth.quadTo(tx + dr, ty - dr * 0.2, tx + dr, ty + dr * 0.35)
                pt.drawPath(pth)
                pt.drawEllipse(QPointF(tx, ty + dr * 0.35), dr, dr)
                pt.setBrush(hsla(40, 0.40, 0.55, 0.35 * a))
                cl = max(0.7, dr * 0.2)
                pt.drawEllipse(QPointF(tx - dr * 0.3, ty + dr * 0.15), cl, cl)

    # ── labels ──────────────────────────────────────────────────────────
    def _draw_labels(self, pt: QPainter):
        f = QFont()
        f.setPointSize(14)
        f.setBold(True)
        pt.setFont(f)
        pt.setPen(QColor(0, 0, 0, 180))
        pt.drawText(12, 26, self._assistant_name)
        pt.setPen(QColor(255, 255, 255, 230))
        pt.drawText(10, 24, self._assistant_name)
        f2 = QFont()
        f2.setPointSize(11)
        f2.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        pt.setFont(f2)
        pt.setPen(QColor(220, 230, 240, 150))
        pt.drawText(QRectF(0, H - 34, W, 24), Qt.AlignmentFlag.AlignHCenter, self._state.label)
        emo = self._emotion_mode if self._emotion_mode != "auto" else self._auto_emotion
        pt.setPen(QColor(200, 190, 150, 130))
        pt.drawText(QRectF(0, H - 16, W, 16), Qt.AlignmentFlag.AlignHCenter, emo.upper())
        if self._face_name and self._face_timer > 0:
            pt.setFont(f)
            pt.setPen(QColor(140, 255, 200, min(255, self._face_timer * 3)))
            pt.drawText(QRectF(0, 74, W, 24), Qt.AlignmentFlag.AlignHCenter,
                        f"{self._face_name} ({self._face_confidence:.0%})")
