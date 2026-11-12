"""
Smoke Face HUD — emotive female face sculpted from smoke (PyQt6 port of
smoke-avatar/index_v2.html).

Implements the same public interface as HUDOverlay so the agent/main code can
swap frontends freely:

  set_state / set_activity / set_voice_level / set_palette_hue /
  clear_palette_hue / set_contrast_factor / set_assistant_name /
  set_face_detected / clear_face / set_profile / clear_profile

Smoke-specific extras:
  set_emotion(name)      - explicit emotion or 'auto' (state-driven)
  set_level_source(cb)   - callable -> (level, playing) fed by AudioPlayer RMS
                           for viseme-driven mouth movement
  emotion_changed signal - fires when the LLM/emotion tag changes emotion

Rendering mirrors the web version: particle layers (haze/hair/outline/wisp/
eyes/brows/mouth/nose/tears/breath) blitted from a sprite atlas with additive
blending onto a half-res QImage, plus a QPainter anatomy underlayer (almond
eyes with iris/pupil/catchlight, nose with nostrils, Cupid's-bow mouth with
dark oral cavity, dark hair mass + flowing strands).
"""

from __future__ import annotations

import math
import random
import time

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QComboBox, QWidget

from .state import JarvisState

# ── Emotions (identical parameter sets to the web canvas v2) ────────────
EMOTIONS = {
    "neutral":   dict(mouthCurve=0.05, mouthWidth=0.50, mouthOpen=0.02, browAngle=-0.03, browHeight=0.02, browCurve=0.42, eyeOpen=0.48, pupilY=0.00, cheekLift=0.00, turb=0.35, hue=212, spd=1.0, tear=0),
    "happy":     dict(mouthCurve=0.90, mouthWidth=0.78, mouthOpen=0.28, browAngle=-0.10, browHeight=0.06, browCurve=0.60, eyeOpen=0.34, pupilY=-0.02, cheekLift=0.30, turb=0.40, hue=40, spd=1.2, tear=0),
    "laugh":     dict(mouthCurve=1.00, mouthWidth=0.88, mouthOpen=0.85, browAngle=-0.25, browHeight=0.16, browCurve=0.65, eyeOpen=0.10, pupilY=-0.04, cheekLift=0.45, turb=0.60, hue=48, spd=1.5, tear=0),
    "sad":       dict(mouthCurve=-0.75, mouthWidth=0.42, mouthOpen=0.10, browAngle=0.60, browHeight=-0.04, browCurve=0.10, eyeOpen=0.46, pupilY=0.06, cheekLift=0.00, turb=0.22, hue=222, spd=0.7, tear=1),
    "angry":     dict(mouthCurve=-0.55, mouthWidth=0.62, mouthOpen=0.18, browAngle=-0.72, browHeight=-0.10, browCurve=0.05, eyeOpen=0.38, pupilY=0.00, cheekLift=0.00, turb=0.95, hue=2, spd=1.9, tear=0),
    "surprised": dict(mouthCurve=0.00, mouthWidth=0.28, mouthOpen=0.82, browAngle=0.12, browHeight=0.38, browCurve=0.45, eyeOpen=0.98, pupilY=-0.02, cheekLift=0.00, turb=0.70, hue=280, spd=1.4, tear=0),
    "fear":      dict(mouthCurve=-0.30, mouthWidth=0.38, mouthOpen=0.55, browAngle=0.55, browHeight=0.32, browCurve=0.20, eyeOpen=1.05, pupilY=-0.05, cheekLift=0.00, turb=1.00, hue=160, spd=2.3, tear=0),
    "calm":      dict(mouthCurve=0.30, mouthWidth=0.48, mouthOpen=0.00, browAngle=0.00, browHeight=0.03, browCurve=0.50, eyeOpen=0.06, pupilY=0.00, cheekLift=0.10, turb=0.15, hue=190, spd=0.5, tear=0),
}
EMOTION_ORDER = ["neutral", "happy", "laugh", "sad", "angry", "surprised", "fear", "calm"]
PKEYS = list(EMOTIONS["neutral"].keys())

# State -> default emotion (used in 'auto' mode when no LLM tag arrives)
STATE_EMOTION = {
    JarvisState.IDLE: "calm",
    JarvisState.LISTENING: "neutral",
    JarvisState.THINKING: "surprised",
    JarvisState.SPEAKING: "happy",
}
# visemes: (mouthOpen, mouthWidth, mouthCurve)
VISEMES = [
    (0.75, 0.55, 0.15),   # ah
    (0.25, 0.85, 0.25),   # ee
    (0.45, 0.30, 0.05),   # oo
    (0.90, 0.60, -0.05),  # jaw pop
    (0.10, 0.60, 0.10),   # closed
]

# ── value noise (port of the web mulberry32+perlin-ish grid) ────────────
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

_RND = _mulberry32(20261111)
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

def hsla(h, s, l, a):
    """QColor from HSLA floats (PyQt6 has no fromHsla)."""
    c = QColor.fromHslF(h % 360 / 360.0 if h >= 1 else h, s, l)
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


def fbm(x, y):
    return 0.6 * noise2(x, y) + 0.3 * noise2(x * 2.1 + 11, y * 2.1) + 0.1 * noise2(x * 4.7, y * 4.7 + 5)


W = H = 720.0          # logical canvas
CX, CY = W * 0.5, H * 0.52
RX, RY = W * 0.185, W * 0.255
EX, EY = W * 0.078, CY - W * 0.055
EW = W * 0.048
BY = EY - W * 0.052

FG = 26
CELL = W / FG


class SmokeFaceHUD(QWidget):
    """Emotive smoke-face frontend (drop-in HUDOverlay interface)."""

    emotion_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = JarvisState.IDLE
        self._activity = 0.0
        self._voice_level = 0.0
        self._assistant_name = "Jarvis"
        self._face_name = ""
        self._face_confidence = 0.0
        self._face_timer = 0
        self._profile_name = ""
        self._palette_hue: int | None = None
        self._contrast_factor = 1.0

        self._emotion_mode = "auto"     # 'auto' or explicit emotion name
        self._auto_emotion = "calm"
        self.target = dict(EMOTIONS["calm"])
        self.P = dict(EMOTIONS["calm"])
        self._speak_level = 0.0
        self._speak_target = 0.0
        self._level_source = None       # callable -> (level: float, playing: bool)
        self._vis_i = 0
        self._in_word = False
        self._word_t = 0.0
        self._speech_active = False

        self._t0 = time.monotonic()
        self._blink_at = self._next_blink()
        self._blink_left = 0.0

        self._build_particles()
        self._sprite_hue = -1
        self._build_sprites(212)
        self._fx = [0.0] * (FG * FG)
        self._fy = [0.0] * (FG * FG)

        # particle layer at half resolution (fill-rate), blitted up at the end
        self._buf = QImage(int(W / 2), int(H / 2), QImage.Format.Format_ARGB32_Premultiplied)
        self._hair = QImage(int(W), int(H), QImage.Format.Format_ARGB32_Premultiplied)

        # 30 fps — Python particle loop needs a gentler budget than the JS 60
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

        # emotion selector (only visible in smoke frontend)
        self.emotion_combo = QComboBox(self)
        self.emotion_combo.addItem("Auto", "auto")
        for e in EMOTION_ORDER:
            self.emotion_combo.addItem(e.capitalize(), e)
        self.emotion_combo.setFixedHeight(26)
        self.emotion_combo.setStyleSheet(
            "QComboBox{background:rgba(0,0,0,140);color:#dfe7ef;border:1px solid "
            "rgba(140,160,190,90);border-radius:5px;padding:3px 8px;font-size:12px;}"
        )
        self.emotion_combo.currentIndexChanged.connect(self._on_combo_emotion)
        self.emotion_combo.move(10, 40)
        self.emotion_combo.setFixedWidth(110)

    # ── HUDOverlay-compatible interface ─────────────────────────────
    def showEvent(self, event):
        self._timer.start(33)
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def set_palette_hue(self, hue: int):
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
                self.target = dict(EMOTIONS[emo])
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

    # ── smoke-specific ───────────────────────────────────────────────
    def set_emotion(self, name: str):
        """Manual emotion pick, or 'auto' to follow state machine."""
        if name in EMOTIONS:
            self._emotion_mode = name
            self.emotion_combo.blockSignals(True)
            self.emotion_combo.setCurrentIndex(EMOTION_ORDER.index(name) + 1)
            self.emotion_combo.blockSignals(False)
            self.target = dict(EMOTIONS[name])
            self.emotion_changed.emit(name)
        else:
            self._emotion_mode = "auto"
            self.emotion_combo.blockSignals(True)
            self.emotion_combo.setCurrentIndex(0)
            self.emotion_combo.blockSignals(False)
            self.target = dict(EMOTIONS[self._auto_emotion])

    def set_level_source(self, cb):
        """cb() -> (rms_level 0..1, is_playing). Polled each frame (GUI thread)."""
        self._level_source = cb

    def _on_combo_emotion(self, idx: int):
        self.set_emotion("auto" if idx == 0 else EMOTION_ORDER[idx - 1])

    # ── speech envelope (RMS-driven visemes) ─────────────────────────
    def _speech_tick(self, dt):
        if self._level_source is not None:
            try:
                lvl, playing = self._level_source()
            except Exception:
                lvl, playing = 0.0, False
            self._speech_active = playing
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
            # fallback: activity-based fake speech (when not wired to AudioPlayer)
            self._speak_target = self._activity * 0.5 if self._state == JarvisState.SPEAKING else 0.0
        self._speak_level += (self._speak_target - self._speak_level) * min(1.0, dt * 20.0)

    def _eff(self):
        P, mix = self.P, min(1.0, self._speak_level * 1.4)
        v = VISEMES[self._vis_i]
        e = dict(P)
        e["mouthOpen"] = P["mouthOpen"] * (1 - mix) + v[0] * mix + self._speak_level * 0.10
        e["mouthWidth"] = P["mouthWidth"] * (1 - mix) + v[1] * mix
        e["mouthCurve"] = P["mouthCurve"] * (1 - mix * 0.7) + v[2] * mix
        return e

    # ── particles ────────────────────────────────────────────────────
    WEIGHTS = [
        ("haze", 0.18), ("hair", 0.27), ("outline", 0.16), ("wisp", 0.13),
        ("eyeL", 0.036), ("eyeR", 0.036), ("browL", 0.022), ("browR", 0.022),
        ("mouth", 0.07), ("nose", 0.02), ("tear", 0.014), ("breath", 0.05),
    ]
    N = 1100  # reduced from the JS 3100 for the Python particle loop

    def _build_particles(self):
        self.parts = []
        self.strand_defs = []
        pool, acc = [], 0.0
        for f, w in self.WEIGHTS:
            acc += w
            pool.append((f, acc))
        for _ in range(self.N):
            r = _RND()
            feat = next((f for f, c in pool if r <= c), "haze")
            p = dict(
                feat=feat, t=_RND(), u=_RND(), seed=_RND() * 97, spr=int(_RND() * 4),
                size=0.0, alpha=0.0, life=0.0,
                sp=0.5 + _RND(), ph=_RND() * 6.283, w=0.6 + _RND() * 0.9,
                x=CX + (_RND() - 0.5) * W * 0.6, y=CY + (_RND() - 0.5) * H * 0.8,
                vx=0.0, vy=0.0, lock=None,
            )
            if feat == "hair":
                self._assign_hair(p)
                if len(self.strand_defs) < 30:
                    self.strand_defs.append(p["lock"])
            s, a = {"haze": (0.045 + _RND() * 0.035, 0.020),
                    "wisp": (0.012 + _RND() * 0.010, 0.048),
                    "hair": (0.016 + _RND() * 0.016, 0.058),
                    "outline": (0.014 + _RND() * 0.011, 0.052),
                    "mouth": (0.011 + _RND() * 0.008, 0.080),
                    "breath": (0.009 + _RND() * 0.009, 0.065),
                    "tear": (0.007, 0.11)}.get(feat, (0.008 + _RND() * 0.006, 0.085))
            p["size"], p["alpha"] = W * s, a
            self.parts.append(p)

    def _assign_hair(self, p):
        r = _mulberry32(int(p["seed"] * 1771 + 13))
        side = -1 if r() < 0.5 else 1
        kind = r()
        if kind < 0.42:
            s = r()
            topA = math.pi * (1.12 + s * 0.55)
            sx = CX + RX * (1.06 + s * 0.05) * math.cos(topA)
            sy = CY + RY * 1.02 * math.sin(topA)
            flare = RX * (1.22 + s * 0.28)
            endY = CY + RY * (1.62 + r() * 0.3)
            p["lock"] = [sx, sy, CX + side * flare * 0.52, endY, 0.8 + r(), 4 + r() * 5]
        elif kind < 0.72:
            s = r()
            a1 = math.pi * (1.06 if side < 0 else 0.94) + (s - 0.5) * 0.4
            rr = RY * 1.14
            p["lock"] = [CX + RX * 0.04, CY - rr,
                         CX + RX * (1.02 + s * 0.12) * math.cos(a1), CY + RY * 1.02 * math.sin(a1),
                         1.2 + r(), 3 + r() * 4]
        elif kind < 0.86:
            off = RX * (0.72 + r() * 0.16)
            p["lock"] = [CX + side * off * 0.9, CY - RY * 0.55,
                         CX + side * off * 1.12, CY + RY * 0.85, 1.5 + r(), 6 + r() * 4]
        else:
            a = math.pi * (1 + r()) + (r() - 0.5) * 0.6
            sx = CX + RX * 1.12 * math.cos(a)
            sy = CY + RY * 1.18 * math.sin(a)
            p["lock"] = [sx, sy, sx + (r() - 0.5) * W * 0.09, sy - (0.03 + r() * 0.06) * W,
                         2 + r() * 2, 2 + r() * 3]

    # ── feature targets (ported 1:1 from the web version) ────────────
    def _outline(self, t):
        th = t * math.pi * 2
        w = 1 - 0.28 * max(0.0, math.sin(th))
        return CX + RX * w * math.cos(th), CY + RY * math.sin(th)

    def _haze(self, p):
        r = _mulberry32(int(p["seed"]))()
        a = _mulberry32(int(p["seed"] + 7))() * math.pi * 2
        rr = math.sqrt(r) * 0.95
        return CX + RX * rr * math.cos(a), CY + RY * rr * math.sin(a)

    def _hair_tgt(self, p, t_now):
        L = p["lock"]
        u = p["u"]
        mx, my = (L[0] + L[2]) / 2, (L[1] + L[3]) / 2
        cxq, cyq = mx + (mx - CX) * 0.42, my * 1.02 + CY * (-0.02)
        bu = 1 - u
        x = bu * bu * L[0] + 2 * bu * u * cxq + u * u * L[2]
        y = bu * bu * L[1] + 2 * bu * u * cyq + u * u * L[3]
        x += math.sin(t_now * 1.2 * L[4] + u * L[5] + p["seed"]) * W * 0.006 * (1 + u * 2)
        y += math.cos(t_now * L[4] + p["seed"] * 2) * W * 0.003
        return x, y

    def _nr(self, p):
        return noise2(p["seed"] * 13.7, p["ph"] * 0.4 + p["t"] * 5)

    def _f01(self, p):
        return 0.5 + 0.5 * noise2(p["seed"] * 7.1 + 31, p["ph"] * 0.3 + p["t"] * 3 + 7)

    def _eye(self, side, t, e):
        pr = {"seed": side * 31.7 + t * 3, "t": t, "ph": self._tnow}
        if t < 0.3:
            a = self._nr(pr) * math.pi * 2
            r = self._f01(pr) * EW * 0.30
            return CX + side * EX + math.cos(a) * r, EY + e["pupilY"] * W + math.sin(a) * r
        th = self._nr(pr) * math.pi * 2
        eh = max(EW * 0.10 * e["eyeOpen"], EW * 0.06)
        lift = 0 if e["eyeOpen"] > 0.2 else -0.008 * W
        return CX + side * EX + math.cos(th) * EW, EY + math.sin(th) * eh + lift

    def _brow(self, side, t, e):
        bh = e["browHeight"] * W * 0.05
        x_out, x_in = CX + side * EX * 1.32, CX + side * EX * 0.42
        y_out = BY + bh + e["browAngle"] * W * 0.012
        y_in = BY + bh - e["browAngle"] * W * 0.038
        u = t
        cym = (y_out + y_in) / 2 - e["browCurve"] * W * 0.022 + e["cheekLift"] * W * 0.010
        pr = {"seed": side * 31.7 + t * 3, "t": t, "ph": self._tnow}
        return (x_out + (x_in - x_out) * u + self._nr(pr) * 0.004 * W,
                (1 - u) ** 2 * y_out + 2 * (1 - u) * u * cym + u * u * y_in)

    def _mouth(self, t, e, p, my):
        half = (0.042 + 0.034 * e["mouthWidth"]) * W
        lift = e["mouthCurve"] * W * 0.014
        open_h = e["mouthOpen"] * W * 0.13
        lx, ly = CX - half, my - lift
        rx = CX + half
        if open_h < W * 0.02:
            u, cym = t, my - e["mouthCurve"] * W * 0.048
            return lx + (rx - lx) * u, (1 - u) ** 2 * ly + 2 * (1 - u) * u * cym + u * u * ly
        if t < 0.33:
            u = self._f01(p)
            x = lx + (rx - lx) * u
            top = my - open_h * 0.42 - e["mouthCurve"] * W * 0.02
            bot = my + open_h * 0.62
            return x, top + self._f01(p) * (bot - top)
        u = (t - 0.33) * 3 if t < 0.66 else (t - 0.66) * 3
        if t < 0.66:
            cym = my - open_h * 0.55 - e["mouthCurve"] * W * 0.02
            return lx + (rx - lx) * u, (1 - u) ** 2 * ly + 2 * (1 - u) * u * cym + u * u * ly
        cym = my + open_h * 1.15
        return lx + (rx - lx) * u, (1 - u) ** 2 * ly + 2 * (1 - u) * u * cym + u * u * ly

    def _nose(self, t, p):
        y0, y1 = CY - W * 0.012, CY + W * 0.078
        u = t
        return (CX + math.sin(u * math.pi) * W * 0.012 - 0.004 * W
                + (0.06 * W * (u - 0.8) if u > 0.8 else 0)
                + fbm(p["seed"], self._tnow * 1) * 0.004 * W,
                y0 + (y1 - y0) * u)

    def _tear(self, t, p):
        side = 1 if int(p["seed"]) % 2 else -1
        ph = (self._tnow * 0.6 * (0.6 + 0.8 * self._f01(p)) + t * 0.35) % 1
        return CX + side * (EX + EW * 0.72), EY + EW * 0.5 + ph * W * 0.16

    # ── sprites (soft puffs with internal wispy erosion) ─────────────
    def _build_sprites(self, hue):
        self.sprites = []
        for v in range(4):
            S = 96
            img = QImage(S, S, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            pt = QPainter(img)
            jx = S / 2 + (_RND() - 0.5) * S * 0.12
            jy = S / 2 + (_RND() - 0.5) * S * 0.12
            g = QRadialGradient(jx, jy, S / 2)  # simple radial: focal ctor no-ops on some Qt builds
            g.setColorAt(0.0, hsla(hue, 0.34, 0.90, 0.95))
            g.setColorAt(0.35, hsla(hue, 0.31, 0.76, 0.38))
            g.setColorAt(0.7, hsla(hue, 0.26, 0.62, 0.10))
            g.setColorAt(1.0, hsla(hue, 0.22, 0.55, 0.0))
            pt.fillRect(0, 0, S, S, QBrush(g))
            pt.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            for _ in range(5):
                a = _RND() * math.pi * 2
                r = S * (0.15 + _RND() * 0.22)
                cx2, cy2 = S / 2 + math.cos(a) * r, S / 2 + math.sin(a) * r
                g2 = QRadialGradient(cx2, cy2, S * (0.1 + _RND() * 0.12))
                g2.setColorAt(0.0, QColor(0, 0, 0, 140))
                g2.setColorAt(1.0, QColor(0, 0, 0, 0))
                pt.fillRect(0, 0, S, S, QBrush(g2))
            pt.end()
            self.sprites.append(img)
        self._sprite_hue = hue

    # ── flow field ───────────────────────────────────────────────────
    def _build_flow(self, t):
        for gy in range(FG):
            for gx in range(FG):
                i = gy * FG + gx
                x, y = gx * 0.35, gy * 0.35
                self._fx[i] = noise2(x + t * 0.8, y)
                self._fy[i] = noise2(y - 17, x + t * 0.8) - 0.25

    def _flow_at(self, px, py):
        gx = px / CELL
        gy = py / CELL
        gx = min(max(gx, 0.0), FG - 1.001)
        gy = min(max(gy, 0.0), FG - 1.001)
        x0, y0 = int(gx), int(gy)
        u, v = gx - x0, gy - y0
        i = y0 * FG + x0
        a, b = self._fx[i], self._fx[i + 1]
        c, d = self._fx[i + FG], self._fx[i + FG + 1]
        a2, b2 = self._fy[i], self._fy[i + 1]
        c2, d2 = self._fy[i + FG], self._fy[i + FG + 1]
        r0 = a + (b - a) * u
        r1 = c + (d - c) * u
        r0b = a2 + (b2 - a2) * u
        r1b = c2 + (d2 - c2) * u
        return r0 + (r1 - r0) * v, r0b + (r1b - r0b) * v

    # ── tick + paint ─────────────────────────────────────────────────
    def _tick(self):
        t_now = time.monotonic() - self._t0
        dt = 1 / 30
        self._speech_tick(dt)
        for k in PKEYS:
            d = self.target[k] - self.P[k]
            self.P[k] += d if abs(d) < 0.002 else d * 0.045
        if abs(self.P["hue"] - self._sprite_hue) > 1.5:
            self._build_sprites(self.P["hue"])
        if self._face_timer > 0:
            self._face_timer -= 1
        self.update()

    def paintEvent(self, event):
        t_now = time.monotonic() - self._t0
        self._tnow = t_now
        e = self._eff()
        my = CY + W * 0.115 + e["mouthOpen"] * W * 0.02

        swx = fbm(t_now * 0.22, 11) * W * 0.014
        swy = fbm(t_now * 0.28, 77) * W * 0.011
        breath = 1 + fbm(t_now * 0.45, 5) * 0.014 + self._speak_level * 0.004

        pt = QPainter(self)
        pt.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        scale = min(self.width() / W, self.height() / H)
        pt.translate((self.width() - W * scale) / 2, (self.height() - H * scale) / 2)
        pt.scale(scale, scale)

        # bg vignette
        bg = QRadialGradient(CX, CY, W * 0.75)
        bg.setColorAt(0.0, QColor(10, 15, 22))
        bg.setColorAt(1.0, QColor(3, 5, 10))
        pt.fillRect(QRectF(0, 0, W, H), QBrush(bg))

        # ── particle layer -> half-res buffer ──
        self._buf.fill(Qt.GlobalColor.transparent)
        bp = QPainter(self._buf)
        bp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        bp.scale(0.5, 0.5)
        bp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        self._build_flow(t_now)
        turb = self.P["turb"] + self._speak_level * 0.15
        spd = self.P["spd"]
        breath_puffs = self._speak_level > 0.25
        for p in self.parts:
            p["t"] += p["sp"] * 0.003 * spd * 33 / 1000 * 30
            if p["t"] > 1:
                p["t"] -= 1
            elif p["t"] < 0:
                p["t"] += 1
            feat = p["feat"]
            if feat == "breath":
                if breath_puffs and p["life"] <= 0 and _RND() < 0.12:
                    p["life"] = 1.0
                    p["x"] = CX + (_RND() - 0.5) * W * 0.05
                    p["y"] = my + W * 0.02
                    p["vx"] = (_RND() - 0.5) * 0.5
                    p["vy"] = 0.35 + _RND() * 0.5
                if p["life"] <= 0:
                    continue
                p["life"] -= 0.0009 * 33
                p["x"] += p["vx"] * 33 * 0.06 + fbm(p["x"] * 0.02, t_now * 2) * 0.5
                p["y"] += p["vy"] * 33 * 0.06
                bl = max(0.0, p["life"])
                bs = p["size"] * (1.6 - bl * 0.6)
                bp.setOpacity(min(1.0, p["alpha"] * bl))
                bp.drawImage(QRectF(p["x"] - bs, p["y"] - bs, bs * 2, bs * 2), self.sprites[p["spr"]])
                continue
            if feat == "haze":
                tgt = self._haze(p)
            elif feat == "hair":
                tgt = self._hair_tgt(p, t_now)
            elif feat in ("outline", "wisp"):
                tgt = self._outline(p["t"])
            elif feat == "eyeL":
                tgt = self._eye(-1, p["t"], e)
            elif feat == "eyeR":
                tgt = self._eye(1, p["t"], e)
            elif feat == "browL":
                tgt = self._brow(-1, p["t"], e)
            elif feat == "browR":
                tgt = self._brow(1, p["t"], e)
            elif feat == "mouth":
                tgt = self._mouth(p["t"], e, p, my)
            elif feat == "nose":
                tgt = self._nose(p["t"], p)
            else:
                tgt = self._tear(p["t"], p)
            fx = CX + (tgt[0] - CX) * breath + swx
            fy = CY + (tgt[1] - CY) * breath + swy
            wscale = 2.2 if feat == "wisp" else 1.0
            flx, fly = self._flow_at(p["x"], p["y"])
            m = W * 0.019 * turb * wscale
            k = 1 - math.exp(-0.0075 * 33 * (0.6 if feat == "tear" else (1.6 if feat == "wisp" else 1.0)))
            p["x"] += (fx + flx * m - p["x"]) * k
            p["y"] += (fy + fly * m - p["y"]) * k
            p["ph"] += 0.006 * 33 * p["w"]
            flick = 0.72 + 0.28 * math.sin(p["ph"])
            a = p["alpha"] * flick * (0.9 + self._speak_level * 0.25)
            if feat == "tear":
                a *= max(0.0, self.P["tear"])
            bp.setOpacity(min(1.0, a))
            s = p["size"]
            bp.drawImage(QRectF(p["x"] - s, p["y"] - s, s * 2, s * 2), self.sprites[p["spr"]])
        bp.end()

        # ── hair mass + strands buffer ──
        self._draw_hair(pt, t_now)

        # ── anatomy detail (dark strokes BEFORE additive smoke) ──
        self._draw_anatomy(pt, e, my, t_now)

        # ── smoke blit on top, additive ──
        pt.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        pt.drawImage(QRectF(0, 0, W, H), self._buf)
        pt.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # ── text overlays ──
        self._draw_labels(pt)
        pt.end()

    # ── hair mass / halo / strands / shine ──────────────────────────
    def _draw_hair(self, pt: QPainter, t_now):
        self._hair.fill(Qt.GlobalColor.transparent)
        hp = QPainter(self._hair)
        hp.setRenderHint(QPainter.CompositionMode and QPainter.RenderHint.Antialiasing, True)
        hue = int(self.P["hue"]) % 360
        path = QPainterPath()
        path.moveTo(CX - RX * 1.34, CY + RY * 1.55)
        path.cubicTo(CX - RX * 1.5, CY + RY * 0.2, CX - RX * 1.3, CY - RY * 1.05, CX - RX * 0.6, CY - RY * 1.32)
        path.quadTo(CX, CY - RY * 1.5, CX + RX * 0.6, CY - RY * 1.32)
        path.cubicTo(CX + RX * 1.3, CY - RY * 1.05, CX + RX * 1.5, CY + RY * 0.2, CX + RX * 1.34, CY + RY * 1.55)
        path.quadTo(CX, CY + RY * 1.25, CX - RX * 1.34, CY + RY * 1.55)
        path.closeSubpath()
        grad = QLinearGradient(CX, CY - RY * 1.5, CX, CY + RY * 1.9)
        grad.setColorAt(0.0, hsla(hue, 0.26, 0.62, 0.22))
        grad.setColorAt(0.6, hsla(hue, 0.24, 0.52, 0.17))
        grad.setColorAt(1.0, hsla(hue, 0.22, 0.44, 0.10))
        hp.fillPath(path, QBrush(grad))
        hp.setPen(QPen(hsla(hue, 0.28, 0.80, 0.07), W * 0.02))
        hp.drawPath(path)
        # feather-erase face opening
        hp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        go = QRadialGradient(CX, CY, RX * 1.02)
        go.setColorAt(0.0, QColor(0, 0, 0, 255))
        go.setColorAt(0.72, QColor(0, 0, 0, 242))
        go.setColorAt(1.0, QColor(0, 0, 0, 0))
        hp.setBrush(QBrush(go))
        hp.setPen(Qt.PenStyle.NoPen)
        hp.drawEllipse(QPointF(CX, CY + RY * 0.02), RX * 0.98, RY * 1.02)
        hp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        # flowing strands (light/dark alternating, tapering)
        for i, L in enumerate(self.strand_defs):
            mx, my2 = (L[0] + L[2]) / 2, (L[1] + L[3]) / 2
            cxq, cyq = mx + (mx - CX) * 0.42, my2 * 1.02 + CY * (-0.02)
            pts = []
            for s in range(15):
                u, bu = s / 14, 1 - s / 14
                x = bu * bu * L[0] + 2 * bu * u * cxq + u * u * L[2]
                y = bu * bu * L[1] + 2 * bu * u * cyq + u * u * L[3]
                x += math.sin(t_now * 1.2 * L[4] + u * L[5] + i) * W * 0.006 * (1 + u * 2)
                y += math.cos(t_now * L[4] + i * 1.7) * W * 0.003
                pts.append(QPointF(x, y))
            for seg in range(3):
                a0, a1 = seg * 5, min(14, seg * 5 + 5)
                t = (seg + 0.5) / 3
                light = i % 3 != 0
                al = 0.05 * (1 - t * 0.5)
                col = hsla(hue, 0.26, 0.78, al) if light else hsla(hue, 0.20, 0.20, al)
                hp.setPen(QPen(col, W * 0.004 * (1 - t * 0.55), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                path2 = QPainterPath(pts[a0])
                for s in range(a0 + 1, a1 + 1):
                    path2.lineTo(pts[s])
                hp.drawPath(path2)
        # crown shine
        for sgn in (-1, 1):
            gS = QLinearGradient(CX + sgn * RX * 0.1, CY - RY * 1.4, CX + sgn * RX * 1.05, CY - RY * 0.6)
            gS.setColorAt(0.0, QColor(255, 255, 255, 0))
            gS.setColorAt(0.5, QColor(255, 255, 255, 26))
            gS.setColorAt(1.0, QColor(255, 255, 255, 0))
            hp.setPen(QPen(QBrush(gS), W * 0.011, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            sp = QPainterPath(QPointF(CX + sgn * RX * 0.12, CY - RY * 1.34))
            sp.quadTo(CX + sgn * RX * 0.7, CY - RY * 1.3, CX + sgn * RX * 1.06, CY - RY * 0.55)
            hp.drawPath(sp)
        hp.setPen(QPen(hsla(hue, 0.10, 0.02, 0.4), W * 0.004, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        hp.drawLine(QPointF(CX, CY - RY * 1.42), QPointF(CX + 2, CY - RY * 1.2))
        hp.end()
        pt.drawImage(QRectF(0, 0, W, H), self._hair)

    # ── eyes / nose / mouth anatomy ──────────────────────────────────
    def _blink(self, t_now):
        if t_now >= self._blink_at and self._blink_left <= 0:
            self._blink_left = 0.13
        if self._blink_left > 0:
            self._blink_left -= 1 / 30
            if self._blink_left < 0:
                self._blink_left = 0.0
            return math.sin(math.pi * (1 - max(self._blink_left, 0.0) / 0.13))
        return 0.0

    def _next_blink(self):
        # blink rate -20% vs the web canvas (2.5-7.0s -> 3.1-8.75s)
        return time.monotonic() - self._t0 + random.uniform(3.1, 8.75)

    def _draw_anatomy(self, pt: QPainter, e, my, t_now):
        hue = int(self.P["hue"]) % 360
        blink = self._blink(t_now)
        pt.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for side in (-1, 1):
            open_ = max(e["eyeOpen"] * (1 - blink * 0.96), 0.02)
            eh = EW * 0.44 * open_
            lift = 0 if e["eyeOpen"] > 0.2 else -0.008 * W
            ecx = CX + side * EX
            ecy = EY + e["pupilY"] * W * 0.5 + lift
            tilt = side * EW * 0.14
            y_top = ecy - eh * (0.62 + 0.1 * e["eyeOpen"])
            y_bot = ecy + eh * (0.95 - e["cheekLift"] * 0.3)
            y_top0 = ecy - EW * 0.44 * max(e["eyeOpen"], 0.02) * (0.62 + 0.1 * e["eyeOpen"])
            inn = QPointF(ecx - side * EW * 0.94, ecy + EW * 0.06 + tilt * 0.3)
            out = QPointF(ecx + side * EW, ecy - EW * 0.10 - tilt * 0.5)
            upper = QPainterPath(inn)
            upper.quadTo(ecx, y_top, out.x(), out.y())
            lower = QPainterPath(inn)
            lower.quadTo(ecx, y_bot, out.x(), out.y())
            pt.setPen(QPen(hsla(hue, 0.12, 0.08, 0.43), EW * 0.11,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pt.drawPath(upper)
            pt.setPen(QPen(hsla(hue, 0.12, 0.08, 0.24), EW * 0.08,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pt.drawPath(lower)
            # crease
            cr = QPainterPath(QPointF(ecx - EW * 0.8, y_top0 + EW * 0.22))
            cr.quadTo(ecx, y_top0 - EW * 0.16, ecx + EW * 0.85, y_top0 + EW * 0.10)
            pt.setPen(QPen(hsla(hue, 0.10, 0.06, 0.22), EW * 0.09,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            pt.drawPath(cr)
            # lashes
            pt.setPen(QPen(hsla(hue, 0.10, 0.05, 0.4), EW * 0.055,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for k in range(3):
                tpar = 0.86 - k * 0.14
                bx = ecx + side * EW * tpar
                by = ecy - (0 if tpar == 0.86 else (1 - tpar) * EW * 0.3) + tilt * (tpar - 0.5) * 0.8
                pt.drawLine(QPointF(bx, by),
                            QPointF(bx + side * EW * (0.18 - k * 0.03), by - EW * (0.16 - k * 0.02)))
            # iris/pupil/catchlight clipped to the almond
            if open_ > 0.12:
                almond = QPainterPath(upper)
                almond.lineTo(out)
                almond = QPainterPath()
                almond.moveTo(inn)
                almond.quadTo(ecx, y_top, out.x(), out.y())
                almond.quadTo(ecx, y_bot, inn.x(), inn.y())
                almond.closeSubpath()
                pt.save()
                pt.setClipPath(almond)
                ir = EW * (0.46 + 0.06 * blink)
                iy = ecy + e["pupilY"] * W * 0.35 + ir * 0.04
                gi = QRadialGradient(ecx, iy, ir)
                gi.setColorAt(0.0, hsla(hue, 0.20, 0.88, 0.80))
                gi.setColorAt(0.55, hsla(hue, 0.22, 0.80, 0.55))
                gi.setColorAt(0.8, hsla(hue, 0.15, 0.20, 0.50))
                gi.setColorAt(1.0, hsla(hue, 0.10, 0.10, 0.0))
                pt.setBrush(QBrush(gi))
                pt.setPen(Qt.PenStyle.NoPen)
                pt.drawEllipse(QPointF(ecx, iy), ir, ir)
                pr = max(1.0, ir * (0.42 - 0.10 * e["eyeOpen"]))
                pt.setBrush(hsla(hue, 0.10, 0.02, 0.86))
                pt.drawEllipse(QPointF(ecx, iy), pr, pr)
                pt.setPen(QPen(hsla(hue, 0.10, 0.03, 0.35), EW * 0.05))
                pt.setBrush(Qt.BrushStyle.NoBrush)
                pt.drawEllipse(QPointF(ecx, iy), ir * 0.97, ir * 0.97)
                pt.setBrush(QColor(255, 255, 255, 170))
                pt.setPen(Qt.PenStyle.NoPen)
                cl = max(1.0, ir * 0.16)
                pt.drawEllipse(QPointF(ecx - ir * 0.38, iy - ir * 0.42), cl, cl)
                pt.restore()
        # nose
        tip = CY + W * 0.078
        gB = QLinearGradient(CX - W * 0.02, tip, CX + W * 0.02, tip)
        gB.setColorAt(0.0, QColor(255, 255, 255, 0))
        gB.setColorAt(0.45, QColor(255, 255, 255, 42))
        gB.setColorAt(1.0, QColor(255, 255, 255, 0))
        bridge = QPainterPath(QPointF(CX - W * 0.004, CY - W * 0.006))
        bridge.quadTo(CX + W * 0.002, (CY + tip) / 2, CX, tip - W * 0.006)
        pt.setPen(QPen(QBrush(gB), W * 0.007, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        pt.drawPath(bridge)
        pt.setPen(Qt.PenStyle.NoPen)
        pt.setBrush(hsla(hue, 0.10, 0.03, 0.43))
        for sgn in (-1, 1):
            pt.save()
            pt.translate(CX + sgn * W * 0.0135, tip - W * 0.002)
            pt.rotate(math.degrees(0.35 * sgn))
            pt.drawEllipse(QPointF(0, 0), W * 0.0042, W * 0.0026)
            pt.restore()
        pt.setBrush(hsla(hue, 0.12, 0.80, 0.24))
        pt.drawEllipse(QPointF(CX, tip - W * 0.009), W * 0.009, W * 0.0045)
        # mouth
        half = (0.042 + 0.034 * e["mouthWidth"]) * W
        lift = e["mouthCurve"] * W * 0.014
        open_h = e["mouthOpen"] * W * 0.13
        lx, ly = CX - half, my - lift
        rx = CX + half
        ctop = my - open_h * 0.55 - e["mouthCurve"] * W * 0.02
        cbot = my + open_h * 1.05 + e["mouthCurve"] * W * 0.008
        cav = QPainterPath(QPointF(lx, ly))
        cav.quadTo(CX, ctop, rx, ly)
        cav.quadTo(CX, cbot, lx, ly)
        cav.closeSubpath()
        gC = QLinearGradient(0, my - open_h * 0.6, 0, my + open_h * 1.1)
        gC.setColorAt(0.0, hsla(hue, 0.10, 0.02, 0.75))
        gC.setColorAt(0.6, hsla(hue, 0.10, 0.02, 0.59))
        gC.setColorAt(1.0, hsla(hue, 0.10, 0.02, 0.0))
        pt.setBrush(QBrush(gC))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPath(cav)
        # Cupid's bow
        bow = my - open_h * 0.62 - max(lift, W * 0.004)
        lip = QPainterPath(QPointF(lx, ly))
        lip.quadTo(CX - half * 0.52, bow - W * 0.002, CX - W * 0.006, bow - W * 0.006)
        lip.quadTo(CX, bow + W * 0.004, CX + W * 0.006, bow - W * 0.006)
        lip.quadTo(CX + half * 0.52, bow - W * 0.002, rx, ly)
        pt.setPen(QPen(hsla(hue, 0.14, 0.88, 0.33), W * 0.004,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        pt.setBrush(Qt.BrushStyle.NoBrush)
        pt.drawPath(lip)
        low = QPainterPath(QPointF(lx + half * 0.15, my + open_h * 0.85))
        low.quadTo(CX, cbot + W * 0.010, rx - half * 0.15, my + open_h * 0.85)
        pt.setPen(QPen(hsla(hue, 0.14, 0.92, 0.22), W * 0.0035,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        pt.drawPath(low)

    # ── labels (name / state / face / profile / emotion) ────────────
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
        label = self._state.label
        pt.setPen(QColor(220, 230, 240, 150))
        pt.drawText(QRectF(0, H - 34, W, 24), Qt.AlignmentFlag.AlignHCenter, label)
        emo = self._emotion_mode if self._emotion_mode != "auto" else self._auto_emotion
        pt.setPen(QColor(150, 190, 230, 120))
        pt.drawText(QRectF(0, H - 16, W, 16), Qt.AlignmentFlag.AlignHCenter, emo.upper())
        if self._face_name and self._face_timer > 0:
            pt.setFont(f)
            pt.setPen(QColor(140, 255, 200, min(255, self._face_timer * 3)))
            pt.drawText(QRectF(0, 44, W, 24), Qt.AlignmentFlag.AlignHCenter,
                        f"{self._face_name} ({self._face_confidence:.0%})")
