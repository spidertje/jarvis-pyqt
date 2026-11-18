"""Brow-motion calmness regression for the solid HUD frontends.

Bug fixed 2026-11-18 (user: 'distracted by the movement of the eyebrows'):
_micro()'s timer was armed once in __init__ and never re-armed, so after
the first pass the gate re-fired every 0.26s — brows bounced non-stop.
Fix: re-arm (6-16s between raises), slower rise (0.5s instead of 0.26s),
and amplitude damped by turbulence (idle brows ~15% amplitude).
Tests measure the simulated micro signal, not the constants.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import random  # noqa: E402

import pytest  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from jarvis.solid_face import SolidFaceHUD  # noqa: E402


def _simulate(hud, emotion, seconds=30, seed=7):
    random.seed(seed)
    hud.set_emotion(emotion)
    for k in hud.P:
        hud.P[k] = hud.target[k]
    hud._micro_at = 3.0
    hud._micro_left = 0.0
    vals = []
    t = 0.0
    dt = 1 / 30
    for _ in range(int(seconds / dt)):
        vals.append(hud._micro(t))
        t += dt
    raises = 0
    in_r = False
    active = 0
    for v in vals:
        if v > 0.05:
            active += 1
            if not in_r:
                raises += 1
                in_r = True
        else:
            in_r = False
    damp = hud._brow_damp(hud._eff())
    return raises, active / len(vals), damp


@pytest.mark.parametrize("figure", ["smiley", "goddess", "bender"])
class TestBrowCalm:
    def test_brows_mostly_still(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        _, active_frac, _ = _simulate(hud, "calm")
        # old bug: active_frac ~1.0 (continuous bounce)
        assert active_frac < 0.2, f"{figure}: brows {active_frac:.0%} of frames moving"

    def test_raise_cadence_is_rare(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        raises, _, _ = _simulate(hud, "neutral", seconds=30)
        assert 1 <= raises <= 5, f"{figure}: {raises} brow raises in 30s"

    def test_idle_amplitude_damped(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        _, _, damp_calm = _simulate(hud, "calm")
        _, _, damp_fear = _simulate(hud, "fear")
        assert damp_calm < 0.3, f"{figure}: idle brow damp {damp_calm:.2f} too strong"
        assert damp_fear > 0.8, f"{figure}: expressive damp {damp_fear:.2f} too weak"
