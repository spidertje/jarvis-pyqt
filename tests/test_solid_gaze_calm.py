"""Gaze-calmness regression for the solid HUD frontends.

The web-port saccade baseline (retarget every 0.7-2.9s at full amplitude,
fast glide) read as constant eye-darting at HUD scale. The fix retargets
rarer (1.5-5s), scales amplitude with emotional turbulence, and glides
more gently. These tests simulate the saccade tick directly and assert
the measured behaviour, not the constants.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import random  # noqa: E402

import pytest  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

# QWidget construction aborts without a QApplication instance
_app = QApplication.instance() or QApplication([])

from jarvis.solid_face import SolidFaceHUD  # noqa: E402


def _simulate(hud, emotion, seconds=30, seed=1234):
    """Run _saccade at the 30fps tick budget with the emotion's turb settled."""
    random.seed(seed)
    hud.set_emotion(emotion)
    # settle the P-springs toward the emotion target (turb drives amplitude)
    tgt = hud.target
    for k in hud.P:
        hud.P[k] = tgt[k]
    # deterministic fresh gaze state
    hud._sac_at = 0.0
    hud._sac_x = hud._sac_y = hud._sac_tx = hud._sac_ty = 0.0
    t = 0.0
    dt = 1 / 30
    xs = []
    target_changes = 0
    prev_tx = hud._sac_tx
    for _ in range(int(seconds / dt)):
        hud._saccade(t, dt)
        if hud._sac_tx != prev_tx:
            target_changes += 1
            prev_tx = hud._sac_tx
        xs.append(hud._sac_x)
    t += dt
    peak = max(abs(x) for x in xs)
    return peak, target_changes


@pytest.mark.parametrize("figure", ["smiley", "goddess", "bender"])
class TestGazeCalm:
    def test_calm_gaze_stays_near_center(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        peak, changes = _simulate(hud, "calm")
        # old baseline peak approached 0.25; calm (turb<=0.10) must stay
        # well under half that
        assert peak < 0.08, f"{figure} calm gaze peak excursion {peak:.3f}"

    def test_saccade_rate_is_calm(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        _, changes = _simulate(hud, "neutral", seconds=30)
        # old cadence: ~13-16 retargets per 30s; new: ~6-12 expected
        assert changes <= 12, f"{figure}: {changes} retargets in 30s (too darty)"
        assert changes >= 1, f"{figure}: gaze frozen (no life)"

    def test_emotion_modulates_amplitude(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        peak_calm, _ = _simulate(hud, "calm")
        peak_fear, _ = _simulate(hud, "fear")
        assert peak_fear > peak_calm * 1.5, (
            f"{figure}: fear ({peak_fear:.3f}) should dart clearly more "
            f"than calm ({peak_calm:.3f})"
        )
        # still bounded: fear peak must not exceed the old baseline max
        assert peak_fear < 0.25, f"{figure}: fear excursion {peak_fear:.3f}"

    def test_widget_constructible_with_new_code(self, figure):
        # smoke: full constructor path (not __new__+manual init) still works
        hud = SolidFaceHUD(figure)
        hud.set_emotion("happy")
        for _ in range(30):
            hud._saccade(1.0 + _ / 30, 1 / 30)
        assert -1.0 < hud._sac_x < 1.0
