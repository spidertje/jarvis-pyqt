"""Blink-calmness regression for the solid HUD frontends.

Two bugs fixed 2026-11-18 (user: 'the speed of the eyelids is still the same'):
1. _blink_at was armed ONCE in __init__ and never re-armed, so the gate
   `t_now >= _blink_at and _blink_left <= 0` re-fired the instant a blink
   ended: eyelids fluttered continuously.
2. The blink itself was 0.13s (~4 frames at 30fps) — a snap that read as
   twitching; now 0.32s, near-natural tempo.
Tests measure the simulated blink signal, not the constants.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import random  # noqa: E402

import pytest  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from jarvis.solid_face import SolidFaceHUD  # noqa: E402


def _simulate(hud, seconds=30, seed=99):
    """Sample _blink() at the 30fps tick budget; return (blinks, closed_frac)."""
    random.seed(seed)
    hud._blink_at = hud._next_blink()
    hud._blink_left = 0.0
    vals = []
    t = 0.0
    dt = 1 / 30
    for _ in range(int(seconds / dt)):
        vals.append(hud._blink(t))
        t += dt
    # a blink = contiguous run of nonzero samples
    blinks = 0
    in_b = False
    closed = 0
    for v in vals:
        if v > 0.05:
            closed += 1
            if not in_b:
                blinks += 1
                in_b = True
        else:
            in_b = False
    return blinks, closed / len(vals), vals


@pytest.mark.parametrize("figure", ["smiley", "goddess", "bender"])
class TestBlinkCalm:
    def test_lids_mostly_open(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        _, closed_frac, _ = _simulate(hud)
        # flutter-bug baseline had closed_frac ~1.0; real blinking ~5-10%
        assert closed_frac < 0.25, f"{figure}: lids {closed_frac:.0%} closed (flutter)"

    def test_blink_cadence_humanlike(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        blinks, _, _ = _simulate(hud, seconds=30)
        # ~1 blink every 3.5-8.5s -> 3-9 per 30s; flutter would be dozens
        assert 1 <= blinks <= 9, f"{figure}: {blinks} blinks in 30s"

    def test_blink_duration_is_gentle(self, figure):
        hud = SolidFaceHUD.__new__(SolidFaceHUD)
        SolidFaceHUD.__init__(hud, figure)
        _, _, vals = _simulate(hud)
        runs = []
        run = 0
        for v in vals:
            if v > 0.05:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        assert runs, f"{figure}: no blinks observed"
        d = max(runs) / 30
        assert d >= 0.2, f"{figure}: longest blink {d:.2f}s too snappy"
