"""Render one screenshot per HUD frontend (and a couple of emotions each)
into docs/screenshots/. Runs fully headless (offscreen QPA).

    QT_QPA_PLATFORM=offscreen python tools/make_screenshots.py
"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtWidgets import QApplication
from jarvis.hud_overlay import HUDOverlay
from jarvis.smoke_face import SmokeFaceHUD
from jarvis.solid_face import SolidFaceHUD
from jarvis.state import JarvisState

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)

app = QApplication.instance() or QApplication([])


def pump(w, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        app.processEvents()
        time.sleep(0.01)


def shot(w, name):
    w.grab().save(os.path.join(OUT, name))
    print("saved", name)


def main():
    # ── Arc Reactor ──
    arc = HUDOverlay()
    arc.resize(800, 600)
    arc.show()
    arc.set_state(JarvisState.SPEAKING)
    arc.set_activity(0.9)
    arc.set_assistant_name("Jarvis")
    pump(arc, 1.2)
    shot(arc, "arc-reactor.png")
    arc._timer.stop() if hasattr(arc, "_timer") else None
    arc.close()

    # ── Smoke Face (emotive particle face) ──
    smoke = SmokeFaceHUD()
    smoke.resize(800, 600)
    smoke.show()
    smoke.set_assistant_name("Jarvis")
    smoke.set_level_source(lambda: (0.75, True))
    smoke.set_emotion("happy")
    pump(smoke, 1.4)
    shot(smoke, "smoke-face-happy.png")
    smoke.set_emotion("calm")
    smoke.set_level_source(None)
    pump(smoke, 1.4)
    shot(smoke, "smoke-face-calm.png")
    smoke._timer.stop()
    smoke.close()

    # ── Solid figures: smiley / goddess / bender ──
    for figure, emotions in (
        ("smiley", [("happy", "happy"), ("sad", "sad")]),
        ("goddess", [("happy", "happy"), ("angry", "angry")]),
        ("bender", [("laugh", "laugh"), ("angry", "angry")]),
    ):
        w = SolidFaceHUD(figure)
        w.resize(800, 600)
        w.show()
        w.set_assistant_name("Jarvis")
        for emo, tag in emotions:
            w.set_emotion(emo)
            w.set_level_source(lambda: (0.75, True) if emo in ("happy", "laugh", "angry") else None)
            pump(w, 1.4)
            name = f"solid-{figure}-{tag}.png" if len(emotions) > 1 else f"solid-{figure}.png"
            shot(w, name)
        w._timer.stop()
        w.close()

    print("done:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
