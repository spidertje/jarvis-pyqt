"""Headless verification of SmokeFaceHUD in jarvis-pyqt."""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
from PyQt6.QtCore import QCoreApplication

app = QApplication(sys.argv)

# The package __init__ is lazy (PEP 562), so plain imports are cheap now and
# pull in only what is used. No fake-module shim needed (the old shim poisoned
# sys.modules["jarvis"] for every later test and broke monkeypatch targets).
from jarvis.smoke_face import SmokeFaceHUD
from jarvis.state import JarvisState

w = SmokeFaceHUD()
w.resize(800, 600)
w.show()

def pump(seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        QCoreApplication.processEvents()
        time.sleep(0.01)

import numpy as np

def snap(tag):
    img = w.grab().toImage()
    img.save(f"/tmp/smoke_{tag}.png")
    d = img.convertToFormat(QImage.Format.Format_RGBA8888)
    W2, H2 = d.width(), d.height()
    ptr = d.constBits()
    ptr.setsize(d.bytesPerLine() * d.height())
    buf = np.frombuffer(ptr, np.uint8).reshape(d.height(), d.bytesPerLine())
    px = buf[:, : W2 * 4].reshape(H2, W2, 4)[:, ::3, :3][:, ::3, :]
    v = px.max(axis=2)
    n = v.size
    sat = int((v >= 250).sum()); mid = int((v > 90).sum()); dark = int((v <= 20).sum())
    print(f"[{tag}] sat={100*sat/n:.1f}% mid={100*mid/n:.1f}% dark={100*dark/n:.1f}%")

pump(1.0)
snap("idle")

# emotion switch
w.set_emotion("angry")
pump(2.0)
snap("angry")

# speaking with fake level source
levels = {"lvl": 0.0, "play": False}
w.set_level_source(lambda: (levels["lvl"], levels["play"]))
levels["play"] = True
import random
def fluctuate():
    levels["lvl"] = random.uniform(0.2, 0.9)
for _ in range(40):
    fluctuate()
    pump(0.06)
# sample mouth region during speech (center-lower face)
img = w.grab().toImage()
img.save("/tmp/smoke_speaking.png")
print("[speaking] level_source active, captured")

# back to auto
levels["play"] = False
w.set_emotion("auto")
w.set_state(JarvisState.SPEAKING)
pump(1.0)
snap("auto_speaking_state")

# performance
from PyQt6.QtCore import QTimer
frames = {"n": 0}
w.paint_count = 0
orig_pe = w.paintEvent
def counting_pe(ev):
    w.paint_count += 1
    orig_pe(ev)
w.paintEvent = counting_pe
pump(3.0)
print(f"[perf] ~{w.paint_count/3:.1f} fps headless")

# emotion tag module tests
from jarvis.emotion import parse_emotion, classify_emotion, EmotionTagFilter
assert parse_emotion("[emo:happy] hello there") == ("happy", "hello there")
assert parse_emotion("no tag here") == (None, "no tag here")
assert classify_emotion("that is dangerous, be careful") == "fear"
f = EmotionTagFilter()
out = ""
for tok in ["[emo", ":sad", "] sorry about ", "that"]:
    out += f.feed(tok)
out += f.flush()
assert f.emotion == "sad", f.emotion
assert out == "sorry about that", repr(out)
print("[emotion] tag filter OK:", out)
print("ALL_OK")
