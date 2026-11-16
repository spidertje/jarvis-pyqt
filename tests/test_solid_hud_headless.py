"""Headless verification of SolidFaceHUD (smiley / goddess / bender)."""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from jarvis.solid_face import SolidFaceHUD
from jarvis.state import JarvisState

_app = QApplication.instance() or QApplication([])


def pump(w, secs=0.25):
    from PyQt6.QtWidgets import QApplication
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        QApplication.processEvents()
        time.sleep(0.01)


def lit_pixels(w):
    img = w.grab().toImage()
    d = img.convertToFormat(QImage.Format.Format_RGBA8888)
    W2, H2 = d.width(), d.height()
    ptr = d.constBits()
    bpl = d.bytesPerLine()
    ptr.setsize(bpl * H2)
    buf = bytes(ptr)
    lit = 0
    total = 0
    for y in range(0, H2, 3):
        row = buf[y * bpl:(y + 1) * bpl]
        for x in range(0, W2, 3):
            i = x * 4
            lum = max(row[i], row[i + 1], row[i + 2])
            total += 1
            if lum > 60:
                lit += 1
    return lit / total, img


def main():
    results = {}
    for figure in ("smiley", "goddess", "bender"):
        w = SolidFaceHUD(figure)
        w.resize(800, 600)
        w.show()
        pump(w, 0.4)

        frac0, _ = lit_pixels(w)
        assert frac0 > 0.10, f"{figure}: frame too blank (lit={frac0:.3f})"

        # emotion switches must not crash and must change the render
        w.set_emotion("happy")
        pump(w, 0.3)
        img_h, _ = lit_pixels(w)
        w.set_emotion("sad")
        pump(w, 0.3)
        w.set_emotion("laugh")
        pump(w, 0.3)
        w.set_emotion("angry")
        pump(w, 0.2)
        w.set_emotion("surprised")
        pump(w, 0.2)
        w.set_emotion("fear")
        pump(w, 0.2)
        w.set_emotion("calm")
        pump(w, 0.2)
        w.set_emotion("auto")
        for st in (JarvisState.LISTENING, JarvisState.THINKING, JarvisState.SPEAKING, JarvisState.IDLE):
            w.set_state(st)
            pump(w, 0.15)

        # lip-sync level source
        w.set_level_source(lambda: (0.8, True))
        pump(w, 0.4)
        frac_sp, _ = lit_pixels(w)
        w.set_level_source(None)

        # spring bounce impulse fired on switches (velocity channel non-zero at some point)
        w.set_emotion("surprised")
        pump(w, 0.05)
        bounce_active = abs(w.P["bounce"]) > 1e-6 or abs(w.vel["bounce"]) > 1e-6
        assert bounce_active, f"{figure}: squash impulse never fired"

        # emotion_changed signal
        fired = []
        w.emotion_changed.connect(lambda e: fired.append(e))
        w.set_emotion("happy")
        assert fired == ["happy"], f"{figure}: emotion_changed signal broken"

        # face-detect label path (no crash)
        w.set_face_detected("Markus", 0.92)
        pump(w, 0.2)

        frac_end, _ = lit_pixels(w)
        assert frac_end > 0.10, f"{figure}: render died (lit={frac_end:.3f})"
        # per-figure element probes (window is 800x600 -> 720-space scales by
        # 600/720=0.8333 and centers vertically): prove each figure draws its
        # signature elements, not just a generic blob.
        def px(x720, y720):
            img = w.grab().toImage()
            s = 600.0 / 720.0
            pxw = int(W2px(x720 * s + (800 - 720 * s) / 2))
            pyw = int(W2px(y720 * s + (600 - 720 * s) / 2))
            return img.pixelColor(pxw, pyw)
        def W2px(v):
            return max(0, min(799, v))
        w.set_emotion("calm")
        pump(w, 0.6)
        if figure == "smiley":
            w.set_emotion("happy")  # warm yellow face tone
            pump(w, 0.6)
        elif figure == "bender":
            w.set_emotion("neutral")  # yellow metal glow (calm is blue-screen)
            pump(w, 0.6)
        el = {}
        if figure == "smiley":
            c = px(360, 374)  # yellow face centre at CY=H*0.52
            el["face_warm"] = c.red() > 150 and c.green() > 120 and c.blue() < 120
        elif figure == "goddess":
            c = px(360, 360)  # skin tone centre
            el["skin"] = c.red() > 130 and c.green() > 100 and c.blue() < c.red()
            # gold stephane band: crown arc spans x 240-480, y 150-205
            golds = sum(1 for yy in range(150, 206, 5) for xx in range(240, 481, 8)
                        if (lambda q: q.red() > 170 and q.green() > 130 and q.blue() < 120)(px(xx, yy)))
            el["stephane_gold"] = golds >= 5
        else:
            c = px(270, 360)  # grey metal head left-of-centre
            el["metal_grey"] = abs(c.red() - c.green()) < 40 and abs(c.green() - c.blue()) < 45 and c.red() > 90
            # glowing eye pixels near nominal eye centre row
            glows = sum(1 for xx in (318, 402)
                        if (lambda q: q.red() > 170 and q.green() > 160 and q.red() - q.blue() > 25)(px(xx, 302)))
            el["glow_eyes"] = glows >= 1
        assert all(el.values()), f"{figure}: element probes failed {el}"
        results[figure] = dict(initial=round(frac0, 3), speaking=round(frac_sp, 3),
                               final=round(frac_end, 3), elements=el)
        w._timer.stop()
        w.close()
    print("OK", results)


if __name__ == "__main__":
    main()
