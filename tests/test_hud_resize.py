"""HUD paints correctly at any window size (audit G2: fixed 800x600 geometry)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from jarvis.hud_overlay import HUDOverlay, JarvisState


def test_hud_renders_at_any_size(qapp):
    hud = HUDOverlay()
    hud.set_state(JarvisState.SPEAKING)
    for w, h in ((800, 600), (1280, 720), (2560, 1440), (500, 400)):
        hud.resize(w, h)
        pm = hud.grab()
        assert not pm.isNull()
        assert pm.width() == w and pm.height() == h
        # Non-black pixels exist = something was drawn, not a blank frame.
        img = pm.toImage()
        lit = sum(
            1 for x in range(0, w, 40) for y in range(0, h, 40)
            if img.pixelColor(x, y).lightness() > 40
        )
        assert lit > 0, f"blank HUD frame at {w}x{h}"
