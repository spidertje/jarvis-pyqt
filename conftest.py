"""Pytest bootstrap: make the ``src/jarvis`` package importable.

Tests import ``jarvis.*`` directly, but the package lives under ``src/`` and
is not pip-installed. Putting ``src`` on ``sys.path`` here is the minimal,
standard way to make ``python -m pytest`` work from a clean checkout without
requiring ``pip install -e .`` or a manual ``PYTHONPATH`` export.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Headless boxes without system OpenGL/EGL (GPU-less CI/servers): PyQt6.QtGui
# fails to import on libEGL.so.1. If local .deb-extracted libs exist, preload
# them RTLD_GLOBAL before anything imports PyQt6. No-op on systems that have
# the real libs (dev laptops / machines with GPU support).
_TESTLIBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".testlibs", "root", "usr", "lib", "x86_64-linux-gnu")
if os.path.isdir(_TESTLIBS):
    import ctypes

    for _lib in ("libglvnd0.so.1", "libEGL.so.1", "libGL.so.1", "libGLESv2.so.2",
                 "libGLX.so.0", "libGLdispatch.so.0", "libEGL_mesa.so.0", "libGLX_mesa.so.0"):
        _p = os.path.join(_TESTLIBS, _lib)
        if os.path.exists(_p):
            try:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
