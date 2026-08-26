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
