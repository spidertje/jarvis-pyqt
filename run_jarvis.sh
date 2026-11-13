#!/usr/bin/env bash
# Script to run Jarvis PyQt6 application.
#
# Usage:
#   1. Copy .env.template to .env
#   2. Edit .env with your configuration
#   3. ./run_jarvis.sh

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    set -a  # export all variables
    source .env
    set +a
else
    echo "⚠️  No .env file found. Copy .env.template to .env and configure it."
    echo "   Falling back to environment variables or defaults."
fi

# Change to the directory where you cloned jarvis-pyqt (if not already there)
cd "$(dirname "$0")"

# Prefer the project venv if present: wake-word deps (openwakeword,
# onnxruntime) are installed only there — bare system python3 silently
# disables the wake word ("No module named 'openwakeword'").
PY="python3"
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
fi

# Run the application
exec "$PY" src/main.py
