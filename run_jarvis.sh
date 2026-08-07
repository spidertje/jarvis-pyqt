#!/usr/bin/env bash
# Script to run Jarvis PyQt6 application with environment variables set for the user's environment.

# Database connection
export JARVIS_DB_HOST=192.168.55.41
export JARVIS_DB_USER=root
export JARVIS_DB_PASSWORD=rocklobster
export JARVIS_DB_NAME=jarvis

# LLM endpoint (defaults to local Hermes API)
export JARVIS_LLM_URL="http://192.168.55.179:8642/v1"
export JARVIS_LLM_BASE_URL="http://192.168.55.179:8642/v1"
# LLM API key (from Hermes gateway)
export JARVIS_LLM_API_KEY="freellmapi-0d9e3106805c5ef86625a1a4256b96ca20e494f775ace34f"

# STT endpoint (Wyoming faster-whisper on main server 192.168.55.41)
export JARVIS_STT_HOST="192.168.55.41"
export JARVIS_STT_PORT="10300"

# Change to the directory where you cloned jarvis-pyqt (if not already there)
cd "$(dirname "$0")"

# Run the application
python3 src/main.py