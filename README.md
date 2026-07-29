# Jarvis Desktop

PyQt6 desktop app with Arc Reactor HUD, face recognition, TTS/STT, and chat with Hermes Agent backend.

## Status

- [x] Phase 1: Arc Reactor HUD overlay with state machine
- [ ] Phase 2: TTS (piper) + STT (whisper)
- [ ] Phase 3: Chat with Hermes backend
- [ ] Phase 4: Face recognition (OpenCV LBPH + MariaDB)
- [ ] Phase 5: Profile switching + polish

## Requirements

- Python 3.10+
- PyQt6
- opencv-contrib-python
- pymysql
- numpy
- requests
- (piper-tts, openai-whisper — Phase 2+)

## Development

```bash
# Create venv
uv venv
source .venv/bin/activate

# Install deps
uv pip install -e ".[dev]"

# Run
python -m src.main
```

## Architecture

```
src/
├── main.py              # Entry point
└── jarvis/
    ├── __init__.py
    ├── state.py         # State machine (IDLE, LISTENING, THINKING, SPEAKING)
    ├── hud_overlay.py   # Arc Reactor HUD widget
    ├── tts.py           # Piper TTS integration
    ├── stt.py           # Whisper STT integration
    ├── chat.py          # Hermes backend chat
    └── face_recognition.py  # OpenCV face recognition
```

## License

MIT
