# Jarvis Desktop

PyQt6 Arc Reactor HUD with voice interaction.

## Stack

- **HUD**: PyQt6 custom widget with Arc Reactor animation (60fps)
- **STT**: Faster-whisper via Wyoming protocol (port 10300 on 192.168.55.41)
- **TTS**: Piper via Wyoming protocol (port 10200 on 192.168.55.41)
- **Chat**: OpenAI-compatible API (port 3001 on 192.168.55.43)
- **Audio**: sounddevice for PCM playback

## Phases

1. **Phase 1** — Arc Reactor HUD overlay with state machine
2. **Phase 2** — TTS/STT integration (piper + whisper via Wyoming)
3. **Phase 3** — Chat with LLM backend + voice mode
4. **Phase 4** — Face recognition (OpenCV LBPH + MariaDB)
5. **Phase 5** — Profile switching + polish

## Requirements

```
pip install PyQt6 aiohttp numpy sounddevice
```

## Usage

```bash
cd /opt/data/jarvis-pyqt
python3 src/main.py
```

## Architecture

```
┌─────────────────────────────────────────┐
│  JarvisApp (QWidget)                    │
│  ┌─────────────────────────────────┐    │
│  │ HUDOverlay (background)         │    │
│  │ - Arc Reactor animation         │    │
│  │ - 60fps state machine           │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ Bottom bar: mic, status, input  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  JarvisAgent                            │
│  STT → Chat → TTS                       │
│  State: IDLE ↔ LISTENING → THINKING →   │
│         SPEAKING → IDLE                 │
└─────────────────────────────────────────┘
```

## Services

| Service | Host | Port | Protocol |
|---------|------|------|----------|
| Piper TTS | 192.168.55.41 | 10200 | Wyoming |
| Whisper STT | 192.168.55.41 | 10300 | Wyoming |
| LLM API | 192.168.55.43 | 3001 | OpenAI |

## Configuration

Edit `src/main.py` → `JarvisApp.__init__()` → `AgentConfig()` for:
- LLM endpoint (`ChatConfig`)
- Wyoming hosts/ports (`STTWyomingConfig`, `TTSWyomingConfig`)
- System prompt (`AgentConfig.system_prompt`)
