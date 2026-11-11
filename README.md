# Jarvis Desktop

PyQt6 Arc Reactor HUD with voice interaction.

## Stack

- **HUD**: switchable frontends — Arc Reactor (60fps) or **Smoke Face** (emotive face sculpted from smoke particles; mouth lip-syncs to TTS PCM, emotion set by LLM `[emo:x]` tags)
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

## HUD Frontends

Two interchangeable faces, selectable live:

- **Arc Reactor** — the original concentric-ring HUD.
- **Smoke Face** (`src/jarvis/smoke_face.py`) — a female face built from ~1,100
  drifting smoke particles with painted anatomy (almond eyes with iris/catchlight,
  Cupid's-bow mouth, hair locks). It reacts to the conversation:
  - **Emotion from the LLM**: the agent appends an instruction to the system prompt;
    replies start with `[emo:happy]`-style tags which are stripped before TTS and
    drive the face. A keyword classifier catches replies with no tag.
  - **Lip-sync**: per-playback-chunk RMS from `AudioPlayer` (`get_level()`) drives
    viseme blending — the mouth moves with what is actually spoken.
  - State machine mapping (auto mode): IDLE→calm, LISTENING→neutral,
    THINKING→surprised, SPEAKING→happy.

Switch via **Settings → Appearance → HUD Frontend**, press **F2** to cycle,
or set `frontend` in config / `JARVIS_FRONTEND=smoke`.

Headless check: `python tests/test_smoke_hud_headless.py`

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
| Piper TTS | 192.168.xx.xx | 10200 | Wyoming |
| Whisper STT | 192.168.xx.xx | 10300 | Wyoming |
| LLM API | 192.168.xx.xx | 8642 | OpenAI |

## Configuration

Edit `src/main.py` → `JarvisApp.__init__()` → `AgentConfig()` for:
- LLM endpoint (`ChatConfig`)
- Wyoming hosts/ports (`STTWyomingConfig`, `TTSWyomingConfig`)
- System prompt (`AgentConfig.system_prompt`)
