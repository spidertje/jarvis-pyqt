-- Per-HUD-canvas TTS voice overrides (jarvis database).
-- A row exists ONLY when the user overrode that canvas's voice; absence
-- means "inherit the TTS-tab default voice" (AppConfig.tts_voice).
-- The app also self-creates this table on first use (hud_voice_store.py).
-- Canvas keys mirror config.FRONTS order: arc, smoke, smiley, goddess, bender.

CREATE TABLE IF NOT EXISTS hud_voice (
    frontend VARCHAR(20) PRIMARY KEY,
    voice VARCHAR(100) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
