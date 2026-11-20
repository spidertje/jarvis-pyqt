"""
Jarvis HUD canvas voices — per-frontend TTS voice overrides.

Every HUD canvas (frontend key, see ``config.FRONTS``) speaks with the
TTS-tab default voice *unless* the user set a per-canvas override.
Overrides live in the MariaDB ``hud_voice`` table (source of truth); a
small in-memory cache — primed from the ``config.json`` ``hud_voices``
mirror — keeps resolution working while the DB is unreachable.

Resolution chain (implemented by ``Agent.effective_voice``):
    DB override for the active canvas -> AppConfig/AgentConfig.tts_voice

The table is created lazily on first DB write/read (same self-healing
DDL pattern as ``chat_messages``) — NOT inside ``ProfileManager._get_db``,
because mock-based profile tests count ``execute()`` calls there.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS hud_voice (
    frontend VARCHAR(20) PRIMARY KEY,
    voice VARCHAR(100) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


class HudVoiceStore:
    """DB-primary store for per-canvas voice overrides.

    ``db_provider`` is a zero-arg callable returning an open pymysql
    connection (normally ``ProfileManager.get_db`` so one connection and
    its reconnect logic serve every jarvis table). When it is ``None`` or
    raises, the store degrades to its in-memory cache — the app must
    never crash over a missing voice override.
    """

    def __init__(
        self,
        db_provider: Callable[[], Any] | None = None,
        fallback: dict[str, str] | None = None,
    ):
        self._db_provider = db_provider
        self._cache: dict[str, str] = dict(fallback or {})
        self._table_ready = False
        self._db_warned = False

    # ── internals ────────────────────────────────────────────────────
    def _db(self):
        if self._db_provider is None:
            return None
        try:
            return self._db_provider()
        except Exception as e:  # noqa: BLE001 — voice overrides are optional
            self._warn(e)
            return None

    def _warn(self, e: Exception) -> None:
        if not self._db_warned:
            logger.warning(f"hud_voice DB unavailable, using cache: {e}")
            self._db_warned = True

    def _ensure_table(self, db) -> None:
        if self._table_ready:
            return
        with db.cursor() as cur:
            cur.execute(CREATE_SQL)
        self._table_ready = True

    # ── public API ───────────────────────────────────────────────────
    def load_all(self) -> dict[str, str]:
        """Refresh the cache from the DB; return the merged overrides."""
        db = self._db()
        if db is None:
            return dict(self._cache)
        try:
            self._ensure_table(db)
            with db.cursor() as cur:
                cur.execute("SELECT frontend, voice FROM hud_voice")
                for row in cur.fetchall():
                    self._cache[row["frontend"]] = row["voice"]
            self._db_warned = False
        except Exception as e:  # noqa: BLE001
            self._warn(e)
        return dict(self._cache)

    def get(self, frontend: str) -> str | None:
        """Override voice for a canvas, or None (= inherit default)."""
        return self._cache.get(frontend)

    def set(self, frontend: str, voice: str) -> None:
        """Persist an override (cache first so the UI stays responsive)."""
        self._cache[frontend] = voice
        db = self._db()
        if db is None:
            return
        try:
            self._ensure_table(db)
            with db.cursor() as cur:
                cur.execute(
                    "INSERT INTO hud_voice (frontend, voice) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE voice = VALUES(voice)",
                    (frontend, voice),
                )
            db.commit()
            self._db_warned = False
            logger.info(f"Saved canvas voice override: {frontend} -> {voice}")
        except Exception as e:  # noqa: BLE001
            self._warn(e)

    def clear(self, frontend: str) -> None:
        """Remove an override — the canvas inherits the TTS-tab default."""
        self._cache.pop(frontend, None)
        db = self._db()
        if db is None:
            return
        try:
            self._ensure_table(db)
            with db.cursor() as cur:
                cur.execute("DELETE FROM hud_voice WHERE frontend = %s", (frontend,))
            db.commit()
            self._db_warned = False
            logger.info(f"Cleared canvas voice override for {frontend}")
        except Exception as e:  # noqa: BLE001
            self._warn(e)

    def snapshot(self) -> dict[str, str]:
        """Cache copy — mirrored into config.json as the offline fallback."""
        return dict(self._cache)
