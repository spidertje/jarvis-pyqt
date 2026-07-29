"""
Jarvis Profiles — Person-specific chat profiles.

When a face is recognized, switch to that person's profile:
- Custom system prompt (tone, knowledge, personality)
- Separate chat history
- HUD theme accent color

Profiles stored in MariaDB `jarvis` DB `profiles` table.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import pymysql

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    """A person's chat profile."""
    name: str
    system_prompt: str = "You are Jarvis, a helpful AI assistant."
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    accent_hue: int = 182  # Default cyan
    enabled: bool = True

    def to_db_row(self) -> tuple:
        """Convert to DB INSERT/UPDATE row."""
        import json
        return (
            self.name,
            self.system_prompt,
            json.dumps(self.chat_history),
            self.accent_hue,
            int(self.enabled),
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "Profile":
        """Create Profile from DB row."""
        import json
        history = json.loads(row.get("chat_history", "[]"))
        return cls(
            name=row["name"],
            system_prompt=row.get("system_prompt", "You are Jarvis, a helpful AI assistant."),
            chat_history=history,
            accent_hue=row.get("accent_hue", 182),
            enabled=bool(row.get("enabled", 1)),
        )


class ProfileManager:
    """Manage profiles: load, switch, save."""

    def __init__(self, db_host="192.168.55.41", db_port=3306,
                 db_user="root", db_password="rocklobster", db_name="jarvis"):
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self._db = None
        self._profiles: Dict[str, Profile] = {}
        self._active_name: Optional[str] = None

    def _get_db(self) -> pymysql.Connection:
        """Get or create DB connection."""
        if self._db is None or self._db.closed:
            self._db = pymysql.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name,
                cursorclass=pymysql.cursors.DictCursor,
            )
        return self._db

    def load_all(self) -> List[Profile]:
        """Load all profiles from DB."""
        self._profiles.clear()
        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute("SELECT * FROM profiles ORDER BY name")
                for row in cur.fetchall():
                    profile = Profile.from_db_row(row)
                    self._profiles[profile.name.lower()] = profile
            logger.info(f"Loaded {len(self._profiles)} profiles")
            return list(self._profiles.values())
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return []

    def get(self, name: str) -> Optional[Profile]:
        """Get a profile by name (case-insensitive)."""
        return self._profiles.get(name.lower())

    def get_active(self) -> Optional[Profile]:
        """Get the currently active profile."""
        if self._active_name:
            return self._profiles.get(self._active_name.lower())
        return None

    def switch(self, name: str) -> bool:
        """
        Switch to a profile by name.

        Returns:
            True if switched successfully.
        """
        name_lower = name.lower()
        if name_lower not in self._profiles:
            logger.warning(f"Profile not found: {name}")
            return False

        self._active_name = name_lower
        profile = self._profiles[name_lower]

        if not profile.enabled:
            logger.warning(f"Profile disabled: {name}")
            return False

        logger.info(f"Switched to profile: {name} (hue={profile.accent_hue})")
        return True

    def clear(self):
        """Clear active profile (return to idle)."""
        self._active_name = None
        logger.info("Active profile cleared")

    def save(self, profile: Profile) -> bool:
        """Save a profile to DB (upsert)."""
        try:
            db = self._get_db()
            row = profile.to_db_row()
            with db.cursor() as cur:
                cur.execute(
                    "INSERT INTO profiles (name, system_prompt, chat_history, accent_hue, enabled) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE "
                    "system_prompt=VALUES(system_prompt), "
                    "chat_history=VALUES(chat_history), "
                    "accent_hue=VALUES(accent_hue), "
                    "enabled=VALUES(enabled)",
                    row,
                )
            db.commit()
            self._profiles[profile.name.lower()] = profile
            return True
        except Exception as e:
            logger.error(f"Failed to save profile {profile.name}: {e}")
            return False

    def delete(self, name: str) -> bool:
        """Delete a profile."""
        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute("DELETE FROM profiles WHERE name = %s", (name,))
                db.commit()
            if name.lower() in self._profiles:
                del self._profiles[name.lower()]
            if self._active_name == name.lower():
                self._active_name = None
            return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete profile {name}: {e}")
            return False

    def list_names(self) -> List[str]:
        """List all profile names."""
        return list(self._profiles.keys())

    def close(self):
        """Close DB connection."""
        if self._db and not self._db.closed:
            self._db.close()
