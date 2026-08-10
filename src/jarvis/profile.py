"""
Jarvis Profiles — Person-specific chat profiles.

When a face is recognized, switch to that person's profile:
- Custom system prompt (tone, knowledge, personality)
- Separate chat history
- HUD theme accent color

Profiles stored in MariaDB `jarvis` DB `profiles` table.
"""

import logging
import os
from dataclasses import dataclass, field

import pymysql

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    """A person's chat profile."""

    name: str
    assistant_name: str = "Jarvis"
    system_prompt: str = "You are Jarvis, a helpful AI assistant."
    chat_history: list[dict[str, str]] = field(default_factory=list)
    accent_hue: int = 182  # Default cyan
    enabled: bool = True
    face_name: str | None = None  # Links face identity to profile

    def to_db_row(self) -> tuple:
        """Convert to DB INSERT/UPDATE row."""
        import json

        # Map to web frontend schema:
        # name, assistant_name, system_prompt->assistant_full,
        # chat_history, accent_hue, enabled, face_name
        return (
            self.name,
            self.assistant_name,
            self.system_prompt,
            json.dumps(self.chat_history),
            self.accent_hue,
            int(self.enabled),
            self.face_name,
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "Profile":
        """Create Profile from DB row."""
        import json

        # Map actual DB columns (from web frontend schema) to Profile fields
        raw_history = row.get("chat_history")
        history = json.loads(raw_history) if raw_history else []
        # enabled: if column exists use it, else default to True
        enabled_val = row.get("enabled")
        if enabled_val is None:
            enabled_val = True  # Default enabled if column doesn't exist

        # Build system prompt from assistant_name and assistant_full
        assistant_name = row.get("assistant_name", "Jarvis")
        assistant_full = row.get("assistant_full", "")
        # If system_prompt column exists use it, else generate from name
        system_prompt = row.get("system_prompt")
        if not system_prompt:
            if assistant_full and assistant_full != assistant_name:
                system_prompt = (
                    f"You are {assistant_name}, {assistant_full}. "
                    "Be helpful, knowledgeable, and direct."
                )
            else:
                system_prompt = f"You are {assistant_name}, a helpful AI assistant."

        return cls(
            name=row["name"],
            assistant_name=assistant_name,
            system_prompt=system_prompt,
            chat_history=history,
            accent_hue=row.get("accent_hue") or cls._palette_to_hue(row.get("palette", "cyan")),
            enabled=bool(enabled_val),
            face_name=row.get("face_name"),
        )

    @classmethod
    def create_from_default(cls, default_profile: "Profile", face_name: str) -> "Profile":
        """Create a new profile for a recognized face, copying defaults from default profile.

        The new profile uses the face_name as its profile name (per user preference:
        'name the profile the same as the face').
        """
        return cls(
            name=face_name,
            assistant_name=default_profile.assistant_name,
            system_prompt=default_profile.system_prompt,
            chat_history=[],
            accent_hue=default_profile.accent_hue,
            enabled=True,
            face_name=face_name,
        )

    @staticmethod
    def _palette_to_hue(palette: str) -> int:
        """Map palette name to hue."""
        hues = {
            "cyan": 182,
            "copper": 30,
            "emerald": 120,
            "violet": 250,
            "matrix": 200,
            "red": 0,
            "green": 80,
            "blue": 220,
            "yellow": 40,
            "orange": 60,
        }
        return hues.get(palette, 182)

    @staticmethod
    def _hue_to_palette(hue: int) -> str:
        """Map hue to palette name (reverse lookup)."""
        hues = {
            182: "cyan",
            30: "copper",
            120: "emerald",
            250: "violet",
            200: "matrix",
            0: "red",
            80: "green",
            220: "blue",
            40: "yellow",
            60: "orange",
        }
        return hues.get(hue, "cyan")


class ProfileManager:
    """Manage profiles: load, switch, save."""

    def __init__(self, db_host=None, db_port=None, db_user=None, db_password=None, db_name=None):
        self.db_host = db_host or os.environ.get("JARVIS_DB_HOST")
        self.db_port = (
            db_port if db_port is not None else int(os.environ.get("JARVIS_DB_PORT", "3306"))
        )
        self.db_user = db_user or os.environ.get("JARVIS_DB_USER")
        self.db_password = db_password or os.environ.get("JARVIS_DB_PASSWORD")
        self.db_name = db_name or os.environ.get("JARVIS_DB_NAME")
        self._db = None
        self._profiles: dict[str, Profile] = {}
        self._active_name: str | None = None

    def _get_db(self) -> pymysql.Connection:
        """Get or create DB connection."""
        if self._db is None or not self._db.open:
            self._db = pymysql.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password or "",
                database=self.db_name or "jarvis",
                cursorclass=pymysql.cursors.DictCursor,
            )
        return self._db

    def load_all(self) -> list[Profile]:
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

    def get(self, name: str) -> Profile | None:
        """Get a profile by name (case-insensitive)."""
        return self._profiles.get(name.lower())

    def get_by_face_name(self, face_name: str) -> Profile | None:
        """Get a profile by face_name (case-insensitive)."""
        for profile in self._profiles.values():
            if profile.face_name and profile.face_name.lower() == face_name.lower():
                return profile
        return None

    def get_active(self) -> Profile | None:
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
        """Save a profile to DB (upsert).

        ponytail: writes only columns present in the live web-frontend schema
        (accent_hue/enabled/system_prompt aren't columns there; they map to
        palette/assistant_full). chat_history guarded separately since older
        tables lack it.
        """
        try:
            db = self._get_db()
            import json

            palette = Profile._hue_to_palette(profile.accent_hue)
            with db.cursor() as cur:
                cur.execute(
                    (
                        "INSERT INTO profiles "
                        "(name, assistant_name, assistant_full, "
                        "palette, chat_history, face_name, api_key) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE "
                        "assistant_name=VALUES(assistant_name), "
                        "assistant_full=VALUES(assistant_full), "
                        "palette=VALUES(palette), "
                        "chat_history=VALUES(chat_history), "
                        "face_name=VALUES(face_name)"
                    ),
                    (
                        profile.name,
                        profile.assistant_name,
                        profile.system_prompt,
                        palette,
                        json.dumps(profile.chat_history),
                        profile.face_name,
                        "",  # api_key NOT-NULL — not used by PyQt profiles
                    ),
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

    def list_names(self) -> list[str]:
        """List all profile names."""
        return list(self._profiles.keys())

    def get_default_assistant_name(self) -> str | None:
        """Load assistant_name from the default profile row (is_default=1)."""
        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute("SELECT assistant_name FROM profiles WHERE is_default=1 LIMIT 1")
                row = cur.fetchone()
                return row["assistant_name"] if row else None
        except Exception as e:
            logger.error(f"Failed to load default assistant name: {e}")
            return None

    def get_default(self) -> Profile | None:
        """Get the default profile (is_default=1) as a Profile object."""
        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute("SELECT * FROM profiles WHERE is_default=1 LIMIT 1")
                row = cur.fetchone()
                if row:
                    return Profile.from_db_row(row)
                return None
        except Exception as e:
            logger.error(f"Failed to load default profile: {e}")
            return None

    def set_default_assistant_name(self, name: str) -> bool:
        """Persist assistant_name to the default profile row (is_default=1)."""
        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute(
                    "UPDATE profiles SET assistant_name=%s WHERE is_default=1",
                    (name,),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        "UPDATE profiles SET assistant_name=%s ORDER BY id LIMIT 1",
                        (name,),
                    )
            db.commit()
            logger.info(f"Saved assistant name to DB: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save default assistant name: {e}")
            return False

    def close(self):
        """Close DB connection."""
        if self._db is not None and self._db.open:
            self._db.close()
