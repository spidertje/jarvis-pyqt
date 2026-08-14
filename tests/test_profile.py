"""Tests for the Profile dataclass and its helpers."""

import json

from jarvis.profile import Profile, ProfileManager


class TestProfile:
    def test_defaults(self):
        """Profile should have sensible defaults."""
        p = Profile(name="testuser")
        assert p.name == "testuser"
        assert p.assistant_name == "Jarvis"
        assert p.system_prompt.startswith("You are Jarvis")
        assert p.accent_hue == 182
        assert p.enabled is True
        assert p.chat_history == []

    def test_palette_to_hue(self):
        """_palette_to_hue should map known palettes to correct hues."""
        assert Profile._palette_to_hue("cyan") == 182
        assert Profile._palette_to_hue("copper") == 30
        assert Profile._palette_to_hue("violet") == 250
        assert Profile._palette_to_hue("nonexistent") == 182  # default

    def test_hue_to_palette(self):
        """_hue_to_palette should map known hues to correct palettes."""
        assert Profile._hue_to_palette(182) == "cyan"
        assert Profile._hue_to_palette(30) == "copper"
        assert Profile._hue_to_palette(0) == "red"
        assert Profile._hue_to_palette(999) == "cyan"  # default

    def test_from_db_row_minimal(self):
        """from_db_row should work with minimal DB row."""
        row = {"name": "alice"}
        p = Profile.from_db_row(row)
        assert p.name == "alice"
        assert p.assistant_name == "Jarvis"
        assert p.enabled is True
        assert p.chat_history == []

    def test_from_db_row_full(self):
        """from_db_row should parse all fields from a full DB row."""
        history = [{"role": "user", "content": "hello"}]
        row = {
            "name": "bob",
            "assistant_name": "Bob",
            "assistant_full": "a friendly assistant",
            "system_prompt": "You are Bob",
            "chat_history": json.dumps(history),
            "accent_hue": 45,
            "palette": "copper",
            "enabled": 1,
        }
        p = Profile.from_db_row(row)
        assert p.name == "bob"
        assert p.assistant_name == "Bob"
        assert p.system_prompt == "You are Bob"
        assert p.chat_history == history
        assert p.accent_hue == 45
        assert p.enabled is True

    def test_from_db_row_disabled(self):
        """from_db_row should handle enabled=0."""
        row = {"name": "disabled_user", "enabled": 0}
        p = Profile.from_db_row(row)
        assert p.enabled is False


class TestProfileManagerSave:
    """ProfileManager.save() upsert behavior (mocked DB)."""

    @staticmethod
    def _manager_with_mock_db():
        from unittest.mock import MagicMock

        mgr = ProfileManager.__new__(ProfileManager)
        mgr._db = MagicMock()
        mgr._active_name = None
        mgr._profiles = {}
        mgr.db_host = None
        mgr.db_port = 3306
        mgr.db_user = None
        mgr.db_password = None
        mgr.db_name = None
        return mgr

    def _capture_execute(self, manager):
        """Return the SQL string+tuple passed to cursor.execute."""
        cur = manager._db.cursor.return_value.__enter__.return_value
        # execute was called exactly once
        assert cur.execute.call_count == 1
        args, kwargs = cur.execute.call_args
        sql = args[0]
        params = args[1] if len(args) > 1 else kwargs.get("args")
        return sql, params

    def test_save_persists_accent_hue(self):
        manager = self._manager_with_mock_db()
        profile = Profile(name="alice", accent_hue=45, face_name="alice")
        assert manager.save(profile) is True
        sql, params = self._capture_execute(manager)
        # accent_hue must appear in both the INSERT column list and the UPDATE clause
        assert "accent_hue" in sql
        # Parameter order: name(0), assistant_name(1), system_prompt->assistant_full(2),
        # palette(3), accent_hue(4), chat_history(5), face_name(6), api_key(7)
        assert params[4] == 45

    def test_save_persists_face_name(self):
        manager = self._manager_with_mock_db()
        profile = Profile(name="bob", face_name="robert-face")
        assert manager.save(profile) is True
        sql, params = self._capture_execute(manager)
        assert "face_name" in sql
        assert params[6] == "robert-face"

    def test_save_returns_false_on_db_error(self):
        manager = self._manager_with_mock_db()
        manager._db.cursor.side_effect = Exception("boom")
        profile = Profile(name="alice")
        assert manager.save(profile) is False
