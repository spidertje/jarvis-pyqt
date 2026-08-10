"""Tests for the Profile dataclass and its helpers."""

import json

from jarvis.profile import Profile


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
