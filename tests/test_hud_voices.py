"""Tests for per-HUD-canvas TTS voice overrides (hud_voice table + store).

Covers: DB-primary store CRUD (fake DB), degradation when the DB is down,
FRONTS canonical order pinning, AppConfig mirror round-trip, the agent's
resolution chain (DB override -> TTS-tab default), live voice swap on
canvas change, and the Settings canvas-voice dropdowns.
"""

from unittest.mock import MagicMock

import pytest

from jarvis.agent import AgentConfig, JarvisAgent
from jarvis.config import DEFAULT_FRONT, FRONT_KEYS, FRONTS, AppConfig
from jarvis.hud_voice_store import HudVoiceStore


# ── fake DB ──────────────────────────────────────────────────────────
class FakeCursor:
    """Understands exactly the statements HudVoiceStore emits."""

    def __init__(self, db):
        self._db = db

    def execute(self, sql, params=None):
        self._db.executed.append((" ".join(sql.split()), params))
        s = sql.strip().upper()
        if s.startswith("SELECT"):
            self._rows = [
                {"frontend": k, "voice": v} for k, v in self._db.rows.items()
            ]
        elif s.startswith("INSERT"):
            self._db.rows[params[0]] = params[1]
        elif s.startswith("DELETE"):
            self._db.rows.pop(params[0], None)

    def fetchall(self):
        return getattr(self, "_rows", [])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDB:
    def __init__(self):
        self.rows: dict[str, str] = {}
        self.executed: list[tuple] = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


# ── canonical canvas order (dropdown/cycle pin, per skill rule) ──────
class TestFrontsOrder:
    def test_canonical_order_pinned(self):
        assert FRONT_KEYS == ("arc", "smoke", "smiley", "goddess", "bender")
        assert DEFAULT_FRONT == "arc"  # first canvas is the default

    def test_every_key_has_a_label(self):
        keys = [k for k, _ in FRONTS]
        labels = [lab for _, lab in FRONTS]
        assert keys == list(FRONT_KEYS)
        assert all(isinstance(lab, str) and lab for lab in labels)


# ── store CRUD / degradation ─────────────────────────────────────────
class TestHudVoiceStore:
    def test_set_then_load_roundtrip(self):
        db = FakeDB()
        store = HudVoiceStore(lambda: db)
        store.set("smoke", "en_GB-alan-medium")
        assert store.get("smoke") == "en_GB-alan-medium"  # cache-first
        fresh = HudVoiceStore(lambda: db)
        assert fresh.load_all() == {"smoke": "en_GB-alan-medium"}

    def test_set_upserts_one_row_per_canvas(self):
        db = FakeDB()
        store = HudVoiceStore(lambda: db)
        store.set("arc", "en_US-ryan-medium")
        store.set("arc", "en_US-joe-medium")
        assert db.rows == {"arc": "en_US-joe-medium"}

    def test_clear_restores_default(self):
        db = FakeDB()
        store = HudVoiceStore(lambda: db)
        store.set("bender", "en_GB-vctk-medium")
        store.clear("bender")
        assert store.get("bender") is None
        assert db.rows == {}

    def test_create_table_is_idempotent(self):
        db = FakeDB()
        store = HudVoiceStore(lambda: db)
        store.set("smiley", "en_US-amy-medium")
        store.load_all()
        creates = [s for s, _ in db.executed if s.upper().startswith("CREATE")]
        assert len(creates) == 1  # ensured once, not per operation

    def test_db_down_degrades_to_cache(self):
        def boom():
            raise ConnectionError("mariadb down")

        store = HudVoiceStore(boom, fallback={"goddess": "en_GB-cori-medium"})
        assert store.load_all() == {"goddess": "en_GB-cori-medium"}
        store.set("smoke", "en_US-kusal-medium")  # must not raise
        assert store.get("smoke") == "en_US-kusal-medium"

    def test_no_db_uses_fallback_mirror(self):
        store = HudVoiceStore(None, fallback={"arc": "en_US-lessac-high"})
        assert store.get("arc") == "en_US-lessac-high"
        assert store.snapshot() == {"arc": "en_US-lessac-high"}


# ── config.json mirror ───────────────────────────────────────────────
class TestAppConfigMirror:
    def test_default_empty_dict(self):
        assert AppConfig().hud_voices == {}

    def test_from_dict_roundtrip(self):
        cfg = AppConfig.from_dict({"hud_voices": {"smoke": "en_GB-alba-medium"}})
        assert cfg.hud_voices == {"smoke": "en_GB-alba-medium"}


# ── agent resolution chain ───────────────────────────────────────────
class TestAgentVoiceResolution:
    def _agent(self):
        agent = JarvisAgent()  # DB unreachable here -> cache-only store
        agent.hud_voices = HudVoiceStore(None)
        return agent

    def test_default_voice_without_override(self):
        agent = self._agent()
        assert agent.effective_voice("arc") == "en_US-lessac-medium"
        assert agent.effective_voice("smoke") == "en_US-lessac-medium"

    def test_override_wins_per_canvas(self):
        agent = self._agent()
        agent.hud_voices.set("smoke", "en_GB-alan-medium")
        assert agent.effective_voice("smoke") == "en_GB-alan-medium"
        assert agent.effective_voice("arc") == "en_US-lessac-medium"

    def test_apply_frontend_voice_swaps_tts(self):
        agent = self._agent()
        agent.hud_voices.set("bender", "en_US-joe-medium")
        agent.apply_frontend_voice("bender")
        assert agent.tts.voice == "en_US-joe-medium"
        # back to a canvas without override -> default again
        agent.apply_frontend_voice("arc")
        assert agent.tts.voice == "en_US-lessac-medium"

    def test_startup_voice_uses_active_canvas_override(self):
        cfg = AgentConfig()
        cfg.frontend = "smiley"
        cfg.hud_voices = {"smiley": "en_US-kusal-medium"}  # config mirror
        agent = JarvisAgent(cfg)
        assert agent.tts.voice == "en_US-kusal-medium"


# ── settings dialog wiring ───────────────────────────────────────────
@pytest.fixture
def canvas_settings(qtbot):
    from jarvis.settings import SettingsDialog

    agent_config = AgentConfig()
    store = HudVoiceStore(None, fallback={"goddess": "en_GB-alan-medium"})
    agent = MagicMock()
    agent.hud = None
    agent.tts = MagicMock()
    agent.tts.voice = "en_US-lessac-medium"
    agent.profiles = MagicMock()
    agent.hud_voices = store

    dialog = SettingsDialog(
        agent_config=agent_config, app_config=AppConfig(), agent=agent
    )
    qtbot.addWidget(dialog)
    yield dialog, agent, store
    dialog.deleteLater()


class TestSettingsCanvasVoices:
    def test_one_combo_per_canvas_in_canonical_order(self, canvas_settings):
        dialog, _, _ = canvas_settings
        assert list(dialog.canvas_voice_combos) == list(FRONT_KEYS)

    def test_frontend_combo_itemdata_matches_fronts(self, canvas_settings):
        dialog, _, _ = canvas_settings
        keys = [
            dialog.frontend_combo.itemData(i)
            for i in range(dialog.frontend_combo.count())
        ]
        assert keys == list(FRONT_KEYS)  # pin: no index-branching drift

    def test_first_entry_is_use_default(self, canvas_settings):
        dialog, _, _ = canvas_settings
        for combo in dialog.canvas_voice_combos.values():
            assert combo.itemData(0) == {"type": "default"}

    def test_stored_override_is_preselected(self, canvas_settings):
        dialog, _, _ = canvas_settings
        combo = dialog.canvas_voice_combos["goddess"]
        data = combo.currentData()
        assert isinstance(data, dict) and data.get("name") == "en_GB-alan-medium"
        plain = dialog.canvas_voice_combos["smoke"]
        assert plain.currentData() == {"type": "default"}

    def test_choosing_voice_writes_override_to_store(self, canvas_settings):
        dialog, _, store = canvas_settings
        combo = dialog.canvas_voice_combos["smoke"]
        for i in range(combo.count()):
            d = combo.itemData(i)
            if isinstance(d, dict) and d.get("name") == "en_US-ryan-medium":
                combo.setCurrentIndex(i)
                break
        assert store.get("smoke") == "en_US-ryan-medium"

    def test_use_default_clears_override(self, canvas_settings):
        dialog, _, store = canvas_settings
        combo = dialog.canvas_voice_combos["bender"]
        # Set an override through the UI first (store.set path)...
        for i in range(combo.count()):
            d = combo.itemData(i)
            if isinstance(d, dict) and d.get("name") == "en_GB-vctk-medium":
                combo.setCurrentIndex(i)
                break
        assert store.get("bender") == "en_GB-vctk-medium"
        # ...then back to "Use default" (store.clear path).
        combo.setCurrentIndex(0)
        assert store.get("bender") is None

    def test_live_applied_only_for_active_canvas(self, canvas_settings):
        dialog, agent, _ = canvas_settings
        dialog.agent_config.frontend = "smiley"
        combo = dialog.canvas_voice_combos["smiley"]
        for i in range(combo.count()):
            d = combo.itemData(i)
            if isinstance(d, dict) and d.get("name") == "en_US-kusal-medium":
                combo.setCurrentIndex(i)
                break
        agent.apply_frontend_voice.assert_called_once_with("smiley")
        other = dialog.canvas_voice_combos["arc"]
        other.setCurrentIndex(1)  # active canvas is 'smiley', not 'arc'
        agent.apply_frontend_voice.assert_called_once_with("smiley")
