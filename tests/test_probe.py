"""Tests for the Settings 'Test Connection' multi-service prober."""
import asyncio

from jarvis.probe import probe_services


def test_all_fail_when_nothing_configured():
    out = asyncio.run(probe_services({}))
    assert set(out) == {"DB", "LLM", "STT", "TTS"}
    assert all(v.startswith("FAIL") for v in out.values())


def test_voice_link_skipped_without_hosts():
    out = asyncio.run(probe_services({"stt_host": "", "tts_host": ""}))
    assert "VOICE LINK" not in out


def test_llm_probe_fails_clean_on_unreachable():
    from jarvis.probe import probe_llm

    r = asyncio.run(probe_llm(
        {"llm_base_url": "http://127.0.0.1:1/v1", "llm_api_key": "x", "llm_model": "m"}))
    assert r.startswith("FAIL")


def test_stt_probe_fails_clean_on_unreachable():
    from jarvis.probe import probe_stt

    r = asyncio.run(probe_stt({"stt_host": "127.0.0.1", "stt_port": 1}))
    assert r.startswith("FAIL") and "127.0.0.1" in r


def test_tts_probe_fails_clean_on_unreachable():
    from jarvis.probe import probe_tts

    r = asyncio.run(probe_tts({"tts_host": "127.0.0.1", "tts_port": 1}))
    assert r.startswith("FAIL")


def test_db_probe_fails_clean_on_unreachable():
    from jarvis.probe import _probe_db_sync

    r = _probe_db_sync({"db_host": "127.0.0.1", "db_port": 1,
                        "db_user": "u", "db_password": "p", "db_name": "d"})
    assert r.startswith("FAIL")


def test_settings_dialog_has_result_signal():
    """Test results must be marshalled back to the GUI thread via a signal."""
    from jarvis.settings import SettingsDialog

    assert hasattr(SettingsDialog, "_test_results")
