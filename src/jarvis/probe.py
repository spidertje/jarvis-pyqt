"""Service connectivity probes for the Settings 'Test Connection' button.

Probes ALL four services Jarvis depends on, using the (unsaved) form
values: MariaDB, the LLM endpoint, and the two Wyoming voice servers.

- DB: real pymysql connect + SELECT VERSION().
- LLM: GET {base_url}/models (fast), fall back to a 1-token chat if the
  endpoint has no /models route.
- STT: connect to Wyoming, run a known 0.3s PCM clip through transcribe
  (proves the whole framing path, not just the TCP port).
- TTS: connect, synthesize 'test', require non-empty PCM back — then a
  STT round trip if both are up (the definitive voice-pipeline check).

Each probe is failure-isolated: one dead service never hides the others'
results. Returns {name: "OK: ..." | "FAIL: ..."} for the UI to render.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT = 8.0  # per-service budget


async def probe_llm(values: dict) -> str:
    base = (values.get("llm_base_url") or "").strip()
    if not base:
        return "FAIL: no LLM base URL configured"
    key = values.get("llm_api_key") or ""
    model = values.get("llm_model") or "auto"
    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(total=_PROBE_TIMEOUT)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        url = base.rstrip("/") + "/models"
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, headers=headers) as r:
                if r.status == 200:
                    try:
                        data = await r.json()
                        ids = [m.get("id") for m in data.get("data", [])][:5]
                        return f"OK: {base} ({len(data.get('data', []))} models: {', '.join(map(str, ids)) or 'list empty'})"
                    except Exception:
                        return f"OK: {base} (/models reachable)"
                # No /models route — fall back to a minimal chat call.
                chat_url = base.rstrip("/") + "/chat/completions"
                async with s.post(
                    chat_url,
                    headers={**headers, "Content-Type": "application/json"},
                    json={"model": model, "messages": [
                        {"role": "user", "content": "ping"}],
                        "max_tokens": 1},
                ) as r2:
                    if r2.status == 200:
                        return f"OK: {base} (chat completion works, model={model})"
                    body = (await r2.text())[:200]
                    return f"FAIL: {base} → HTTP {r2.status}: {body}"
    except asyncio.TimeoutError:
        return f"FAIL: {base} timed out after {_PROBE_TIMEOUT:.0f}s"
    except Exception as e:
        return f"FAIL: {base} → {type(e).__name__}: {e}"


def _probe_db_sync(values: dict) -> str:
    host = values.get("db_host") or "localhost"
    port = int(values.get("db_port") or 3306)
    try:
        import pymysql

        conn = pymysql.connect(
            host=host, port=port,
            user=values.get("db_user") or "root",
            password=values.get("db_password") or "",
            database=values.get("db_name") or "",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION()")
                row = cur.fetchone() or {}
                version = row.get("VERSION()", "?")
        finally:
            conn.close()
        return f"OK: MariaDB {version} ({host}:{port}/{values.get('db_name')})"
    except Exception as e:
        return f"FAIL: {host}:{port} → {type(e).__name__}: {e}"


async def probe_stt(values: dict) -> str:
    host = values.get("stt_host") or ""
    port = int(values.get("stt_port") or 10300)
    if not host:
        return "FAIL: no STT host configured"
    from jarvis.stt import WhisperSTT, WyomingConfig

    stt = WhisperSTT(WyomingConfig(host=host, port=port))
    try:
        ok = await asyncio.wait_for(stt.connect(timeout=5.0), timeout=8.0)
        if not ok:
            return f"FAIL: cannot reach Wyoming server {host}:{port}"
        # Silence clip transcribes to '' quickly — proves the protocol
        # round trip end-to-end (framing, audio payload, transcript event).
        clip = b"\x00\x00" * 4800  # 0.3s of silence @16k mono 16-bit
        txt = await asyncio.wait_for(stt.transcribe(clip), timeout=_PROBE_TIMEOUT)
        if txt is None:
            return f"FAIL: {host}:{port} connected but transcribe returned None"
        return f"OK: faster-whisper {host}:{port} (round trip ok)"
    except (asyncio.TimeoutError, asyncio.TimeoutError):
        return f"FAIL: {host}:{port} timed out after {_PROBE_TIMEOUT:.0f}s"
    except Exception as e:
        return f"FAIL: {host}:{port} → {type(e).__name__}: {e}"
    finally:
        try:
            await stt.disconnect()
        except Exception:
            pass


async def probe_tts(values: dict) -> str:
    host = values.get("tts_host") or ""
    port = int(values.get("tts_port") or 10200)
    voice = values.get("tts_voice") or "en_US-lessac-medium"
    if not host:
        return "FAIL: no TTS host configured"
    from jarvis.tts import PiperTTS, WyomingConfig as TTSWyomingConfig

    tts = PiperTTS(TTSWyomingConfig(host=host, port=port, voice=voice))
    try:
        ok = await asyncio.wait_for(tts.connect(timeout=5.0), timeout=8.0)
        if not ok:
            return f"FAIL: cannot reach Wyoming server {host}:{port}"
        pcm = await asyncio.wait_for(tts.speak("test"), timeout=_PROBE_TIMEOUT)
        if not pcm:
            return f"FAIL: {host}:{port} connected but synthesized nothing (bad voice name?)"
        secs = len(pcm) / (16000 * 2)
        return f"OK: piper {host}:{port} ('{voice}' → {secs:.1f}s audio)"
    except asyncio.TimeoutError:
        return f"FAIL: {host}:{port} timed out after {_PROBE_TIMEOUT:.0f}s"
    except Exception as e:
        return f"FAIL: {host}:{port} → {type(e).__name__}: {e}"
    finally:
        try:
            await tts.disconnect()
        except Exception:
            pass


async def probe_voice_link(values: dict) -> str | None:
    """TTS→STT round trip: speak a known sentence, transcribe it back.

    Only runs when both Wyoming hosts are configured. Returns None to
    skip (hosts missing). This is the end-to-end proof the whole voice
    path works — framing, audio, model loading, everything.
    """
    stt_host = values.get("stt_host") or ""
    tts_host = values.get("tts_host") or ""
    if not stt_host or not tts_host:
        return None
    from jarvis.stt import WhisperSTT, WyomingConfig
    from jarvis.tts import PiperTTS, WyomingConfig as TTSWyomingConfig

    stt = WhisperSTT(WyomingConfig(host=stt_host, port=int(values.get("stt_port") or 10300)))
    tts = PiperTTS(TTSWyomingConfig(host=tts_host, port=int(values.get("tts_port") or 10200),
                                   voice=values.get("tts_voice") or "en_US-lessac-medium"))
    try:
        ok1 = await asyncio.wait_for(stt.connect(timeout=5.0), timeout=8.0)
        ok2 = await asyncio.wait_for(tts.connect(timeout=5.0), timeout=8.0)
        if not (ok1 and ok2):
            return "FAIL: could not open both Wyoming links for round trip"
        pcm = await asyncio.wait_for(tts.speak("The quick brown fox"), timeout=_PROBE_TIMEOUT)
        if not pcm:
            return "FAIL: round trip — TTS produced no audio"
        txt = await asyncio.wait_for(stt.transcribe(pcm), timeout=_PROBE_TIMEOUT)
        if txt is None:
            return "FAIL: round trip — STT returned nothing for TTS audio"
        if not txt.strip():
            return "FAIL: round trip — STT transcribed empty text (server up, model issue?)"
        return f"OK: voice link — 'the quick brown fox' → {txt.strip()!r}"
    except asyncio.TimeoutError:
        return f"FAIL: round trip timed out after {_PROBE_TIMEOUT:.0f}s"
    except Exception as e:
        return f"FAIL: round trip → {type(e).__name__}: {e}"
    finally:
        for svc in (stt, tts):
            try:
                await svc.disconnect()
            except Exception:
                pass


async def probe_services(values: dict) -> dict[str, str]:
    """Probe DB + LLM + STT + TTS (+ voice round trip) concurrently."""
    db_task = asyncio.to_thread(_probe_db_sync, values)
    llm, stt, tts, link = await asyncio.gather(
        probe_llm(values), probe_stt(values), probe_tts(values),
        probe_voice_link(values),
        return_exceptions=True,
    )
    def _safe(r):
        return f"FAIL: probe crashed: {r!r}" if isinstance(r, Exception) else str(r)

    out = {
        "DB": await db_task,
        "LLM": _safe(llm),
        "STT": _safe(stt),
        "TTS": _safe(tts),
    }
    if link is not None:
        out["VOICE LINK"] = _safe(link)
    return out
