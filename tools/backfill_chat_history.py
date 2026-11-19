#!/usr/bin/env python3
"""One-time backfill: profiles.chat_history JSON blobs -> chat_messages rows.

Before the 2026-11-19 chat-log rebuild, turns lived only inside the
per-profile chat_history JSON blob (capped at the in-memory history limit,
no timestamps). This inserts those older turns as chat_messages rows so
they join the searchable window history.

Idempotent: rows are matched per profile by (role, content) multiset, so
running it twice inserts nothing new. Backfilled rows get synthetic UTC
created_at values placed just BEFORE the profile's earliest existing row
(one second apart), keeping chronology sane.

Run from the repo root:  .venv/bin/python tools/backfill_chat_history.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jarvis.config import AppConfig  # noqa: E402
from jarvis.profile import ProfileManager  # noqa: E402


def main() -> int:
    cfg = AppConfig.load()
    pm = ProfileManager(
        db_host=cfg.db_host, db_port=cfg.db_port, db_user=cfg.db_user,
        db_password=cfg.db_password, db_name=cfg.db_name,
    )
    profiles = pm.load_all()
    if not profiles:
        print("no profiles loaded (DB reachable?) — nothing to do")
        return 1

    db = pm._get_db()
    total = 0
    with db.cursor() as cur:
        for prof in profiles:
            hist = [
                (m.get("role", ""), m.get("content", ""))
                for m in (prof.chat_history or [])
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            if not hist:
                continue
            cur.execute(
                "SELECT role, content FROM chat_messages WHERE profile_name = %s",
                (prof.name,),
            )
            existing_rows = list(cur.fetchall())
            existing = Counter((r["role"], r["content"]) for r in existing_rows)
            cur.execute(
                "SELECT MIN(created_at) AS min_ts FROM chat_messages WHERE profile_name = %s",
                (prof.name,),
            )
            min_ts = cur.fetchone()["min_ts"]

            blob = Counter(hist)
            to_insert = []
            seen = Counter()
            for role, content in hist:  # keep original chronological order
                key = (role, content)
                if blob[key] > existing[key] and seen[key] < blob[key] - existing[key]:
                    to_insert.append(key)
                    seen[key] += 1
            if not to_insert:
                print(f"{prof.name}: up to date ({len(hist)} in blob, no new rows)")
                continue

            if min_ts is None:
                min_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            if min_ts.tzinfo is not None:
                min_ts = min_ts.astimezone(timezone.utc).replace(tzinfo=None)
            base = min_ts - timedelta(seconds=len(to_insert) + 1)

            for i, (role, content) in enumerate(to_insert):
                cur.execute(
                    "INSERT INTO chat_messages (profile_name, role, content, created_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (prof.name, role, content, base + timedelta(seconds=i)),
                )
            total += len(to_insert)
            print(f"{prof.name}: backfilled {len(to_insert)} row(s) (blob had {len(hist)})")
    db.commit()
    print(f"DONE: {total} row(s) inserted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
