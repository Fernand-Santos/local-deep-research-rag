"""
Auto-sync legal_titles registry from parsed documents.

Ensures every distinct (jurisdiction, title_number) in documents
has a corresponding legal_titles row with a deterministic collection_key.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_collection_key(jurisdiction: str, title_number: str) -> str:
    jur = re.sub(r"[^a-z0-9]", "_", (jurisdiction or "").lower()).strip("_")
    num = re.sub(r"[^a-z0-9]", "_", (title_number or "").lower()).strip("_")
    return f"laws_{jur}_title_{num.zfill(2)}"


def sync_legal_titles(config) -> dict:
    """
    Read distinct (jurisdiction, title_number, title_name) from documents
    and upsert into legal_titles.  Returns summary counts.
    """
    from core.db import connect_db

    ts = _utc_now_iso()
    added = 0
    updated = 0

    with connect_db(config) as conn:
        rows = conn.execute(
            """
            SELECT LOWER(jurisdiction) as jur, title_number, title_name,
                   COUNT(*) as doc_count
            FROM documents
            WHERE jurisdiction IS NOT NULL AND title_number IS NOT NULL
            GROUP BY LOWER(jurisdiction), title_number
            ORDER BY LOWER(jurisdiction), title_number
            """
        ).fetchall()

        for r in rows:
            jur = r["jur"]
            tnum = r["title_number"]
            tname = r["title_name"]
            ckey = _make_collection_key(jur, tnum)

            existing = conn.execute(
                "SELECT id FROM legal_titles WHERE LOWER(jurisdiction) = ? AND title_number = ?",
                (jur, tnum),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE legal_titles SET title_name = ?, collection_key = ?, updated_at = ? WHERE id = ?",
                    (tname, ckey, ts, existing["id"]),
                )
                updated += 1
            else:
                conn.execute(
                    "INSERT INTO legal_titles (id, jurisdiction, title_number, title_name, collection_key, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                    (uuid.uuid4().hex, jur, tnum, tname, ckey, ts, ts),
                )
                added += 1

        conn.commit()

    return {"ok": True, "added": added, "updated": updated, "total_docs_scanned": len(rows)}
