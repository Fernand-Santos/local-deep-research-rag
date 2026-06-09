"""
Automatic state-root ingestion pipeline.

Discovers state folders, registers sources, parses PDFs, chunks/embeds,
mirrors to Chroma, and tracks per-jurisdiction readiness.

Chroma writes: callers that mirror or reset the vector store should hold
`indexer.chroma_lock.chroma_write_lock` so only one process mutates
`CHROMA_PERSIST_DIR` at a time (Streamlit sync and `ingestion.cli` do this).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_AUTO_WORKSPACE_NAME = "_states_root_auto"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_auto_workspace(config) -> str:
    """Get or create the auto-managed workspace for states root sync."""
    from core.db import connect_db

    with connect_db(config) as conn:
        row = conn.execute("SELECT id FROM workspaces WHERE name = ?", (_AUTO_WORKSPACE_NAME,)).fetchone()
        if row:
            return row["id"]

        ws_id = uuid.uuid4().hex
        ts = _utc_now_iso()
        conn.execute(
            "INSERT INTO workspaces (id, name, description, created_at, updated_at) VALUES (?,?,?,?,?)",
            (ws_id, _AUTO_WORKSPACE_NAME, "Auto-managed workspace for States root sync", ts, ts),
        )
        conn.commit()
        return ws_id


def _ensure_source_registered(config, ws_id: str, state_path: str) -> str:
    """Get or create a source for the given state folder path."""
    from core.db import connect_db

    norm = str(Path(state_path).resolve())
    with connect_db(config) as conn:
        row = conn.execute(
            "SELECT id FROM sources WHERE workspace_id = ? AND path = ?",
            (ws_id, norm),
        ).fetchone()
        if row:
            return row["id"]

        src_id = uuid.uuid4().hex
        ts = _utc_now_iso()
        conn.execute(
            "INSERT INTO sources (id, workspace_id, path, source_type, is_temp, created_at) VALUES (?,?,?,?,?,?)",
            (src_id, ws_id, norm, "folder", 0, ts),
        )
        conn.commit()
        return src_id


_INGESTION_ALLOWED_COLS = frozenset({
    "documents_count", "parsed_count", "chunked_count", "mirrored_count",
    "status", "last_error", "last_scan_at",
})


def _update_ingestion_status(config, jurisdiction: str, source_root: str, **kwargs):
    """Upsert jurisdiction_ingestion_status row. Only whitelisted columns accepted."""
    from core.db import connect_db

    ts = _utc_now_iso()
    jur = jurisdiction.strip().lower()

    safe_kwargs = {k: v for k, v in kwargs.items() if k in _INGESTION_ALLOWED_COLS}

    with connect_db(config) as conn:
        existing = conn.execute(
            "SELECT jurisdiction FROM jurisdiction_ingestion_status WHERE jurisdiction = ?",
            (jur,),
        ).fetchone()

        if existing:
            sets = ["updated_at = ?"]
            vals: list = [ts]
            for k, v in safe_kwargs.items():
                sets.append(f"{k} = ?")
                vals.append(v)
            vals.append(jur)
            conn.execute(
                f"UPDATE jurisdiction_ingestion_status SET {', '.join(sets)} WHERE jurisdiction = ?",
                vals,
            )
        else:
            col_list = ["jurisdiction", "source_root", "updated_at"]
            vals = [jur, source_root, ts]
            for k, v in safe_kwargs.items():
                col_list.append(k)
                vals.append(v)
            ph = ", ".join("?" * len(vals))
            conn.execute(
                f"INSERT INTO jurisdiction_ingestion_status ({', '.join(col_list)}) VALUES ({ph})",
                vals,
            )
        conn.commit()


def get_all_ingestion_statuses(config) -> list[dict]:
    """Return all jurisdiction_ingestion_status rows."""
    from core.db import connect_db

    with connect_db(config) as conn:
        rows = conn.execute(
            "SELECT * FROM jurisdiction_ingestion_status ORDER BY jurisdiction"
        ).fetchall()
    return [dict(r) for r in rows]


def sync_single_state(
    config,
    jurisdiction: str,
    state_path: str,
    progress_callback: Callable | None = None,
) -> dict:
    """
    Full incremental sync for one state folder:
    discover → parse → chunk/embed → sync titles → mirror catalog + chunks.

    Expects `chroma_write_lock` to be held by the caller while mirroring to Chroma.

    progress_callback(phase, detail, current, total) is called at each stage
    so UIs can render a live progress bar.

    Returns summary dict.
    """
    from core.db import connect_db
    from core.sources import list_source_files, parse_source_file, scan_source
    from core.title_registry import sync_legal_titles
    from indexer.chroma_health import is_chroma_fatal_store_error
    from indexer.chroma_mirror import mirror_title_catalog_to_chroma, mirror_title_chunks_to_chroma
    from indexer.sql_indexer import index_document_chunks_to_sql, purge_stale_embeddings

    jur = jurisdiction.strip().lower()
    cb = progress_callback or (lambda *_: None)
    summary = {
        "jurisdiction": jur,
        "path": state_path,
        "pdfs_discovered": 0,
        "pdfs_parsed": 0,
        "pdfs_skipped_unchanged": 0,
        "docs_chunked": 0,
        "titles_mirrored": 0,
        "status": "discovered",
        "error": None,
    }

    try:
        ws_id = _ensure_auto_workspace(config)
        src_id = _ensure_source_registered(config, ws_id, state_path)

        cb("scan", jur, 0, 1)
        scan_result = scan_source(config, workspace_id=ws_id, source_id=src_id,
                                  ignore_globs=getattr(config, "IGNORE_GLOBS", None))
        if not scan_result.get("ok"):
            summary["error"] = scan_result.get("error", "scan failed")
            summary["status"] = "error"
            _update_ingestion_status(config, jur, state_path, status="error",
                                     last_error=summary["error"], last_scan_at=_utc_now_iso())
            return summary
        cb("scan", jur, 1, 1)

        files = list_source_files(config, src_id)
        pdf_files = [f for f in files if f["file_type"] == "pdf"]
        summary["pdfs_discovered"] = len(pdf_files)

        _update_ingestion_status(config, jur, state_path,
                                 documents_count=len(pdf_files), last_scan_at=_utc_now_iso())

        with connect_db(config) as conn:
            parsed_sf_ids = {r[0] for r in conn.execute(
                "SELECT source_file_id FROM documents WHERE parse_status = 'parsed'"
            ).fetchall()}

        for idx, pf in enumerate(pdf_files):
            cb("parse", pf["rel_path"], idx, len(pdf_files))
            if pf["status"] == "unchanged" and pf["id"] in parsed_sf_ids:
                summary["pdfs_skipped_unchanged"] += 1
                continue
            res = parse_source_file(config, src_id, pf["id"])
            if res.get("ok"):
                summary["pdfs_parsed"] += 1
            else:
                logger.warning("parse failed %s: %s", pf["rel_path"], res.get("error"))
        cb("parse", "done", len(pdf_files), len(pdf_files))

        sync_legal_titles(config)

        parsed_total = summary["pdfs_parsed"] + summary["pdfs_skipped_unchanged"]
        _update_ingestion_status(config, jur, state_path, parsed_count=parsed_total)

        target_model = getattr(config, "DEFAULT_EMBED_MODEL", "nomic-embed-text-v2-moe:latest")
        if ":" not in target_model:
            target_model += ":latest"
        purge_stale_embeddings(config, target_model)

        with connect_db(config) as conn:
            docs = conn.execute(
                "SELECT id, rel_path FROM documents WHERE LOWER(jurisdiction) = LOWER(?) AND parse_status = 'parsed'",
                (jur,),
            ).fetchall()
            fully_embedded = set()
            for d in docs:
                has_embed = conn.execute(
                    "SELECT 1 FROM embeddings WHERE model = ? AND chunk_id IN (SELECT id FROM chunks WHERE document_id = ?) LIMIT 1",
                    (target_model, d["id"]),
                ).fetchone()
                if has_embed:
                    fully_embedded.add(d["id"])

        chunked = 0
        for idx, d in enumerate(docs):
            doc_label = d["rel_path"] if "rel_path" in d.keys() else d["id"][:8]
            cb("chunk_embed", doc_label, idx, len(docs))
            if d["id"] in fully_embedded:
                chunked += 1
                continue

            def _doc_progress(step, current, total, _idx=idx, _total=len(docs)):
                phase_detail = f"{doc_label} ({step} {current}/{total})"
                cb("chunk_embed", phase_detail, _idx, _total)

            res = index_document_chunks_to_sql(config, d["id"], progress_callback=_doc_progress)
            if res.get("ok"):
                chunked += 1
            else:
                logger.warning("chunk failed doc %s: %s", d["id"][:8], res.get("error"))
        cb("chunk_embed", "done", len(docs), len(docs))

        summary["docs_chunked"] = chunked
        _update_ingestion_status(config, jur, state_path, chunked_count=chunked)

        cb("mirror", jur, 0, 1)
        cat_res = mirror_title_catalog_to_chroma(config, jur)

        if not cat_res.get("ok"):
            err_msg = str(cat_res.get("error") or "catalog mirror failed")
            persist = getattr(config, "CHROMA_PERSIST_DIR", "")
            if is_chroma_fatal_store_error(err_msg):
                summary["error"] = (
                    f"Chroma persist/tenant error. Path: `{persist}`. "
                    "Use **Knowledge Base** → **Maintenance** → **Reset Chroma persist folder**, then sync again. "
                    f"Detail: {err_msg[:200]}"
                )
                summary["status"] = "error"
            else:
                summary["error"] = f"Catalog mirror failed: {err_msg[:280]}"
                summary["status"] = "partial"
            summary["titles_mirrored"] = 0
            _update_ingestion_status(
                config, jur, state_path,
                mirrored_count=0,
                status=summary["status"],
                last_error=summary["error"][:500],
            )
            cb("mirror", "aborted", 0, 1)
            return summary

        with connect_db(config) as conn:
            titles = conn.execute(
                "SELECT id, title_number FROM legal_titles WHERE LOWER(jurisdiction) = LOWER(?)",
                (jur,),
            ).fetchall()

        mirrored = 0
        for idx, t in enumerate(titles):
            cb("mirror", f"Title {t['title_number']}", idx, len(titles))
            res = mirror_title_chunks_to_chroma(config, t["id"])
            if res.get("ok"):
                mirrored += 1
        cb("mirror", "done", len(titles), len(titles))

        summary["titles_mirrored"] = mirrored

        if mirrored > 0 and cat_res.get("ok"):
            summary["status"] = "ready"
        elif parsed_total > 0:
            summary["status"] = "partial"
        else:
            summary["status"] = "discovered"

        _update_ingestion_status(config, jur, state_path,
                                 mirrored_count=mirrored, status=summary["status"], last_error=None)

    except Exception as exc:
        summary["error"] = str(exc)[:300]
        summary["status"] = "error"
        _update_ingestion_status(config, jur, state_path, status="error",
                                 last_error=summary["error"])

    return summary


def sync_states_root(
    config,
    states_root: str,
    progress_callback: Callable | None = None,
) -> dict:
    """
    Discover all state folders under *states_root* and sync each one.

    Callers should hold `chroma_write_lock(config)` while this runs if Chroma
    mirroring is enabled (UI and CLI do this).

    progress_callback(jurisdiction_idx, jurisdiction_total, phase, detail, phase_cur, phase_total)
    is called throughout so UIs can show a nested progress bar.

    Returns:
      {
        "ok": bool,
        "states_root": str,
        "discovered": int,
        "synced": int,
        "ready": int,
        "errors": int,
        "state_summaries": list[dict],
      }
    """
    from core.state_discovery import discover_state_folders

    cb = progress_callback or (lambda *_: None)
    folders = discover_state_folders(states_root)
    if not folders:
        return {"ok": False, "states_root": states_root, "discovered": 0, "synced": 0,
                "ready": 0, "errors": 0, "state_summaries": [],
                "error": "No state folders found under the States root."}

    summaries: list[dict] = []
    ready = 0
    errors = 0

    for jur_idx, sf in enumerate(folders):
        logger.info("sync_state %s (%d PDFs)", sf["jurisdiction"], sf["pdf_count"])

        def _state_cb(phase, detail, cur, total, _ji=jur_idx, _jt=len(folders)):
            cb(_ji, _jt, phase, detail, cur, total)

        result = sync_single_state(config, sf["jurisdiction"], sf["path"],
                                   progress_callback=_state_cb)
        summaries.append(result)
        if result["status"] == "ready":
            ready += 1
        if result.get("error"):
            errors += 1

    return {
        "ok": True,
        "states_root": states_root,
        "discovered": len(folders),
        "synced": len(summaries),
        "ready": ready,
        "errors": errors,
        "state_summaries": summaries,
    }
