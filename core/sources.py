from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_source_path(path: str, config) -> dict:
    raw = (path or "").strip()
    if not raw:
        return {"ok": False, "error": "Path is required", "normalized_path": None}

    try:
        p = Path(raw).expanduser()
        # Resolve if possible; fall back to absolute normalization.
        try:
            resolved = p.resolve(strict=True)
        except Exception:
            resolved = p.absolute()

        normalized = os.path.normpath(str(resolved))
        if not os.path.exists(normalized):
            return {"ok": False, "error": "Path does not exist", "normalized_path": normalized}
        if not os.path.isdir(normalized):
            return {"ok": False, "error": "Path must be a directory", "normalized_path": normalized}
        return {"ok": True, "error": None, "normalized_path": normalized}
    except Exception:
        return {"ok": False, "error": "Invalid path", "normalized_path": None}


def add_source(
    config,
    workspace_id: str,
    path: str,
    source_type: str = "folder",
    is_temp: bool = False,
) -> dict:
    from core.db import connect_db

    ws_id = (workspace_id or "").strip()
    if not ws_id:
        return {"ok": False, "error": "Workspace is required"}

    v = validate_source_path(path, config)
    if not v.get("ok"):
        return {"ok": False, "error": v.get("error"), "normalized_path": v.get("normalized_path")}

    normalized = v["normalized_path"]
    src_id = uuid.uuid4().hex
    ts = _utc_now_iso()
    stype = (source_type or "folder").strip() or "folder"
    is_temp_int = 1 if bool(is_temp) else 0

    try:
        with connect_db(config) as conn:
            # Reject duplicates within the same workspace.
            exists = conn.execute(
                "SELECT 1 FROM sources WHERE workspace_id = ? AND path = ?",
                (ws_id, normalized),
            ).fetchone()
            if exists:
                return {"ok": False, "error": "Source already registered in this workspace", "normalized_path": normalized}

            conn.execute(
                """
                INSERT INTO sources (id, workspace_id, path, source_type, is_temp, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (src_id, ws_id, normalized, stype, is_temp_int, ts),
            )
            conn.commit()

        return {
            "ok": True,
            "source": {
                "id": src_id,
                "workspace_id": ws_id,
                "path": normalized,
                "source_type": stype,
                "is_temp": bool(is_temp),
                "created_at": ts,
            },
        }
    except Exception:
        return {"ok": False, "error": "Failed to add source", "normalized_path": normalized}


def list_sources(config, workspace_id: str) -> list[dict]:
    from core.db import connect_db

    ws_id = (workspace_id or "").strip()
    if not ws_id:
        return []

    with connect_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, workspace_id, path, source_type, is_temp, created_at
            FROM sources
            WHERE workspace_id = ?
            ORDER BY created_at DESC
            """,
            (ws_id,),
        ).fetchall()

    out: list[dict] = []
    for r in rows:
        d = dict(r)
        d["is_temp"] = bool(d.get("is_temp"))
        out.append(d)
    return out


def scan_source(config, workspace_id: str, source_id: str, ignore_globs: list[str] | None = None) -> dict:
    """
    Discover files under a registered source and upsert into source_files.
    Creates an indexing_runs record and returns summary counts.
    """
    from core.db import connect_db
    from core.discovery import discover_files

    ws_id = (workspace_id or "").strip()
    src_id = (source_id or "").strip()
    if not ws_id or not src_id:
        return {"ok": False, "error": "workspace_id and source_id are required"}

    run_id = uuid.uuid4().hex
    started = _utc_now_iso()

    counts = {
        "files_scanned": 0,
        "files_added": 0,
        "files_updated": 0,
        "files_unchanged": 0,
        "files_skipped": 0,
    }

    try:
        with connect_db(config) as conn:
            src = conn.execute(
                "SELECT id, workspace_id, path FROM sources WHERE id = ? AND workspace_id = ?",
                (src_id, ws_id),
            ).fetchone()
            if not src:
                return {"ok": False, "error": "Source not found for workspace"}

            conn.execute(
                """
                INSERT INTO indexing_runs (id, workspace_id, source_id, status, files_scanned, files_added, files_updated,
                                          files_unchanged, files_skipped, started_at, finished_at)
                VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, ?, NULL)
                """,
                (run_id, ws_id, src_id, "running", started),
            )
            conn.commit()

            discovered = discover_files(str(src["path"]), ignore_globs=ignore_globs)
            counts["files_scanned"] = len(discovered)

            existing_rows = conn.execute(
                "SELECT id, path, size_bytes, mtime_ns FROM source_files WHERE source_id = ?",
                (src_id,),
            ).fetchall()
            existing = {r["path"]: dict(r) for r in existing_rows}

            finished = _utc_now_iso()
            for f in discovered:
                p = f["path"]
                rel_path = f["rel_path"]
                size_bytes = int(f["size_bytes"])
                mtime_ns = int(f["mtime_ns"])
                file_type = f["file_type"]

                if p in existing:
                    prev = existing[p]
                    if int(prev["mtime_ns"]) != mtime_ns or int(prev["size_bytes"]) != size_bytes:
                        conn.execute(
                            """
                            UPDATE source_files
                            SET rel_path = ?, file_type = ?, size_bytes = ?, mtime_ns = ?, status = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (rel_path, file_type, size_bytes, mtime_ns, "updated", finished, prev["id"]),
                        )
                        counts["files_updated"] += 1
                    else:
                        conn.execute(
                            "UPDATE source_files SET status = ?, updated_at = ? WHERE id = ?",
                            ("unchanged", finished, prev["id"]),
                        )
                        counts["files_unchanged"] += 1
                else:
                    fid = uuid.uuid4().hex
                    conn.execute(
                        """
                        INSERT INTO source_files (id, source_id, path, rel_path, file_type, size_bytes, mtime_ns, sha256,
                                                  status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                        """,
                        (fid, src_id, p, rel_path, file_type, size_bytes, mtime_ns, "discovered", finished, finished),
                    )
                    counts["files_added"] += 1

            conn.execute(
                """
                UPDATE indexing_runs
                SET status = ?, files_scanned = ?, files_added = ?, files_updated = ?, files_unchanged = ?, files_skipped = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (
                    "completed",
                    counts["files_scanned"],
                    counts["files_added"],
                    counts["files_updated"],
                    counts["files_unchanged"],
                    counts["files_skipped"],
                    finished,
                    run_id,
                ),
            )
            conn.commit()

        return {"ok": True, "run_id": run_id, "started_at": started, "finished_at": finished, **counts}
    except Exception:
        try:
            with connect_db(config) as conn:
                conn.execute(
                    "UPDATE indexing_runs SET status = ?, finished_at = ? WHERE id = ?",
                    ("failed", _utc_now_iso(), run_id),
                )
                conn.commit()
        except Exception:
            pass
        return {"ok": False, "error": "Scan failed"}


def list_source_files(config, source_id: str) -> list[dict]:
    from core.db import connect_db

    src_id = (source_id or "").strip()
    if not src_id:
        return []

    with connect_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, source_id, path, rel_path, file_type, size_bytes, mtime_ns, sha256, status, created_at, updated_at
            FROM source_files
            WHERE source_id = ?
            ORDER BY rel_path ASC
            """,
            (src_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def parse_source_file(config, source_id: str, source_file_id: str) -> dict:
    """
    Parse a single PDF source_file: extract text, derive metadata, persist
    documents + document_sections rows.  Idempotent on re-parse.
    """
    from core.db import connect_db
    from core.metadata import parse_filename_metadata, parse_path_metadata
    from core.pdf_parser import extract_pdf_text, parse_legal_sections

    src_id = (source_id or "").strip()
    sf_id = (source_file_id or "").strip()
    if not src_id or not sf_id:
        return {"ok": False, "error": "source_id and source_file_id are required"}

    ts = _utc_now_iso()

    try:
        with connect_db(config) as conn:
            sf = conn.execute(
                "SELECT id, source_id, path, rel_path, file_type FROM source_files WHERE id = ? AND source_id = ?",
                (sf_id, src_id),
            ).fetchone()
            if not sf:
                return {"ok": False, "error": "Source file not found"}

            if sf["file_type"] != "pdf":
                return {"ok": False, "error": f"Unsupported file_type: {sf['file_type']}"}

            path_meta = parse_path_metadata(sf["path"])
            file_meta = parse_filename_metadata(os.path.basename(sf["path"]))

            extraction = extract_pdf_text(sf["path"])
            if not extraction.get("ok"):
                conn.execute(
                    """
                    INSERT INTO documents (id, source_file_id, path, rel_path, corpus, jurisdiction_type,
                        jurisdiction, document_family, title_number, title_name, file_type, extraction_mode,
                        page_count, parse_status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_file_id) DO UPDATE SET
                        extraction_mode=excluded.extraction_mode, parse_status=excluded.parse_status,
                        updated_at=excluded.updated_at
                    """,
                    (
                        uuid.uuid4().hex, sf_id, sf["path"], sf["rel_path"],
                        path_meta.get("corpus"), path_meta.get("jurisdiction_type"),
                        path_meta.get("jurisdiction"), file_meta.get("document_family"),
                        file_meta.get("title_number"), file_meta.get("title_name"),
                        sf["file_type"], extraction.get("extraction_mode", "none"),
                        0, "failed", ts, ts,
                    ),
                )
                conn.commit()
                return {"ok": False, "error": extraction.get("error") or "Extraction failed"}

            sections = parse_legal_sections(extraction["text"], pages=extraction.get("pages"))

            # Upsert document (unique on source_file_id).
            # Check for existing doc first.
            existing_doc = conn.execute(
                "SELECT id FROM documents WHERE source_file_id = ?", (sf_id,)
            ).fetchone()

            if existing_doc:
                doc_id = existing_doc["id"]
                conn.execute(
                    """
                    UPDATE documents SET path=?, rel_path=?, corpus=?, jurisdiction_type=?,
                        jurisdiction=?, document_family=?, title_number=?, title_name=?,
                        file_type=?, extraction_mode=?, page_count=?, parse_status=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        sf["path"], sf["rel_path"],
                        path_meta.get("corpus"), path_meta.get("jurisdiction_type"),
                        path_meta.get("jurisdiction"), file_meta.get("document_family"),
                        file_meta.get("title_number"), file_meta.get("title_name"),
                        sf["file_type"], extraction["extraction_mode"],
                        extraction["page_count"], "parsed", ts, doc_id,
                    ),
                )
            else:
                doc_id = uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO documents (id, source_file_id, path, rel_path, corpus, jurisdiction_type,
                        jurisdiction, document_family, title_number, title_name, file_type, extraction_mode,
                        page_count, parse_status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id, sf_id, sf["path"], sf["rel_path"],
                        path_meta.get("corpus"), path_meta.get("jurisdiction_type"),
                        path_meta.get("jurisdiction"), file_meta.get("document_family"),
                        file_meta.get("title_number"), file_meta.get("title_name"),
                        sf["file_type"], extraction["extraction_mode"],
                        extraction["page_count"], "parsed", ts, ts,
                    ),
                )

            # Replace sections for this document.
            conn.execute("DELETE FROM document_sections WHERE document_id = ?", (doc_id,))
            for sec in sections:
                conn.execute(
                    """
                    INSERT INTO document_sections (id, document_id, section_number, catchline,
                        history, body_text, page_start, page_end, section_order, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex, doc_id, sec.get("section_number"),
                        sec.get("catchline"), sec.get("history"),
                        sec["body_text"], sec.get("page_start"), sec.get("page_end"),
                        sec["section_order"], ts, ts,
                    ),
                )

            conn.commit()

        return {
            "ok": True,
            "document_id": doc_id,
            "parse_status": "parsed",
            "extraction_mode": extraction["extraction_mode"],
            "page_count": extraction["page_count"],
            "sections_created": len(sections),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_documents_for_source(config, source_id: str) -> list[dict]:
    from core.db import connect_db

    src_id = (source_id or "").strip()
    if not src_id:
        return []

    with connect_db(config) as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.source_file_id, d.path, d.rel_path, d.corpus, d.jurisdiction_type,
                   d.jurisdiction, d.document_family, d.title_number, d.title_name,
                   d.file_type, d.extraction_mode, d.page_count, d.parse_status,
                   d.created_at, d.updated_at
            FROM documents d
            JOIN source_files sf ON d.source_file_id = sf.id
            WHERE sf.source_id = ?
            ORDER BY d.rel_path ASC
            """,
            (src_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_sections_for_document(config, document_id: str) -> list[dict]:
    from core.db import connect_db

    doc_id = (document_id or "").strip()
    if not doc_id:
        return []

    with connect_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, document_id, section_number, catchline, history, body_text,
                   page_start, page_end, section_order, created_at, updated_at
            FROM document_sections
            WHERE document_id = ?
            ORDER BY section_order ASC
            """,
            (doc_id,),
        ).fetchall()
    return [dict(r) for r in rows]
