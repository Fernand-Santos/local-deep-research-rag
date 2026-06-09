"""
Persist chunks + embeddings into SQL (canonical store).

Supports a progress_callback for UI progress bars. The callback receives
(step: str, current: int, total: int) at each major stage.
"""
from __future__ import annotations

import logging
import struct
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def index_document_chunks_to_sql(
    config,
    document_id: str,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    """
    Chunk all sections of a document, embed, and persist to SQL.

    progress_callback(step, current, total) is called for UI updates:
      step="estimate" → initial estimate
      step="chunk"    → chunking progress
      step="embed"    → embedding progress
    """
    from core.db import connect_db
    from indexer.chunker import (
        chunk_document_section,
        compute_dynamic_chunk_params,
        estimate_chunks_for_sections,
        total_body_chars,
    )
    from indexer.embedder import embed_texts

    doc_id = (document_id or "").strip()
    if not doc_id:
        return {"ok": False, "error": "document_id is required"}

    ts = _utc_now_iso()
    embed_model = getattr(config, "DEFAULT_EMBED_MODEL", "nomic-embed-text-v2-moe:latest")
    if ":" not in embed_model:
        embed_model = embed_model + ":latest"
    ollama_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    cb = progress_callback or (lambda *_: None)

    try:
        with connect_db(config) as conn:
            conn.execute("PRAGMA defer_foreign_keys = ON;")

            doc = conn.execute(
                "SELECT id, jurisdiction, title_number FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if not doc:
                return {"ok": False, "error": "Document not found"}

            jurisdiction = doc["jurisdiction"]
            title_number = doc["title_number"]

            sections = conn.execute(
                """
                SELECT id, document_id, section_number, catchline, history,
                       body_text, page_start, page_end, section_order
                FROM document_sections
                WHERE document_id = ?
                ORDER BY section_order ASC
                """,
                (doc_id,),
            ).fetchall()

            if not sections:
                return {"ok": False, "error": "No sections to chunk"}

            sec_dicts = [dict(s) for s in sections]
            doc_chars = total_body_chars(sec_dicts)
            chunk_params = compute_dynamic_chunk_params(doc_chars)
            logger.info(
                "chunk: doc=%s chars=%d bucket=%s max_tokens=%d overlap=%d",
                doc_id, doc_chars, chunk_params["bucket"],
                chunk_params["max_tokens"], chunk_params["overlap_tokens"],
            )
            estimated_total = estimate_chunks_for_sections(sec_dicts, chunk_params)
            cb("estimate", 0, estimated_total)

            all_chunks: list[dict] = []
            strategy_counts: Counter = Counter()
            for i, sec_dict in enumerate(sec_dicts):
                chunks = chunk_document_section(sec_dict, chunk_params)
                for c in chunks:
                    c["jurisdiction"] = jurisdiction
                    c["title_number"] = title_number
                    c["document_id"] = doc_id
                all_chunks.extend(chunks)
                if chunks:
                    strategy_counts[chunks[0]["chunk_strategy"]] += len(chunks)
                cb("chunk", len(all_chunks), estimated_total)

            if not all_chunks:
                return {"ok": False, "error": "Chunking produced 0 chunks"}

            actual_total = len(all_chunks)

            _purge_document_chunks(conn, doc_id)

            chunk_ids: list[str] = []
            chunk_texts: list[str] = []
            for c in all_chunks:
                cid = uuid.uuid4().hex
                chunk_ids.append(cid)
                chunk_texts.append(c["text"])
                conn.execute(
                    """
                    INSERT INTO chunks (id, document_id, document_section_id, chunk_index,
                        chunk_strategy, text, token_estimate, jurisdiction, title_number,
                        section_number, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cid, doc_id, c.get("document_section_id"), c["chunk_index"],
                     c["chunk_strategy"], c["text"], c["token_estimate"],
                     c.get("jurisdiction"), c.get("title_number"), c.get("section_number"), ts, ts),
                )
            conn.commit()

            EMBED_BATCH = 64
            vectors: list[list[float]] = []
            for batch_start in range(0, len(chunk_texts), EMBED_BATCH):
                batch = chunk_texts[batch_start:batch_start + EMBED_BATCH]
                vecs = embed_texts(batch, model=embed_model, ollama_url=ollama_url)
                vectors.extend(vecs)
                cb("embed", len(vectors), actual_total)

            dims = len(vectors[0]) if vectors else 0

            for cid, vec in zip(chunk_ids, vectors):
                conn.execute(
                    """
                    INSERT INTO embeddings (id, chunk_id, model, dimensions, vector, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (uuid.uuid4().hex, cid, embed_model, dims, _pack_vector(vec), ts),
                )
            conn.commit()

        return {
            "ok": True,
            "document_id": doc_id,
            "chunks_created": actual_total,
            "embeddings_created": len(vectors),
            "embed_model": embed_model,
            "dimensions": dims,
            "strategies": dict(strategy_counts),
            "estimated_chunks": estimated_total,
        }
    except Exception as exc:
        err = str(exc)
        try:
            from core.analytics import record_document_index
            if doc_id:
                em = getattr(config, "DEFAULT_EMBED_MODEL", "nomic-embed-text-v2-moe:latest")
                record_document_index(
                    config, document_id=doc_id, chunks_created=0,
                    embed_model=em, duration_ms=None, ok=False, error=err[:500],
                )
        except Exception:
            pass
        return {"ok": False, "error": err}


def index_title_chunks_to_sql(config, jurisdiction: str, title_number: str) -> dict:
    from core.db import connect_db

    if not jurisdiction or not title_number:
        return {"ok": False, "error": "jurisdiction and title_number are required"}

    with connect_db(config) as conn:
        rows = conn.execute(
            "SELECT id FROM documents WHERE jurisdiction = ? AND title_number = ?",
            (jurisdiction.strip(), title_number.strip()),
        ).fetchall()

    if not rows:
        return {"ok": False, "error": "No documents found for this title"}

    total_chunks = 0
    total_embeddings = 0
    errors: list[str] = []
    for row in rows:
        res = index_document_chunks_to_sql(config, row["id"])
        if res.get("ok"):
            total_chunks += res.get("chunks_created", 0)
            total_embeddings += res.get("embeddings_created", 0)
        else:
            errors.append(f"{row['id'][:8]}: {res.get('error')}")

    return {
        "ok": len(errors) == 0,
        "documents_processed": len(rows),
        "chunks_created": total_chunks,
        "embeddings_created": total_embeddings,
        "errors": errors or None,
    }


def _purge_document_chunks(conn, doc_id: str) -> int:
    """Delete all chunks, embeddings, and index_map entries for a document. Returns count deleted."""
    old_chunk_ids = [
        r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
        ).fetchall()
    ]
    if not old_chunk_ids:
        return 0
    for batch_start in range(0, len(old_chunk_ids), 500):
        batch = old_chunk_ids[batch_start:batch_start + 500]
        placeholders = ",".join("?" * len(batch))
        conn.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", batch)
        conn.execute(f"DELETE FROM index_map WHERE chunk_id IN ({placeholders})", batch)
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    return len(old_chunk_ids)


def purge_stale_embeddings(config, target_model: str) -> dict:
    """Delete embeddings whose model doesn't match *target_model*, enabling clean re-embed."""
    from core.db import connect_db

    m = target_model.strip()
    if ":" not in m:
        m += ":latest"

    with connect_db(config) as conn:
        conn.execute("PRAGMA defer_foreign_keys = ON;")
        stale = conn.execute(
            "SELECT id, chunk_id FROM embeddings WHERE model != ?", (m,)
        ).fetchall()
        if not stale:
            return {"ok": True, "deleted": 0, "target_model": m}
        for row in stale:
            conn.execute("DELETE FROM embeddings WHERE id = ?", (row["id"],))
        conn.commit()
    logger.info("purged %d stale embeddings (not model=%s)", len(stale), m)
    return {"ok": True, "deleted": len(stale), "target_model": m}


def purge_all_index_data(config) -> dict:
    """
    Delete ALL mirrored/indexed retrieval payloads in SQL:
    embeddings, index_map, and chunks.

    Documents/sections remain intact so a subsequent sync can cleanly re-chunk
    and re-embed everything.
    """
    from core.db import connect_db

    with connect_db(config) as conn:
        conn.execute("PRAGMA defer_foreign_keys = ON;")
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        map_count = conn.execute("SELECT COUNT(*) FROM index_map").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM index_map")
        conn.execute("DELETE FROM chunks")
        conn.commit()

    logger.info(
        "purge_all_index_data: deleted embeddings=%d index_map=%d chunks=%d",
        emb_count, map_count, chunk_count,
    )
    return {
        "ok": True,
        "deleted_embeddings": emb_count,
        "deleted_index_map": map_count,
        "deleted_chunks": chunk_count,
    }


def list_chunks_for_document(config, document_id: str) -> list[dict]:
    from core.db import connect_db

    doc_id = (document_id or "").strip()
    if not doc_id:
        return []

    with connect_db(config) as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.document_id, c.document_section_id, c.chunk_index,
                   c.chunk_strategy, c.text, c.token_estimate, c.jurisdiction,
                   c.title_number, c.section_number, c.created_at, c.updated_at,
                   e.model as embed_model, e.dimensions as embed_dimensions
            FROM chunks c
            LEFT JOIN embeddings e ON e.chunk_id = c.id
            WHERE c.document_id = ?
            ORDER BY c.chunk_index ASC
            """,
            (doc_id,),
        ).fetchall()
    return [dict(r) for r in rows]
