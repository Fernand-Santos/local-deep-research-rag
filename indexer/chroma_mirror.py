"""
Mirror SQL canonical data into Chroma collections.

- Catalog collection: one per jurisdiction (compact routing entries from legal_titles).
- Chunk collection: one per legal title (full chunks with pre-computed embeddings from SQL).
- index_map rows provide SQL-side traceability for every mirrored chunk.

Collections are deleted and recreated on each mirror to avoid corrupt HNSW state.

Tested against chromadb 1.4.x (PersistentClient, list_collections returns Collection objects).
"""
from __future__ import annotations

import logging
import struct
import time
import uuid
from datetime import datetime, timezone

from indexer.chroma_store import (
    get_chroma_client,
    get_scoped_chunk_collection_name,
    get_state_catalog_collection_name,
)

logger = logging.getLogger(__name__)

_LAST_FATAL_LOG: dict[str, float] = {}
_FATAL_LOG_COOLDOWN_S = 60.0


def _log_mirror_failure(operation: str, subject: str, exc: BaseException, config) -> None:
    """Log mirror errors with persist path; throttle identical fatal store errors in tight loops."""
    from indexer.chroma_health import is_chroma_fatal_store_error

    err = str(exc)
    persist = getattr(config, "CHROMA_PERSIST_DIR", "?")
    if is_chroma_fatal_store_error(err):
        fp = f"{operation}|{err[:120]}"
        now = time.time()
        if now - _LAST_FATAL_LOG.get(fp, 0.0) >= _FATAL_LOG_COOLDOWN_S:
            _LAST_FATAL_LOG[fp] = now
            logger.error(
                "%s failed for %s: %s | CHROMA_PERSIST_DIR=%s",
                operation, subject, err, persist,
            )
        else:
            logger.debug("%s fatal store error (suppressed repeat): %s", operation, err[:160])
    else:
        logger.error(
            "%s failed for %s: %s | CHROMA_PERSIST_DIR=%s",
            operation, subject, err, persist,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _safe_delete_collection(client, name: str) -> None:
    """Delete a collection if it exists, ignoring errors on corrupt collections."""
    try:
        client.delete_collection(name)
    except ValueError:
        pass
    except Exception as exc:
        logger.debug("_safe_delete_collection('%s') ignored: %s", name, exc)


# chromadb 1.4 defaults sync_threshold=1000: HNSW binary segments are not flushed
# to disk until 1000 mutations. Most title collections are smaller, so a fresh
# Python process (Streamlit restart) sees empty HNSW files → "nothing found on disk".
_HNSW_PERSIST_CONFIG = {"hnsw:sync_threshold": 1}
_MIRROR_ADD_BATCH_SIZE = 400


def _create_mirror_collection(client, name: str):
    """Create a collection configured to flush HNSW segments after every add."""
    return client.create_collection(
        name=name,
        embedding_function=None,
        configuration=_HNSW_PERSIST_CONFIG,
    )


def _as_float_list(vec) -> list[float]:
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        return list(vec.tolist())
    return list(vec)


def _flush_collection_vectors(col, embeddings: list[list[float]] | None) -> None:
    """
    Force HNSW persistence by issuing a minimal vector query after writes.

    With sync_threshold=1 this is mostly belt-and-suspenders, but it also
    validates the collection is queryable before we return from mirror.
    """
    if not embeddings:
        return
    probe = _as_float_list(embeddings[0])
    if not probe:
        return
    try:
        n = col.count()
        if n <= 0:
            return
        col.query(
            query_embeddings=[probe],
            n_results=min(1, n),
            include=[],
        )
    except Exception as exc:
        logger.warning("_flush_collection_vectors failed for '%s': %s", col.name, exc)
        raise


def _add_to_collection_batched(
    col,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Add records in batches and flush HNSW to disk after the final batch."""
    total = len(ids)
    if total == 0:
        return
    batch = _MIRROR_ADD_BATCH_SIZE
    for start in range(0, total, batch):
        end = min(start + batch, total)
        col.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
    _flush_collection_vectors(col, embeddings)


# ------------------------------------------------------------------
# A. Catalog mirroring
# ------------------------------------------------------------------

def mirror_title_catalog_to_chroma(config, jurisdiction: str) -> dict:
    """
    Create / replace a catalog collection for the given jurisdiction.
    Deletes the old collection first to avoid corrupt HNSW state.
    """
    from core.db import connect_db
    from indexer.embedder import embed_texts

    jur = (jurisdiction or "").strip()
    if not jur:
        return {"ok": False, "error": "jurisdiction is required"}

    col_name = get_state_catalog_collection_name(config, jur)
    ts = _utc_now_iso()
    embed_model = getattr(config, "DEFAULT_EMBED_MODEL", "nomic-embed-text-v2-moe:latest")
    if ":" not in embed_model:
        embed_model += ":latest"
    ollama_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        with connect_db(config) as conn:
            rows = conn.execute(
                "SELECT id, jurisdiction, title_number, title_name, collection_key FROM legal_titles WHERE LOWER(jurisdiction) = LOWER(?)",
                (jur,),
            ).fetchall()

        if not rows:
            return {"ok": False, "error": f"No legal_titles found for jurisdiction={jur}"}

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for r in rows:
            ids.append(r["id"])
            routing_text = f"{r['jurisdiction']} Title {r['title_number']}: {r['title_name']}"
            docs.append(routing_text)
            metas.append({
                "legal_title_id": r["id"],
                "jurisdiction": r["jurisdiction"],
                "title_number": str(r["title_number"]),
                "title_name": r["title_name"] or "",
                "collection_key": r["collection_key"] or "",
            })

        vectors = embed_texts(docs, model=embed_model, ollama_url=ollama_url)

        client = get_chroma_client(config)
        _safe_delete_collection(client, col_name)
        col = _create_mirror_collection(client, col_name)
        _add_to_collection_batched(col, ids, docs, metas, vectors)

        logger.info("mirror_title_catalog_to_chroma: '%s' → %d records", col_name, len(ids))
        return {
            "ok": True,
            "collection_name": col_name,
            "records_mirrored": len(ids),
            "embed_model": embed_model,
        }
    except Exception as exc:
        _log_mirror_failure("mirror_title_catalog_to_chroma", jur, exc, config)
        return {"ok": False, "error": str(exc)}


# ------------------------------------------------------------------
# B. Chunk mirroring
# ------------------------------------------------------------------

def mirror_title_chunks_to_chroma(config, legal_title_id: str) -> dict:
    """
    Mirror all SQL chunks (with pre-computed embeddings) for a legal title
    into a scoped Chroma collection. Deletes old collection first.
    Writes index_map rows for traceability.
    """
    from core.db import connect_db

    lt_id = (legal_title_id or "").strip()
    if not lt_id:
        return {"ok": False, "error": "legal_title_id is required"}

    ts = _utc_now_iso()

    try:
        with connect_db(config) as conn:
            lt = conn.execute(
                "SELECT id, jurisdiction, title_number, title_name, collection_key FROM legal_titles WHERE id = ?",
                (lt_id,),
            ).fetchone()
            if not lt:
                return {"ok": False, "error": "Legal title not found"}

            jurisdiction = lt["jurisdiction"]
            title_number = lt["title_number"]
            title_name = lt["title_name"] or ""
            col_name = get_scoped_chunk_collection_name(config, jurisdiction, title_number)

            rows = conn.execute(
                """
                SELECT c.id as chunk_id, c.document_id, c.document_section_id,
                       c.chunk_index, c.chunk_strategy, c.text, c.token_estimate,
                       c.jurisdiction, c.title_number, c.section_number,
                       ds.catchline, ds.page_start, ds.page_end,
                       d.rel_path as source_path,
                       e.vector as emb_blob, e.model as embed_model, e.dimensions as embed_dims
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                LEFT JOIN document_sections ds ON c.document_section_id = ds.id
                LEFT JOIN embeddings e ON e.chunk_id = c.id
                WHERE LOWER(c.jurisdiction) = LOWER(?) AND c.title_number = ?
                ORDER BY c.chunk_index ASC
                """,
                (jurisdiction, title_number),
            ).fetchall()

            if not rows:
                return {"ok": False, "error": "No chunks found for this title"}

            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict] = []
            embs: list[list[float]] = []
            embed_model_name = ""
            skipped_no_embed = 0

            for r in rows:
                blob = r["emb_blob"]
                if not blob:
                    skipped_no_embed += 1
                    continue
                vec = _unpack_vector(blob)
                ids.append(r["chunk_id"])
                docs.append(r["text"])
                embs.append(vec)
                embed_model_name = r["embed_model"] or embed_model_name
                metas.append({
                    "chunk_id": r["chunk_id"],
                    "jurisdiction": r["jurisdiction"] or "",
                    "title_number": str(r["title_number"] or ""),
                    "title_name": title_name,
                    "section_number": r["section_number"] or "",
                    "catchline": r["catchline"] or "",
                    "source_path": r["source_path"] or "",
                    "page_start": int(r["page_start"]) if r["page_start"] is not None else -1,
                    "page_end": int(r["page_end"]) if r["page_end"] is not None else -1,
                    "document_id": r["document_id"] or "",
                    "document_section_id": r["document_section_id"] or "",
                    "chunk_strategy": r["chunk_strategy"] or "",
                })

            if not ids:
                if skipped_no_embed > 0:
                    logger.warning("mirror %s: %d chunks but ALL lack embeddings — skipping collection creation",
                                   col_name, skipped_no_embed)
                return {"ok": False, "error": f"No chunks with embeddings found ({skipped_no_embed} without embeddings)"}

            if skipped_no_embed > 0:
                logger.info("mirror %s: skipped %d chunks without embeddings", col_name, skipped_no_embed)

            client = get_chroma_client(config)
            _safe_delete_collection(client, col_name)
            col = _create_mirror_collection(client, col_name)
            _add_to_collection_batched(col, ids, docs, metas, embs)

            conn.execute(
                "DELETE FROM index_map WHERE collection_key = ? AND index_type = ?",
                (col_name, "chroma"),
            )
            for cid in ids:
                conn.execute(
                    "INSERT INTO index_map (id, chunk_id, index_type, external_id, collection_key, created_at) VALUES (?,?,?,?,?,?)",
                    (uuid.uuid4().hex, cid, "chroma", cid, col_name, ts),
                )
            conn.commit()

        logger.info("mirror_title_chunks_to_chroma: '%s' → %d records (skipped %d without embeddings)",
                     col_name, len(ids), skipped_no_embed)
        return {
            "ok": True,
            "collection_name": col_name,
            "records_mirrored": len(ids),
            "embed_model": embed_model_name,
            "skipped_no_embed": skipped_no_embed,
        }
    except Exception as exc:
        _log_mirror_failure("mirror_title_chunks_to_chroma", lt_id, exc, config)
        return {"ok": False, "error": str(exc)}


# ------------------------------------------------------------------
# C. Utility
# ------------------------------------------------------------------

def get_jurisdiction_catalog_status(config, jurisdiction: str) -> dict:
    """
    Whether the per-jurisdiction Chroma collection used for title routing exists and has rows.

    Used by retrieval and UI to detect missing `catalog_<jurisdiction>` after reset or partial sync.

    Returns:
        ok: True if collection exists and count > 0
        collection_name: e.g. catalog_alabama
        count: routing entries
        error: None if ok, else a short user-facing message
    """
    jur = (jurisdiction or "").strip()
    name = get_state_catalog_collection_name(config, jur)
    if not jur:
        return {"ok": False, "collection_name": name, "count": 0, "error": "jurisdiction is empty"}

    try:
        client = get_chroma_client(config)
        col = client.get_collection(name)
        n = col.count()
        if n <= 0:
            return {
                "ok": False,
                "collection_name": name,
                "count": 0,
                "error": "catalog collection exists but is empty — run Sync States Root or Mirror catalog to Chroma",
            }
        return {"ok": True, "collection_name": name, "count": n, "error": None}
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()
        if "does not exist" in low or "not found" in low:
            hint = (
                "catalog collection missing — run **Knowledge Base** → **Sync States Root**, "
                "or **Advanced** → **Chroma Mirror** → **Mirror catalog to Chroma**"
            )
        else:
            hint = msg
        return {"ok": False, "collection_name": name, "count": 0, "error": hint}


def list_chroma_collections(config) -> list[dict]:
    """List all Chroma collections with their record counts.

    chromadb 1.x: list_collections() returns Collection objects with .name attribute.
    We always use get_collection(name) for a fresh reference before calling count().
    """
    try:
        client = get_chroma_client(config)
        raw_cols = client.list_collections()
        results: list[dict] = []
        for item in raw_cols:
            name = item.name if hasattr(item, "name") else str(item)
            try:
                col = client.get_collection(name)
                results.append({"name": name, "count": col.count()})
            except Exception as inner:
                logger.warning("list_chroma_collections: collection '%s' unreadable: %s", name, inner)
                results.append({"name": name, "count": -1})
        logger.info("list_chroma_collections: found %d collections", len(results))
        return results
    except Exception as exc:
        persist = getattr(config, "CHROMA_PERSIST_DIR", "?")
        logger.error("list_chroma_collections failed: %s | CHROMA_PERSIST_DIR=%s", exc, persist)
        return []


def delete_all_chroma_collections(config) -> dict:
    """Delete ALL Chroma collections. Used for full reset."""
    try:
        client = get_chroma_client(config)
        raw_cols = client.list_collections()
        deleted = 0
        for item in raw_cols:
            name = item.name if hasattr(item, "name") else str(item)
            _safe_delete_collection(client, name)
            deleted += 1
        logger.info("delete_all_chroma_collections: deleted %d", deleted)
        return {"ok": True, "deleted": deleted}
    except Exception as exc:
        persist = getattr(config, "CHROMA_PERSIST_DIR", "?")
        logger.error("delete_all_chroma_collections failed: %s | CHROMA_PERSIST_DIR=%s", exc, persist)
        return {"ok": False, "error": str(exc)}
