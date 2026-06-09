"""
Chroma index health probing, circuit breaker for corrupt HNSW segments,
and unified repair (re-mirror catalog + title collections from SQL).

Detects errors such as:
  Error creating hnsw segment reader: Nothing found on disk
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from indexer.chroma_store import (
    get_chroma_client,
    get_scoped_chunk_collection_name,
    get_state_catalog_collection_name,
)

logger = logging.getLogger(__name__)

_HNSW_MARKERS = (
    "nothing found on disk",
    "hnsw segment reader",
    "error creating hnsw",
    "segment reader",
)

# Chroma system / tenant bootstrap failures (persist dir corruption, bad sqlite, etc.)
_FATAL_STORE_MARKERS = (
    "could not connect to tenant",
    "default_tenant",
    "unable to open database",
    "database disk image is malformed",
    "readonly database",
)

# In-process Rust binding init failures (chromadb 1.x). Once this fires, the
# Python process is poisoned: SharedSystemClient may have cached a partially
# constructed RustBindingsAPI whose `.bindings` attribute was never assigned.
# Recovery requires clearing chromadb's shared system cache OR restarting
# the Python process.
_RUNTIME_POISONED_MARKERS = (
    "rustbindingsapi",
    "has no attribute 'bindings'",
    "object has no attribute 'bindings'",
)

_UNHEALTHY: set[str] = set()
_UNHEALTHY_LOCK = threading.Lock()
_LOGGED_BREAKER: set[str] = set()


def is_chroma_disk_corruption_error(message: str | None) -> bool:
    if not message:
        return False
    low = message.lower()
    return any(m in low for m in _HNSW_MARKERS)


def is_chroma_fatal_store_error(message: str | None) -> bool:
    """True when Chroma's persist store / tenant layer is broken (not a single bad collection)."""
    if not message:
        return False
    low = message.lower()
    return any(m in low for m in _FATAL_STORE_MARKERS) or is_chroma_runtime_poisoned(message)


def is_chroma_fatal_store_exception(exc: BaseException) -> bool:
    return is_chroma_fatal_store_error(str(exc))


def is_chroma_runtime_poisoned(message: str | None) -> bool:
    """
    True when chromadb's in-process Rust runtime is poisoned (RustBindingsAPI
    constructed without a `.bindings` attribute). Symptom:
        AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'

    A persist-dir reset alone CANNOT recover from this — the Python process must
    either clear chromadb's SharedSystemClient cache or be restarted.
    """
    if not message:
        return False
    low = message.lower()
    return any(m in low for m in _RUNTIME_POISONED_MARKERS)


def is_chroma_runtime_poisoned_exception(exc: BaseException) -> bool:
    return is_chroma_runtime_poisoned(str(exc))


def mark_collection_unhealthy(collection_name: str) -> None:
    """Open circuit for this collection until repair clears state."""
    name = (collection_name or "").strip()
    if not name:
        return
    with _UNHEALTHY_LOCK:
        _UNHEALTHY.add(name)
        if name not in _LOGGED_BREAKER:
            _LOGGED_BREAKER.add(name)
            logger.warning(
                "Chroma circuit open for '%s' (disk/index error); skipping queries until repair.",
                name,
            )


def is_collection_marked_unhealthy(collection_name: str) -> bool:
    name = (collection_name or "").strip()
    if not name:
        return False
    with _UNHEALTHY_LOCK:
        return name in _UNHEALTHY


def clear_unhealthy_collections() -> None:
    """Call after successful repair or full Chroma reset."""
    with _UNHEALTHY_LOCK:
        _UNHEALTHY.clear()
        _LOGGED_BREAKER.clear()
    logger.info("Chroma unhealthy collection circuit cleared")


def _embed_probe(config) -> list[list[float]]:
    from indexer.embedder import embed_texts

    model = getattr(config, "DEFAULT_EMBED_MODEL", "nomic-embed-text-v2-moe:latest")
    if ":" not in model:
        model += ":latest"
    url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    return embed_texts(["statute retrieval index probe"], model=model, ollama_url=url)


def _stored_embedding_probe(col) -> list[list[float]] | None:
    """
    Prefer a vector already stored in the collection (fast, no Ollama round-trip).
    Falls back to None so caller can use _embed_probe.
    """
    try:
        got = col.get(limit=1, include=["embeddings"])
        embs = got.get("embeddings") if isinstance(got, dict) else None
        if not embs:
            return None
        first = embs[0]
        if first is None:
            return None
        if hasattr(first, "tolist"):
            first = first.tolist()
        vec = list(first)
        return [vec] if vec else None
    except Exception as exc:
        logger.debug("stored embedding probe failed: %s", exc)
        return None


def probe_collection_queryable(config, collection_name: str) -> tuple[bool, str | None]:
    """
    True if collection exists, has rows, and a minimal vector query succeeds.
    """
    name = (collection_name or "").strip()
    if not name:
        return False, "empty name"

    try:
        client = get_chroma_client(config)
        col = client.get_collection(name)
        n = col.count()
        if n <= 0:
            return False, "empty collection"
        vec = _stored_embedding_probe(col) or _embed_probe(config)
        if not vec:
            return False, "embedding probe failed"
        k = min(1, n)
        col.query(
            query_embeddings=vec,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return True, None
    except Exception as exc:
        msg = str(exc)
        return False, msg


def assess_jurisdiction_index_health(
    config,
    jurisdiction: str,
    *,
    max_title_samples: int = 5,
) -> dict[str, Any]:
    """
    Probe catalog + a sample of title chunk collections for one jurisdiction.

    Marks unhealthy collections when corruption errors are detected.
    """
    jur = (jurisdiction or "").strip().lower()
    out: dict[str, Any] = {
        "jurisdiction": jur,
        "ok": True,
        "catalog_name": "",
        "catalog_ok": True,
        "catalog_error": None,
        "sampled_titles": [],
    }
    if not jur:
        out["ok"] = False
        return out

    cat_name = get_state_catalog_collection_name(config, jur)
    out["catalog_name"] = cat_name

    ok, err = probe_collection_queryable(config, cat_name)
    if not ok:
        out["catalog_ok"] = False
        out["catalog_error"] = err
        out["ok"] = False
        if err and is_chroma_disk_corruption_error(err):
            mark_collection_unhealthy(cat_name)

    from core.db import connect_db

    try:
        with connect_db(config) as conn:
            rows = conn.execute(
                """
                SELECT id, title_number FROM legal_titles
                WHERE LOWER(jurisdiction) = LOWER(?)
                ORDER BY title_number
                LIMIT ?
                """,
                (jur, max_title_samples),
            ).fetchall()
    except Exception as exc:
        out["sample_error"] = str(exc)
        return out

    for r in rows:
        tnum = r["title_number"] or ""
        col_name = get_scoped_chunk_collection_name(config, jur, tnum)
        tok, terr = probe_collection_queryable(config, col_name)
        entry = {"title_number": tnum, "collection_name": col_name, "ok": tok, "error": terr}
        out["sampled_titles"].append(entry)
        if not tok:
            out["ok"] = False
            if terr and is_chroma_disk_corruption_error(terr):
                mark_collection_unhealthy(col_name)

    return out


def assess_search_index_health(
    config,
    jurisdictions: list[str],
    *,
    max_title_samples: int = 5,
) -> dict[str, Any]:
    """Run assess_jurisdiction_index_health for each jurisdiction."""
    uniq = sorted({(j or "").strip().lower() for j in jurisdictions if j and str(j).strip()})
    per: list[dict[str, Any]] = []
    overall_ok = True
    repair_targets: list[str] = []

    for jur in uniq:
        row = assess_jurisdiction_index_health(config, jur, max_title_samples=max_title_samples)
        per.append(row)
        if not row.get("ok"):
            overall_ok = False
            repair_targets.append(jur)

    return {
        "ok": overall_ok,
        "jurisdictions_checked": uniq,
        "needs_repair": repair_targets,
        "details": per,
    }


def repair_search_index_for_jurisdictions(
    config,
    jurisdictions: list[str],
    *,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Re-mirror catalog + all titled chunk collections for each jurisdiction from SQL.

    Deletes/recreates collections via chroma_mirror (avoids stale HNSW state).

    progress_callback signature: fn(jur_idx: int, jur_total: int, phase: str,
    detail: str, cur: int, total: int). phase is one of: "lock", "catalog",
    "titles", "done".
    """
    from indexer.chroma_lock import chroma_write_lock
    from indexer.chroma_mirror import mirror_title_catalog_to_chroma, mirror_title_chunks_to_chroma
    from indexer.chroma_store import close_all_chroma_clients

    uniq = sorted({(j or "").strip().lower() for j in jurisdictions if j and str(j).strip()})
    if not uniq:
        return {"ok": True, "details": [], "message": "no jurisdictions supplied"}

    def _emit(jur_idx, jur_total, phase, detail, cur, total):
        if progress_callback is None:
            return
        try:
            progress_callback(jur_idx, jur_total, phase, detail, cur, total)
        except Exception as exc:  # progress is cosmetic; never break the repair
            logger.debug("repair progress_callback failed: %s", exc)

    summary: list[dict[str, Any]] = []

    _emit(0, len(uniq), "lock", "acquiring write lock", 0, 1)

    with chroma_write_lock(config):
        clear_unhealthy_collections()
        close_all_chroma_clients()

        for j_idx, jur in enumerate(uniq):
            _emit(j_idx, len(uniq), "catalog", jur, 0, 1)
            cat_res = mirror_title_catalog_to_chroma(config, jur)
            _emit(j_idx, len(uniq), "catalog", jur, 1, 1)

            from core.db import connect_db

            try:
                with connect_db(config) as conn:
                    title_rows = conn.execute(
                        """
                        SELECT id, title_number FROM legal_titles
                        WHERE LOWER(jurisdiction) = LOWER(?)
                        ORDER BY title_number
                        """,
                        (jur,),
                    ).fetchall()
            except Exception as exc:
                summary.append({
                    "jurisdiction": jur,
                    "catalog": cat_res,
                    "titles_ok": 0,
                    "titles_total": 0,
                    "error": str(exc),
                })
                _emit(j_idx, len(uniq), "titles", f"{jur} (db error)", 0, 0)
                continue

            titles_ok = 0
            title_errors: list[tuple[str, str]] = []
            total_titles = len(title_rows)
            for t_idx, tr in enumerate(title_rows):
                lt_id = tr["id"]
                tnum = str(tr["title_number"])
                _emit(j_idx, len(uniq), "titles", f"{jur} Title {tnum}", t_idx, total_titles)
                tres = mirror_title_chunks_to_chroma(config, lt_id)
                if tres.get("ok"):
                    titles_ok += 1
                else:
                    title_errors.append((tnum, tres.get("error", "?")))
            _emit(j_idx, len(uniq), "titles", jur, total_titles, total_titles)

            summary.append({
                "jurisdiction": jur,
                "catalog": cat_res,
                "titles_ok": titles_ok,
                "titles_total": total_titles,
                "title_errors": title_errors[:12],
            })

        close_all_chroma_clients()
        _emit(len(uniq), len(uniq), "done", "repair complete", len(uniq), len(uniq))

    catalog_all_ok = all(
        isinstance(s.get("catalog"), dict) and s["catalog"].get("ok")
        for s in summary
        if "catalog" in s
    )
    titles_all_ok = all(s.get("titles_total", 0) == s.get("titles_ok", 0) for s in summary)

    return {
        "ok": catalog_all_ok and titles_all_ok,
        "details": summary,
    }
