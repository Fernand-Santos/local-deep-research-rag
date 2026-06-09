"""
Chroma client and deterministic collection naming.

Uses a module-level singleton so every call reuses the same PersistentClient
instance (chromadb 1.x does not support multiple clients on the same path).
"""
from __future__ import annotations

import gc
import logging
import os
import re
import shutil
import stat
import threading
import time
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()
_client_cache: dict[str, chromadb.ClientAPI] = {}

# Cooldown so we don't spam identical 30-line tracebacks on every retrieval call
# while the in-process Rust binding is poisoned. Keyed by (persist_dir, error).
_LAST_LOG_LOCK = threading.Lock()
_LAST_LOG: dict[tuple[str, str], float] = {}
_LOG_COOLDOWN_SECONDS = 60.0


def _should_log_full_trace(persist_dir: str, error_msg: str) -> bool:
    key = (persist_dir, error_msg[:240])
    now = time.time()
    with _LAST_LOG_LOCK:
        last = _LAST_LOG.get(key, 0.0)
        if (now - last) < _LOG_COOLDOWN_SECONDS:
            return False
        _LAST_LOG[key] = now
    return True


class ChromaRuntimeError(RuntimeError):
    """
    Raised when chromadb's in-process Rust runtime is poisoned and cannot be
    recovered without restarting the Python process.

    Carries `actionable_message` with copy-pasteable remediation steps for UIs.
    """

    def __init__(self, message: str, *, persist_dir: str = "") -> None:
        super().__init__(message)
        self.persist_dir = persist_dir
        self.actionable_message = (
            "Chroma's in-process Rust binding is poisoned in this Python "
            "process (RustBindingsAPI partially initialized). Your on-disk "
            "data is intact — only the running process is broken.\n"
            "  1. Stop this Streamlit process (Ctrl+C in its terminal).\n"
            "  2. Restart Streamlit:\n"
            "       streamlit run main.py\n"
            "  3. The new process will reattach to your existing persist "
            "directory; no re-ingestion needed.\n"
            "DO NOT run `--reset-chroma` for this error — that erases data.\n"
            f"\nDetails: {message}"
            + (f"\nPersist dir: {persist_dir}" if persist_dir else "")
        )


def _try_clear_chroma_shared_state() -> bool:
    """
    Best-effort: clear chromadb's process-wide SharedSystemClient cache.

    chromadb 1.x caches Systems by identifier inside SharedSystemClient. After a
    failed PersistentClient construction it can leave a partially constructed
    RustBindingsAPI in that cache, causing every later PersistentClient call on
    the same path to surface the same `'RustBindingsAPI' object has no
    attribute 'bindings'` error. Clearing the cache forces a clean rebuild.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        clear_fn = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear_fn):
            clear_fn()
            logger.info("Cleared chromadb SharedSystemClient cache")
            return True
        identifier_dict = getattr(SharedSystemClient, "_identifier_to_system", None)
        if isinstance(identifier_dict, dict):
            identifier_dict.clear()
            logger.info("Cleared chromadb SharedSystemClient._identifier_to_system")
            return True
    except Exception as exc:  # cache-clear is best effort, never raise
        logger.debug("Could not clear chromadb shared system cache: %s", exc)
    return False


def force_clear_chroma_runtime() -> dict:
    """
    Public recovery: drop our cached clients AND chromadb's shared system cache.
    Returns dict suitable for surfacing in UI: {clients_closed, shared_cleared}.
    """
    closed = close_all_chroma_clients()
    cleared = _try_clear_chroma_shared_state()
    return {"clients_closed": closed, "shared_cleared": cleared}


def verify_chroma_health(config) -> dict:
    """
    Cheap end-to-end probe of the persist dir: open a PersistentClient, call
    list_collections, and confirm we can read at least one collection's count.

    Returns:
        {
            "ok": bool,
            "persist_dir": str,
            "collections": int | None,
            "sample": {"name": str, "count": int} | None,
            "error": str | None,
            "actionable": str | None,   # set when ok=False, recovery hint
        }

    Pure read-only — never deletes or mutates anything on disk. Safe to call at
    app startup to surface poisoned-runtime issues before any retrieval call.
    """
    persist_dir = str(getattr(config, "CHROMA_PERSIST_DIR", "") or "")
    out: dict = {
        "ok": False,
        "persist_dir": persist_dir,
        "collections": None,
        "sample": None,
        "error": None,
        "actionable": None,
    }
    if not persist_dir:
        out["error"] = "CHROMA_PERSIST_DIR not configured"
        return out
    try:
        client = get_chroma_client(config)
        cols = client.list_collections()
        out["collections"] = len(cols)
        if cols:
            first = cols[0]
            try:
                out["sample"] = {"name": first.name, "count": int(first.count())}
            except Exception as exc:
                out["sample"] = {"name": getattr(first, "name", "?"), "count": -1}
                logger.debug("verify_chroma_health: sample count failed: %s", exc)
        out["ok"] = True
        return out
    except ChromaRuntimeError as exc:
        out["error"] = str(exc)
        out["actionable"] = exc.actionable_message
        return out
    except Exception as exc:
        out["error"] = str(exc)
        from indexer.chroma_health import is_chroma_runtime_poisoned_exception
        if is_chroma_runtime_poisoned_exception(exc):
            out["actionable"] = (
                "chromadb's in-process Rust binding is poisoned in this Python "
                "process. Your on-disk data is intact — just restart Streamlit."
            )
        return out


def get_chroma_client(config) -> chromadb.ClientAPI:
    persist_dir = str(getattr(config, "CHROMA_PERSIST_DIR"))
    with _client_lock:
        if persist_dir in _client_cache:
            return _client_cache[persist_dir]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Local import to avoid circular dependency at module load.
        from indexer.chroma_health import is_chroma_runtime_poisoned_exception

        try:
            client = chromadb.PersistentClient(path=persist_dir)
        except Exception as exc:
            full_trace = _should_log_full_trace(persist_dir, str(exc))
            if full_trace:
                logger.error(
                    "Failed to create Chroma PersistentClient at %s: %s",
                    persist_dir, exc, exc_info=True,
                )
            else:
                logger.warning(
                    "Chroma PersistentClient failure (suppressed traceback for "
                    "%.0fs) at %s: %s",
                    _LOG_COOLDOWN_SECONDS, persist_dir, exc,
                )
            if not is_chroma_runtime_poisoned_exception(exc):
                raise
            # Poisoned in-process runtime: try once to clear chromadb's
            # SharedSystemClient cache, then rebuild from scratch. If that
            # still fails, raise ChromaRuntimeError — caller's banner tells
            # the user to RESTART STREAMLIT (data on disk is untouched).
            if full_trace:
                logger.warning(
                    "Detected poisoned Chroma runtime; attempting in-process recovery "
                    "(SharedSystemClient cache clear) before raising."
                )
            _try_clear_chroma_shared_state()
            try:
                client = chromadb.PersistentClient(path=persist_dir)
                logger.info(
                    "Recovered Chroma PersistentClient at %s after shared cache clear",
                    persist_dir,
                )
            except Exception as exc2:
                if full_trace:
                    logger.error(
                        "Chroma runtime still poisoned after recovery attempt at %s: %s. "
                        "On-disk data is intact; restart Streamlit to recover.",
                        persist_dir, exc2,
                    )
                raise ChromaRuntimeError(str(exc2), persist_dir=persist_dir) from exc2

        _client_cache[persist_dir] = client
        logger.info("Created PersistentClient at %s", persist_dir)
        return client


def _close_single_client(client, persist_dir: str) -> None:
    """
    Best-effort close for chromadb PersistentClient.

    Chroma 1.x does not expose a stable close API across releases, so we call
    known methods if available and swallow close-time errors.
    """
    for method_name in ("close", "persist"):
        fn = getattr(client, method_name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:
                logger.debug("Chroma client %s.%s failed: %s", persist_dir, method_name, exc)

    system = getattr(client, "_system", None)
    stop_fn = getattr(system, "stop", None) if system is not None else None
    if callable(stop_fn):
        try:
            stop_fn()
        except Exception as exc:
            logger.debug("Chroma client %s system stop failed: %s", persist_dir, exc)


def close_all_chroma_clients() -> int:
    """
    Close and clear all cached PersistentClient instances.
    Returns number of cached clients that were closed.
    """
    with _client_lock:
        items = list(_client_cache.items())
        _client_cache.clear()

    for persist_dir, client in items:
        _close_single_client(client, persist_dir)

    if items:
        logger.info("Closed %d cached Chroma client(s)", len(items))
    return len(items)


def reset_chroma_persist_dir(
    config,
    *,
    retries: int = 6,
    sleep_seconds: float = 0.6,
) -> dict:
    """
    Forcefully reset CHROMA_PERSIST_DIR.

    Steps:
      1. Close & drop all cached Chroma clients (releases sqlite + HNSW handles).
      2. Run gc to drop any lingering Python references that pin OS file handles.
      3. Recursively delete the persist directory, with retries to ride out
         transient Windows file-locking races (WinError 32) after handle release.
         Read-only files are chmod'd writable before retry.
      4. Recreate an empty persist directory.

    Returns dict: {"ok": bool, "path": str, "attempts": int, "error": str | None}.
    Caller MUST hold chroma_write_lock while invoking this.
    """
    persist = Path(getattr(config, "CHROMA_PERSIST_DIR", "") or "")
    if not str(persist):
        return {"ok": False, "path": "", "attempts": 0, "error": "CHROMA_PERSIST_DIR not configured"}

    closed = close_all_chroma_clients()
    gc.collect()
    logger.info("reset_chroma_persist_dir start path=%s closed_clients=%d", persist, closed)

    def _on_rm_error(func, path, exc_info):
        # Try to make the path writable, then retry the removal once.
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            pass
        try:
            func(path)
        except OSError as exc:
            logger.debug("rmtree on_error retry failed for %s: %s", path, exc)

    last_err: str | None = None
    attempts = 0
    if persist.exists():
        for attempt in range(1, retries + 1):
            attempts = attempt
            try:
                shutil.rmtree(persist, onerror=_on_rm_error)
                last_err = None
                break
            except OSError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "rmtree(%s) attempt %d/%d failed: %s",
                    persist, attempt, retries, exc,
                )
                gc.collect()
                time.sleep(sleep_seconds * attempt)
                continue

    if persist.exists():
        return {
            "ok": False,
            "path": str(persist),
            "attempts": attempts,
            "error": last_err or "directory still present after retries",
        }

    try:
        persist.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "ok": False,
            "path": str(persist),
            "attempts": attempts,
            "error": f"mkdir failed: {exc}",
        }

    logger.info("reset_chroma_persist_dir done path=%s attempts=%d", persist, attempts)
    return {"ok": True, "path": str(persist), "attempts": attempts, "error": None}


def _sanitise(s: str) -> str:
    """Lowercase, replace non-alphanum with underscore, collapse runs."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", s.lower())).strip("_")


def get_state_catalog_collection_name(config, jurisdiction: str) -> str:
    prefix = getattr(config, "CATALOG_COLLECTION_PREFIX", "catalog_")
    return f"{prefix}{_sanitise(jurisdiction)}"


def get_scoped_chunk_collection_name(config, jurisdiction: str, title_number: str) -> str:
    prefix = getattr(config, "TITLE_COLLECTION_PREFIX", "title_")
    return f"{prefix}{_sanitise(jurisdiction)}_{_sanitise(title_number)}"
