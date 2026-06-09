"""
Cross-process lock for Chroma persistence writes.

Only one writer should mutate CHROMA_PERSIST_DIR at a time (UI sync, CLI ingest,
repair job). Readers may still race corrupted segments if a writer crashes;
prefer closing clients after writes.

Uses an exclusive lock file SIBLING to the persist directory (NOT inside it),
so a "reset persist folder" workflow can safely shutil.rmtree the persist dir
while still holding the lock — the lock file itself lives one level up.
Stale locks (> stale_seconds, default 2h) are removed automatically.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STALE_SECONDS = 7200
DEFAULT_POLL_SECONDS = 0.35
DEFAULT_WAIT_TIMEOUT_SECONDS = 600


class ChromaLockBusy(RuntimeError):
    """Another process holds the Chroma write lock and wait timed out."""


def _lock_path(config) -> Path:
    """
    Lock file path. Located OUTSIDE CHROMA_PERSIST_DIR (sibling to it) so that a
    full reset (shutil.rmtree on the persist dir) does not try to delete a file
    that the current process has open — which on Windows raises WinError 32.
    """
    persist = Path(getattr(config, "CHROMA_PERSIST_DIR", "") or "")
    parent = persist.parent if str(persist) else Path.cwd()
    return parent / f".{persist.name or 'chroma'}.write.lock"


def _legacy_lock_path(config) -> Path:
    """Old location (inside persist dir) — cleaned up opportunistically."""
    persist = Path(getattr(config, "CHROMA_PERSIST_DIR", "") or "")
    return persist / ".chroma_write.lock"


def _remove_stale_lock(path: Path, stale_seconds: int) -> None:
    try:
        if not path.is_file():
            return
        age = time.time() - path.stat().st_mtime
        if age > stale_seconds:
            path.unlink(missing_ok=True)
            logger.warning(
                "Removed stale Chroma write lock (%s, age=%.0fs)",
                path,
                age,
            )
    except OSError as exc:
        logger.debug("Stale lock check failed for %s: %s", path, exc)


@contextmanager
def chroma_write_lock(
    config,
    *,
    blocking: bool = True,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
) -> Generator[int | None, None, None]:
    """
    Acquire exclusive write lock for Chroma persist dir.

    Yields file descriptor int when acquired (Linux/macOS); None on Windows after create-only lock file pattern.

    On Windows we rely on exclusive create + delete without holding fd across unlink quirks.
    """
    persist = Path(getattr(config, "CHROMA_PERSIST_DIR", "") or "")
    persist.mkdir(parents=True, exist_ok=True)
    path = _lock_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Best-effort cleanup of any legacy lock left inside the persist dir.
    legacy = _legacy_lock_path(config)
    if legacy != path:
        try:
            if legacy.is_file():
                age = time.time() - legacy.stat().st_mtime
                if age > stale_seconds:
                    legacy.unlink(missing_ok=True)
                    logger.info("Removed legacy in-dir Chroma lock %s", legacy)
        except OSError:
            pass

    deadline = time.time() + wait_timeout_seconds if blocking else time.time()
    fd: int | None = None

    while True:
        _remove_stale_lock(path, stale_seconds)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8", errors="replace"))
            logger.debug("Acquired Chroma write lock %s", path)
            break
        except FileExistsError:
            if not blocking:
                raise ChromaLockBusy(
                    f"Chroma write lock already held ({path}). Close other ingestion apps or wait."
                ) from None
            if time.time() >= deadline:
                raise ChromaLockBusy(
                    f"Timed out waiting for Chroma write lock ({path}). "
                    "Stop parallel Streamlit/CLI runs or delete stale lock if no process is indexing."
                ) from None
            time.sleep(poll_seconds)

    try:
        yield fd
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except OSError:
            pass
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not remove Chroma lock file %s: %s", path, exc)
