"""
Offline full ingestion: same work as Knowledge Base → Sync States Root, without Streamlit.

Run from project root:
  python -m ingestion.cli
  python -m ingestion.cli --states-root "C:\\path\\to\\States"
  python -m ingestion.cli -v

Schedule overnight (Task Scheduler) so the UI opens with SQLite + Chroma already built.
Logs: APP_DATA_DIR/logs/app.log (and stderr).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path


def _progress_logger(root_logger: logging.Logger) -> Callable[..., None]:
    """Build progress_callback for sync_states_root (console-friendly)."""

    def _cb(jur_idx: int, jur_total: int, phase: str, detail: str, cur: int, total: int) -> None:
        phase_labels = {
            "scan": "scan",
            "parse": "parse",
            "chunk_embed": "chunk/embed",
            "mirror": "mirror",
        }
        label = phase_labels.get(phase, phase)
        detail_s = (detail or "")[:120]
        root_logger.info(
            "[%d/%d] %s | %s/%s | %s",
            jur_idx + 1,
            jur_total,
            label,
            cur,
            total,
            detail_s,
        )

    return _cb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run full states-root sync (parse, chunk, embed, Chroma mirror) outside the UI.",
    )
    parser.add_argument(
        "--states-root",
        type=str,
        default=None,
        help="Override STATES_ROOT (default: config / env STATES_ROOT).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging.",
    )
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="Delete CHROMA_PERSIST_DIR before sync and rebuild collections from scratch.",
    )
    parser.add_argument(
        "--reset-sql-index",
        action="store_true",
        help="Delete SQL chunks/embeddings/index_map before sync to force full re-chunk/re-embed.",
    )
    args = parser.parse_args(argv)

    from core.config import load_config
    from core.db import init_db
    from core.logging import setup_logging
    from core.title_registry import sync_legal_titles
    from core.workspace import ensure_app_dirs
    from ingestion.state_sync import sync_states_root
    from indexer.chroma_lock import ChromaLockBusy, chroma_write_lock
    from indexer.chroma_store import close_all_chroma_clients, reset_chroma_persist_dir
    from indexer.sql_indexer import purge_all_index_data

    config = load_config()
    ensure_app_dirs(config)
    init_db(config)
    sync_legal_titles(config)

    lr = setup_logging(config.APP_DATA_DIR)
    if args.verbose:
        lr.setLevel(logging.DEBUG)
        for h in lr.handlers:
            h.setLevel(logging.DEBUG)
        logging.getLogger("ingestion").setLevel(logging.DEBUG)
        logging.getLogger("indexer").setLevel(logging.DEBUG)
        logging.getLogger("core").setLevel(logging.DEBUG)

    states_root = (args.states_root or "").strip() or getattr(config, "STATES_ROOT", "")
    if not states_root:
        lr.error("No states root: pass --states-root or set STATES_ROOT / config.")
        return 1

    lr.info("Starting offline sync | states_root=%s", states_root)
    lr.info("APP_DATA_DIR=%s | CHROMA_PERSIST_DIR=%s", config.APP_DATA_DIR, config.CHROMA_PERSIST_DIR)

    if args.reset_sql_index:
        purge_res = purge_all_index_data(config)
        lr.info(
            "Reset SQL index data: chunks=%s embeddings=%s index_map=%s",
            purge_res.get("deleted_chunks", 0),
            purge_res.get("deleted_embeddings", 0),
            purge_res.get("deleted_index_map", 0),
        )

    if args.reset_chroma:
        chroma_dir = Path(config.CHROMA_PERSIST_DIR)
        try:
            with chroma_write_lock(config):
                reset_res = reset_chroma_persist_dir(config)
        except ChromaLockBusy as exc:
            lr.error(
                "Cannot reset %s — write lock busy: %s. "
                "Stop Streamlit / other ingestion processes and retry.",
                chroma_dir, exc,
            )
            return 1
        if not reset_res.get("ok"):
            lr.error(
                "Failed to reset Chroma persist dir (%s) after %d attempt(s): %s. "
                "Stop Streamlit/other ingestion processes and retry.",
                chroma_dir, reset_res.get("attempts", 0), reset_res.get("error"),
            )
            return 1
        lr.info("Reset Chroma persist dir: %s (attempts=%d)", chroma_dir, reset_res.get("attempts", 0))

    result = {"ok": False, "error": "not started"}
    try:
        t0 = time.perf_counter()

        try:
            with chroma_write_lock(config):
                result = sync_states_root(config, states_root, progress_callback=_progress_logger(lr))
        except ChromaLockBusy as exc:
            lr.error("%s", exc)
            return 1
        elapsed = time.perf_counter() - t0
    finally:
        # Ensure buffers are flushed and local process releases file handles.
        close_all_chroma_clients()

    if not result.get("ok"):
        lr.error("Sync failed: %s", result.get("error", result))
        return 1

    lr.info(
        "Done in %.1fs | discovered=%s synced=%s ready=%s errors=%s",
        elapsed,
        result.get("discovered"),
        result.get("synced"),
        result.get("ready"),
        result.get("errors"),
    )
    for ss in result.get("state_summaries") or []:
        jur = ss.get("jurisdiction", "?")
        st = ss.get("status", "?")
        lr.info("  %s: %s | mirrored_titles=%s err=%s", jur, st, ss.get("titles_mirrored"), ss.get("error") or "")

    if result.get("errors", 0) > 0:
        lr.warning("Completed with one or more state errors; check logs.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
