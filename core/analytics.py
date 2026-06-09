"""
Development analytics: persisted events in SQLite for prompts, retrieval, and runs.

Used for tuning RAG quality and monitoring ingestion. Not a substitute for audit logs.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("local_deep_research")

_ANALYTICS_TABLE = "dev_analytics_events"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def record_dev_event(
    config,
    event_type: str,
    *,
    session_run_id: str | None = None,
    user_prompt: str | None = None,
    orchestrator_model: str | None = None,
    answer_model: str | None = None,
    embedding_model: str | None = None,
    corpus_family: str | None = None,
    jurisdiction: str | None = None,
    task_count: int | None = None,
    evidence_chunks_total: int | None = None,
    retrieval_iterations_max: int | None = None,
    unavailable_scopes_count: int | None = None,
    duration_ms: int | None = None,
    status: str = "ok",
    error_message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    Insert one analytics row. Returns event id.
    Also logs a single structured line for file-based review.
    """
    from core.db import connect_db

    eid = uuid.uuid4().hex
    ts = _utc_now_iso()
    extra_json = json.dumps(extra, ensure_ascii=False)[:16000] if extra else None

    row = {
        "id": eid,
        "session_run_id": session_run_id,
        "event_type": event_type,
        "created_at": ts,
        "user_prompt": (user_prompt or "")[:8000],
        "orchestrator_model": orchestrator_model,
        "answer_model": answer_model,
        "embedding_model": embedding_model,
        "corpus_family": corpus_family,
        "jurisdiction": jurisdiction,
        "task_count": task_count,
        "evidence_chunks_total": evidence_chunks_total,
        "retrieval_iterations_max": retrieval_iterations_max,
        "unavailable_scopes_count": unavailable_scopes_count,
        "duration_ms": duration_ms,
        "status": status,
        "error_message": (error_message or "")[:2000],
        "extra_json": extra_json,
    }

    try:
        with connect_db(config) as conn:
            conn.execute(
                f"""
                INSERT INTO {_ANALYTICS_TABLE} (
                    id, session_run_id, event_type, created_at, user_prompt,
                    orchestrator_model, answer_model, embedding_model,
                    corpus_family, jurisdiction, task_count, evidence_chunks_total,
                    retrieval_iterations_max, unavailable_scopes_count,
                    duration_ms, status, error_message, extra_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["id"], row["session_run_id"], row["event_type"], row["created_at"],
                    row["user_prompt"], row["orchestrator_model"], row["answer_model"],
                    row["embedding_model"], row["corpus_family"], row["jurisdiction"],
                    row["task_count"], row["evidence_chunks_total"],
                    row["retrieval_iterations_max"], row["unavailable_scopes_count"],
                    row["duration_ms"], row["status"], row["error_message"], row["extra_json"],
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("dev_analytics insert failed: %s", exc)
        return eid

    logger.info(
        "dev_analytics event=%s type=%s tasks=%s evidence_chunks=%s duration_ms=%s status=%s embed=%s",
        eid[:8],
        event_type,
        task_count,
        evidence_chunks_total,
        duration_ms,
        status,
        embedding_model or "",
    )
    return eid


def list_dev_analytics_events(config, limit: int = 200) -> list[dict[str, Any]]:
    """Recent analytics rows, newest first."""
    from core.db import connect_db

    limit = max(1, min(limit, 500))
    with connect_db(config) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM {_ANALYTICS_TABLE}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def record_states_sync_summary(
    config,
    session_run_id: str | None,
    summary: dict[str, Any],
    duration_ms: int | None = None,
) -> str:
    """Log a states_root sync result."""
    ok = bool(summary.get("ok", True)) and not summary.get("error")
    return record_dev_event(
        config,
        "states_sync",
        session_run_id=session_run_id,
        user_prompt=f"states_root={summary.get('states_root', '')}"[:8000],
        task_count=summary.get("discovered"),
        evidence_chunks_total=summary.get("synced") or summary.get("ready"),
        duration_ms=duration_ms,
        status="ok" if ok else "error",
        error_message=str(summary.get("error") or "")[:2000] or None,
        extra={
            "discovered": summary.get("discovered"),
            "synced": summary.get("synced"),
            "ready": summary.get("ready"),
            "errors": summary.get("errors"),
        },
    )


def record_document_index(
    config,
    *,
    document_id: str,
    chunks_created: int,
    embed_model: str,
    duration_ms: int | None,
    ok: bool,
    error: str | None = None,
) -> str:
    """Optional: chunk+embed completion per document (can be high volume)."""
    return record_dev_event(
        config,
        "document_index",
        user_prompt=f"document_id={document_id}",
        embedding_model=embed_model,
        task_count=chunks_created,
        duration_ms=duration_ms,
        status="ok" if ok else "error",
        error_message=error,
        extra={"document_id": document_id},
    )


def record_ask_session(
    config,
    *,
    session_run_id: str | None,
    user_prompt: str,
    orchestrator_model: str | None,
    answer_model: str | None,
    embedding_model: str | None,
    compiled_request: dict | None,
    task_results: list[dict],
    unavailable_scopes: list[str],
    duration_ms: int,
    status: str = "ok",
    error_message: str | None = None,
    report_model: str | None = None,
    final_report_chars: int = 0,
) -> str:
    """Aggregate metrics after an Ask run (streaming or batch)."""
    evidence_total = sum(len(t.get("evidence_chunks") or []) for t in task_results)
    iters = 0
    if task_results:
        iters = max(int(t.get("retrieval_iterations") or 1) for t in task_results)
    corp = None
    if compiled_request and compiled_request.get("tasks"):
        corp = compiled_request["tasks"][0].get("corpus_family")
    jur = None
    if len(task_results) == 1:
        jur = (task_results[0].get("jurisdiction") or "") or None
    return record_dev_event(
        config,
        "ask",
        session_run_id=session_run_id,
        user_prompt=user_prompt,
        orchestrator_model=orchestrator_model,
        answer_model=answer_model,
        embedding_model=embedding_model,
        corpus_family=corp,
        jurisdiction=jur,
        task_count=len(task_results),
        evidence_chunks_total=evidence_total,
        retrieval_iterations_max=iters,
        unavailable_scopes_count=len(unavailable_scopes or []),
        duration_ms=duration_ms,
        status=status,
        error_message=error_message,
        extra={
            "unavailable_scopes": unavailable_scopes[:50],
            "task_jurisdictions": [t.get("jurisdiction") for t in task_results[:50]],
            "report_model": report_model,
            "final_report_chars": final_report_chars,
        },
    )
