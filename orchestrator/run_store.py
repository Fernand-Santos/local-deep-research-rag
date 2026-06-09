"""
Persist and query orchestrator runs, specs, and queue in SQL.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_run(config, prompt: str, spec: dict, queue_items: list[dict]) -> dict:
    """
    Persist a new run + its spec + queue items in a single transaction.
    Returns {"ok": True, "run_id": ..., "queue_count": ...} or error dict.
    """
    from core.db import connect_db

    run_id = uuid.uuid4().hex
    ts = _utc_now_iso()

    try:
        with connect_db(config) as conn:
            conn.execute(
                """INSERT INTO runs
                   (id, user_prompt, run_mode, corpus_family, status,
                    clarification_needed, stop_requested, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,0,?,?)""",
                (
                    run_id,
                    prompt,
                    spec.get("run_mode", "single"),
                    spec.get("corpus_family", ""),
                    "compiled",
                    1 if spec.get("clarification_needed") else 0,
                    ts,
                    ts,
                ),
            )

            spec_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO run_specs
                   (id, run_id, spec_json, created_at, updated_at)
                   VALUES (?,?,?,?,?)""",
                (spec_id, run_id, json.dumps(spec, default=str), ts, ts),
            )

            for qi in queue_items:
                qi_id = uuid.uuid4().hex
                conn.execute(
                    """INSERT INTO run_queue
                       (id, run_id, queue_order, corpus_family,
                        jurisdiction_or_issuer, scope_key, status,
                        created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        qi_id,
                        run_id,
                        qi.get("queue_order", 0),
                        qi.get("corpus_family", ""),
                        qi.get("jurisdiction_or_issuer", ""),
                        qi.get("scope_key", ""),
                        qi.get("status", "pending"),
                        ts,
                        ts,
                    ),
                )

            conn.commit()

        return {"ok": True, "run_id": run_id, "queue_count": len(queue_items)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_runs(config) -> list[dict]:
    from core.db import connect_db
    with connect_db(config) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(config, run_id: str) -> dict | None:
    from core.db import connect_db
    with connect_db(config) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_run_spec(config, run_id: str) -> dict | None:
    from core.db import connect_db
    with connect_db(config) as conn:
        row = conn.execute(
            "SELECT * FROM run_specs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["spec_json"])
    except (json.JSONDecodeError, KeyError):
        return None


def list_run_queue(config, run_id: str) -> list[dict]:
    from core.db import connect_db
    with connect_db(config) as conn:
        rows = conn.execute(
            "SELECT * FROM run_queue WHERE run_id = ? ORDER BY queue_order ASC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def request_stop(config, run_id: str) -> None:
    from core.db import connect_db
    ts = _utc_now_iso()
    with connect_db(config) as conn:
        conn.execute(
            "UPDATE runs SET stop_requested = 1, updated_at = ? WHERE id = ?",
            (ts, run_id),
        )
        conn.commit()


def get_next_pending_queue_item(config, run_id: str) -> dict | None:
    from core.db import connect_db
    with connect_db(config) as conn:
        row = conn.execute(
            "SELECT * FROM run_queue WHERE run_id = ? AND status = 'pending' ORDER BY queue_order ASC LIMIT 1",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def update_queue_item_status(config, queue_item_id: str, status: str) -> None:
    from core.db import connect_db
    ts = _utc_now_iso()
    with connect_db(config) as conn:
        conn.execute(
            "UPDATE run_queue SET status = ?, updated_at = ? WHERE id = ?",
            (status, ts, queue_item_id),
        )
        conn.commit()


def save_queue_result(config, run_id: str, queue_item_id: str, result: dict) -> dict:
    from core.db import connect_db
    ts = _utc_now_iso()
    result_id = uuid.uuid4().hex
    try:
        with connect_db(config) as conn:
            conn.execute(
                """INSERT INTO run_results
                   (id, run_id, queue_item_id, result_json, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (result_id, run_id, queue_item_id, json.dumps(result, default=str), ts, ts),
            )
            conn.commit()
        return {"ok": True, "result_id": result_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def mark_run_status(config, run_id: str, status: str) -> None:
    from core.db import connect_db
    ts = _utc_now_iso()
    with connect_db(config) as conn:
        conn.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
            (status, ts, run_id),
        )
        conn.commit()


def summarize_run_progress(config, run_id: str) -> dict:
    from core.db import connect_db
    with connect_db(config) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM run_queue WHERE run_id = ? GROUP BY status",
            (run_id,),
        ).fetchall()
    counts = {r["status"]: r["cnt"] for r in rows}
    total = sum(counts.values())
    return {
        "total": total,
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
    }


def list_run_results(config, run_id: str) -> list[dict]:
    from core.db import connect_db
    with connect_db(config) as conn:
        rows = conn.execute(
            "SELECT * FROM run_results WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        entry = dict(r)
        try:
            entry["result"] = json.loads(entry.pop("result_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            entry["result"] = {}
        out.append(entry)
    return out


def get_ordered_run_results(config, run_id: str) -> list[dict]:
    """Return run_results joined to run_queue, ordered by queue_order ASC."""
    from core.db import connect_db
    with connect_db(config) as conn:
        rows = conn.execute(
            """SELECT rq.queue_order, rq.jurisdiction_or_issuer, rq.corpus_family,
                      rq.scope_key, rq.status AS queue_status,
                      rr.id AS result_id, rr.result_json, rr.created_at AS result_created
               FROM run_queue rq
               LEFT JOIN run_results rr ON rr.queue_item_id = rq.id AND rr.run_id = rq.run_id
               WHERE rq.run_id = ?
               ORDER BY rq.queue_order ASC""",
            (run_id,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        entry = dict(r)
        raw = entry.pop("result_json", None)
        if raw:
            try:
                entry["result"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                entry["result"] = {}
        else:
            entry["result"] = None
        out.append(entry)
    return out


def build_cumulative_run_output(config, run_id: str) -> dict:
    """Assemble a cumulative output from persisted run_results in queue order."""
    run = get_run(config, run_id)
    if not run:
        return {"ok": False, "error": "Run not found"}

    ordered = get_ordered_run_results(config, run_id)
    progress = summarize_run_progress(config, run_id)

    sections: list[dict] = []
    missing: list[dict] = []
    text_parts: list[str] = []

    for item in ordered:
        qo = item["queue_order"]
        target = item["jurisdiction_or_issuer"]
        qstatus = item["queue_status"]
        result = item.get("result")

        section: dict = {
            "queue_order": qo,
            "target": target,
            "corpus_family": item["corpus_family"],
            "queue_status": qstatus,
        }

        if result is not None:
            evidence_chunks = result.get("evidence_chunks", [])
            section["execution_status"] = "completed" if result.get("ok") else "failed"
            section["evidence_count"] = len(evidence_chunks)
            section["searched_collections"] = result.get("searched_collections", [])
            section["synthesized_notes"] = result.get("synthesized_notes", "")
            section["result"] = result

            heading = f"--- {target} ({item['corpus_family']}) ---"
            text_parts.append(heading)
            if result.get("ok") and evidence_chunks:
                text_parts.append(f"  {len(evidence_chunks)} evidence chunk(s) found.")
                if section["synthesized_notes"]:
                    text_parts.append(f"  {section['synthesized_notes']}")
            elif not result.get("ok"):
                notes = result.get("synthesized_notes", "No results.")
                text_parts.append(f"  [FAILED / NO EVIDENCE] {notes}")
            text_parts.append("")
        elif qstatus == "failed":
            section["execution_status"] = "failed"
            section["evidence_count"] = 0
            section["searched_collections"] = []
            section["synthesized_notes"] = ""
            section["result"] = None
            text_parts.append(f"--- {target} ({item['corpus_family']}) ---")
            text_parts.append("  [FAILED] No result data persisted.")
            text_parts.append("")
        elif qstatus in ("pending", "running"):
            section["execution_status"] = "pending"
            section["evidence_count"] = 0
            section["searched_collections"] = []
            section["synthesized_notes"] = ""
            section["result"] = None
            missing.append({"queue_order": qo, "target": target, "queue_status": qstatus})
            continue
        elif qstatus == "skipped":
            section["execution_status"] = "skipped"
            section["evidence_count"] = 0
            section["searched_collections"] = []
            section["synthesized_notes"] = ""
            section["result"] = None
            text_parts.append(f"--- {target} ({item['corpus_family']}) ---")
            text_parts.append("  [SKIPPED]")
            text_parts.append("")
        else:
            continue

        sections.append(section)

    return {
        "ok": True,
        "run_id": run_id,
        "run_status": run["status"],
        "completed_count": progress["completed"],
        "total_count": progress["total"],
        "assembled_sections": sections,
        "assembled_text": "\n".join(text_parts),
        "missing_queue_items": missing,
    }


def build_structured_run_report(config, run_id: str) -> dict:
    """Build a structured report from persisted run_results, delegating to report_builder."""
    run = get_run(config, run_id)
    if not run:
        return {"ok": False, "error": "Run not found"}

    ordered = get_ordered_run_results(config, run_id)
    progress = summarize_run_progress(config, run_id)

    from orchestrator.report_builder import (
        build_report_text,
        format_scope_report_section,
        save_report_file,
        summarize_result_payload,
    )

    report_sections: list[dict] = []
    missing: list[dict] = []

    for item in ordered:
        qo = item["queue_order"]
        target = item["jurisdiction_or_issuer"]
        qstatus = item["queue_status"]
        result = item.get("result")

        sec: dict = {
            "queue_order": qo,
            "target": target,
            "corpus_family": item["corpus_family"],
        }

        if result is not None:
            evidence = result.get("evidence_chunks", [])
            sec["execution_status"] = "completed" if result.get("ok") else "failed"
            sec["evidence_count"] = len(evidence)
            sec["searched_collections"] = result.get("searched_collections", [])
            sec["findings"] = summarize_result_payload(result)
            sec["result"] = result
        elif qstatus == "failed":
            sec["execution_status"] = "failed"
            sec["evidence_count"] = 0
            sec["searched_collections"] = []
            sec["findings"] = "No result data persisted."
            sec["result"] = None
        elif qstatus in ("pending", "running"):
            sec["execution_status"] = "pending"
            sec["evidence_count"] = 0
            sec["searched_collections"] = []
            sec["findings"] = ""
            sec["result"] = None
            missing.append({"queue_order": qo, "target": target, "queue_status": qstatus})
        elif qstatus == "skipped":
            sec["execution_status"] = "skipped"
            sec["evidence_count"] = 0
            sec["searched_collections"] = []
            sec["findings"] = ""
            sec["result"] = None
        else:
            continue

        report_sections.append(sec)

    report_text = build_report_text(run, report_sections, progress, missing)

    return {
        "ok": True,
        "run_id": run_id,
        "run_status": run["status"],
        "total_scopes": progress["total"],
        "completed_scopes": progress["completed"],
        "failed_scopes": progress["failed"],
        "missing_scopes": missing,
        "report_sections": report_sections,
        "report_text": report_text,
    }
