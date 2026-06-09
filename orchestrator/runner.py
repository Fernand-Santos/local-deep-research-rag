"""
Sequential queue runner — processes pending queue items one at a time.

Stateless between calls: reads state from SQL, executes step(s), writes results back.
"""
from __future__ import annotations

from orchestrator.run_store import (
    get_next_pending_queue_item,
    get_run,
    get_run_spec,
    mark_run_status,
    save_queue_result,
    summarize_run_progress,
    update_queue_item_status,
)


def run_steps_until_pause(config, run_id: str, max_steps: int = 5) -> dict:
    """
    Execute up to *max_steps* pending queue items sequentially.

    Stops when:
    - max_steps reached
    - run completes (no pending items)
    - stop_requested becomes true
    - a step raises an unrecoverable exception
    """
    run = get_run(config, run_id)
    if not run:
        return _batch_result(run_id, "not_found", 0, 0, [], summarize_run_progress(config, run_id))

    if run["stop_requested"]:
        if run["status"] not in ("stopped", "completed"):
            mark_run_status(config, run_id, "stopped")
        return _batch_result(run_id, "stopped", 0, 0, [], summarize_run_progress(config, run_id))

    if run["status"] in ("completed", "stopped", "failed"):
        return _batch_result(run_id, run["status"], 0, 0, [], summarize_run_progress(config, run_id))

    step_results: list[dict] = []
    steps_attempted = 0
    steps_completed = 0
    batch_status = "completed"

    for _ in range(max_steps):
        # Re-check stop flag from SQL each iteration.
        fresh = get_run(config, run_id)
        if fresh and fresh["stop_requested"]:
            if fresh["status"] not in ("stopped", "completed"):
                mark_run_status(config, run_id, "stopped")
            batch_status = "stopped"
            break

        result = run_next_queue_item(config, run_id)
        steps_attempted += 1

        es = result.get("execution_status", "")
        step_results.append({
            "queue_target": result.get("queue_target", ""),
            "execution_status": es,
            "ok": result.get("ok", False),
            "evidence_count": result.get("evidence_count", 0),
        })

        if es == "completed":
            steps_completed += 1
        elif es == "all_done":
            batch_status = "completed"
            break
        elif es == "stopped":
            batch_status = "stopped"
            break
        elif es == "failed":
            batch_status = "paused_on_failure"
            break
        elif es == "not_found":
            batch_status = "not_found"
            break

    else:
        # max_steps reached without another break
        batch_status = "max_steps_reached"

    progress = summarize_run_progress(config, run_id)
    final_run = get_run(config, run_id)
    return _batch_result(
        run_id, batch_status, steps_attempted, steps_completed,
        step_results, progress,
        final_run_status=final_run["status"] if final_run else "unknown",
        last_qi_id=step_results[-1].get("queue_target", "") if step_results else "",
    )


def _batch_result(
    run_id: str, batch_status: str,
    steps_attempted: int, steps_completed: int,
    step_results: list[dict], progress: dict,
    final_run_status: str = "", last_qi_id: str = "",
) -> dict:
    return {
        "run_id": run_id,
        "batch_status": batch_status,
        "steps_attempted": steps_attempted,
        "steps_completed": steps_completed,
        "final_run_status": final_run_status or batch_status,
        "last_queue_target": last_qi_id,
        "progress": progress,
        "step_results": step_results,
    }


def run_next_queue_item(config, run_id: str) -> dict:
    """
    Execute exactly one pending queue item for the given run.

    Returns a transparent dict with execution outcome and progress.
    """
    run = get_run(config, run_id)
    if not run:
        return {"ok": False, "run_id": run_id, "execution_status": "not_found", "error": "Run not found"}

    # -- stop gate --
    if run["stop_requested"]:
        if run["status"] not in ("stopped", "completed"):
            mark_run_status(config, run_id, "stopped")
        return {
            "ok": False,
            "run_id": run_id,
            "execution_status": "stopped",
            "progress": summarize_run_progress(config, run_id),
        }

    # -- next item --
    qi = get_next_pending_queue_item(config, run_id)
    if qi is None:
        if run["status"] != "completed":
            mark_run_status(config, run_id, "completed")
        return {
            "ok": True,
            "run_id": run_id,
            "execution_status": "all_done",
            "progress": summarize_run_progress(config, run_id),
        }

    qi_id = qi["id"]
    corpus_family = qi["corpus_family"]
    jur = qi["jurisdiction_or_issuer"]

    # Resolve concepts and query from the persisted RunSpec.
    spec = get_run_spec(config, run_id) or {}
    concepts = spec.get("concepts", [])
    query = spec.get("original_prompt", run.get("user_prompt", ""))

    # -- mark running --
    mark_run_status(config, run_id, "running")
    update_queue_item_status(config, qi_id, "running")

    try:
        from retrieval.scoped_retriever import run_single_scope_query

        result = run_single_scope_query(
            config,
            corpus_family=corpus_family,
            jurisdiction_or_issuer=jur,
            concepts=concepts,
            query=query,
        )

        # Persist result.
        save_queue_result(config, run_id, qi_id, result)
        update_queue_item_status(config, qi_id, "completed")

    except Exception as exc:
        update_queue_item_status(config, qi_id, "failed")
        save_queue_result(config, run_id, qi_id, {"ok": False, "error": str(exc)})

        progress = summarize_run_progress(config, run_id)
        if progress["pending"] == 0 and progress["running"] == 0:
            mark_run_status(config, run_id, "completed")
        return {
            "ok": False,
            "run_id": run_id,
            "queue_item_id": qi_id,
            "queue_target": jur,
            "execution_status": "failed",
            "error": str(exc),
            "progress": progress,
        }

    # -- check if run is fully done --
    progress = summarize_run_progress(config, run_id)
    if progress["pending"] == 0 and progress["running"] == 0:
        mark_run_status(config, run_id, "completed")

    return {
        "ok": result.get("ok", False),
        "run_id": run_id,
        "queue_item_id": qi_id,
        "queue_target": jur,
        "execution_status": "completed",
        "searched_collections": result.get("searched_collections", []),
        "evidence_count": len(result.get("evidence_chunks", [])),
        "progress": progress,
    }
