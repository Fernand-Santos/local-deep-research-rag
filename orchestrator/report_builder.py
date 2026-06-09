"""
Build a structured Markdown report from persisted run results.

All content is derived strictly from SQL data — no model memory, no hallucination.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def summarize_result_payload(result: dict | None) -> str:
    """Produce a concise findings block from a single-scope result dict."""
    if result is None:
        return "No result data."
    if not result.get("ok"):
        notes = result.get("synthesized_notes", "")
        return f"No evidence found. {notes}".strip()

    chunks = result.get("evidence_chunks", [])
    if not chunks:
        return result.get("synthesized_notes", "No evidence chunks returned.")

    lines: list[str] = []
    notes = result.get("synthesized_notes", "")
    if notes:
        lines.append(notes)

    for i, ch in enumerate(chunks[:5]):
        sec = ch.get("section_number", "?")
        catchline = ch.get("catchline", "")
        score = ch.get("score", "?")
        text_preview = (ch.get("chunk_text", "") or "")[:250].replace("\n", " ")
        lines.append(f"  - Section {sec} ({catchline}) [score={score}]: {text_preview}…")

    if len(chunks) > 5:
        lines.append(f"  - … and {len(chunks) - 5} more evidence chunk(s)")

    return "\n".join(lines)


def format_scope_report_section(item: dict) -> tuple[str, str]:
    """
    Format one scope into a Markdown heading + body block.

    Returns (md_section, status_tag).
    """
    target = item.get("target", item.get("jurisdiction_or_issuer", "?"))
    corpus = item.get("corpus_family", "")
    es = item.get("execution_status", "pending")
    ev_count = item.get("evidence_count", 0)
    searched = item.get("searched_collections", [])
    result = item.get("result")

    if es == "completed" and ev_count > 0:
        tag = "COMPLETED"
    elif es == "completed" and ev_count == 0:
        tag = "NO EVIDENCE"
    elif es == "failed":
        tag = "FAILED"
    elif es == "skipped":
        tag = "SKIPPED"
    elif es == "pending":
        tag = "PENDING"
    else:
        tag = es.upper()

    lines: list[str] = []
    lines.append(f"### {target} ({corpus})")
    lines.append("")
    lines.append(f"**Status:** {tag} | **Evidence chunks:** {ev_count}")
    if searched:
        lines.append(f"**Collections searched:** {', '.join(searched)}")
    lines.append("")

    if es in ("completed", "failed") and result is not None:
        findings = summarize_result_payload(result)
        lines.append("**Findings:**")
        lines.append("")
        lines.append(findings)
    elif es == "failed" and result is None:
        lines.append("*No result data persisted for this scope.*")
    elif es == "pending":
        lines.append("*This scope has not been executed yet.*")
    elif es == "skipped":
        lines.append("*This scope was skipped.*")

    lines.append("")
    return "\n".join(lines), tag


def build_report_text(
    run: dict,
    ordered_items: list[dict],
    progress: dict,
    missing: list[dict],
) -> str:
    """Assemble a full Markdown report from structured data."""
    lines: list[str] = []

    lines.append(f"# Run Report: {run.get('user_prompt', '(no prompt)')}")
    lines.append("")
    lines.append(f"- **Run ID:** `{run['id']}`")
    lines.append(f"- **Status:** {run['status']}")
    lines.append(f"- **Mode:** {run.get('run_mode', '?')}")
    lines.append(f"- **Corpus:** {run.get('corpus_family', '?')}")
    lines.append(f"- **Created:** {run.get('created_at', '?')}")
    lines.append(f"- **Generated:** {_utc_now_iso()}")
    lines.append("")

    total = progress.get("total", 0)
    completed = progress.get("completed", 0)
    failed = progress.get("failed", 0)
    pending = progress.get("pending", 0)
    lines.append(f"## Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total scopes | {total} |")
    lines.append(f"| Completed | {completed} |")
    lines.append(f"| Failed | {failed} |")
    lines.append(f"| Pending | {pending} |")
    lines.append("")

    if missing:
        lines.append(f"**Pending scopes ({len(missing)}):** " + ", ".join(m["target"] for m in missing[:20]))
        if len(missing) > 20:
            lines.append(f"… and {len(missing) - 20} more")
        lines.append("")

    lines.append("## Scope Results")
    lines.append("")

    for item in ordered_items:
        section_md, _ = format_scope_report_section(item)
        lines.append(section_md)
        lines.append("---")
        lines.append("")

    lines.append("*End of report.*")
    return "\n".join(lines)


def build_ask_report_text(
    user_prompt: str,
    task_results: list[dict],
    unavailable_scopes: list[str] | None = None,
    orchestrator_model: str | None = None,
    answer_model: str | None = None,
    embedding_model: str | None = None,
    duration_ms: int | None = None,
) -> str:
    """Assemble a KB-style structured Markdown report from Ask tab task_results.

    Mirrors the layout of `build_report_text` (Summary table + per-scope
    Findings) but also includes the answer_text synthesized by the LLM on
    the Ask page, plus the explicit title/section hints the user supplied.
    """
    unavailable = list(unavailable_scopes or [])
    total = len(task_results)
    completed = sum(1 for t in task_results if t.get("answer_status") in ("strong", "weak"))
    no_ev = sum(1 for t in task_results if t.get("answer_status") == "no_evidence")
    errors = sum(1 for t in task_results if t.get("answer_status") == "error")

    lines: list[str] = []
    lines.append(f"# Ask Report: {user_prompt or '(no prompt)'}")
    lines.append("")
    lines.append(f"- **Generated:** {_utc_now_iso()}")
    if orchestrator_model:
        lines.append(f"- **Orchestrator:** `{orchestrator_model}`")
    if answer_model:
        lines.append(f"- **Answer model:** `{answer_model}`")
    if embedding_model:
        lines.append(f"- **Embedding model:** `{embedding_model}`")
    if duration_ms is not None:
        lines.append(f"- **Total duration:** {duration_ms} ms")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total tasks | {total} |")
    lines.append(f"| Completed (strong/weak) | {completed} |")
    lines.append(f"| No evidence | {no_ev} |")
    lines.append(f"| Errors | {errors} |")
    lines.append(f"| Unavailable scopes | {len(unavailable)} |")
    lines.append("")

    if unavailable:
        lines.append(f"**Unavailable (not indexed):** {', '.join(unavailable)}")
        lines.append("")

    lines.append("## Scope Results")
    lines.append("")

    for tr in task_results:
        jur = (tr.get("jurisdiction") or "?").strip() or "?"
        corpus = tr.get("corpus_family", "")
        status = tr.get("answer_status", "unknown")
        evidence = tr.get("evidence_chunks", [])
        citations = tr.get("citations", [])
        searched = tr.get("searched_collections", [])
        searched_titles = tr.get("searched_titles", [])
        iters = tr.get("retrieval_iterations", 1)
        weak_rej = tr.get("weak_evidence_rejected_count", 0)
        title_hints = tr.get("title_hints", [])
        section_hints = tr.get("section_hints", [])

        tag_map = {"strong": "STRONG EVIDENCE", "weak": "WEAK EVIDENCE",
                   "no_evidence": "NO EVIDENCE", "error": "ERROR"}
        tag = tag_map.get(status, status.upper())

        lines.append(f"### {jur} ({corpus})")
        lines.append("")
        lines.append(f"**Status:** {tag} | **Evidence chunks:** {len(evidence)} | "
                     f"**Iterations:** {iters} | **Weak rejected:** {weak_rej}")
        if title_hints:
            lines.append(f"**Pinned titles:** {', '.join(title_hints)}")
        if section_hints:
            lines.append(f"**Pinned sections:** {', '.join(section_hints)}")
        if searched:
            lines.append(f"**Collections searched:** {', '.join(searched)}")
        if searched_titles:
            titles_str = ", ".join(
                f"Title {st.get('title_number', '?')}" for st in searched_titles[:8]
            )
            lines.append(f"**Titles searched:** {titles_str}")
        lines.append("")

        answer_text = (tr.get("answer_text") or "").strip()
        if answer_text:
            lines.append("**Answer:**")
            lines.append("")
            lines.append(answer_text)
            lines.append("")

        if citations:
            lines.append(f"**Citations ({len(citations)}):**")
            for ci in citations[:15]:
                lines.append(
                    f"- § {ci.get('section_number', '?')} "
                    f"{ci.get('catchline', '')} | "
                    f"Title {ci.get('title_number', '?')}: {ci.get('title_name', '')} | "
                    f"pp. {ci.get('page_start', '?')}\u2013{ci.get('page_end', '?')}"
                )
            lines.append("")

        if evidence:
            lines.append("**Top evidence chunks:**")
            for ev in evidence[:5]:
                sec = ev.get("section_number", "?")
                cl = ev.get("catchline", "")
                score = ev.get("score", "?")
                preview = (ev.get("chunk_text") or "")[:250].replace("\n", " ")
                lines.append(f"  - § {sec} ({cl}) [score={score}]: {preview}…")
            if len(evidence) > 5:
                lines.append(f"  - … and {len(evidence) - 5} more evidence chunk(s)")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("*End of report.*")
    return "\n".join(lines)


def save_report_file(config, run_id: str, report_text: str) -> str:
    """Save report Markdown to APP_DATA_DIR/reports/<run_id>.md. Returns the path."""
    reports_dir = Path(getattr(config, "APP_DATA_DIR")) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{run_id}.md"
    path.write_text(report_text, encoding="utf-8")
    return str(path)
