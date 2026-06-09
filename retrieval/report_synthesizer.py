"""
Third-stage report agent: merges orchestrator output + retrieval evidence + per-task
answers into one executive Markdown report (user-selectable model in the UI).

Does not use raw embedding vectors; passes chunk text, scores, retrieval_method, and
collection metadata so the model can reason over what was actually retrieved.
"""
from __future__ import annotations

import json
import logging
from typing import Generator

logger = logging.getLogger(__name__)

MAX_ORCHESTRATOR_THINKING_CHARS = 6000
MAX_LLM_RAW_CHARS = 8000
MAX_CHUNK_TEXT_PER_ITEM = 900
MAX_TOTAL_USER_PAYLOAD_CHARS = 100_000
MAX_CHUNKS_ACROSS_TASKS = 80


def _truncate(s: str, limit: int) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    return s[: limit - 20] + "\n… [truncated]"


def _compact_compiled_for_report(compiled: dict | None) -> str:
    """Serialize orchestrator output for the report model (bounded size)."""
    if not compiled:
        return "(no compiled request)"
    parts: list[str] = []
    parts.append(f"raw_query: {compiled.get('raw_query', '')}")
    refs = compiled.get("structural_refs") or {}
    if refs:
        parts.append(f"structural_refs: {json.dumps(refs)}")
    thinking = compiled.get("llm_thinking") or ""
    if thinking:
        parts.append("orchestrator_reasoning (may include model chain-of-thought):\n" + _truncate(thinking, MAX_ORCHESTRATOR_THINKING_CHARS))
    raw = compiled.get("llm_raw") or ""
    if raw:
        parts.append("orchestrator_raw_json:\n" + _truncate(raw, MAX_LLM_RAW_CHARS))
    tasks = compiled.get("tasks") or []
    slim_tasks = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        slim_tasks.append({
            "jurisdiction": t.get("jurisdiction"),
            "corpus_family": t.get("corpus_family"),
            "title_hints": t.get("title_hints"),
            "section_hints": t.get("section_hints"),
            "concepts": (t.get("concepts") or [])[:12],
            "query": _truncate(str(t.get("query") or ""), 500),
            "raw_query": _truncate(str(t.get("raw_query") or ""), 500),
        })
    parts.append("tasks:\n" + json.dumps(slim_tasks, indent=2))
    return "\n\n".join(parts)


def _dedupe_chunks(task_results: list[dict]) -> list[dict]:
    """Merge evidence across tasks by chunk_id; keep best score."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for tr in task_results:
        jur = (tr.get("jurisdiction") or "").strip()
        for ch in tr.get("evidence_chunks") or []:
            if not isinstance(ch, dict):
                continue
            cid = str(ch.get("chunk_id") or "")
            if not cid:
                cid = f"hash:{hash(ch.get('chunk_text') or '')}_{jur}_{ch.get('section_number')}"
            score = float(ch.get("score") or 0.0)
            prev = best.get(cid)
            if prev is None or score > float(prev.get("_score", 0.0)):
                copy_ch = dict(ch)
                copy_ch["_score"] = score
                copy_ch["_task_jurisdiction"] = jur
                best[cid] = copy_ch
                if cid not in order:
                    order.append(cid)
    merged = [best[c] for c in order]
    merged.sort(key=lambda x: float(x.get("_score", 0.0)), reverse=True)
    out = merged[:MAX_CHUNKS_ACROSS_TASKS]
    for ch in out:
        ch.pop("_score", None)
    return out


def _format_retrieval_digest(task_results: list[dict], deduped: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"Total deduplicated evidence chunks (top {len(deduped)}): {len(deduped)}")
    for i, ch in enumerate(deduped):
        meta = (
            f"jurisdiction={ch.get('_task_jurisdiction') or ch.get('jurisdiction', '')} | "
            f"title={ch.get('title_number', '')} {ch.get('title_name', '')} | "
            f"section={ch.get('section_number', '')} | {ch.get('catchline', '')} | "
            f"pp.{ch.get('page_start', '?')}-{ch.get('page_end', '?')} | "
            f"score={ch.get('score')} | method={ch.get('retrieval_method', '')} | "
            f"collection={ch.get('collection_name', '')}"
        )
        body = _truncate((ch.get("chunk_text") or "").strip(), MAX_CHUNK_TEXT_PER_ITEM)
        lines.append(f"[Chunk {i+1}] {meta}\n{body}")
    return "\n\n".join(lines)


def _format_per_task_answers(task_results: list[dict]) -> str:
    blocks: list[str] = []
    for tr in task_results:
        jur = (tr.get("jurisdiction") or "").strip() or "?"
        status = tr.get("answer_status", "")
        ans = _truncate(str(tr.get("answer_text") or ""), 8000)
        blocks.append(f"### Jurisdiction: {jur} (answer_status={status})\n{ans}")
    return "\n\n".join(blocks)


def build_final_report_user_content(
    user_prompt: str,
    compiled: dict | None,
    task_results: list[dict],
    unavailable_scopes: list[str] | None,
) -> str:
    """Single user message body for the report generator (size-capped)."""
    unavail = ", ".join(unavailable_scopes or []) or "(none)"
    compiled_block = _compact_compiled_for_report(compiled)
    deduped = _dedupe_chunks(task_results)
    digest = _format_retrieval_digest(task_results, deduped)
    answers = _format_per_task_answers(task_results)

    body = (
        f"# User question\n{user_prompt}\n\n"
        f"# Unavailable scopes (not indexed)\n{unavail}\n\n"
        f"# Orchestrator and task plan\n{compiled_block}\n\n"
        f"# Per-jurisdiction draft answers (retrieval / answer model)\n{answers}\n\n"
        f"# Retrieved evidence (deduplicated, ranked by score)\n"
        f"Use these excerpts as the primary ground truth. Scores and retrieval_method "
        f"reflect how each item was found (vector catalog, SQL fallback, pinned title, etc.).\n\n"
        f"{digest}"
    )
    if len(body) > MAX_TOTAL_USER_PAYLOAD_CHARS:
        body = _truncate(body, MAX_TOTAL_USER_PAYLOAD_CHARS)
    return body


def build_final_report_messages(
    user_prompt: str,
    compiled: dict | None,
    task_results: list[dict],
    unavailable_scopes: list[str] | None,
) -> list[dict]:
    system = (
        "You are a senior legal research editor. You receive the user's question, the "
        "orchestrator's task decomposition and reasoning notes, per-jurisdiction draft answers, "
        "and a deduplicated set of retrieved statutory excerpts with metadata and relevance scores.\n\n"
        "Write ONE cohesive Markdown report for the user, including:\n"
        "1. **Executive summary** — direct answer to the question where the evidence allows.\n"
        "2. **Findings by jurisdiction** — integrate draft answers with the strongest supporting excerpts.\n"
        "3. **Evidence table or bullet list** — key sections (cite title, section, pages, source file if present).\n"
        "4. **Gaps and limitations** — weak evidence, missing scopes, or conflicts between chunks.\n\n"
        "Rules: Ground every legal conclusion in the provided excerpts. If evidence is insufficient, say so. "
        "Do not invent citations or statutory text not present in the excerpts. "
        "If multiple jurisdictions disagree, note that clearly."
    )
    user_content = build_final_report_user_content(
        user_prompt, compiled, task_results, unavailable_scopes
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]


def synthesize_final_report_stream(
    *,
    model: str,
    ollama_url: str,
    user_prompt: str,
    compiled: dict | None,
    task_results: list[dict],
    unavailable_scopes: list[str] | None = None,
    temperature: float = 0.25,
    max_tokens: int = 8192,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool = False,
) -> Generator[dict, None, None]:
    """Stream the final integrated report (thinking optional, off by default)."""
    from core.llm import ollama_chat_stream

    if not task_results:
        yield {"type": "content", "text": "*No tasks completed; nothing to report.*"}
        yield {"type": "done", "text": ""}
        return

    messages = build_final_report_messages(user_prompt, compiled, task_results, unavailable_scopes)
    try:
        for chunk in ollama_chat_stream(
            model=model,
            messages=messages,
            ollama_url=ollama_url,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_p=top_p,
            top_k=top_k,
        ):
            yield chunk
    except Exception as exc:
        logger.exception("synthesize_final_report_stream failed: %s", exc)
        yield {"type": "error", "text": str(exc)[:500]}


def synthesize_final_report(
    *,
    model: str,
    ollama_url: str,
    user_prompt: str,
    compiled: dict | None,
    task_results: list[dict],
    unavailable_scopes: list[str] | None = None,
    temperature: float = 0.25,
    max_tokens: int = 8192,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool = False,
) -> dict:
    """Non-streaming final report (for batch/tests)."""
    from core.llm import ollama_chat

    if not task_results:
        return {"ok": True, "content": "", "thinking": None, "model": model, "error": None}
    messages = build_final_report_messages(user_prompt, compiled, task_results, unavailable_scopes)
    return ollama_chat(
        model=model,
        messages=messages,
        ollama_url=ollama_url,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        top_p=top_p,
        top_k=top_k,
    )
