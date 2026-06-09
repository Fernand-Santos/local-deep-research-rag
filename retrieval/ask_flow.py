"""
User-facing Ask flow: discovers indexed scopes, gates queries, runs retrieval,
and synthesizes answers using a selected LLM.

Supports both blocking (ollama_chat) and streaming (ollama_chat_stream) modes.
All state is read from SQL/Chroma — no model memory.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Generator

logger = logging.getLogger(__name__)

_EMBED_KEYWORDS = {"embed", "nomic", "moe", "mxbai", "bge", "e5", "gte", "minilm", "all-minilm"}
_CHAT_KEYWORDS = {"llama", "mistral", "gemma", "phi", "qwen", "deepseek", "codellama", "wizard", "vicuna", "yi", "solar", "command", "orca", "neural", "zephyr", "dolphin", "nous", "tinyllama", "starcoder", "granite"}

MAX_EVIDENCE_FOR_ANSWER = 8
MAX_EVIDENCE_CHARS = 6000


def classify_ollama_models(model_names: list[str]) -> dict:
    """Split Ollama model names into likely chat and embedding candidates."""
    chat: list[str] = []
    embed: list[str] = []
    for m in model_names:
        low = m.lower()
        if any(kw in low for kw in _EMBED_KEYWORDS):
            embed.append(m)
        elif any(kw in low for kw in _CHAT_KEYWORDS):
            chat.append(m)
        elif ":" in m:
            chat.append(m)
        else:
            chat.append(m)
    return {"chat": sorted(set(chat)), "embed": sorted(set(embed))}


def get_indexed_scopes(config) -> dict:
    from core.db import connect_db
    from indexer.chroma_mirror import list_chroma_collections

    result: dict = {
        "jurisdictions": [],
        "titles": [],
        "chroma_collections": [],
        "indexed_embed_model": None,
        "catalog_jurisdictions": [],
    }

    try:
        with connect_db(config) as conn:
            jurs = [r[0] for r in conn.execute(
                "SELECT DISTINCT jurisdiction FROM legal_titles ORDER BY jurisdiction"
            ).fetchall() if r[0]]
            result["jurisdictions"] = jurs

            titles = [dict(r) for r in conn.execute(
                "SELECT id, jurisdiction, title_number, title_name, collection_key FROM legal_titles ORDER BY jurisdiction, title_number"
            ).fetchall()]
            result["titles"] = titles

            emb_row = conn.execute("SELECT DISTINCT model FROM embeddings LIMIT 1").fetchone()
            if emb_row:
                result["indexed_embed_model"] = emb_row[0]
    except Exception as exc:
        logger.error("get_indexed_scopes: SQL query failed: %s", exc)

    try:
        cols = list_chroma_collections(config)
        result["chroma_collections"] = cols
        catalog_prefix = getattr(config, "CATALOG_COLLECTION_PREFIX", "catalog_")
        for c in cols:
            name = c.get("name", "")
            if name.startswith(catalog_prefix) and c.get("count", 0) > 0:
                result["catalog_jurisdictions"].append(name[len(catalog_prefix):])
    except Exception as exc:
        logger.error("get_indexed_scopes: Chroma listing failed: %s", exc)

    return result


def get_searchable_jurisdictions(indexed_scopes: dict, config=None) -> list[str]:
    catalog_jurs = set(indexed_scopes.get("catalog_jurisdictions", []))
    title_prefix = "title_"
    chunk_jurs: set[str] = set()
    for c in indexed_scopes.get("chroma_collections", []):
        name = c.get("name", "")
        if name.startswith(title_prefix) and c.get("count", 0) > 0:
            parts = name[len(title_prefix):].rsplit("_", 1)
            if parts:
                chunk_jurs.add(parts[0])

    chroma_ready = catalog_jurs & chunk_jurs

    if config is not None:
        try:
            from core.db import connect_db
            with connect_db(config) as conn:
                rows = conn.execute(
                    "SELECT jurisdiction FROM jurisdiction_ingestion_status WHERE status = 'ready'"
                ).fetchall()
                if rows:
                    db_ready = {r[0] for r in rows}
                    if chroma_ready:
                        return sorted(chroma_ready & db_ready)
                    logger.info("get_searchable_jurisdictions: Chroma empty, falling back to DB-ready: %s", db_ready)
                    return sorted(db_ready)
        except Exception as exc:
            logger.error("get_searchable_jurisdictions: DB fallback failed: %s", exc)

    return sorted(chroma_ready)


def run_gated_ask(config, query, corpus_family, jurisdiction_or_issuer, concepts, indexed_scopes):
    from retrieval.scoped_retriever import run_single_scope_query

    searchable = get_searchable_jurisdictions(indexed_scopes, config=config)
    jur_lower = (jurisdiction_or_issuer or "").strip().lower()

    if jur_lower and jur_lower not in searchable:
        return {"ok": False, "status": "unavailable", "query": query, "corpus_family": corpus_family, "jurisdiction_or_issuer": jurisdiction_or_issuer,
                "message": f"'{jurisdiction_or_issuer}' is not currently indexed. Indexed: {', '.join(searchable) or 'none'}.",
                "evidence_chunks": [], "candidate_targets": [], "searched_collections": []}

    if not jur_lower and not searchable:
        return {"ok": False, "status": "no_index", "query": query,
                "message": "No scopes are currently indexed.", "evidence_chunks": [], "candidate_targets": [], "searched_collections": []}

    target_jur = jurisdiction_or_issuer.strip() if jur_lower else (searchable[0] if searchable else "")

    result = run_single_scope_query(config, corpus_family=corpus_family or "state_requirements",
                                    jurisdiction_or_issuer=target_jur, concepts=concepts, query=query)

    evidence = result.get("evidence_chunks", [])
    answer_status = result.get("answer_status", "no_evidence")

    return {
        "ok": result.get("ok", False), "status": answer_status, "answer_status": answer_status,
        "query": query, "corpus_family": corpus_family, "jurisdiction_or_issuer": target_jur,
        "message": result.get("synthesized_notes", ""), "evidence_chunks": evidence,
        "candidate_targets": result.get("candidate_targets", []),
        "searched_collections": result.get("searched_collections", []),
        "retrieval_iterations": result.get("retrieval_iterations", 1),
        "searched_titles": result.get("searched_titles", []),
        "weak_evidence_rejected_count": result.get("weak_evidence_rejected_count", 0),
    }


# ------------------------------------------------------------------
# Evidence context builder
# ------------------------------------------------------------------

def _build_evidence_context(evidence: list[dict]) -> str:
    parts: list[str] = []
    total_chars = 0
    for i, ev in enumerate(evidence[:MAX_EVIDENCE_FOR_ANSWER]):
        text = (ev.get("chunk_text") or "")[:1500]
        sec = ev.get("section_number", "?")
        catchline = ev.get("catchline", "")
        title = ev.get("title_number", "?")
        title_name = ev.get("title_name", "")
        pages = f"pp. {ev.get('page_start', '?')}-{ev.get('page_end', '?')}"

        block = f"[Evidence {i+1}] Title {title} ({title_name}), Section {sec} ({catchline}), {pages}:\n{text}"
        if total_chars + len(block) > MAX_EVIDENCE_CHARS:
            break
        parts.append(block)
        total_chars += len(block)
    return "\n\n".join(parts)


def _build_answer_messages(query: str, evidence: list[dict], jurisdiction: str, answer_status: str) -> list[dict]:
    """Build the messages list for the answer synthesis call."""
    context = _build_evidence_context(evidence)
    confidence_note = ""
    if answer_status == "weak":
        confidence_note = " Note: the retrieved evidence is weak. Be cautious and indicate uncertainty."

    system = (
        "You are a legal research assistant. Answer the user's question using ONLY the provided evidence. "
        "Cite specific sections and page numbers. If the evidence is insufficient, say so clearly. "
        "Do not invent information not present in the evidence." + confidence_note
    )
    user_msg = f"Question: {query}\n\nJurisdiction: {jurisdiction}\n\nEvidence:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]


def _build_citations(evidence: list[dict]) -> list[dict]:
    citations = []
    for ev in evidence[:MAX_EVIDENCE_FOR_ANSWER]:
        citations.append({
            "section_number": ev.get("section_number", ""),
            "catchline": ev.get("catchline", ""),
            "title_number": ev.get("title_number", ""),
            "title_name": ev.get("title_name", ""),
            "page_start": ev.get("page_start"),
            "page_end": ev.get("page_end"),
            "source_path": ev.get("source_path", ""),
            "score": ev.get("score"),
        })
    return citations


# ------------------------------------------------------------------
# Non-streaming answer synthesis (backend / batch use)
# ------------------------------------------------------------------

def synthesize_answer(
    query: str,
    evidence: list[dict],
    jurisdiction: str,
    answer_model: str,
    ollama_url: str,
    answer_status: str = "strong",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    from core.llm import ollama_chat

    if not evidence:
        return {"answer_text": "No evidence was found for this query. The requested scope may not be indexed.",
                "thinking": None, "answer_status": "no_evidence", "citations": [], "model": answer_model}

    messages = _build_answer_messages(query, evidence, jurisdiction, answer_status)
    citations = _build_citations(evidence)

    result = ollama_chat(model=answer_model, messages=messages, ollama_url=ollama_url,
                         temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)

    if result.get("ok"):
        return {"answer_text": result["content"], "thinking": result.get("thinking"),
                "answer_status": answer_status, "citations": citations, "model": answer_model}
    else:
        return {"answer_text": f"Answer model error: {result.get('error', 'unknown')}",
                "thinking": None, "answer_status": "error", "citations": citations, "model": answer_model}


# ------------------------------------------------------------------
# Streaming answer synthesis (for live UI rendering)
# ------------------------------------------------------------------

def synthesize_answer_stream(
    query: str,
    evidence: list[dict],
    jurisdiction: str,
    answer_model: str,
    ollama_url: str,
    answer_status: str = "strong",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Generator[dict, None, None]:
    """
    Streaming version: yields {"type": "thinking"|"content"|"done"|"error", "text": str}
    plus a final {"type": "meta", "citations": [...], "answer_status": str} at the end.
    """
    from core.llm import ollama_chat_stream

    if not evidence:
        yield {"type": "content", "text": "No evidence was found for this query."}
        yield {"type": "meta", "citations": [], "answer_status": "no_evidence"}
        yield {"type": "done", "text": ""}
        return

    messages = _build_answer_messages(query, evidence, jurisdiction, answer_status)
    citations = _build_citations(evidence)

    for chunk in ollama_chat_stream(model=answer_model, messages=messages, ollama_url=ollama_url,
                                    temperature=temperature, max_tokens=max_tokens,
                                    top_p=top_p, top_k=top_k):
        yield chunk

    yield {"type": "meta", "citations": citations, "answer_status": answer_status}


# ------------------------------------------------------------------
# Multi-task Ask execution (non-streaming, used by batch/Advanced)
# ------------------------------------------------------------------

def run_ask_tasks(
    config,
    compiled_request: dict,
    answer_model: str,
    embedding_model: str | None = None,
    max_iterations: int = 3,
    session_run_id: str | None = None,
    orchestrator_model: str | None = None,
    user_prompt_for_log: str | None = None,
) -> dict:
    from retrieval.scoped_retriever import run_single_scope_query

    t0 = time.perf_counter()
    ollama_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    tasks = compiled_request.get("tasks", [])
    unavailable = compiled_request.get("unavailable_scopes", [])

    task_results: list[dict] = []
    execution_log: list[str] = []
    cumulative_parts: list[str] = []

    for i, task in enumerate(tasks):
        jur = task.get("jurisdiction", "")
        concepts = task.get("concepts", [])
        query = task.get("query", "")
        raw_query = task.get("raw_query") or query
        title_hints = task.get("title_hints", [])
        section_hints = task.get("section_hints", [])
        corpus = task.get("corpus_family", "state_requirements")

        # Prefer the raw user prompt for retrieval — it preserves explicit
        # Title/Section phrasing and keywords the LLM might rewrite away.
        retrieval_query = raw_query if raw_query else query

        log_hints = f" hints={title_hints}" if title_hints else ""
        execution_log.append(f"Task {i+1}: {jur} — {retrieval_query[:80]}{log_hints}")

        retrieval_result = run_single_scope_query(
            config, corpus_family=corpus, jurisdiction_or_issuer=jur,
            concepts=concepts, query=retrieval_query,
            title_hints=title_hints, section_hints=section_hints,
        )

        evidence = retrieval_result.get("evidence_chunks", [])
        ret_status = retrieval_result.get("answer_status", "no_evidence")
        execution_log.append(f"  Retrieval: {len(evidence)} chunks, status={ret_status}")

        answer = synthesize_answer(
            query=query, evidence=evidence, jurisdiction=jur,
            answer_model=answer_model, ollama_url=ollama_url, answer_status=ret_status,
        )
        execution_log.append(f"  Answer: status={answer['answer_status']}")

        task_result = {
            "task_index": i, "jurisdiction": jur, "query": query, "corpus_family": corpus,
            "concepts": concepts, "answer_text": answer["answer_text"],
            "answer_status": answer["answer_status"], "thinking": answer.get("thinking"),
            "citations": answer.get("citations", []), "answer_model": answer["model"],
            "evidence_chunks": evidence,
            "searched_collections": retrieval_result.get("searched_collections", []),
            "searched_titles": retrieval_result.get("searched_titles", []),
            "candidate_targets": retrieval_result.get("candidate_targets", []),
            "retrieval_iterations": retrieval_result.get("retrieval_iterations", 1),
            "weak_evidence_rejected_count": retrieval_result.get("weak_evidence_rejected_count", 0),
        }
        task_results.append(task_result)

        header = f"## {jur.title()}" if jur else f"## Task {i+1}"
        cumulative_parts.append(f"{header}\n\n{answer['answer_text']}")

    if unavailable:
        unavail_block = "## Unavailable Scopes\n\n" + "\n".join(f"- {s}" for s in unavailable) + "\n\nThese jurisdictions are not currently indexed."
        cumulative_parts.append(unavail_block)

    duration_ms = int((time.perf_counter() - t0) * 1000)
    log_prompt = user_prompt_for_log or ""
    if not log_prompt and tasks:
        log_prompt = (tasks[0].get("query") or "") if isinstance(tasks[0], dict) else ""
    try:
        from core.analytics import record_ask_session
        record_ask_session(
            config,
            session_run_id=session_run_id,
            user_prompt=log_prompt,
            orchestrator_model=orchestrator_model,
            answer_model=answer_model,
            embedding_model=embedding_model,
            compiled_request=compiled_request,
            task_results=task_results,
            unavailable_scopes=unavailable,
            duration_ms=duration_ms,
            status="ok",
        )
    except Exception as exc:
        logger.warning("record_ask_session failed: %s", exc)

    return {
        "task_results": task_results,
        "cumulative_output": "\n\n---\n\n".join(cumulative_parts),
        "unavailable_scopes": unavailable,
        "execution_log": execution_log,
    }
