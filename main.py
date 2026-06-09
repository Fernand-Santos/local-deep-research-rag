"""
Local Deep Research RAG — Streamlit UI.

Run:
- streamlit run main.py
- python -m app

Tabs:
  Ask            — primary orchestrated query flow with live streaming
  Knowledge Base — what is indexed and searchable
  Advanced       — operator/admin ingestion, indexing, orchestration tools
"""
from __future__ import annotations

import json as _json
import time
import uuid
from pathlib import Path

import streamlit as st

from core.config import load_config
from core.db import connect_db, init_db
from core.health import check_ollama
from core.logging import setup_logging
from core.sources import (
    add_source,
    list_documents_for_source,
    list_sections_for_document,
    list_source_files,
    list_sources,
    parse_source_file,
    scan_source,
)
from core.workspace import create_workspace, ensure_app_dirs, list_workspaces
from tracing.events import emit_event
import html as _html


# ======================================================================
# Helpers
# ======================================================================

def _init_state() -> None:
    defaults = {
        "run_id": uuid.uuid4().hex,
        "selected_workspace_id": "",
        "selected_source_id": "",
        "selected_document_id": "",
        "selected_run_id": "",
        "last_scan_summary": None,
        "last_parse_summary": None,
        "last_chunk_summary": None,
        "last_catalog_mirror": None,
        "last_title_mirror": None,
        "last_retrieval_result": None,
        "last_compiled_spec": None,
        "last_step_result": None,
        "last_batch_result": None,
        "last_report": None,
        "last_report_path": None,
        "ask_result": None,
        "orchestrator_model": None,
        "answer_model": None,
        "report_model": None,
        "embed_model": None,
        "last_sync_result": None,
        "index_health_snapshot": None,
        "index_health_checked_at": 0.0,
        "llm_temperature": 0.3,
        "llm_max_tokens": 4096,
        "llm_top_p": 0.9,
        "llm_top_k": 40,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_model_params() -> dict:
    """Return current model generation parameters from session state."""
    return {
        "temperature": st.session_state.get("llm_temperature", 0.3),
        "max_tokens": st.session_state.get("llm_max_tokens", 4096),
        "top_p": st.session_state.get("llm_top_p", 0.9),
        "top_k": st.session_state.get("llm_top_k", 40),
    }


def _ensure_index_health_snapshot(config, searchable: list[str], ttl_seconds: float = 90.0) -> dict:
    """Throttle Chroma probe queries across Streamlit reruns."""
    import time as _time

    if not searchable:
        return {"ok": True, "needs_repair": [], "details": [], "jurisdictions_checked": []}

    now = _time.time()
    snap = st.session_state.get("index_health_snapshot")
    last = float(st.session_state.get("index_health_checked_at") or 0)
    if isinstance(snap, dict) and (now - last) < ttl_seconds:
        return snap

    from indexer.chroma_health import assess_search_index_health

    snap = assess_search_index_health(config, searchable)
    st.session_state.index_health_snapshot = snap
    st.session_state.index_health_checked_at = now
    return snap


def _invalidate_index_health_snapshot() -> None:
    st.session_state.index_health_snapshot = None
    st.session_state.index_health_checked_at = 0.0
    st.session_state._indexed_scopes_cache = None
    st.session_state._indexed_scopes_at = 0.0


def _get_indexed_scopes_cached(config, ttl_seconds: float = 180.0) -> dict:
    """Cache Chroma list_collections work across Streamlit reruns (expensive)."""
    import time as _time

    now = _time.time()
    cached = st.session_state.get("_indexed_scopes_cache")
    last = float(st.session_state.get("_indexed_scopes_at") or 0)
    if isinstance(cached, dict) and (now - last) < ttl_seconds:
        return cached

    from retrieval.ask_flow import get_indexed_scopes

    data = get_indexed_scopes(config)
    st.session_state._indexed_scopes_cache = data
    st.session_state._indexed_scopes_at = now
    return data


def _flag_chroma_runtime_poisoned(detail: str = "") -> None:
    """Record that this Python process saw a poisoned chromadb Rust runtime."""
    st.session_state.chroma_runtime_poisoned = True
    if detail:
        st.session_state.chroma_runtime_poisoned_detail = detail


def _detect_runtime_poisoned_in_snapshot(snap: dict | None) -> str | None:
    """
    Walk a health snapshot looking for chromadb Rust binding init failures and
    return a short diagnostic string when one is found.
    """
    if not isinstance(snap, dict):
        return None
    from indexer.chroma_health import is_chroma_runtime_poisoned

    for det in snap.get("details", []) or []:
        for key in ("catalog_error", "sample_error"):
            err = det.get(key)
            if err and is_chroma_runtime_poisoned(err):
                return f"{det.get('jurisdiction', '?')}: {err[:240]}"
        for sub in det.get("sampled_titles", []) or []:
            err = sub.get("error")
            if err and is_chroma_runtime_poisoned(err):
                return f"{det.get('jurisdiction', '?')} {sub.get('title_number', '?')}: {err[:240]}"
    return None


def _render_runtime_poisoned_banner(config, *, where: str) -> bool:
    """
    Show a sticky, actionable banner whenever chromadb's in-process runtime is
    poisoned. Returns True when banner was rendered (caller may want to skip
    further calls into Chroma).
    """
    if not st.session_state.get("chroma_runtime_poisoned"):
        return False

    detail = st.session_state.get("chroma_runtime_poisoned_detail", "")
    persist_dir = getattr(config, "CHROMA_PERSIST_DIR", "")

    st.error(
        "**Chroma runtime is poisoned in this Streamlit process.**\n\n"
        "Symptom: `'RustBindingsAPI' object has no attribute 'bindings'` / "
        "`Could not connect to tenant default_tenant`. This is a chromadb 1.x "
        "in-process bug — **your on-disk data is intact**. Restarting Streamlit "
        "fixes it without losing anything.\n\n"
        "**Recommended steps (data-safe):**\n"
        "1. Press `Ctrl+C` in the terminal running `streamlit run main.py`.\n"
        "2. Start Streamlit again:\n"
        "   ```\n"
        "   streamlit run main.py\n"
        "   ```\n"
        "3. The next process will open the existing persist directory and "
        "resume serving queries against your already-ingested corpus.\n\n"
        "**Do NOT** click *Reset Chroma persist folder* or run "
        "`python -m ingestion.cli --reset-chroma` for this error — those wipe "
        "your existing embeddings and force a full re-ingestion. They are only "
        "for actual on-disk corruption."
    )
    st.caption(f"Persist dir: `{persist_dir}` · Triggered in: {where}")
    if detail:
        with st.expander("Last runtime error"):
            st.code(detail)

    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        if st.button(
            "Attempt in-process recovery",
            key=f"runtime_recover_{where}",
            help="Best-effort: clears chromadb's SharedSystemClient cache and our cached clients. "
            "May succeed for transient glitches; restart is the reliable fix.",
        ):
            from indexer.chroma_store import force_clear_chroma_runtime
            res = force_clear_chroma_runtime()
            st.session_state.chroma_runtime_poisoned = False
            st.session_state.chroma_runtime_poisoned_detail = ""
            _invalidate_index_health_snapshot()
            st.success(
                f"Cleared {res['clients_closed']} cached client(s); "
                f"shared cache cleared = {res['shared_cleared']}. "
                "If errors persist, restart Streamlit."
            )
            st.rerun()
    with rcol2:
        st.caption(
            "Restart-then-CLI-reset is faster and 100% reliable. "
            "In-process recovery only works for transient glitches."
        )
    return True


def _render_index_health_banner(config, searchable: list[str], *, compact: bool = False) -> None:
    """Enterprise-friendly status; avoids exposing HNSW internals."""
    snap = _ensure_index_health_snapshot(config, searchable)

    poisoned = _detect_runtime_poisoned_in_snapshot(snap)
    if poisoned:
        _flag_chroma_runtime_poisoned(poisoned)
        _render_runtime_poisoned_banner(config, where="health_banner")
        return

    if snap.get("ok"):
        if compact:
            st.caption("Search index: ready")
        else:
            st.success("Search index is healthy (catalog + sample collections verified).")
        return

    need = snap.get("needs_repair") or []
    hint = ", ".join(n.title() for n in need[:6]) if need else "one or more jurisdictions"
    if compact:
        msg = (
            f"The local vector search store needs repair ({hint}). "
            "Use **Repair search index** below."
        )
        st.warning(msg)
        return

    msg = (
        f"The local vector search store needs repair ({hint}). "
        "This can happen after an interrupted update or running two indexing sessions at once. "
        "Use **Repair search index** below."
    )
    st.error(msg)


def _run_repair_search_index(
    config,
    searchable: list[str],
    jurisdictions: list[str] | None,
    *,
    progress_callback=None,
) -> dict:
    from indexer.chroma_health import repair_search_index_for_jurisdictions

    targets = jurisdictions if jurisdictions else searchable
    res = repair_search_index_for_jurisdictions(
        config, targets, progress_callback=progress_callback,
    )
    _invalidate_index_health_snapshot()
    return res


def _repair_with_live_status(
    config,
    searchable: list[str],
    targets: list[str],
    *,
    label: str = "Repairing search index",
) -> dict:
    """
    Drive _run_repair_search_index with a Streamlit st.status panel so users see
    live phase / per-jurisdiction / per-title progress while the broken Chroma
    state is rebuilt.
    """
    targets = [t for t in (targets or []) if t]
    if not targets:
        targets = list(searchable)

    status_label = f"{label} — {len(targets)} jurisdiction(s)"
    with st.status(status_label, expanded=True) as status:
        phase_line = st.empty()
        detail_line = st.empty()
        bar = st.progress(0, text="Starting…")

        phase_labels = {
            "lock": "Acquiring write lock",
            "catalog": "Rebuilding title catalog",
            "titles": "Re-mirroring title collections",
            "done": "Finalizing",
        }

        def _cb(j_idx: int, j_total: int, phase: str, detail: str, cur: int, total: int) -> None:
            phase_line.markdown(
                f"**Step {min(j_idx + 1, j_total)}/{j_total}** — "
                f"{phase_labels.get(phase, phase).title()}"
            )
            sub = f"{detail}" if not total else f"{detail} ({cur}/{total})"
            detail_line.caption(sub)
            denom = max(j_total, 1)
            local = (cur / total) if total else 0.0
            overall = min(1.0, (j_idx + local) / denom)
            bar.progress(overall, text=f"{int(overall * 100)}% — {phase_labels.get(phase, phase)}")

        try:
            rep = _run_repair_search_index(
                config, searchable, targets, progress_callback=_cb,
            )
        except Exception as exc:
            status.update(label=f"{label} failed", state="error", expanded=True)
            from indexer.chroma_health import is_chroma_runtime_poisoned_exception
            if is_chroma_runtime_poisoned_exception(exc):
                _flag_chroma_runtime_poisoned(str(exc))
                _render_runtime_poisoned_banner(config, where="repair")
            else:
                st.error(f"Repair raised: {exc}")
            logger.error("repair_search_index error: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

        bar.progress(1.0, text="100% — done")
        if rep.get("ok"):
            status.update(label=f"{label} — complete", state="complete", expanded=False)
            st.success("Search index rebuilt successfully.")
        else:
            status.update(label=f"{label} — finished with issues", state="error", expanded=True)
            st.warning("Repair finished with one or more failures (see details).")
            with st.expander("Repair details"):
                st.json(rep)
        return rep


def _reset_chroma_persist_dir_with_status(config) -> dict:
    """
    Drive a full CHROMA_PERSIST_DIR reset with a live st.status panel so the
    user can see each step (lock → close clients → delete → recreate → clear
    circuit) progress, instead of a blocking spinner that can fail with an
    opaque WinError 32.
    """
    from indexer.chroma_health import clear_unhealthy_collections
    from indexer.chroma_lock import ChromaLockBusy, chroma_write_lock
    from indexer.chroma_store import close_all_chroma_clients, reset_chroma_persist_dir

    chroma_dir = Path(getattr(config, "CHROMA_PERSIST_DIR", ""))
    steps = [
        "Acquire exclusive write lock",
        "Close cached Chroma clients",
        "Delete persist directory",
        "Recreate empty persist directory",
        "Clear unhealthy-collection circuit",
    ]
    total_steps = len(steps)

    with st.status(f"Resetting Chroma persist folder `{chroma_dir}`", expanded=True) as status:
        bar = st.progress(0, text=f"0/{total_steps} — preparing")
        log_box = st.empty()
        log_lines: list[str] = []

        def _step(idx: int, line: str) -> None:
            log_lines.append(f"- **{idx}/{total_steps}** {line}")
            log_box.markdown("\n".join(log_lines))
            bar.progress(idx / total_steps, text=f"{idx}/{total_steps} — {steps[idx - 1]}")

        try:
            _step(1, steps[0])
            with chroma_write_lock(config):
                _step(2, steps[1])
                close_all_chroma_clients()

                _step(3, steps[2])
                res = reset_chroma_persist_dir(config)

                if not res.get("ok"):
                    status.update(label="Reset failed", state="error", expanded=True)
                    detail = res.get("error", "unknown error")
                    log_lines.append(f"- ❌ Could not delete `{chroma_dir}` ({detail})")
                    log_box.markdown("\n".join(log_lines))
                    st.error(
                        "Could not remove the Chroma folder. Close any other Streamlit "
                        "or CLI sessions that may have it open, then retry. "
                        f"Detail: {detail}"
                    )
                    return res

                _step(4, steps[3])
                # reset_chroma_persist_dir already recreated the dir; this confirms.

                _step(5, steps[4])
                clear_unhealthy_collections()
                _invalidate_index_health_snapshot()
        except ChromaLockBusy as exc:
            status.update(label="Reset blocked — lock busy", state="error", expanded=True)
            st.error(str(exc))
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            status.update(label="Reset failed", state="error", expanded=True)
            from indexer.chroma_health import is_chroma_runtime_poisoned_exception
            if is_chroma_runtime_poisoned_exception(exc):
                _flag_chroma_runtime_poisoned(str(exc))
                _render_runtime_poisoned_banner(config, where="reset_dir")
            else:
                st.error(f"Reset raised: {exc}")
            logger.error("chroma_persist_dir_reset failed: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

        # On-disk reset succeeded; clear any in-process poison flag too.
        st.session_state.chroma_runtime_poisoned = False
        st.session_state.chroma_runtime_poisoned_detail = ""
        try:
            from indexer.chroma_store import force_clear_chroma_runtime
            force_clear_chroma_runtime()
        except Exception as exc:
            logger.debug("post-reset shared-cache clear failed: %s", exc)

        status.update(label="Chroma persist folder reset — ready to re-sync", state="complete", expanded=False)
        st.success(
            f"Removed and recreated `{chroma_dir}`. Run **Sync States Root** next "
            "to rebuild collections from your indexed SQLite chunks."
        )
        return {"ok": True, "path": str(chroma_dir)}


def _reset_chroma_collections_with_status(config) -> dict:
    """Driven analog of the legacy 'Reset Chroma' button (collections only)."""
    from indexer.chroma_lock import ChromaLockBusy, chroma_write_lock
    from indexer.chroma_mirror import delete_all_chroma_collections

    steps = [
        "Acquire exclusive write lock",
        "Delete all Chroma collections",
        "Refresh health snapshot",
    ]
    total_steps = len(steps)

    with st.status("Resetting Chroma collections", expanded=True) as status:
        bar = st.progress(0, text=f"0/{total_steps} — preparing")
        log_box = st.empty()
        log_lines: list[str] = []

        def _step(idx: int, line: str) -> None:
            log_lines.append(f"- **{idx}/{total_steps}** {line}")
            log_box.markdown("\n".join(log_lines))
            bar.progress(idx / total_steps, text=f"{idx}/{total_steps} — {steps[idx - 1]}")

        try:
            _step(1, steps[0])
            with chroma_write_lock(config):
                _step(2, steps[1])
                res = delete_all_chroma_collections(config)
            _step(3, steps[2])
            _invalidate_index_health_snapshot()
        except ChromaLockBusy as exc:
            status.update(label="Reset blocked — lock busy", state="error", expanded=True)
            st.error(str(exc))
            return {"ok": False, "error": str(exc)}

        if res.get("ok"):
            status.update(label="Chroma collections cleared", state="complete", expanded=False)
            st.success(f"Deleted {res['deleted']} collection(s). Run **Sync States Root** next.")
        else:
            status.update(label="Reset failed", state="error", expanded=True)
            st.error(res.get("error", "Reset failed"))
        return res


def _classify_models(model_names: list[str]) -> tuple[list[str], list[str]]:
    from retrieval.ask_flow import classify_ollama_models
    c = classify_ollama_models(model_names)
    return c["chat"], c["embed"]


# ======================================================================
# Sidebar — Ollama connection + 3-role model selection
# ======================================================================

def _validate_url(url: str) -> str:
    """Basic URL validation — only allow http/https schemes."""
    u = url.strip()
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        return ""
    if any(c in u for c in ("\n", "\r", "\x00", " ")):
        return ""
    return u


def _sidebar(config, logger):
    st.sidebar.markdown("### Connection")
    raw_url = st.sidebar.text_input(
        "Ollama URL", value=config.OLLAMA_BASE_URL, help="e.g. http://localhost:11434",
    ).strip() or config.OLLAMA_BASE_URL
    base_url = _validate_url(raw_url) or config.OLLAMA_BASE_URL

    status = check_ollama(base_url)
    if status["ok"]:
        st.sidebar.success("Ollama connected")
    else:
        st.sidebar.error(status.get("error") or "Not connected")
        st.sidebar.info("Start Ollama (`ollama serve`) and reload.")

    all_models = status.get("models", [])
    chat_models, embed_models = _classify_models(all_models)

    st.sidebar.markdown("### Models")

    if chat_models:
        def _pick_default(models, pref):
            return models.index(pref) if pref in models else 0

        st.sidebar.selectbox(
            "Orchestrator model",
            chat_models,
            index=_pick_default(chat_models, config.DEFAULT_CHAT_MODEL),
            help="Parses your question into tasks. Thinking-capable models (e.g. DeepSeek-R1) show reasoning logs.",
            key="orchestrator_model",
        )

        st.sidebar.selectbox(
            "Retrieval / Answer model",
            chat_models,
            index=_pick_default(chat_models, config.DEFAULT_CHAT_MODEL),
            help="Reads retrieved evidence and writes the final answer.",
            key="answer_model",
        )

        st.sidebar.selectbox(
            "Report generator model",
            chat_models,
            index=_pick_default(chat_models, config.DEFAULT_CHAT_MODEL),
            help="Third pass: merges orchestrator reasoning, retrieved chunks (scores/methods), and per-task answers into one executive report.",
            key="report_model",
        )
    elif all_models:
        st.sidebar.selectbox("Chat model", all_models, key="orchestrator_model", help="Used for orchestration and answers.")
        st.session_state.answer_model = st.session_state.orchestrator_model
        st.session_state.report_model = st.session_state.orchestrator_model
    else:
        st.sidebar.caption("No models available. Pull one: `ollama pull llama3.2`")

    if embed_models:
        default_embed = config.DEFAULT_EMBED_MODEL
        if ":" not in default_embed:
            default_embed += ":latest"
        idx = embed_models.index(default_embed) if default_embed in embed_models else 0
        st.sidebar.selectbox(
            "Embedding model",
            embed_models,
            index=idx,
            help="Powers semantic search. Must match the model used during indexing.",
            key="embed_model",
        )
    elif all_models:
        st.sidebar.selectbox("Embedding model", all_models, key="embed_model", help="No dedicated embedding models detected.")

    st.sidebar.markdown("### Generation Parameters")
    st.sidebar.slider(
        "Temperature", 0.0, 2.0, 0.3, 0.05,
        help="Higher = more creative, lower = more deterministic.",
        key="llm_temperature",
    )
    st.sidebar.number_input(
        "Max tokens", 256, 32768, 4096, step=256,
        help="Maximum tokens the model can generate per response.",
        key="llm_max_tokens",
    )
    st.sidebar.slider(
        "Top P (nucleus sampling)", 0.0, 1.0, 0.9, 0.05,
        help="Cumulative probability cutoff for token sampling.",
        key="llm_top_p",
    )
    st.sidebar.number_input(
        "Top K", 1, 200, 40,
        help="Number of highest-probability tokens to consider.",
        key="llm_top_k",
    )

    return base_url, status


# ======================================================================
# Tab: Ask (orchestrated multi-task flow with LIVE STREAMING)
# ======================================================================

def _tab_ask(config, logger):
    st.markdown(
        "Ask a question about your indexed legal corpus. "
        "The orchestrator decomposes your question into tasks, retrieves evidence, "
        "the answer model writes per-jurisdiction answers, and the **report generator** "
        "produces one executive report from orchestrator reasoning plus all retrieved chunks."
    )

    if _render_runtime_poisoned_banner(config, where="tab_ask"):
        return

    from indexer.chroma_mirror import get_jurisdiction_catalog_status
    from retrieval.ask_flow import (
        get_indexed_scopes, get_searchable_jurisdictions,
        synthesize_answer_stream,
    )
    from retrieval.report_synthesizer import synthesize_final_report_stream
    from retrieval.scoped_retriever import run_single_scope_query
    from orchestrator.ask_compiler import compile_ask_request

    indexed = _get_indexed_scopes_cached(config)
    searchable = get_searchable_jurisdictions(indexed, config=config)
    indexed_embed = indexed.get("indexed_embed_model")

    if st.session_state.embed_model and indexed_embed:
        if st.session_state.embed_model != indexed_embed:
            st.warning(
                f"Embedding mismatch: selected `{st.session_state.embed_model}` vs indexed `{indexed_embed}`. "
                f"Go to **Knowledge Base** tab → **Maintenance** → **Purge stale embeddings**, then sync."
            )

    all_jurs = indexed.get("jurisdictions", [])
    available_jurs = searchable or all_jurs

    if not searchable:
        if all_jurs:
            st.warning(
                "Jurisdictions are indexed but Chroma collections may need rebuilding. "
                "Go to **Knowledge Base** → **Sync States Root** to rebuild. "
                "You can still try queries below."
            )
        else:
            st.info("No scopes indexed yet. Go to **Knowledge Base** tab to sync, or **Advanced** to ingest documents.")

    if searchable:
        st.caption(f"Searchable: {', '.join(searchable)}")
        startup_health = st.session_state.get("_chroma_startup_health")
        if isinstance(startup_health, dict) and startup_health.get("ok"):
            st.caption("Search index: ready")
        else:
            _render_index_health_banner(config, searchable, compact=True)
            snap = _ensure_index_health_snapshot(config, searchable, ttl_seconds=300.0)
            if not snap.get("ok"):
                col_fix_a, col_fix_b = st.columns(2)
                with col_fix_a:
                    if st.button("Repair search index", key="ask_repair_index", help="Rebuilds catalog and title collections from your database."):
                        need = snap.get("needs_repair") or searchable
                        _repair_with_live_status(config, searchable, need, label="Repairing search index")
                        st.rerun()
                with col_fix_b:
                    st.caption("Tip: avoid running offline **python -m ingestion.cli** while this app syncs.")

    CORPUS_FAMILIES = ["Auto"] + [
        "state_requirements", "federal_requirements",
        "gse_requirements", "investor_requirements",
    ]

    c1, c2 = st.columns(2)
    with c1:
        ask_corpus = st.selectbox("Corpus family", CORPUS_FAMILIES, help="'Auto' = state_requirements.", key="ask_corpus")
    with c2:
        ask_jur = st.selectbox("Jurisdiction", ["(detect from question)"] + available_jurs,
                               help="Leave as 'detect' for multi-jurisdiction questions.", key="ask_jur")

    if available_jurs:
        _sel = st.session_state.get("ask_jur", "(detect from question)")
        if _sel and _sel != "(detect from question)":
            _cat = get_jurisdiction_catalog_status(config, _sel)
            if not _cat.get("ok"):
                st.warning(
                    f"**Chroma title catalog missing** for `{_sel}` (`{_cat.get('collection_name', '')}`). "
                    f"{_cat.get('error', '')} "
                    "Until fixed, retrieval uses SQL keyword fallback only (weaker). "
                    "Use **Knowledge Base** → **Sync States Root**, or **Advanced** → **Chroma Mirror** → **Mirror catalog to Chroma**."
                )

    ask_query = st.text_area(
        "Your question",
        height=90,
        placeholder="e.g. Can I send or request Pay Off Fees and Lien Release Fees in Alabama?",
        key="ask_query",
    )

    orch_model = st.session_state.get("orchestrator_model")
    ans_model = st.session_state.get("answer_model")
    report_model = (st.session_state.get("report_model") or ans_model or "").strip()
    ollama_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    mparams = _get_model_params()

    if st.button("Ask", disabled=not (ask_query.strip() and ans_model), type="primary"):
        t_ask_start = time.perf_counter()
        corpus = ask_corpus if ask_corpus != "Auto" else None
        jur = ask_jur if ask_jur != "(detect from question)" else None

        with st.spinner("Compiling tasks…"):
            compiled = compile_ask_request(
                config,
                user_query=ask_query.strip(),
                corpus_family=corpus,
                jurisdiction_or_issuer=jur,
                orchestrator_model=orch_model,
                searchable_jurisdictions=searchable,
                temperature=mparams["temperature"],
                max_tokens=mparams["max_tokens"],
                top_p=mparams["top_p"],
                top_k=mparams["top_k"],
            )

        task_count = len(compiled.get("tasks", []))
        unavail = compiled.get("unavailable_scopes", [])

        if compiled.get("llm_thinking"):
            with st.expander("Orchestrator reasoning", expanded=False):
                st.text(compiled["llm_thinking"])

        if unavail:
            st.warning("Unavailable scopes (not indexed): " + ", ".join(unavail))

        if task_count == 0 and unavail:
            st.session_state.ask_result = {
                "task_results": [], "cumulative_output": "No indexed scopes matched.",
                "unavailable_scopes": unavail, "execution_log": [], "compiled": compiled,
            }
            _dur = int((time.perf_counter() - t_ask_start) * 1000)
            try:
                from core.analytics import record_ask_session
                record_ask_session(
                    config,
                    session_run_id=st.session_state.run_id,
                    user_prompt=ask_query.strip(),
                    orchestrator_model=orch_model,
                    answer_model=ans_model,
                    embedding_model=st.session_state.get("embed_model"),
                    compiled_request=compiled,
                    task_results=[],
                    unavailable_scopes=unavail,
                    duration_ms=_dur,
                    status="ok",
                )
            except Exception:
                pass
        else:
            all_task_results: list[dict] = []
            cumulative_parts: list[str] = []

            for i, task in enumerate(compiled.get("tasks", [])):
                task_jur = task.get("jurisdiction", "")
                task_concepts = task.get("concepts", [])
                task_query = task.get("query", "")
                task_raw_query = task.get("raw_query") or ask_query.strip()
                task_title_hints = task.get("title_hints", [])
                task_section_hints = task.get("section_hints", [])
                task_corpus = task.get("corpus_family", "state_requirements")

                # Anchor retrieval on the raw user prompt so explicit
                # Title/Section/keyword references are never stripped by the
                # LLM orchestrator.
                retrieval_query = task_raw_query or task_query

                st.markdown(f"### {task_jur.title()}" if task_jur else f"### Task {i+1}")
                if task_title_hints:
                    st.caption(f"Pinned titles: {', '.join(task_title_hints)}")

                with st.spinner(f"Retrieving evidence for {task_jur.title() or 'query'}…"):
                    retrieval_result = run_single_scope_query(
                        config, corpus_family=task_corpus,
                        jurisdiction_or_issuer=task_jur, concepts=task_concepts,
                        query=retrieval_query,
                        title_hints=task_title_hints,
                        section_hints=task_section_hints,
                    )

                evidence = retrieval_result.get("evidence_chunks", [])
                ret_status = retrieval_result.get("answer_status", "no_evidence")

                if retrieval_result.get("chroma_catalog_ok") is False:
                    st.error(
                        f"Title routing catalog unavailable: `{retrieval_result.get('chroma_catalog_collection', '?')}` — "
                        f"{retrieval_result.get('chroma_catalog_error', 'unknown error')}"
                    )

                if ret_status == "strong":
                    st.success(f"Strong evidence ({len(evidence)} chunks)")
                elif ret_status == "weak":
                    st.warning(f"Weak evidence ({len(evidence)} chunks)")
                else:
                    st.error("No evidence found")

                # --- LIVE STREAMING ANSWER ---
                thinking_placeholder = st.empty()
                answer_placeholder = st.empty()
                thinking_buf = []
                content_buf = []
                citations: list[dict] = []
                final_answer_status = ret_status

                for chunk in synthesize_answer_stream(
                    query=retrieval_query, evidence=evidence, jurisdiction=task_jur,
                    answer_model=ans_model, ollama_url=ollama_url, answer_status=ret_status,
                    temperature=mparams["temperature"], max_tokens=mparams["max_tokens"],
                    top_p=mparams["top_p"], top_k=mparams["top_k"],
                ):
                    ctype = chunk.get("type", "")
                    text = chunk.get("text", "")

                    if ctype == "thinking":
                        thinking_buf.append(text)
                        _escaped = _html.escape("".join(thinking_buf))
                        thinking_placeholder.markdown(
                            f'<div style="background:#2d2d2d;color:#888;padding:8px 12px;'
                            f'border-radius:6px;font-family:monospace;font-size:0.85em;'
                            f'max-height:300px;overflow-y:auto;white-space:pre-wrap;">'
                            f'{_escaped}</div>',
                            unsafe_allow_html=True,
                        )

                    elif ctype == "content":
                        content_buf.append(text)
                        answer_placeholder.markdown("".join(content_buf))

                    elif ctype == "meta":
                        citations = chunk.get("citations", [])
                        final_answer_status = chunk.get("answer_status", ret_status)

                    elif ctype == "error":
                        answer_placeholder.error(f"Answer error: {text}")
                        final_answer_status = "error"

                full_thinking = "".join(thinking_buf) if thinking_buf else None
                full_answer = "".join(content_buf)

                if full_thinking:
                    thinking_placeholder.empty()
                    with st.expander("Model reasoning / thinking log", expanded=False):
                        _esc_think = _html.escape(full_thinking)
                        st.markdown(
                            f'<div style="background:#2d2d2d;color:#aaa;padding:10px 14px;'
                            f'border-radius:6px;font-family:monospace;font-size:0.85em;'
                            f'white-space:pre-wrap;">{_esc_think}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    thinking_placeholder.empty()

                if citations:
                    with st.expander(f"Citations ({len(citations)})", expanded=False):
                        for ci in citations:
                            st.write(
                                f"- \u00a7 {ci.get('section_number', '?')} {ci.get('catchline', '')} | "
                                f"Title {ci.get('title_number', '?')}: {ci.get('title_name', '')} | "
                                f"pp. {ci.get('page_start', '?')}\u2013{ci.get('page_end', '?')}"
                            )

                with st.expander("Searched titles & collections", expanded=False):
                    if retrieval_result.get("chroma_catalog_ok") is False:
                        st.caption(
                            f"Chroma catalog: `{retrieval_result.get('chroma_catalog_collection', '?')}` — "
                            f"{retrieval_result.get('chroma_catalog_error', 'unavailable')}"
                        )
                    s_titles = retrieval_result.get("searched_titles", [])
                    for t in s_titles:
                        st.write(f"- Title {t['title_number']}: {t['title_name']} (score={t.get('score')})")
                    searched = retrieval_result.get("searched_collections", [])
                    if searched:
                        st.caption("Collections: " + ", ".join(f"`{s}`" for s in searched))
                    st.caption(f"Iterations: {retrieval_result.get('retrieval_iterations', 1)} | "
                              f"Weak rejected: {retrieval_result.get('weak_evidence_rejected_count', 0)}")

                if evidence:
                    with st.expander(f"Evidence chunks ({len(evidence)})", expanded=False):
                        for j, ev in enumerate(evidence[:10]):
                            st.write(
                                f"**#{j+1}** \u00a7 {ev.get('section_number', '?')} {ev.get('catchline', '')} "
                                f"(score={ev.get('score')}) pp.{ev.get('page_start', '?')}\u2013{ev.get('page_end', '?')}"
                            )
                            st.text((ev.get("chunk_text") or "")[:300])
                        if len(evidence) > 10:
                            st.caption(f"Showing 10 of {len(evidence)}.")

                _render_page_previews(evidence, config=config)

                task_result = {
                    "task_index": i, "jurisdiction": task_jur, "query": task_query,
                    "raw_query": task_raw_query,
                    "title_hints": task_title_hints,
                    "section_hints": task_section_hints,
                    "answer_text": full_answer, "answer_status": final_answer_status,
                    "thinking": full_thinking, "citations": citations,
                    "evidence_chunks": evidence,
                    "searched_collections": retrieval_result.get("searched_collections", []),
                    "searched_titles": retrieval_result.get("searched_titles", []),
                    "retrieval_iterations": retrieval_result.get("retrieval_iterations", 1),
                    "weak_evidence_rejected_count": retrieval_result.get("weak_evidence_rejected_count", 0),
                    "chroma_catalog_ok": retrieval_result.get("chroma_catalog_ok"),
                    "chroma_catalog_collection": retrieval_result.get("chroma_catalog_collection"),
                    "chroma_catalog_error": retrieval_result.get("chroma_catalog_error"),
                }
                all_task_results.append(task_result)

                header = f"## {task_jur.title()}" if task_jur else f"## Task {i+1}"
                cumulative_parts.append(f"{header}\n\n{full_answer}")

                st.markdown("---")

            _dur = int((time.perf_counter() - t_ask_start) * 1000)

            from orchestrator.report_builder import build_ask_report_text
            ask_report_md = build_ask_report_text(
                user_prompt=ask_query.strip(),
                task_results=all_task_results,
                unavailable_scopes=unavail,
                orchestrator_model=orch_model,
                answer_model=ans_model,
                embedding_model=st.session_state.get("embed_model"),
                duration_ms=_dur,
            )

            _render_ask_structured_report(ask_report_md, st.session_state.run_id)

            # --- Third agent: executive report (after structured appendix) ---
            llm_final_report = ""
            llm_final_report_thinking: str | None = None
            final_report_model_used = ""
            if report_model:
                st.markdown("---")
                st.subheader("Executive report (Report generator)")
                st.caption(
                    f"Model: `{report_model}` — single pass over orchestrator output, "
                    "deduplicated evidence (scores / retrieval methods), and per-jurisdiction answers."
                )
                rep_think_ph = st.empty()
                rep_body_ph = st.empty()
                rep_think_buf: list[str] = []
                rep_body_buf: list[str] = []
                rep_max_tokens = min(8192, max(2048, int(mparams["max_tokens"] * 1.5)))
                rep_temp = min(0.6, max(0.1, float(mparams["temperature"]) * 0.85))
                final_report_model_used = report_model
                for chunk in synthesize_final_report_stream(
                    model=report_model,
                    ollama_url=ollama_url,
                    user_prompt=ask_query.strip(),
                    compiled=compiled,
                    task_results=all_task_results,
                    unavailable_scopes=unavail,
                    temperature=rep_temp,
                    max_tokens=rep_max_tokens,
                    top_p=mparams["top_p"],
                    top_k=mparams["top_k"],
                    enable_thinking=False,
                ):
                    ctype = chunk.get("type", "")
                    text = chunk.get("text", "")
                    if ctype == "thinking":
                        rep_think_buf.append(text)
                        _ert = _html.escape("".join(rep_think_buf))
                        rep_think_ph.markdown(
                            f'<div style="background:#2d2d2d;color:#888;padding:8px 12px;'
                            f'border-radius:6px;font-family:monospace;font-size:0.85em;'
                            f'max-height:220px;overflow-y:auto;white-space:pre-wrap;">{_ert}</div>',
                            unsafe_allow_html=True,
                        )
                    elif ctype == "content":
                        rep_body_buf.append(text)
                        rep_body_ph.markdown("".join(rep_body_buf))
                    elif ctype == "error":
                        rep_body_ph.error(f"Report generator error: {text}")
                        break
                llm_final_report = "".join(rep_body_buf).strip()
                llm_final_report_thinking = "".join(rep_think_buf).strip() or None
                rep_think_ph.empty()
                if llm_final_report_thinking:
                    with st.expander("Report generator thinking (if any)", expanded=False):
                        st.text(llm_final_report_thinking)
                if llm_final_report:
                    st.download_button(
                        "Download executive report",
                        data=llm_final_report.encode("utf-8"),
                        file_name=f"executive_report_{st.session_state.run_id[:12]}.md",
                        mime="text/markdown",
                        key="ask_exec_report_dl",
                    )

            st.session_state.ask_result = {
                "task_results": all_task_results,
                "cumulative_output": "\n\n---\n\n".join(cumulative_parts),
                "ask_report_md": ask_report_md,
                "llm_final_report": llm_final_report,
                "llm_final_report_thinking": llm_final_report_thinking,
                "final_report_model": final_report_model_used,
                "user_prompt": ask_query.strip(),
                "unavailable_scopes": unavail,
                "execution_log": [],
                "compiled": compiled,
            }

            try:
                from core.analytics import record_ask_session
                record_ask_session(
                    config,
                    session_run_id=st.session_state.run_id,
                    user_prompt=ask_query.strip(),
                    orchestrator_model=orch_model,
                    answer_model=ans_model,
                    embedding_model=st.session_state.get("embed_model"),
                    compiled_request=compiled,
                    task_results=all_task_results,
                    unavailable_scopes=unavail,
                    duration_ms=_dur,
                    status="ok",
                    report_model=final_report_model_used or None,
                    final_report_chars=len(llm_final_report),
                )
            except Exception:
                pass

        emit_event(
            run_id=st.session_state.run_id, agent_id="ui", event_type="ask_orchestrated",
            payload={"query": ask_query[:200], "tasks": task_count, "unavailable": len(unavail)},
            trace_dir=config.TRACE_DIR,
        )
        logger.info("ask_orchestrated tasks=%d unavail=%d", task_count, len(unavail))

    else:
        _render_ask_result_cached(config)


def _render_ask_result_cached(config):
    """Render cached results from session state (when page re-runs without new Ask)."""
    ar = st.session_state.get("ask_result")
    if not isinstance(ar, dict):
        return

    compiled = ar.get("compiled", {})
    task_results = ar.get("task_results", [])
    unavailable = ar.get("unavailable_scopes", [])

    if compiled.get("llm_thinking"):
        with st.expander("Orchestrator reasoning", expanded=False):
            st.text(compiled["llm_thinking"])

    if unavailable:
        st.warning("Unavailable scopes (not indexed): " + ", ".join(unavailable))

    if not task_results:
        if unavailable:
            st.info("No indexed scopes matched. Index more jurisdictions via Knowledge Base or Advanced tab.")
        return

    for tr in task_results:
        jur = (tr.get("jurisdiction") or "").title()
        st.markdown(f"### {jur}" if jur else f"### Task {tr.get('task_index', 0) + 1}")

        answer_status = tr.get("answer_status", "")
        if answer_status == "strong":
            st.success("Strong evidence")
        elif answer_status == "weak":
            st.warning("Weak evidence — answer may be unreliable")
        elif answer_status == "no_evidence":
            st.error("No evidence found")
        elif answer_status == "error":
            st.error("Answer model error")

        st.markdown(tr.get("answer_text", ""))

        citations = tr.get("citations", [])
        if citations:
            with st.expander(f"Citations ({len(citations)})", expanded=False):
                for ci in citations:
                    st.write(
                        f"- \u00a7 {ci.get('section_number', '?')} {ci.get('catchline', '')} | "
                        f"Title {ci.get('title_number', '?')}: {ci.get('title_name', '')} | "
                        f"pp. {ci.get('page_start', '?')}\u2013{ci.get('page_end', '?')}"
                    )

        if tr.get("thinking"):
            with st.expander("Model reasoning / thinking log", expanded=False):
                _esc_cached = _html.escape(tr["thinking"])
                st.markdown(
                    f'<div style="background:#2d2d2d;color:#aaa;padding:10px 14px;'
                    f'border-radius:6px;font-family:monospace;font-size:0.85em;'
                    f'white-space:pre-wrap;">{_esc_cached}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Searched titles & collections", expanded=False):
            s_titles = tr.get("searched_titles", [])
            for t in s_titles:
                st.write(f"- Title {t['title_number']}: {t['title_name']} (score={t.get('score')})")
            searched = tr.get("searched_collections", [])
            if searched:
                st.caption("Collections: " + ", ".join(f"`{s}`" for s in searched))
            st.caption(f"Iterations: {tr.get('retrieval_iterations', 1)} | "
                      f"Weak rejected: {tr.get('weak_evidence_rejected_count', 0)}")

        evidence = tr.get("evidence_chunks", [])
        if evidence:
            with st.expander(f"Evidence chunks ({len(evidence)})", expanded=False):
                for j, ev in enumerate(evidence[:10]):
                    st.write(
                        f"**#{j+1}** \u00a7 {ev.get('section_number', '?')} {ev.get('catchline', '')} "
                        f"(score={ev.get('score')}) pp.{ev.get('page_start', '?')}\u2013{ev.get('page_end', '?')}"
                    )
                    st.text((ev.get("chunk_text") or "")[:300])
                if len(evidence) > 10:
                    st.caption(f"Showing 10 of {len(evidence)}.")

        _render_page_previews(evidence, config=config)

        st.markdown("---")

    cum = ar.get("cumulative_output", "")
    if cum.strip():
        if len(task_results) > 1:
            with st.expander("Full cumulative output", expanded=False):
                st.markdown(cum)
        st.download_button(
            "Download Report",
            data=cum.encode("utf-8"),
            file_name="research_report.md",
            mime="text/markdown",
            key="ask_download_report_cached",
        )

    ask_report_md = ar.get("ask_report_md", "")
    if ask_report_md:
        _render_ask_structured_report(ask_report_md, st.session_state.get("run_id"))
    else:
        # Older ask_result without the report → build it on the fly.
        try:
            from orchestrator.report_builder import build_ask_report_text
            rebuilt = build_ask_report_text(
                user_prompt=ar.get("user_prompt", ""),
                task_results=task_results,
                unavailable_scopes=unavailable,
            )
            _render_ask_structured_report(rebuilt, st.session_state.get("run_id"))
        except Exception:
            pass

    llm_rep = (ar.get("llm_final_report") or "").strip()
    if llm_rep:
        _render_executive_report_block(
            llm_rep,
            ar.get("llm_final_report_thinking"),
            (ar.get("final_report_model") or "").strip(),
            st.session_state.get("run_id"),
            from_cache=True,
        )


def _render_ask_structured_report(report_md: str, run_id: str | None = None) -> None:
    """Render the KB-style structured report block on the Ask page."""
    if not report_md or not report_md.strip():
        return
    st.markdown("---")
    st.subheader("Structured Research Report")
    st.caption(
        "Same report format as the Knowledge Base page — built from the "
        "SQL/Chroma evidence and the synthesized answers for this question."
    )
    preview_limit = 8000
    st.markdown(report_md[:preview_limit])
    if len(report_md) > preview_limit:
        with st.expander("Full report", expanded=False):
            st.markdown(report_md)
    key_suffix = run_id or "ask"
    st.download_button(
        "Download Structured Report",
        data=report_md.encode("utf-8"),
        file_name=f"ask_report_{key_suffix[:12]}.md",
        mime="text/markdown",
        key=f"ask_structured_download_{key_suffix}",
    )


def _render_executive_report_block(
    report_md: str,
    thinking: str | None,
    model_name: str,
    run_id: str | None,
    *,
    from_cache: bool = False,
) -> None:
    """Render the third-agent executive Markdown (live stream already showed body when not from_cache)."""
    if not report_md or not report_md.strip():
        return
    if from_cache:
        st.markdown("---")
        st.subheader("Executive report (Report generator)")
        if model_name:
            st.caption(f"Model: `{model_name}`")
    preview_limit = 12000
    st.markdown(report_md[:preview_limit])
    if len(report_md) > preview_limit:
        with st.expander("Full executive report", expanded=False):
            st.markdown(report_md)
    if thinking:
        with st.expander("Report generator thinking (if any)", expanded=False):
            st.text(thinking)
    key_suffix = (run_id or "ask") + "_exec"
    st.download_button(
        "Download executive report",
        data=report_md.encode("utf-8"),
        file_name=f"executive_report_{(run_id or 'ask')[:12]}.md",
        mime="text/markdown",
        key=f"ask_exec_dl_cached_{key_suffix}",
    )


def _render_page_previews(evidence: list[dict], config=None):
    """Render PDF page previews for top cited chunks."""
    from core.page_preview import get_evidence_page_previews
    previews = get_evidence_page_previews(evidence, max_previews=2, config=config)
    if not previews:
        return
    with st.expander(f"Cited page previews ({len(previews)})", expanded=False):
        for p in previews:
            st.caption(f"Page {p['page_number']} — \u00a7 {p['section_number']} {p['catchline']} — `{Path(p['source_path']).name}`")
            import base64
            st.image(base64.b64decode(p["image_b64"]), width="stretch")


# ======================================================================
# Report rendering (shared between tabs)
# ======================================================================

def _render_latest_report(config):
    """Show the latest completed run report if one exists — visible from KB and Ask tabs."""
    try:
        from orchestrator.run_store import build_structured_run_report, list_runs
        runs = list_runs(config)
        completed = [r for r in runs if r["status"] in ("completed", "stopped")]
        if not completed:
            return
        latest = completed[0]
        rpt = build_structured_run_report(config, latest["id"])
        if not rpt.get("ok"):
            return
        report_text = rpt.get("report_text", "").strip()
        if not report_text:
            return

        st.subheader("Latest Research Report")
        st.caption(f"Run: {latest['user_prompt'][:80]}… | Status: {rpt['run_status']} | "
                   f"{rpt['completed_scopes']}/{rpt['total_scopes']} scopes")
        st.markdown(report_text[:5000])
        if len(report_text) > 5000:
            with st.expander("Full report", expanded=False):
                st.markdown(report_text)
        st.markdown("---")
    except Exception:
        pass


# ======================================================================
# Tab: Knowledge Base
# ======================================================================

def _tab_kb(config, logger):
    st.markdown(
        "This tab shows what is currently indexed and searchable. "
        "Use the **Advanced** tab to manage the pipeline."
    )

    _render_runtime_poisoned_banner(config, where="tab_kb")

    from retrieval.ask_flow import get_indexed_scopes, get_searchable_jurisdictions
    from ingestion.state_sync import get_all_ingestion_statuses, sync_states_root
    from indexer.chroma_mirror import (
        delete_all_chroma_collections,
        get_jurisdiction_catalog_status,
        mirror_title_catalog_to_chroma,
    )

    indexed = _get_indexed_scopes_cached(config)
    searchable = get_searchable_jurisdictions(indexed, config=config)
    titles = indexed.get("titles", [])
    cols = indexed.get("chroma_collections", [])
    embed_model = indexed.get("indexed_embed_model")
    target_embed = config.DEFAULT_EMBED_MODEL
    if ":" not in target_embed:
        target_embed += ":latest"

    st.subheader("States Root")
    states_root = getattr(config, "STATES_ROOT", "")
    st.caption(
        f"Path: `{states_root}` — For long runs, use **offline sync** so the UI stays fast: "
        f"`python -m ingestion.cli` from the project root (logs in `{config.APP_DATA_DIR}/logs/app.log`), "
        f"or `scripts/run_full_ingest.ps1` on Windows."
    )
    if Path(states_root).is_dir():
        if searchable:
            _render_index_health_banner(config, searchable, compact=False)
            snap_kb = _ensure_index_health_snapshot(config, searchable)
            kb_need = snap_kb.get("needs_repair") or []
            rb1, rb2 = st.columns(2)
            with rb1:
                if st.button(
                    "Repair search index",
                    key="kb_repair_index",
                    type="secondary",
                    help="Rebuilds catalog + title collections from SQLite when vector storage errors occur.",
                ):
                    targets = kb_need if kb_need else searchable
                    _repair_with_live_status(
                        config, searchable, targets,
                        label="Repairing search index",
                    )
                    st.rerun()
            with rb2:
                st.caption(
                    "Runs automatically when samples fail health checks. "
                    "Only one indexing job should run at a time (close CLI ingest while syncing)."
                )

        sc1, sc2 = st.columns([2, 1])
        with sc1:
            if st.button("Sync States Root", key="kb_sync_root", type="primary"):
                from indexer.chroma_lock import ChromaLockBusy, chroma_write_lock

                t_sync = time.perf_counter()

                progress_bar = st.progress(0, text="Discovering state folders…")
                status_text = st.empty()
                detail_text = st.empty()

                def _sync_progress(jur_idx, jur_total, phase, detail, cur, total):
                    jur_frac = jur_idx / max(jur_total, 1)
                    phase_weights = {"scan": 0.05, "parse": 0.25, "chunk_embed": 0.50, "mirror": 0.20}
                    phase_base = 0.0
                    for p, w in phase_weights.items():
                        if p == phase:
                            break
                        phase_base += w
                    phase_w = phase_weights.get(phase, 0.1)
                    phase_frac = cur / max(total, 1)
                    overall = jur_frac + (1.0 / max(jur_total, 1)) * (phase_base + phase_w * phase_frac)
                    overall = min(overall, 1.0)

                    phase_labels = {
                        "scan": "Scanning files",
                        "parse": "Parsing PDFs",
                        "chunk_embed": "Chunking & Embedding",
                        "mirror": "Mirroring to Chroma",
                    }
                    label = phase_labels.get(phase, phase)
                    progress_bar.progress(overall, text=f"[{jur_idx+1}/{jur_total}] {label}")
                    status_text.caption(f"Phase: **{label}** | {cur}/{total}")
                    if detail and detail != "done":
                        detail_text.caption(f"`{detail}`")

                sync_result: dict = {"ok": False, "error": "not started"}
                try:
                    with chroma_write_lock(config):
                        sync_result = sync_states_root(config, states_root, progress_callback=_sync_progress)
                except ChromaLockBusy as exc:
                    sync_result = {"ok": False, "error": str(exc)}
                    st.error(str(exc))

                progress_bar.progress(1.0, text="Sync complete!")
                status_text.empty()
                detail_text.empty()

                st.session_state["last_sync_result"] = sync_result
                sync_ms = int((time.perf_counter() - t_sync) * 1000)
                _invalidate_index_health_snapshot()
                try:
                    from core.analytics import record_states_sync_summary
                    record_states_sync_summary(
                        config, st.session_state.run_id,
                        {**sync_result, "states_root": states_root},
                        duration_ms=sync_ms,
                    )
                except Exception:
                    pass
                emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="states_root_synced",
                            payload={"discovered": sync_result.get("discovered"), "ready": sync_result.get("ready")},
                            trace_dir=config.TRACE_DIR)
                logger.info("states_root_synced discovered=%d ready=%d", sync_result.get("discovered", 0), sync_result.get("ready", 0))
                if sync_result.get("ok"):
                    st.rerun()
        with sc2:
            st.caption("Adds or updates PDFs under States root (exclusive lock prevents corrupt indexes).")

        with st.expander("Maintenance (advanced operators)", expanded=False):
            st.caption(
                "Most users only need **Sync States Root** or **Repair search index**. "
                "The two reset buttons below are **DESTRUCTIVE**: they erase your "
                "embedded corpus and force a full multi-hour re-ingestion. They are "
                "only for actual on-disk corruption — not for the `RustBindingsAPI` / "
                "`default_tenant` in-process error (that one is fixed by restarting "
                "Streamlit, no data loss)."
            )

            # Live snapshot of what would be lost — surface BEFORE any reset.
            try:
                from indexer.chroma_store import verify_chroma_health
                health = verify_chroma_health(config)
                if health.get("ok"):
                    st.info(
                        f"Current Chroma store: **{health['collections']} collections** "
                        f"at `{health['persist_dir']}`. Both reset buttons below will "
                        "delete this data."
                    )
                else:
                    st.warning(
                        f"Chroma probe failed in this process: {health.get('error', 'unknown')}. "
                        "If this is the `RustBindingsAPI` poisoning, **restart Streamlit "
                        "instead of resetting** — your on-disk data is fine."
                    )
            except Exception as exc:  # probe is cosmetic; never block UI
                logger.debug("maintenance health probe failed: %s", exc)

            m1, m2 = st.columns(2)
            with m1:
                with st.expander("Reset Chroma collections (DESTRUCTIVE)", expanded=False):
                    st.caption(
                        "Deletes every collection inside Chroma but leaves the "
                        "persist file in place. After this, the catalog and per-title "
                        "collections must be rebuilt by **Sync States Root** "
                        "(which re-runs chunking + embedding from PDFs)."
                    )
                    confirm = st.text_input(
                        "Type DELETE to confirm",
                        key="kb_reset_chroma_confirm",
                        placeholder="DELETE",
                    )
                    if st.button(
                        "Reset Chroma collections",
                        key="kb_reset_chroma",
                        disabled=(confirm.strip() != "DELETE"),
                        help="Requires typing DELETE in the box above.",
                    ):
                        res = _reset_chroma_collections_with_status(config)
                        if res.get("ok"):
                            logger.info("chroma_reset deleted=%d", res.get("deleted", 0))
                            st.session_state["kb_reset_chroma_confirm"] = ""
                            try:
                                from core.analytics import record_dev_event
                                record_dev_event(
                                    config, "chroma_reset",
                                    session_run_id=st.session_state.run_id,
                                    status="ok",
                                    extra={"deleted": res.get("deleted", 0)},
                                )
                            except Exception:
                                pass

                with st.expander("Reset Chroma persist folder (FULL WIPE)", expanded=False):
                    st.caption(
                        "Deletes the entire on-disk Chroma directory "
                        f"(`{config.CHROMA_PERSIST_DIR}`), including all 50+ "
                        "collections and ~700K metadata rows. After this you "
                        "must re-run the full ingestion pipeline (PDFs → chunks → "
                        "embeddings → mirror), which can take hours. **Only use this "
                        "if `chroma.sqlite3` itself is genuinely corrupt.**"
                    )
                    confirm_dir = st.text_input(
                        "Type WIPE-CHROMA to confirm",
                        key="kb_reset_chroma_dir_confirm",
                        placeholder="WIPE-CHROMA",
                    )
                    if st.button(
                        "Reset Chroma persist folder",
                        key="kb_reset_chroma_dir",
                        disabled=(confirm_dir.strip() != "WIPE-CHROMA"),
                        help="Requires typing WIPE-CHROMA in the box above.",
                    ):
                        res = _reset_chroma_persist_dir_with_status(config)
                        if res.get("ok"):
                            chroma_dir = res.get("path", str(config.CHROMA_PERSIST_DIR))
                            logger.info("chroma_persist_dir_reset path=%s", chroma_dir)
                            st.session_state["kb_reset_chroma_dir_confirm"] = ""
                            try:
                                from core.analytics import record_dev_event
                                record_dev_event(
                                    config, "chroma_persist_reset",
                                    session_run_id=st.session_state.run_id,
                                    status="ok",
                                    extra={"path": str(chroma_dir)},
                                )
                            except Exception:
                                pass
            with m2:
                if st.button(
                    "Purge stale embeddings",
                    key="kb_reembed",
                    help=f"Remove embeddings from older models so re-sync uses `{target_embed}`.",
                ):
                    from indexer.sql_indexer import purge_stale_embeddings
                    purge_res = purge_stale_embeddings(config, target_embed)
                    if purge_res.get("ok"):
                        st.info(
                            f"Purged {purge_res['deleted']} stale embedding row(s). "
                            f"Run **Sync States Root** to rebuild with `{target_embed}`."
                        )
                        _invalidate_index_health_snapshot()
                    else:
                        st.error("Purge failed")

        sr = st.session_state.get("last_sync_result")
        if isinstance(sr, dict):
            if sr.get("ok"):
                st.success(f"Discovered {sr['discovered']} states | {sr['ready']} ready | {sr['errors']} errors")
                summaries = sr.get("state_summaries", [])
                if summaries:
                    with st.expander("Per-state sync details", expanded=False):
                        for ss in summaries:
                            icon = {"ready": "\u2705", "partial": "\u26a0\ufe0f", "error": "\u274c"}.get(ss.get("status", ""), "\u2b55")
                            st.write(
                                f"{icon} **{ss['jurisdiction'].title()}** — {ss['status']} | "
                                f"PDFs={ss.get('pdfs_discovered', 0)} parsed={ss.get('pdfs_parsed', 0)} "
                                f"chunked={ss.get('docs_chunked', 0)} mirrored={ss.get('titles_mirrored', 0)}"
                            )
                            if ss.get("error"):
                                st.caption(f"  Error: {ss['error'][:120]}")
            else:
                st.error(sr.get("error", "Sync failed"))
    else:
        st.warning(f"States root not found: `{states_root}`. Set STATES_ROOT env var or place folders there.")

    statuses = get_all_ingestion_statuses(config)
    if statuses:
        st.subheader("Per-Jurisdiction Ingestion Status")
        for s in statuses:
            status = s.get("status", "?")
            icon = {"ready": "\u2705", "partial": "\u26a0\ufe0f", "error": "\u274c", "discovered": "\u2b55"}.get(status, "\u2753")
            st.write(
                f"{icon} **{s['jurisdiction'].title()}** — {status} | "
                f"docs={s.get('documents_count', 0)} parsed={s.get('parsed_count', 0)} "
                f"chunked={s.get('chunked_count', 0)} mirrored={s.get('mirrored_count', 0)}"
            )
            if s.get("last_error"):
                st.caption(f"  Error: {s['last_error'][:100]}")

    st.markdown("---")

    st.subheader("Indexed Coverage")
    m1, m2, m3 = st.columns(3)
    m1.metric("Searchable jurisdictions", len(searchable))
    m2.metric("Registered titles", len(titles))
    m3.metric("Chroma collections", len(cols))

    if embed_model:
        if embed_model == target_embed:
            st.caption(f"Indexed embedding model: `{embed_model}` \u2705")
        else:
            st.caption(f"Indexed embedding model: `{embed_model}`")
            st.warning(
                f"Embedding mismatch: indexed with `{embed_model}` but target is `{target_embed}`. "
                f"Use **Knowledge Base** → **Maintenance** → **Purge stale embeddings**, then **Sync States Root**."
            )
    else:
        st.caption(f"Target embedding model: `{target_embed}` (no embeddings indexed yet)")

    if searchable:
        st.subheader("Searchable Jurisdictions")
        st.write(", ".join(searchable))
    else:
        st.info("No jurisdictions fully searchable yet.")

    if titles:
        st.subheader("Registered Legal Titles")
        for t in titles:
            st.write(f"- **{t.get('jurisdiction', '?')}** Title {t.get('title_number', '?')}: {t.get('title_name', '-')}")

    if cols:
        with st.expander(f"Chroma Collections ({len(cols)})", expanded=False):
            for c in cols:
                st.write(f"- `{c['name']}` \u2014 {c['count']} records")

    distinct_jurs = sorted({(t.get("jurisdiction") or "").strip() for t in titles if t.get("jurisdiction")})
    if distinct_jurs:
        with st.expander("Chroma title routing catalogs (required for vector search)", expanded=False):
            st.caption(
                "Each state needs a `catalog_<jurisdiction>` collection so Ask can shortlist legal titles. "
                "If missing (e.g. after **Reset Chroma**), mirror below or run **Sync States Root**."
            )
            for jur in distinct_jurs:
                cs = get_jurisdiction_catalog_status(config, jur)
                icon = "\u2705" if cs.get("ok") else "\u274c"
                line = f"{icon} **{jur}** — `{cs.get('collection_name')}`"
                if cs.get("ok"):
                    line += f" ({cs.get('count')} entries)"
                else:
                    err = cs.get("error") or "unavailable"
                    line += f" — {err}"
                st.markdown(line)
            mc_jur = st.selectbox("Repair: mirror catalog for jurisdiction", distinct_jurs, key="kb_mirror_cat_pick")
            if st.button("Mirror title catalog to Chroma", key="kb_mirror_cat_btn", type="primary"):
                from indexer.chroma_lock import ChromaLockBusy, chroma_write_lock

                try:
                    with st.spinner("Mirroring catalog\u2026"):
                        with chroma_write_lock(config):
                            mres = mirror_title_catalog_to_chroma(config, mc_jur)
                except ChromaLockBusy as exc:
                    mres = {"ok": False, "error": str(exc)}
                    st.error(str(exc))

                if mres.get("ok"):
                    st.success(f"Created `{mres['collection_name']}` with {mres['records_mirrored']} routing entries.")
                    _invalidate_index_health_snapshot()
                else:
                    st.error(mres.get("error", "Mirror failed"))

    st.markdown("---")

    _render_latest_report(config)

    st.markdown(
        "**Why is a scope unavailable?** A jurisdiction becomes searchable after: "
        "(1) PDF ingested, (2) chunked/embedded into SQL, (3) mirrored to Chroma, "
        "and (4) its ingestion status is marked **ready**. Use 'Sync States Root' above."
    )


# ======================================================================
# Tab: Advanced (all existing operator panels)
# ======================================================================

def _tab_advanced(config, logger, base_url, ollama_status):
    st.caption("Operator / admin tools for ingestion, indexing, retrieval testing, and orchestration.")

    with st.expander("Development analytics (SQLite)", expanded=False):
        from core.analytics import list_dev_analytics_events
        ev_rows = list_dev_analytics_events(config, limit=150)
        if ev_rows:
            show_cols = (
                "created_at", "event_type", "status", "duration_ms",
                "task_count", "evidence_chunks_total", "unavailable_scopes_count",
                "embedding_model", "answer_model", "user_prompt", "error_message",
            )
            trimmed = [{k: r.get(k) for k in show_cols} for r in ev_rows]
            st.dataframe(trimmed, width="stretch", height=320)
            st.caption(f"Stored in `{config.APP_DATA_DIR}/state.sqlite` table `dev_analytics_events`.")
        else:
            st.info("No analytics rows yet. Use **Ask** or **Sync States Root** to generate events.")

    with st.expander("Ollama Status & Config", expanded=False):
        st.code("\n".join([f"APP_DATA_DIR={config.APP_DATA_DIR}", f"TRACE_DIR={config.TRACE_DIR}",
                           f"DEFAULT_CHAT_MODEL={config.DEFAULT_CHAT_MODEL}", f"DEFAULT_EMBED_MODEL={config.DEFAULT_EMBED_MODEL}"]), language=None)
        if ollama_status["ok"]:
            st.success("Ollama connected")
            models = ollama_status.get("models", [])
            if models:
                st.write("Models: " + ", ".join(f"`{m}`" for m in models))
        else:
            st.error(ollama_status.get("error") or "Not connected")

    with st.expander("Workspaces", expanded=False):
        with st.form("create_workspace_form", clear_on_submit=True):
            ws_name = st.text_input("Workspace name")
            ws_desc = st.text_input("Description", value="")
            ws_create = st.form_submit_button("Create Workspace")
        if ws_create:
            res = create_workspace(config, ws_name, ws_desc)
            if res.get("ok"):
                ws = res["workspace"]
                st.success("Workspace created")
                st.session_state.selected_workspace_id = ws["id"]
                emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="workspace_created", payload={"workspace": ws}, trace_dir=config.TRACE_DIR)
            else:
                st.error(res.get("error") or "Failed")
        workspaces = list_workspaces(config)
        if not workspaces:
            st.info("No workspaces yet.")
            return
        ws_options = {f"{w['name']} ({w['id'][:8]})": w["id"] for w in workspaces}
        labels = list(ws_options.keys())
        ids = [ws_options[l] for l in labels]
        current_id = st.session_state.selected_workspace_id
        if current_id not in ids:
            current_id = ids[0]
            st.session_state.selected_workspace_id = current_id
        selected_label = st.selectbox("Select workspace", options=labels, index=ids.index(current_id), key="adv_ws")
        st.session_state.selected_workspace_id = ws_options[selected_label]

    ws_id = st.session_state.selected_workspace_id
    if not ws_id:
        return

    with st.expander("Sources", expanded=False):
        with st.form("add_source_form", clear_on_submit=True):
            folder_path = st.text_input("Folder path")
            is_temp = st.checkbox("Temporary/session source", value=False)
            add_btn = st.form_submit_button("Add Source")
        if add_btn:
            res = add_source(config, ws_id, folder_path, source_type="folder", is_temp=is_temp)
            if res.get("ok"):
                st.success("Source added")
                emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="source_added", payload={"source": res["source"]}, trace_dir=config.TRACE_DIR)
            else:
                st.error(res.get("error") or "Failed")
        sources = list_sources(config, ws_id)
        if sources:
            for s in sources:
                st.write(f"- `{s['path']}` | type={s['source_type']} | temp={bool(s['is_temp'])}")
        else:
            st.info("No sources registered.")

    sources = list_sources(config, ws_id)

    with st.expander("Discovery / Inventory", expanded=False):
        if sources:
            src_options = {f"{Path(s['path']).name} ({s['id'][:8]})": s["id"] for s in sources}
            src_labels = list(src_options.keys())
            src_ids = [src_options[l] for l in src_labels]
            current_src = st.session_state.selected_source_id
            if current_src not in src_ids:
                current_src = src_ids[0]
                st.session_state.selected_source_id = current_src
            sel_src_label = st.selectbox("Source", src_labels, index=src_ids.index(current_src), key="adv_src")
            st.session_state.selected_source_id = src_options[sel_src_label]
            sel_src_id = st.session_state.selected_source_id
            if st.button("Scan Source", key="adv_scan"):
                res = scan_source(config, workspace_id=ws_id, source_id=sel_src_id, ignore_globs=getattr(config, "IGNORE_GLOBS", None))
                st.session_state.last_scan_summary = res
                emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="source_scanned", payload={"source_id": sel_src_id, "result": res}, trace_dir=config.TRACE_DIR)
            summary = st.session_state.last_scan_summary
            if isinstance(summary, dict) and summary.get("ok"):
                st.success(f"Scanned {summary.get('files_scanned',0)} | added {summary.get('files_added',0)} | updated {summary.get('files_updated',0)} | unchanged {summary.get('files_unchanged',0)}")
            elif isinstance(summary, dict) and summary and not summary.get("ok"):
                st.error(summary.get("error") or "Scan failed")
            files = list_source_files(config, sel_src_id)
            if files:
                for f in files[:200]:
                    st.write(f"- `{f['rel_path']}` | {f['file_type']} | {f['size_bytes']}b | {f['status']}")
                if len(files) > 200:
                    st.caption(f"Showing 200 of {len(files)}.")
            else:
                st.info("No inventory. Scan the source first.")
        else:
            st.info("Register a source first.")

    with st.expander("Document Parsing", expanded=False):
        if sources and st.session_state.selected_source_id:
            sel_src = st.session_state.selected_source_id
            pdf_files = [f for f in list_source_files(config, sel_src) if f["file_type"] == "pdf"]
            if not pdf_files:
                st.info("No PDFs. Scan source first.")
            else:
                pdf_opts = {f"{f['rel_path']} ({f['id'][:8]})": f for f in pdf_files}
                pdf_label = st.selectbox("PDF", list(pdf_opts.keys()), key="adv_pdf")
                chosen = pdf_opts[pdf_label]
                if st.button("Parse PDF", key="adv_parse"):
                    res = parse_source_file(config, sel_src, chosen["id"])
                    st.session_state.last_parse_summary = res
                    emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="pdf_parsed", payload={"source_file_id": chosen["id"], "result": res}, trace_dir=config.TRACE_DIR)
                ps = st.session_state.last_parse_summary
                if isinstance(ps, dict) and ps.get("ok"):
                    st.success(f"Pages={ps.get('page_count')} sections={ps.get('sections_created')} mode={ps.get('extraction_mode')}")
                elif isinstance(ps, dict) and ps and not ps.get("ok"):
                    st.error(ps.get("error") or "Parse failed")
            docs = list_documents_for_source(config, sel_src) if sources and st.session_state.selected_source_id else []
            if docs:
                st.markdown("**Parsed documents**")
                for d in docs:
                    st.write(f"- `{d.get('rel_path','')}` | title={d.get('title_number','-')} | name={d.get('title_name','-')} | status={d.get('parse_status','-')}")
                doc_opts = {f"{d.get('title_number') or '-'}/{d.get('title_name') or d['rel_path']} ({d['id'][:8]})": d["id"] for d in docs}
                sel_doc_label = st.selectbox("Document for sections", list(doc_opts.keys()), key="adv_doc_sec")
                sel_doc_id = doc_opts[sel_doc_label]
                sections = list_sections_for_document(config, sel_doc_id)
                if sections:
                    st.markdown(f"**{len(sections)} section(s)**")
                    for sec in sections[:100]:
                        with st.expander(f"Section {sec.get('section_number') or sec['section_order']} \u2014 {sec.get('catchline') or '(no catchline)'}", expanded=False):
                            st.text((sec["body_text"] or "")[:600])
                    if len(sections) > 100:
                        st.caption(f"Showing 100 of {len(sections)}.")
        else:
            st.info("Select a source first.")

    # Chunking with progress bar
    with st.expander("Chunking / Embedding (SQL)", expanded=False):
        if sources and st.session_state.selected_source_id:
            sel_src = st.session_state.selected_source_id
            docs = list_documents_for_source(config, sel_src)
            if docs:
                from indexer.sql_indexer import index_document_chunks_to_sql, list_chunks_for_document
                chunk_opts = {f"{d.get('title_number') or '-'}/{d.get('title_name') or d['rel_path']} ({d['id'][:8]})": d["id"] for d in docs}
                chunk_label = st.selectbox("Document to chunk+embed", list(chunk_opts.keys()), key="adv_chunk_doc")
                chunk_doc_id = chunk_opts[chunk_label]
                if st.button("Chunk + Embed to SQL", key="adv_chunk"):
                    progress_bar = st.progress(0, text="Estimating chunks…")
                    status_text = st.empty()

                    def _progress_cb(step: str, current: int, total: int):
                        frac = min(current / max(total, 1), 1.0)
                        if step == "estimate":
                            progress_bar.progress(0, text=f"Estimated ~{total} chunks")
                        elif step == "chunk":
                            progress_bar.progress(frac * 0.5, text=f"Chunking: {current}/{total}")
                        elif step == "embed":
                            progress_bar.progress(0.5 + frac * 0.5, text=f"Embedding: {current}/{total}")

                    res = index_document_chunks_to_sql(config, chunk_doc_id, progress_callback=_progress_cb)
                    progress_bar.progress(1.0, text="Done!")
                    st.session_state.last_chunk_summary = res
                    emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="sql_chunking_completed", payload=res, trace_dir=config.TRACE_DIR)

                cs = st.session_state.last_chunk_summary
                if isinstance(cs, dict) and cs.get("ok"):
                    est = cs.get("estimated_chunks", "?")
                    st.success(f"Chunks={cs.get('chunks_created')} (est. {est}) | Embeds={cs.get('embeddings_created')} | Model={cs.get('embed_model')}")
                elif isinstance(cs, dict) and cs and not cs.get("ok"):
                    st.error(cs.get("error") or "Failed")
                stored = list_chunks_for_document(config, chunk_doc_id)
                if stored:
                    st.caption(f"{len(stored)} chunk(s) in SQL")
            else:
                st.info("Parse documents first.")
        else:
            st.info("Select a source first.")

    with st.expander("Chroma Mirror", expanded=False):
        from indexer.chroma_mirror import list_chroma_collections, mirror_title_catalog_to_chroma, mirror_title_chunks_to_chroma
        conn = connect_db(config)
        try:
            jurisdictions = [r[0] for r in conn.execute("SELECT DISTINCT jurisdiction FROM legal_titles ORDER BY jurisdiction").fetchall() if r[0]]
            legal_titles_all = [dict(r) for r in conn.execute("SELECT id, jurisdiction, title_number, title_name, collection_key FROM legal_titles ORDER BY jurisdiction, title_number").fetchall()]
        except Exception:
            jurisdictions, legal_titles_all = [], []
        conn.close()
        if jurisdictions:
            sel_jur = st.selectbox("Jurisdiction", jurisdictions, key="adv_mirror_jur")
            if st.button("Mirror catalog to Chroma", key="adv_mirror_cat"):
                from indexer.chroma_lock import ChromaLockBusy, chroma_write_lock

                try:
                    with chroma_write_lock(config):
                        res = mirror_title_catalog_to_chroma(config, sel_jur)
                except ChromaLockBusy as exc:
                    res = {"ok": False, "error": str(exc)}
                st.session_state.last_catalog_mirror = res
                emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="chroma_catalog_mirrored", payload=res, trace_dir=config.TRACE_DIR)
                if res.get("ok"):
                    _invalidate_index_health_snapshot()
            cat_res = st.session_state.last_catalog_mirror
            if isinstance(cat_res, dict) and cat_res.get("ok"):
                st.success(f"Catalog: {cat_res['collection_name']} | {cat_res['records_mirrored']} entries")
            elif isinstance(cat_res, dict) and cat_res and not cat_res.get("ok"):
                st.error(cat_res.get("error") or "Failed")
            jur_titles = [t for t in legal_titles_all if (t["jurisdiction"] or "").lower() == sel_jur.lower()]
            if jur_titles:
                t_opts = {f"Title {t['title_number']}: {t['title_name'] or '-'} ({t['id'][:8]})": t["id"] for t in jur_titles}
                sel_t_label = st.selectbox("Legal title", list(t_opts.keys()), key="adv_mirror_title")
                sel_t_id = t_opts[sel_t_label]
                if st.button("Mirror title chunks to Chroma", key="adv_mirror_chunks"):
                    from indexer.chroma_lock import ChromaLockBusy, chroma_write_lock

                    try:
                        with chroma_write_lock(config):
                            res = mirror_title_chunks_to_chroma(config, sel_t_id)
                    except ChromaLockBusy as exc:
                        res = {"ok": False, "error": str(exc)}
                    st.session_state.last_title_mirror = res
                    emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="chroma_title_mirrored", payload=res, trace_dir=config.TRACE_DIR)
                    if res.get("ok"):
                        _invalidate_index_health_snapshot()
                tm_res = st.session_state.last_title_mirror
                if isinstance(tm_res, dict) and tm_res.get("ok"):
                    st.success(f"Chunks: {tm_res['collection_name']} | {tm_res['records_mirrored']} records")
                elif isinstance(tm_res, dict) and tm_res and not tm_res.get("ok"):
                    st.error(tm_res.get("error") or "Failed")
        else:
            st.info("No legal titles registered.")
        chroma_cols = list_chroma_collections(config)
        if chroma_cols:
            st.markdown("**Chroma collections**")
            for cc in chroma_cols:
                st.write(f"- `{cc['name']}` | {cc['count']} records")

    with st.expander("Single-Scope Retrieval Test", expanded=False):
        from retrieval.scoped_retriever import run_single_scope_query
        CORPUS_FAMILIES = ["state_requirements", "federal_requirements", "gse_requirements", "investor_requirements"]
        rc1, rc2 = st.columns(2)
        with rc1:
            ret_corpus = st.selectbox("Corpus family", CORPUS_FAMILIES, key="adv_ret_corpus")
        with rc2:
            ret_jur = st.text_input("Jurisdiction / Issuer", key="adv_ret_jur")
        ret_query = st.text_input("Query", key="adv_ret_query")
        ret_concepts = st.text_input("Concepts (comma-sep)", key="adv_ret_concepts")
        if st.button("Run Scoped Retrieval", key="adv_ret_go", disabled=not (ret_jur and ret_query)):
            concepts = [c.strip() for c in ret_concepts.split(",") if c.strip()] if ret_concepts else []
            with st.spinner("Searching…"):
                result = run_single_scope_query(config, corpus_family=ret_corpus, jurisdiction_or_issuer=ret_jur, concepts=concepts, query=ret_query)
            st.session_state.last_retrieval_result = result
            emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="single_scope_query_run", payload={"corpus_family": ret_corpus, "jurisdiction": ret_jur, "ok": result.get("ok"), "evidence": len(result.get("evidence_chunks", []))}, trace_dir=config.TRACE_DIR)
        rr = st.session_state.last_retrieval_result
        if isinstance(rr, dict):
            if rr.get("ok"):
                st.success(rr.get("synthesized_notes", ""))
            elif rr.get("synthesized_notes"):
                st.warning(rr["synthesized_notes"])
            evidence = rr.get("evidence_chunks", [])
            if evidence:
                for j, ev in enumerate(evidence[:10]):
                    with st.expander(f"#{j+1} \u00a7{ev.get('section_number','?')} {ev.get('catchline','')} score={ev.get('score')}", expanded=(j==0)):
                        st.text((ev.get("chunk_text","") or "")[:400])

    with st.expander("Run Compiler / Orchestrator", expanded=False):
        from orchestrator.compiler import build_run_queue, compile_user_prompt
        from orchestrator.run_store import (
            build_cumulative_run_output, build_structured_run_report,
            create_run, get_run, get_run_spec, list_run_queue, list_run_results, list_runs,
            request_stop, summarize_run_progress, get_next_pending_queue_item,
        )
        from orchestrator.runner import run_next_queue_item, run_steps_until_pause
        from orchestrator.report_builder import save_report_file
        user_prompt = st.text_area("Research prompt", height=80, key="adv_run_prompt")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Compile Run Spec", key="adv_compile", disabled=not user_prompt.strip()):
                spec = compile_user_prompt(config, user_prompt)
                st.session_state.last_compiled_spec = spec
                emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="run_compiled", payload={"prompt": user_prompt[:200], "mode": spec.get("run_mode"), "queue": len(spec.get("queue_items", []))}, trace_dir=config.TRACE_DIR)
        compiled = st.session_state.last_compiled_spec
        if isinstance(compiled, dict):
            if compiled.get("clarification_needed"):
                st.warning("Clarification needed:")
                for q in compiled.get("clarification_questions", []):
                    st.write(f"- {q}")
            else:
                st.success(f"Mode={compiled.get('run_mode')} Corpus={compiled.get('corpus_family')} Targets={len(compiled.get('jurisdictions_or_issuers',[]))}")
            with c2:
                if st.button("Create Run", key="adv_create_run", disabled=compiled.get("clarification_needed", True)):
                    qi = build_run_queue(compiled)
                    res = create_run(config, user_prompt, compiled, qi)
                    if res.get("ok"):
                        st.session_state.selected_run_id = res["run_id"]
                        st.success(f"Run created: {res['run_id'][:12]}… queue={res['queue_count']}")
                        emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="run_created", payload=res, trace_dir=config.TRACE_DIR)
                    else:
                        st.error(res.get("error", "Failed"))
        st.markdown("---")
        runs = list_runs(config)
        if runs:
            run_opts = {f"{r['user_prompt'][:40]}… ({r['id'][:8]}) [{r['status']}]": r["id"] for r in runs}
            sel_run_label = st.selectbox("Run to inspect", list(run_opts.keys()), key="adv_run_sel")
            sel_run_id = run_opts[sel_run_label]
            st.session_state.selected_run_id = sel_run_id
            run_detail = get_run(config, sel_run_id)
            if run_detail:
                r = run_detail
                st.write(f"**Status:** {r['status']} | **Mode:** {r['run_mode']} | **Stop:** {bool(r['stop_requested'])}")
                if not r["stop_requested"]:
                    if st.button("Request Stop", key="adv_stop"):
                        request_stop(config, sel_run_id)
                        st.warning("Stop requested.")
            progress = summarize_run_progress(config, sel_run_id)
            st.write(f"Progress: {progress['completed']}/{progress['total']} completed | {progress['pending']} pending | {progress['failed']} failed")
            next_qi = get_next_pending_queue_item(config, sel_run_id)
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                if st.button("Run Next Step", key="adv_next_step", disabled=(next_qi is None)):
                    with st.spinner(f"Executing: {next_qi['jurisdiction_or_issuer']}…"):
                        sr = run_next_queue_item(config, sel_run_id)
                    st.session_state.last_step_result = sr
                    emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="run_step_completed" if sr.get("ok") else "run_step_failed", payload={"target": sr.get("queue_target",""), "status": sr.get("execution_status","")}, trace_dir=config.TRACE_DIR)
            with ec2:
                max_steps = st.number_input("Max steps", min_value=1, max_value=100, value=5, key="adv_max_steps")
            with ec3:
                if st.button("Auto-Run Batch", key="adv_batch", disabled=(next_qi is None)):
                    with st.spinner(f"Auto-running up to {max_steps} step(s)…"):
                        br = run_steps_until_pause(config, sel_run_id, max_steps=max_steps)
                    st.session_state.last_batch_result = br
                    emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="auto_run_batch_completed", payload={"batch_status": br.get("batch_status"), "steps": br.get("steps_completed")}, trace_dir=config.TRACE_DIR)
            sr = st.session_state.last_step_result
            if isinstance(sr, dict) and sr.get("execution_status"):
                es = sr["execution_status"]
                if es == "completed" and sr.get("ok"):
                    st.success(f"Step: {sr.get('queue_target','?')} | evidence={sr.get('evidence_count',0)}")
                elif es == "all_done":
                    st.success("Run complete.")
                elif es == "stopped":
                    st.warning("Stopped.")
                elif es == "failed":
                    st.error(f"Failed: {sr.get('queue_target','?')}")
            br = st.session_state.last_batch_result
            if isinstance(br, dict) and br.get("batch_status"):
                bs = br["batch_status"]
                st.info(f"Batch: {bs} | {br.get('steps_completed',0)}/{br.get('steps_attempted',0)} steps | final={br.get('final_run_status','?')}")
            results = list_run_results(config, sel_run_id)
            if results:
                st.markdown(f"**Results** ({len(results)})")
                for idx, rr in enumerate(results[:15]):
                    rd = rr.get("result", {})
                    with st.expander(f"[{idx}] {rd.get('jurisdiction_or_issuer','?')} evidence={len(rd.get('evidence_chunks',[]))} ok={rd.get('ok')}", expanded=False):
                        st.write(rd.get("synthesized_notes", ""))
        else:
            st.info("No runs yet.")

    with st.expander("Cumulative Run Output", expanded=False):
        from orchestrator.run_store import build_cumulative_run_output, list_runs as lr
        all_r = lr(config)
        with_res = [r for r in all_r if r["status"] in ("running", "completed", "stopped", "failed")]
        if with_res:
            o_opts = {f"{r['user_prompt'][:40]}… ({r['id'][:8]}) [{r['status']}]": r["id"] for r in with_res}
            o_label = st.selectbox("Run", list(o_opts.keys()), key="adv_cum_sel")
            o_id = o_opts[o_label]
            cum = build_cumulative_run_output(config, o_id)
            if cum.get("ok"):
                st.write(f"Status: {cum['run_status']} | {cum['completed_count']}/{cum['total_count']} completed")
                for sec in cum.get("assembled_sections", []):
                    st.write(f"- [{sec.get('execution_status','?')}] {sec['target']} evidence={sec.get('evidence_count',0)}")
                txt = cum.get("assembled_text", "")
                if txt.strip():
                    st.code(txt[:2000], language=None)
        else:
            st.info("No runs with results.")

    with st.expander("Final Report / Synthesis", expanded=False):
        from orchestrator.run_store import build_structured_run_report as bsr, list_runs as lr2
        from orchestrator.report_builder import save_report_file as srf
        all_r2 = lr2(config)
        reportable = [r for r in all_r2 if r["status"] in ("running", "completed", "stopped", "failed")]
        if reportable:
            rp_opts = {f"{r['user_prompt'][:40]}… ({r['id'][:8]}) [{r['status']}]": r["id"] for r in reportable}
            rp_label = st.selectbox("Run", list(rp_opts.keys()), key="adv_rpt_sel")
            rp_id = rp_opts[rp_label]
            if st.button("Build Report", key="adv_build_rpt"):
                rpt = bsr(config, rp_id)
                st.session_state.last_report = rpt
                st.session_state.last_report_path = None
                emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="report_built", payload={"run_id": rp_id, "ok": rpt.get("ok")}, trace_dir=config.TRACE_DIR)
            rpt = st.session_state.last_report
            if isinstance(rpt, dict) and rpt.get("ok"):
                st.write(f"Status: {rpt['run_status']} | {rpt['completed_scopes']}/{rpt['total_scopes']} completed | {rpt['failed_scopes']} failed")
                rt = rpt.get("report_text", "")
                if rt.strip():
                    st.code(rt[:3000], language="markdown")
                if st.button("Save as Markdown", key="adv_save_rpt"):
                    path = srf(config, rp_id, rt)
                    st.session_state.last_report_path = path
                    emit_event(run_id=st.session_state.run_id, agent_id="ui", event_type="report_saved", payload={"path": path}, trace_dir=config.TRACE_DIR)
                if st.session_state.last_report_path:
                    st.success(f"Saved: `{st.session_state.last_report_path}`")
        else:
            st.info("No runs with results.")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    config = load_config()
    ensure_app_dirs(config)
    init_db(config)

    from core.title_registry import sync_legal_titles
    sync_legal_titles(config)

    logger = setup_logging(config.APP_DATA_DIR)

    st.set_page_config(page_title="Local Deep Research RAG", page_icon="\U0001f52c", layout="centered")
    _init_state()
    st.session_state["_config"] = config

    st.title("Local Deep Research RAG")

    base_url, ollama_status = _sidebar(config, logger)

    # Eager startup probe: open the Chroma store exactly once per Streamlit run,
    # BEFORE any tab does retrieval. Surfaces poisoned in-process Rust bindings
    # as a single visible banner instead of dozens of duplicate tracebacks fired
    # mid-query. Pure read-only — never deletes data.
    try:
        from indexer.chroma_store import verify_chroma_health
        from indexer.chroma_health import is_chroma_runtime_poisoned
        if not st.session_state.get("_chroma_startup_probed"):
            health = verify_chroma_health(config)
            st.session_state["_chroma_startup_probed"] = True
            st.session_state["_chroma_startup_health"] = health
            if health.get("ok"):
                from indexer.chroma_health import clear_unhealthy_collections
                clear_unhealthy_collections()
                st.session_state.chroma_runtime_poisoned = False
                st.session_state.chroma_runtime_poisoned_detail = ""
            elif is_chroma_runtime_poisoned(health.get("error")):
                _flag_chroma_runtime_poisoned(health.get("error", "unknown"))
                logger.error(
                    "chroma startup probe poisoned: %s | persist=%s",
                    health.get("error"), health.get("persist_dir"),
                )
    except Exception as exc:
        logger.warning("chroma startup probe raised: %s", exc)

    tab_ask, tab_kb, tab_adv = st.tabs(["Ask", "Knowledge Base", "Advanced"])

    with tab_ask:
        _tab_ask(config, logger)

    with tab_kb:
        _tab_kb(config, logger)

    with tab_adv:
        _tab_advanced(config, logger, base_url, ollama_status)


if __name__ == "__main__":
    main()
