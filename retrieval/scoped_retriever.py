"""
Single-scope retrieval runner with hybrid scoring, RRF fusion, iterative fallback,
and SQL keyword fallback when Chroma is unavailable.

Flow:
1. Query the catalog collection to shortlist candidate titles.
2. Search shortlisted scoped chunk collections for evidence.
3. Apply hybrid scoring (semantic + lexical overlap).
4. Apply Reciprocal Rank Fusion across multiple retrieval signals.
5. Filter malformed / weak evidence.
6. If evidence is weak, widen the title shortlist and retry (bounded).
7. If Chroma returns nothing, fall back to SQL keyword search.
"""
from __future__ import annotations

import logging
import re

from indexer.chroma_health import (
    is_chroma_disk_corruption_error,
    is_collection_marked_unhealthy,
    mark_collection_unhealthy,
)
from indexer.chroma_mirror import get_jurisdiction_catalog_status
from indexer.chroma_store import (
    get_chroma_client,
    get_scoped_chunk_collection_name,
    get_state_catalog_collection_name,
)
from indexer.chunker import is_malformed_chunk
from indexer.embedder import embed_texts
from retrieval.query_normalizer import build_query_text, expand_basic_aliases, normalize_concepts

logger = logging.getLogger(__name__)

MIN_STRONG_SCORE = 0.40
MIN_WEAK_SCORE = 0.20
MAX_RETRIEVAL_ITERATIONS = 5
INITIAL_CATALOG_K = 8
EXPANDED_CATALOG_K = 16
EVIDENCE_PER_COLLECTION = 15
RRF_K = 60
FINAL_EVIDENCE_CAP = 40  # trim post-RRF for memory and LLM context budget


def _embed_model(config) -> str:
    m = getattr(config, "DEFAULT_EMBED_MODEL", "nomic-embed-text-v2-moe:latest")
    return m if ":" in m else f"{m}:latest"


def _ollama_url(config) -> str:
    return getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")


def _lexical_overlap(query: str, text: str) -> float:
    """Fraction of query terms found in text (case-insensitive)."""
    q_words = set(re.findall(r"[a-z]{3,}", query.lower()))
    if not q_words:
        return 0.0
    t_words = set(re.findall(r"[a-z]{3,}", text.lower()))
    return len(q_words & t_words) / len(q_words)


def _phrase_overlap(query: str, text: str) -> float:
    """Check for multi-word phrase matches from the query in the chunk text."""
    q_lower = query.lower()
    t_lower = text.lower()
    q_words = re.findall(r"[a-z]{3,}", q_lower)
    if len(q_words) < 2:
        return 0.0
    bigrams = [f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words) - 1)]
    matches = sum(1 for bg in bigrams if bg in t_lower)
    return matches / len(bigrams) if bigrams else 0.0


def _hybrid_score(semantic_distance: float | None, query: str, chunk_text: str, meta: dict) -> float:
    """Combine semantic similarity with lexical overlap, phrase overlap, and metadata signals.

    Uses a linear L2-to-similarity mapping that is less aggressive than 1/(1+d)
    so that moderately relevant chunks still score meaningfully.
    """
    if semantic_distance is not None and semantic_distance >= 0:
        sem = max(0.0, 1.0 - semantic_distance / 4.0)
    else:
        sem = 0.0

    full_text = (chunk_text or "") + " " + (meta.get("catchline") or "") + " " + (meta.get("section_number") or "")
    lex = _lexical_overlap(query, full_text)
    phrase = _phrase_overlap(query, full_text)

    return round(0.50 * sem + 0.30 * lex + 0.20 * phrase, 4)


def _rrf_fuse(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.
    Each list item must have a "chunk_id" key.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for rlist in ranked_lists:
        for rank, item in enumerate(rlist):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in items:
                items[cid] = item
    for cid in items:
        items[cid]["rrf_score"] = round(scores[cid], 6)
    return sorted(items.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)


def _is_weak_evidence(chunk: dict) -> bool:
    """Check if a chunk is malformed or too low quality to surface."""
    text = chunk.get("chunk_text", "")
    if is_malformed_chunk(text):
        return True
    if chunk.get("score", 0) < MIN_WEAK_SCORE:
        return True
    return False


# ------------------------------------------------------------------
# SQL keyword fallback
# ------------------------------------------------------------------

def _sql_keyword_search(
    config,
    jurisdiction: str,
    query: str,
    max_results: int = 30,
) -> list[dict]:
    """Search chunks directly from SQL using keyword matching.
    Used as a fallback when Chroma is unavailable or returns nothing.
    """
    from core.db import connect_db

    keywords = [w for w in re.findall(r"[a-z]{3,}", query.lower()) if w not in {
        "the", "and", "for", "are", "that", "this", "with", "from", "can", "have",
        "will", "what", "how", "does", "not", "but", "they", "their", "been", "about",
    }]
    if not keywords:
        return []

    where_parts = []
    params: list[str] = [jurisdiction]
    for kw in keywords[:6]:
        where_parts.append("LOWER(c.text) LIKE ?")
        params.append(f"%{kw}%")

    if not where_parts:
        return []

    keyword_clause = " OR ".join(where_parts)
    sql = f"""
        SELECT c.id as chunk_id, c.document_id, c.document_section_id,
               c.chunk_index, c.chunk_strategy, c.text,
               c.jurisdiction, c.title_number, c.section_number,
               ds.catchline, ds.page_start, ds.page_end,
               d.rel_path as source_path
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        LEFT JOIN document_sections ds ON c.document_section_id = ds.id
        WHERE LOWER(c.jurisdiction) = LOWER(?) AND ({keyword_clause})
        ORDER BY c.chunk_index ASC
        LIMIT ?
    """
    params.append(str(max_results))

    try:
        with connect_db(config) as conn:
            rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        logger.error("SQL keyword fallback failed: %s", exc)
        return []

    results: list[dict] = []
    for r in rows:
        text = r["text"] or ""
        lex = _lexical_overlap(query, text)
        phrase = _phrase_overlap(query, text)
        score = round(0.50 * lex + 0.50 * phrase, 4)

        results.append({
            "chunk_id": r["chunk_id"],
            "collection_name": "sql_fallback",
            "corpus_family": "state_requirements",
            "jurisdiction": r["jurisdiction"] or "",
            "title_number": r["title_number"] or "",
            "title_name": "",
            "section_number": r["section_number"] or "",
            "catchline": r["catchline"] or "",
            "source_path": r["source_path"] or "",
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "document_id": r["document_id"] or "",
            "document_section_id": r["document_section_id"] or "",
            "chunk_strategy": r["chunk_strategy"] or "",
            "chunk_text": text,
            "score": score,
            "semantic_distance": None,
            "retrieval_method": "sql_keyword_fallback",
        })

    results.sort(key=lambda e: e.get("score", 0), reverse=True)
    logger.info("SQL keyword fallback for '%s' in %s: %d results", query[:60], jurisdiction, len(results))
    return results


# ------------------------------------------------------------------
# A0. Deterministic title-hint resolution (from user-cited Title numbers)
# ------------------------------------------------------------------

def _normalize_title_token(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (t or "").strip()).upper()


def resolve_title_hints(
    config,
    corpus_family: str,
    jurisdiction_or_issuer: str,
    title_hints: list[str],
) -> list[dict]:
    """Resolve user-cited title numbers (e.g. ['35', '35A']) against SQL
    `legal_titles` and return target dicts shaped like `select_candidate_targets`.

    These are *pinned* targets — they bypass vector catalog routing and give
    retrieval a deterministic anchor even when semantic search misroutes.
    """
    jur = (jurisdiction_or_issuer or "").strip()
    if not jur or not title_hints:
        return []

    wanted = {_normalize_title_token(t) for t in title_hints if t}
    wanted.discard("")
    if not wanted:
        return []

    from core.db import connect_db

    try:
        with connect_db(config) as conn:
            rows = conn.execute(
                """
                SELECT id, title_number, title_name, collection_key
                FROM legal_titles
                WHERE LOWER(jurisdiction) = LOWER(?)
                """,
                (jur,),
            ).fetchall()
    except Exception as exc:
        logger.warning("resolve_title_hints: SQL lookup failed for %s: %s", jur, exc)
        return []

    matches: list[dict] = []
    for r in rows:
        tnum = r["title_number"] or ""
        if _normalize_title_token(tnum) in wanted:
            matches.append({
                "corpus_family": corpus_family,
                "jurisdiction_or_issuer": jur,
                "collection_name": get_scoped_chunk_collection_name(config, jur, tnum),
                "legal_title_id": r["id"] or "",
                "title_number": tnum,
                "title_name": r["title_name"] or "",
                "score": 1.0,
                "distance": 0.0,
                "source": "title_hint",
            })

    if matches:
        logger.info(
            "resolve_title_hints: pinned %d target(s) for %s from hints %s",
            len(matches), jur, sorted(wanted),
        )
    else:
        logger.info(
            "resolve_title_hints: no legal_titles matched hints %s for %s",
            sorted(wanted), jur,
        )
    return matches


# ------------------------------------------------------------------
# A. Candidate selection
# ------------------------------------------------------------------

def select_candidate_targets(
    config,
    corpus_family: str,
    jurisdiction_or_issuer: str,
    concepts: list[str],
    original_query: str = "",
    top_k: int = 10,
) -> list[dict]:
    jur = (jurisdiction_or_issuer or "").strip()
    if not jur:
        return []

    col_name = get_state_catalog_collection_name(config, jur)

    if is_collection_marked_unhealthy(col_name):
        logger.debug("select_candidate_targets: skipping unhealthy catalog '%s'", col_name)
        return []

    try:
        client = get_chroma_client(config)
        col = client.get_collection(col_name)
        count = col.count()
    except Exception as exc:
        msg = str(exc)
        logger.warning("select_candidate_targets: cannot access catalog '%s': %s", col_name, exc)
        if is_chroma_disk_corruption_error(msg):
            mark_collection_unhealthy(col_name)
        return []

    if count == 0:
        logger.warning("select_candidate_targets: catalog '%s' is empty", col_name)
        return []

    normed = normalize_concepts(concepts)
    expanded = expand_basic_aliases(normed)

    if original_query.strip():
        query_text = build_query_text(original_query.strip(), expanded)
    else:
        query_text = build_query_text(" ".join(expanded), [])

    logger.debug("select_candidate_targets: embedding query text (%d chars): %s", len(query_text), query_text[:120])

    vec = embed_texts([query_text], model=_embed_model(config), ollama_url=_ollama_url(config))
    if not vec:
        logger.error("select_candidate_targets: embedding returned empty for query")
        return []

    try:
        actual_k = min(top_k, count)
        results = col.query(query_embeddings=vec, n_results=actual_k, include=["metadatas", "documents", "distances"])
    except Exception as exc:
        msg = str(exc)
        logger.error("select_candidate_targets: query on '%s' failed: %s", col_name, exc)
        if is_chroma_disk_corruption_error(msg):
            mark_collection_unhealthy(col_name)
        return []

    targets: list[dict] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else None
        title_num = meta.get("title_number", "")
        targets.append({
            "corpus_family": corpus_family,
            "jurisdiction_or_issuer": jur,
            "collection_name": get_scoped_chunk_collection_name(config, jur, title_num),
            "legal_title_id": meta.get("legal_title_id", ""),
            "title_number": title_num,
            "title_name": meta.get("title_name", ""),
            "score": round(1.0 / (1.0 + distance), 4) if distance is not None else None,
            "distance": round(distance, 4) if distance is not None else None,
        })

    logger.info("select_candidate_targets: %d targets from catalog '%s'", len(targets), col_name)
    return targets


# ------------------------------------------------------------------
# B. Scoped evidence retrieval (with hybrid scoring + quality filter)
# ------------------------------------------------------------------

def retrieve_scoped_evidence(
    config,
    corpus_family: str,
    collection_name: str,
    query: str,
    top_k: int = EVIDENCE_PER_COLLECTION,
) -> list[dict]:
    col_name = (collection_name or "").strip()
    if not col_name:
        return []

    if is_collection_marked_unhealthy(col_name):
        logger.debug("retrieve_scoped_evidence: skipping unhealthy '%s'", col_name)
        return []

    try:
        client = get_chroma_client(config)
        col = client.get_collection(col_name)
        count = col.count()
    except Exception as exc:
        msg = str(exc)
        logger.warning("retrieve_scoped_evidence: cannot access collection '%s': %s", col_name, exc)
        if is_chroma_disk_corruption_error(msg):
            mark_collection_unhealthy(col_name)
        return []

    if count == 0:
        logger.warning("retrieve_scoped_evidence: collection '%s' is empty", col_name)
        return []

    vec = embed_texts([query], model=_embed_model(config), ollama_url=_ollama_url(config))
    if not vec:
        logger.error("retrieve_scoped_evidence: embedding returned empty for query")
        return []

    try:
        actual_k = min(top_k, count)
        results = col.query(
            query_embeddings=vec,
            n_results=actual_k,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:
        msg = str(exc)
        logger.error("retrieve_scoped_evidence: query on '%s' failed: %s", col_name, exc)
        if is_chroma_disk_corruption_error(msg):
            mark_collection_unhealthy(col_name)
        return []

    evidence: list[dict] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else None
        chunk_text = results["documents"][0][i] if results["documents"] else ""

        score = _hybrid_score(distance, query, chunk_text, meta)

        evidence.append({
            "chunk_id": results["ids"][0][i],
            "collection_name": col_name,
            "corpus_family": corpus_family,
            "jurisdiction": meta.get("jurisdiction", ""),
            "title_number": meta.get("title_number", ""),
            "title_name": meta.get("title_name", ""),
            "section_number": meta.get("section_number", ""),
            "catchline": meta.get("catchline", ""),
            "source_path": meta.get("source_path", ""),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "document_id": meta.get("document_id", ""),
            "document_section_id": meta.get("document_section_id", ""),
            "chunk_strategy": meta.get("chunk_strategy", ""),
            "chunk_text": chunk_text,
            "score": score,
            "semantic_distance": round(distance, 4) if distance is not None else None,
        })

    logger.info("retrieve_scoped_evidence: %d results from '%s' (top score=%.4f)",
                len(evidence), col_name, evidence[0]["score"] if evidence else 0)
    return evidence


# ------------------------------------------------------------------
# C. Iterative single-scope runner with RRF + SQL fallback
# ------------------------------------------------------------------

def run_single_scope_query(
    config,
    corpus_family: str,
    jurisdiction_or_issuer: str,
    concepts: list[str],
    query: str,
    title_hints: list[str] | None = None,
    section_hints: list[str] | None = None,
    always_run_sql: bool = True,
) -> dict:
    """End-to-end retrieval with RRF fusion over vector + SQL signals.

    Parameters
    ----------
    title_hints:
        User-cited Title numbers (e.g. ["35"]). When present, these Titles
        are pinned as targets BEFORE any vector-based catalog routing, so
        Chroma misrouting cannot drop them.
    section_hints:
        User-cited Sections (e.g. ["35-11-10"]). Passed through into the
        SQL keyword signal for precision.
    always_run_sql:
        If True (default), SQL keyword search is always run and RRF-fused
        with vector results. This mirrors how the Knowledge Base runner
        combines Chroma chunks with SQL-backed metadata for precision.
    """
    title_hints = list(title_hints or [])
    section_hints = list(section_hints or [])
    cat_status = get_jurisdiction_catalog_status(config, jurisdiction_or_issuer)

    all_searched: list[str] = []
    all_evidence_lists: list[list[dict]] = []
    weak_rejected = 0
    iteration = 0
    catalog_k = INITIAL_CATALOG_K
    targets: list[dict] = []

    # 1. Pinned targets from deterministic title-hint resolution.
    pinned = resolve_title_hints(
        config,
        corpus_family=corpus_family,
        jurisdiction_or_issuer=jurisdiction_or_issuer,
        title_hints=title_hints,
    )

    if pinned:
        targets.extend(pinned)
        pinned_evidence: list[dict] = []
        for t in pinned:
            col_name = t["collection_name"]
            if col_name in all_searched:
                continue
            all_searched.append(col_name)
            chunks = retrieve_scoped_evidence(
                config,
                corpus_family=corpus_family,
                collection_name=col_name,
                query=query,
                top_k=EVIDENCE_PER_COLLECTION,
            )
            for ch in chunks:
                ch["retrieval_method"] = "vector_title_pinned"
                if _is_weak_evidence(ch):
                    weak_rejected += 1
                else:
                    pinned_evidence.append(ch)
        if pinned_evidence:
            pinned_evidence.sort(key=lambda e: e.get("score", 0), reverse=True)
            all_evidence_lists.append(pinned_evidence)

    # 2. Iterative vector catalog routing (if pinned didn't yield strong evidence).
    pinned_strong = any(
        e.get("score", 0) >= MIN_STRONG_SCORE
        for lst in all_evidence_lists for e in lst
    )

    if not pinned_strong:
        for iteration in range(1, MAX_RETRIEVAL_ITERATIONS + 1):
            catalog_targets = select_candidate_targets(
                config,
                corpus_family=corpus_family,
                jurisdiction_or_issuer=jurisdiction_or_issuer,
                concepts=concepts if iteration == 1 else expand_basic_aliases(normalize_concepts(concepts)),
                original_query=query,
                top_k=catalog_k,
            )

            if not catalog_targets:
                logger.info("run_single_scope_query iter %d: no catalog targets for %s",
                            iteration, jurisdiction_or_issuer)
                break

            targets.extend(catalog_targets)

            iteration_evidence: list[dict] = []
            for t in catalog_targets:
                col_name = t["collection_name"]
                if col_name in all_searched:
                    continue
                all_searched.append(col_name)
                chunks = retrieve_scoped_evidence(
                    config,
                    corpus_family=corpus_family,
                    collection_name=col_name,
                    query=query,
                    top_k=EVIDENCE_PER_COLLECTION,
                )
                for ch in chunks:
                    ch.setdefault("retrieval_method", "vector_catalog")
                    if _is_weak_evidence(ch):
                        weak_rejected += 1
                    else:
                        iteration_evidence.append(ch)

            if iteration_evidence:
                iteration_evidence.sort(key=lambda e: e.get("score", 0), reverse=True)
                all_evidence_lists.append(iteration_evidence)

            strong = [e for e in iteration_evidence if e.get("score", 0) >= MIN_STRONG_SCORE]
            if strong:
                break

            catalog_k = EXPANDED_CATALOG_K

    # 3. SQL keyword signal — deterministic, cheap, high-precision on
    #    explicit Title/Section mentions. Run when:
    #      (a) vector produced nothing, or
    #      (b) the user cited a specific title/section (always_run_sql=True), or
    #      (c) vector's best score is weak.
    vector_top_score = 0.0
    for lst in all_evidence_lists:
        for e in lst:
            s = e.get("score", 0) or 0
            if s > vector_top_score:
                vector_top_score = s

    needs_sql = (
        not all_evidence_lists
        or bool(title_hints) or bool(section_hints)
        or (always_run_sql and vector_top_score < MIN_STRONG_SCORE)
    )

    if needs_sql:
        sql_query = query
        if section_hints:
            sql_query = f"{sql_query} {' '.join(section_hints)}"
        sql_evidence = _sql_keyword_search(
            config, jurisdiction_or_issuer, sql_query,
            max_results=30,
        )
        if sql_evidence:
            if title_hints:
                wanted = {_normalize_title_token(t) for t in title_hints}
                filtered = [
                    ch for ch in sql_evidence
                    if _normalize_title_token(ch.get("title_number", "")) in wanted
                ]
                sql_evidence = filtered or sql_evidence
            kept_sql = [ch for ch in sql_evidence if not _is_weak_evidence(ch)]
            if kept_sql:
                all_evidence_lists.append(kept_sql)

    if all_evidence_lists:
        all_evidence = _rrf_fuse(all_evidence_lists)[:FINAL_EVIDENCE_CAP]
    else:
        all_evidence = []

    strong = [e for e in all_evidence if e.get("score", 0) >= MIN_STRONG_SCORE]
    answer_status = "strong" if strong else ("weak" if all_evidence else "no_evidence")

    notes_parts: list[str] = []
    if not cat_status.get("ok"):
        notes_parts.append(
            f"Chroma title catalog unavailable (`{cat_status.get('collection_name', '?')}`): "
            f"{cat_status.get('error', 'unknown')}. "
            "Vector title routing is skipped; SQL keyword fallback may still apply."
        )
    if all_evidence:
        notes_parts.append(f"Found {len(all_evidence)} evidence chunks across {len(all_searched)} collection(s).")
        if weak_rejected:
            notes_parts.append(f"Rejected {weak_rejected} low-quality chunks.")
        top = all_evidence[0]
        notes_parts.append(
            f"Top hit: Section {top.get('section_number', '?')} "
            f"({top.get('catchline', 'no catchline')}) "
            f"score={top.get('score')}"
        )
        if answer_status == "weak":
            notes_parts.append("Evidence is weak — results may not be reliable.")
    else:
        notes_parts.append("No quality evidence found in searched collections.")

    searched_titles: list[dict] = []
    seen_titles: set[str] = set()
    for t in targets:
        key = t.get("title_number", "")
        if key not in seen_titles:
            seen_titles.add(key)
            searched_titles.append({"title_number": t["title_number"], "title_name": t["title_name"], "score": t.get("score")})

    logger.info("run_single_scope_query: %s/%s → status=%s, evidence=%d, iterations=%d",
                corpus_family, jurisdiction_or_issuer, answer_status, len(all_evidence), iteration)

    return {
        "ok": len(all_evidence) > 0,
        "corpus_family": corpus_family,
        "jurisdiction_or_issuer": jurisdiction_or_issuer,
        "candidate_targets": targets,
        "searched_collections": all_searched,
        "evidence_chunks": all_evidence,
        "synthesized_notes": " ".join(notes_parts),
        "answer_status": answer_status,
        "retrieval_iterations": iteration,
        "searched_titles": searched_titles,
        "weak_evidence_rejected_count": weak_rejected,
        "chroma_catalog_ok": cat_status.get("ok", False),
        "chroma_catalog_collection": cat_status.get("collection_name"),
        "chroma_catalog_count": cat_status.get("count"),
        "chroma_catalog_error": cat_status.get("error"),
    }
