"""
Ask-mode compiler: parse a natural-language question into executable tasks.

Supports multi-jurisdiction / multi-topic queries. Uses LLM when an
orchestrator model is provided, with a deterministic rule-based fallback.
"""
from __future__ import annotations

import json
import re


_US_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
]


def _extract_jurisdictions(text: str) -> list[str]:
    low = text.lower()
    found = []
    for s in _US_STATES:
        if s in low:
            found.append(s)
    if re.search(r"\bevery\s+state", low) or re.search(r"\ball\s+states", low):
        found = list(_US_STATES)
    return found


_TITLE_PATTERNS = [
    re.compile(r"\btitle[\s:\-]*([0-9]{1,3}[A-Z]?)\b", re.IGNORECASE),
    re.compile(r"\bt\.\s*([0-9]{1,3}[A-Z]?)\b", re.IGNORECASE),
]
_SECTION_PATTERNS = [
    re.compile(r"(?:\bsection|\bsec\.?|§)\s*([0-9]{1,3}[A-Z]?[\-\.][0-9]+(?:[\-\.][0-9]+)?)", re.IGNORECASE),
]
_CHAPTER_PATTERN = re.compile(r"\bchapter[\s:\-]*([0-9]{1,3}[A-Z]?)\b", re.IGNORECASE)


def extract_structural_refs(text: str) -> dict:
    """Deterministically pull Title / Chapter / Section references from a user query.

    Used to pin retrieval to the correct legal title even when the LLM
    orchestrator rewrites the search query.

    Returns:
        {
            "titles":   ["35", "35A", ...],
            "chapters": ["4", ...],
            "sections": ["35-11-10", ...],
        }
    """
    if not text:
        return {"titles": [], "chapters": [], "sections": []}

    titles: list[str] = []
    for pat in _TITLE_PATTERNS:
        for m in pat.findall(text):
            num = m.upper().strip()
            if num and num not in titles:
                titles.append(num)

    chapters: list[str] = []
    for m in _CHAPTER_PATTERN.findall(text):
        num = m.upper().strip()
        if num and num not in chapters:
            chapters.append(num)

    sections: list[str] = []
    for pat in _SECTION_PATTERNS:
        for m in pat.findall(text):
            key = m.strip()
            if key and key not in sections:
                sections.append(key)

    return {"titles": titles, "chapters": chapters, "sections": sections}


def _extract_concepts(text: str) -> list[str]:
    """Extract meaningful concepts, preserving multi-word phrases.

    First extracts bigram/trigram noun-like phrases, then adds
    remaining individual content words. This keeps "lien release fees"
    as a phrase rather than fragmenting it.
    """
    stop = {"the", "and", "for", "are", "can", "you", "how", "what", "does",
            "this", "that", "with", "from", "have", "there", "about", "which",
            "when", "where", "will", "been", "they", "their", "would", "could",
            "should", "into", "also", "each", "other", "than", "some"}

    words = re.findall(r"[a-z]{3,}", text.lower())
    content_words = [w for w in words if w not in stop and w not in _US_STATES]

    seen: set[str] = set()
    out: list[str] = []

    for size in (3, 2):
        for i in range(len(content_words) - size + 1):
            phrase = " ".join(content_words[i : i + size])
            if phrase not in seen:
                seen.add(phrase)
                out.append(phrase)

    for w in content_words:
        if w not in seen:
            seen.add(w)
            out.append(w)

    return out[:15]


def _llm_compile(
    query: str,
    orchestrator_model: str,
    ollama_url: str,
    searchable: list[str],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict | None:
    """Use the orchestrator LLM to decompose a question into tasks."""
    from core.llm import ollama_chat

    system = (
        "You are a legal research task planner. Given a user question, decompose it "
        "into one or more search tasks. Each task searches ONE jurisdiction for specific legal concepts.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        '{"tasks": [{"jurisdiction": "...", "concepts": ["..."], "query": "...", '
        '"title_hints": ["..."]}]}\n\n'
        f"Available indexed jurisdictions: {', '.join(searchable) if searchable else 'none'}.\n"
        "If the user mentions jurisdictions not in the available list, still include them.\n"
        "If the user asks about 'every state' or 'all states', create one task per available "
        "jurisdiction only.\n"
        "\n"
        "CRITICAL RULES for the query field:\n"
        "  1. Preserve EVERY explicit legal reference from the user verbatim: e.g. "
        "'Title 35', 'Title 35: Property', 'Chapter 4', 'Section 35-11-10', '§ 5-19-1'.\n"
        "  2. Preserve the exact fee / concept names the user wrote (e.g. 'Lien Release Fees',\n"
        "     'Pay Off Statement Fees', 'Payoff Fees').\n"
        "  3. Do NOT rewrite the question into your own wording if that removes any of the\n"
        "     above references. When in doubt, include the user's phrase exactly.\n"
        "\n"
        "title_hints: list just the bare Title numbers the user referenced "
        "(e.g. ['35', '35A']); empty list if none.\n"
        "concepts: 3-8 short phrases (keep multi-word phrases intact, e.g. 'lien release fees').\n"
    )

    result = ollama_chat(
        model=orchestrator_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        ollama_url=ollama_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    if not result.get("ok"):
        return None

    content = result.get("content", "")
    thinking = result.get("thinking")

    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group())
        parsed["_thinking"] = thinking
        parsed["_raw"] = content
        return parsed
    except json.JSONDecodeError:
        return None


def compile_ask_request(
    config,
    user_query: str,
    corpus_family: str | None = None,
    jurisdiction_or_issuer: str | None = None,
    orchestrator_model: str | None = None,
    searchable_jurisdictions: list[str] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Parse a user question into a structured task list.

    Returns:
      {
        "tasks": [{"jurisdiction": ..., "concepts": [...], "query": ...}],
        "unavailable_scopes": [...],
        "corpus_family": str,
        "llm_used": bool,
        "llm_thinking": str | None,
        "llm_raw": str | None,
      }
    """
    searchable = searchable_jurisdictions or []
    ollama_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    corpus = corpus_family or "state_requirements"
    raw_query = (user_query or "").strip()

    structural = extract_structural_refs(raw_query)
    rule_concepts = _extract_concepts(raw_query)

    llm_result = None
    llm_thinking = None
    llm_raw = None

    if orchestrator_model:
        llm_result = _llm_compile(raw_query, orchestrator_model, ollama_url, searchable,
                                  temperature=temperature, max_tokens=max_tokens,
                                  top_p=top_p, top_k=top_k)

    tasks: list[dict] = []
    unavailable: list[str] = []

    def _merge_unique(*lists) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for lst in lists:
            for item in lst or []:
                key = (item or "").strip()
                low = key.lower()
                if key and low not in seen:
                    seen.add(low)
                    out.append(key)
        return out

    def _augmented_query(llm_query: str | None) -> str:
        """Always anchor retrieval on the raw user query; append any explicit
        Title/Chapter/Section tokens the LLM may have dropped so Chroma/SQL
        routing can key on them."""
        base = (llm_query or "").strip() or raw_query
        missing: list[str] = []
        low = base.lower()
        for t in structural["titles"]:
            token = f"title {t}".lower()
            if token not in low:
                missing.append(f"Title {t}")
        for s in structural["sections"]:
            if s.lower() not in low:
                missing.append(f"Section {s}")
        for c in structural["chapters"]:
            token = f"chapter {c}".lower()
            if token not in low:
                missing.append(f"Chapter {c}")
        # Always fold raw query back in so user phrasing is preserved for embeddings/SQL.
        if raw_query and raw_query.lower() not in low:
            base = f"{raw_query} {base}"
        if missing:
            base = f"{base} {' '.join(missing)}"
        return base.strip()

    if llm_result and "tasks" in llm_result:
        llm_thinking = llm_result.get("_thinking")
        llm_raw = llm_result.get("_raw")
        for t in llm_result["tasks"]:
            jur = (t.get("jurisdiction") or "").strip().lower()
            if jur and jur not in searchable:
                unavailable.append(jur)
                continue

            merged_concepts = _merge_unique(
                t.get("concepts", []),
                rule_concepts,
            )
            merged_title_hints = _merge_unique(
                t.get("title_hints", []),
                structural["titles"],
            )
            tasks.append({
                "jurisdiction": jur or (searchable[0] if searchable else ""),
                "concepts": merged_concepts,
                "query": _augmented_query(t.get("query")),
                "raw_query": raw_query,
                "title_hints": merged_title_hints,
                "section_hints": list(structural["sections"]),
                "chapter_hints": list(structural["chapters"]),
                "corpus_family": corpus,
            })
    else:
        extracted_jurs = _extract_jurisdictions(raw_query)

        def _make_task(jur: str) -> dict:
            return {
                "jurisdiction": jur,
                "concepts": rule_concepts,
                "query": _augmented_query(None),
                "raw_query": raw_query,
                "title_hints": list(structural["titles"]),
                "section_hints": list(structural["sections"]),
                "chapter_hints": list(structural["chapters"]),
                "corpus_family": corpus,
            }

        if jurisdiction_or_issuer:
            jur = jurisdiction_or_issuer.strip().lower()
            if jur in searchable:
                tasks.append(_make_task(jur))
            else:
                unavailable.append(jur)
        elif extracted_jurs:
            for jur in extracted_jurs:
                if jur in searchable:
                    tasks.append(_make_task(jur))
                else:
                    unavailable.append(jur)
        elif searchable:
            tasks.append(_make_task(searchable[0]))

    return {
        "tasks": tasks,
        "unavailable_scopes": sorted(set(unavailable)),
        "corpus_family": corpus,
        "raw_query": raw_query,
        "structural_refs": structural,
        "llm_used": llm_result is not None,
        "llm_thinking": llm_thinking,
        "llm_raw": llm_raw,
    }
