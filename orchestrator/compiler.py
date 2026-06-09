"""
Rule-based prompt compiler.

Parses a user prompt into a structured RunSpec without LLM dependence.
"""
from __future__ import annotations

import re

from orchestrator.run_spec import QueueItemSpec, RunSpec

US_STATES: list[str] = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

_STATES_LOWER = {s.lower(): s for s in US_STATES}

_GSE_ISSUERS = {
    "fannie mae": "Fannie Mae",
    "freddie mac": "Freddie Mac",
    "ginnie mae": "Ginnie Mae",
    "fha": "FHA",
    "va": "VA",
}

_FEDERAL_KEYWORDS = [
    "cfr", "usc", "federal register", "federal regulation",
    "respa", "tila", "ecoa", "hmda", "dodd-frank", "reg x", "reg z",
]

_ALL_STATES_PATTERNS = [
    r"\bevery\s+state",
    r"\ball\s+states",
    r"\ball\s+50\s+states",
    r"\beach\s+state",
    r"\bnationwide\b",
    r"\bstate[\s-]+by[\s-]+state\b",
]


def _extract_concepts(prompt: str) -> list[str]:
    """Pull likely concept phrases from the prompt."""
    stop = {
        "the", "a", "an", "for", "of", "in", "on", "to", "and", "or",
        "is", "are", "what", "how", "which", "every", "all", "each",
        "state", "states", "law", "laws", "requirement", "requirements",
        "regulation", "regulations", "statute", "statutes",
    }
    words = re.findall(r"[a-z][a-z'-]+", prompt.lower())
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w not in stop and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _detect_all_states(prompt: str) -> bool:
    low = prompt.lower()
    return any(re.search(p, low) for p in _ALL_STATES_PATTERNS)


def _detect_jurisdictions(prompt: str) -> list[str]:
    low = prompt.lower()
    found: list[str] = []
    for key, canonical in _STATES_LOWER.items():
        if re.search(r"\b" + re.escape(key) + r"\b", low):
            found.append(canonical)
    return sorted(set(found))


def _detect_gse_issuer(prompt: str) -> str | None:
    low = prompt.lower()
    for key, canonical in _GSE_ISSUERS.items():
        if key in low:
            return canonical
    return None


def _detect_federal(prompt: str) -> bool:
    low = prompt.lower()
    return any(kw in low for kw in _FEDERAL_KEYWORDS)


def compile_user_prompt(config, prompt: str) -> dict:
    """
    Parse a user prompt into a RunSpec dict.
    Returns the spec as a plain dict (JSON-serializable).
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return RunSpec(
            original_prompt=prompt,
            clarification_needed=True,
            clarification_questions=["Please enter a research question or prompt."],
        ).to_dict()

    all_states = _detect_all_states(prompt)
    jurisdictions = _detect_jurisdictions(prompt)
    gse_issuer = _detect_gse_issuer(prompt)
    is_federal = _detect_federal(prompt)
    concepts = _extract_concepts(prompt)

    corpus_family = ""
    issuers_or_jurisdictions: list[str] = []
    run_mode = "single"
    clarification_needed = False
    clarification_questions: list[str] = []

    if gse_issuer:
        corpus_family = "gse_requirements"
        issuers_or_jurisdictions = [gse_issuer]
    elif is_federal:
        corpus_family = "federal_requirements"
        issuers_or_jurisdictions = ["federal"]
    elif all_states:
        corpus_family = "state_requirements"
        issuers_or_jurisdictions = list(US_STATES)
        run_mode = "multi_state"
    elif jurisdictions:
        corpus_family = "state_requirements"
        issuers_or_jurisdictions = jurisdictions
        if len(jurisdictions) > 1:
            run_mode = "multi_state"
    else:
        clarification_needed = True
        clarification_questions.append(
            "Could not determine the jurisdiction, issuer, or corpus family. "
            "Please specify a state (e.g. Texas), issuer (e.g. Fannie Mae), "
            "or indicate federal/all-states scope."
        )

    if not concepts and not clarification_needed:
        clarification_needed = True
        clarification_questions.append(
            "No specific legal concepts or keywords detected. "
            "Please add terms like 'payoff fees', 'lien release', etc."
        )

    spec = RunSpec(
        original_prompt=prompt,
        run_mode=run_mode,
        corpus_family=corpus_family,
        clarification_needed=clarification_needed,
        clarification_questions=clarification_questions,
        jurisdictions_or_issuers=issuers_or_jurisdictions,
        concepts=concepts,
        aliases=[],
        routing_strategy="catalog_first",
        answer_contract="evidence_with_citations",
    )

    spec.queue_items = _build_queue_item_specs(spec)
    return spec.to_dict()


def _build_queue_item_specs(spec: RunSpec) -> list[QueueItemSpec]:
    items: list[QueueItemSpec] = []
    for i, jur in enumerate(spec.jurisdictions_or_issuers):
        items.append(QueueItemSpec(
            corpus_family=spec.corpus_family,
            jurisdiction_or_issuer=jur,
            scope_key=jur.lower().replace(" ", "_"),
            queue_order=i,
        ))
    return items


def build_run_queue(spec: dict) -> list[dict]:
    """
    Turn a RunSpec dict into a list of queue item dicts ready for SQL insertion.
    """
    items = spec.get("queue_items", [])
    queue: list[dict] = []
    for i, item in enumerate(items):
        queue.append({
            "queue_order": item.get("queue_order", i),
            "corpus_family": item.get("corpus_family", spec.get("corpus_family", "")),
            "jurisdiction_or_issuer": item.get("jurisdiction_or_issuer", ""),
            "scope_key": item.get("scope_key", ""),
            "status": "pending",
        })
    return queue
