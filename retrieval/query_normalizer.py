"""Query / concept normalization helpers with expanded legal domain aliases.

Key design decisions:
- Multi-word concepts (e.g. "lien release") are kept intact so the
  embedding model sees them as phrases, not independent words.
- build_query_text prioritises the original user query and only
  appends concepts that add genuinely new terms.
"""
from __future__ import annotations

import re

_ALIASES: dict[str, list[str]] = {
    "general provisions": ["definitions", "construction of statutes", "general provisions", "interpretation"],
    "agriculture": ["farming", "crops", "livestock", "agricultural", "ag law"],
    "animals": ["animal control", "livestock", "pets", "wildlife", "animal cruelty"],
    "aviation": ["aircraft", "airports", "aeronautics", "airspace"],
    "payoff fees": ["payoff statement", "payoff statement fees", "payoff request", "payoff demand",
                    "loan payoff", "mortgage payoff", "payoff amount"],
    "lien release": ["lien release fees", "satisfaction of mortgage", "release of lien",
                     "lien satisfaction", "discharge of lien", "mortgage discharge"],
    "mortgage": ["mortgages", "deed of trust", "mortgage loan", "home loan",
                 "residential mortgage", "mortgage lending"],
    "foreclosure": ["foreclosure process", "foreclosure sale", "judicial foreclosure",
                    "non-judicial foreclosure", "foreclosure proceedings"],
    "real property": ["real estate", "land", "property law", "conveyance", "deed",
                      "title transfer", "property transfer"],
    "title insurance": ["title search", "title examination", "title defect", "title policy"],
    "escrow": ["escrow account", "escrow agent", "escrow funds", "impound account"],
    "closing": ["settlement", "closing costs", "closing disclosure", "settlement statement"],
    "recording": ["recording fees", "recordation", "document recording", "county recorder"],
    "interest rates": ["interest rate", "usury", "maximum rate", "annual percentage rate", "apr"],
    "licensing": ["license", "licensure", "licensed", "license requirement", "licensing board"],
    "notary": ["notarization", "notary public", "notarial act", "acknowledgment"],
    "power of attorney": ["poa", "attorney in fact", "durable power of attorney"],
    "consumer protection": ["consumer rights", "consumer law", "unfair practices",
                            "deceptive practices", "consumer fraud"],
    "banking": ["banks", "financial institutions", "banking law", "depository institution"],
    "insurance": ["insurance law", "insurance regulation", "insurer", "insurance policy"],
    "taxes": ["taxation", "tax law", "property tax", "tax assessment", "tax lien"],
    "fees": ["fee schedule", "service fees", "processing fees", "charges", "costs"],
}


def normalize_concepts(concepts: list[str]) -> list[str]:
    """Lowercase, strip, deduplicate while preserving multi-word phrases."""
    seen: set[str] = set()
    out: list[str] = []
    for c in concepts:
        key = c.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def expand_basic_aliases(concepts: list[str]) -> list[str]:
    """Add common aliases for known concept terms.

    Matching is done both on the whole concept string and on
    individual words so that "lien release fees" triggers "lien release" aliases.
    """
    expanded = list(concepts)
    matched_canonicals: set[str] = set()

    for c in concepts:
        cl = c.lower().strip()
        for canonical, aliases in _ALIASES.items():
            if canonical in matched_canonicals:
                continue
            if cl == canonical or cl in aliases or canonical in cl:
                matched_canonicals.add(canonical)
                for a in aliases:
                    if a not in expanded:
                        expanded.append(a)
                if canonical not in expanded:
                    expanded.append(canonical)

    return expanded


def build_query_text(query: str, concepts: list[str]) -> str:
    """Combine query and concepts into a single embedding-friendly string.

    The original query is always placed first so the embedding model
    captures its full semantic meaning.  Only concepts that introduce
    genuinely new words are appended to avoid diluting the signal.
    """
    base = query.strip()
    if not base:
        return " ".join(c.strip() for c in concepts if c.strip())

    base_lower_words = set(re.findall(r"[a-z]{3,}", base.lower()))
    additions: list[str] = []
    for c in concepts:
        c = c.strip()
        if not c:
            continue
        c_words = set(re.findall(r"[a-z]{3,}", c.lower()))
        if c_words - base_lower_words:
            additions.append(c)
            base_lower_words.update(c_words)

    if additions:
        return base + " " + " ".join(additions)
    return base
