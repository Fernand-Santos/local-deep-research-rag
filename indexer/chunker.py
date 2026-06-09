"""
Dynamic chunking for legal document sections.

Small sections are kept intact. Larger sections are split at subsection,
paragraph, or clause boundaries. A hard character ceiling ensures no chunk
exceeds embedding model context limits.
"""
from __future__ import annotations

import re

SMALL_TOKEN_THRESHOLD = 300
MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50
MIN_CHUNK_CHARS = 30
MIN_CHUNK_WORDS = 5
MAX_CHUNK_CHARS = 8000


# ------------------------------------------------------------------
# Dynamic chunk-sizing based on source document size.
#
# Formula (total body characters across all sections of one document):
#   < 50k  chars  -> (max_tokens=384,  overlap=40)   # short statutes, ~<25 pages
#   50k-250k       -> (max_tokens=512,  overlap=50)   # medium titles
#   250k-1M        -> (max_tokens=768,  overlap=75)   # large titles
#   > 1M chars    -> (max_tokens=1024, overlap=100)  # mega-titles
#
# Rationale: larger legal codes benefit from slightly larger chunks so that
# multi-paragraph concepts stay together (fewer chunks, better recall), while
# short documents use tighter chunks for precision. Overlap scales with size
# to preserve cross-chunk continuity without blowing up storage.
# ------------------------------------------------------------------

def compute_dynamic_chunk_params(total_chars: int) -> dict:
    """Return (max_tokens, overlap_tokens, small_threshold) sized to document.

    Safe for small/empty inputs (returns the medium default).
    """
    t = max(0, int(total_chars or 0))
    if t < 50_000:
        return {"max_tokens": 384, "overlap_tokens": 40, "small_threshold": 250, "bucket": "small"}
    if t < 250_000:
        return {"max_tokens": 512, "overlap_tokens": 50, "small_threshold": 300, "bucket": "medium"}
    if t < 1_000_000:
        return {"max_tokens": 768, "overlap_tokens": 75, "small_threshold": 400, "bucket": "large"}
    return {"max_tokens": 1024, "overlap_tokens": 100, "small_threshold": 500, "bucket": "mega"}


def total_body_chars(sections: list[dict]) -> int:
    """Sum of body_text lengths across section rows."""
    return sum(len((s.get("body_text") or "")) for s in sections)

_WORD_RE = re.compile(r"\S+")
_SUBSECTION_RE = re.compile(r"\n(?=\([a-zA-Z0-9]+\)\s)")
_PARAGRAPH_RE = re.compile(r"\n{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.;])\s+")

_ARTIFACT_RE = re.compile(
    r"^(?:Section\s*:?\s*)?[\d\-\.]+\s*$"
    r"|^(?:Title|Chapter|Article|Part)\s+[\d\-\.]+\s*$"
    r"|^[§\d\s\-\.,:;]+$",
    re.IGNORECASE | re.MULTILINE,
)


def is_malformed_chunk(text: str) -> bool:
    """Return True if the chunk is too short, trivial, or a parser artifact."""
    t = (text or "").strip()
    if len(t) < MIN_CHUNK_CHARS:
        return True
    words = _WORD_RE.findall(t)
    if len(words) < MIN_CHUNK_WORDS:
        return True
    alpha_chars = sum(1 for ch in t if ch.isalpha())
    if alpha_chars < 10:
        return True
    if _ARTIFACT_RE.fullmatch(t):
        return True
    return False


def estimate_token_count(text: str) -> int:
    return max(1, int(len(_WORD_RE.findall(text)) * 1.3))


def estimate_chunks_for_sections(
    sections: list[dict],
    chunk_params: dict | None = None,
) -> int:
    """Pre-scan sections to estimate chunk count. Honors dynamic sizing if given."""
    if chunk_params is None:
        chunk_params = compute_dynamic_chunk_params(total_body_chars(sections))
    max_tok = chunk_params["max_tokens"]
    small_thr = chunk_params["small_threshold"]

    total = 0
    for sec in sections:
        body = (sec.get("body_text") or "").strip()
        if not body or is_malformed_chunk(body):
            continue
        tokens = estimate_token_count(body)
        if tokens <= small_thr:
            total += 1
        else:
            total += max(1, tokens // max_tok)
    return max(total, 1)


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split text that exceeds max_chars at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    segments = _SENTENCE_RE.split(text)
    current: list[str] = []
    current_len = 0
    for seg in segments:
        if current and current_len + len(seg) > max_chars:
            pieces.append(" ".join(current))
            current = [seg]
            current_len = len(seg)
        else:
            current.append(seg)
            current_len += len(seg)
    if current:
        pieces.append(" ".join(current))
    if not pieces:
        for i in range(0, len(text), max_chars):
            pieces.append(text[i:i + max_chars])
    final: list[str] = []
    for p in pieces:
        if len(p) > max_chars:
            for i in range(0, len(p), max_chars):
                final.append(p[i:i + max_chars])
        else:
            final.append(p)
    return final


def _split_with_overlap(segments: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Merge small segments into chunks respecting max_tokens, adding overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for seg in segments:
        seg_len = estimate_token_count(seg)
        if current and current_len + seg_len > max_tokens:
            chunks.append("\n".join(current))
            tail: list[str] = []
            tail_len = 0
            for s in reversed(current):
                slen = estimate_token_count(s)
                if tail_len + slen > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_len += slen
            current = tail + [seg]
            current_len = tail_len + seg_len
        else:
            current.append(seg)
            current_len += seg_len

    if current:
        chunks.append("\n".join(current))

    final: list[str] = []
    for ch in chunks:
        if len(ch) > MAX_CHUNK_CHARS:
            final.extend(_hard_split(ch, MAX_CHUNK_CHARS))
        else:
            final.append(ch)
    return final


def choose_chunk_strategy(section_row: dict, small_threshold: int = SMALL_TOKEN_THRESHOLD) -> str:
    body = section_row.get("body_text") or ""
    tokens = estimate_token_count(body)
    if tokens <= small_threshold:
        return "whole_section"
    if _SUBSECTION_RE.search(body):
        return "subsection_split"
    if _PARAGRAPH_RE.search(body):
        return "paragraph_split"
    return "sentence_split"


def chunk_document_section(
    section_row: dict,
    chunk_params: dict | None = None,
) -> list[dict]:
    """Chunk a single document_section row.

    `chunk_params` may be passed from `compute_dynamic_chunk_params(total_chars)`
    to override the module-default sizing for this document.
    """
    body = (section_row.get("body_text") or "").strip()
    if not body:
        return []

    if chunk_params is None:
        chunk_params = {
            "max_tokens": MAX_CHUNK_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "small_threshold": SMALL_TOKEN_THRESHOLD,
        }
    max_tok = chunk_params["max_tokens"]
    overlap = chunk_params["overlap_tokens"]
    small_thr = chunk_params["small_threshold"]

    strategy = choose_chunk_strategy(section_row, small_threshold=small_thr)
    sec_num = section_row.get("section_number")
    catchline = section_row.get("catchline")
    sec_id = section_row.get("id")

    if strategy == "whole_section":
        if is_malformed_chunk(body):
            return []
        if len(body) > MAX_CHUNK_CHARS:
            strategy = "sentence_split"
        else:
            return [{
                "text": body,
                "chunk_index": 0,
                "chunk_strategy": strategy,
                "token_estimate": estimate_token_count(body),
                "section_number": sec_num,
                "catchline": catchline,
                "document_section_id": sec_id,
            }]

    if strategy == "subsection_split":
        segments = [s.strip() for s in _SUBSECTION_RE.split(body) if s.strip()]
    elif strategy == "paragraph_split":
        segments = [s.strip() for s in _PARAGRAPH_RE.split(body) if s.strip()]
    else:
        segments = [s.strip() for s in _SENTENCE_RE.split(body) if s.strip()]

    if not segments:
        segments = [body]

    raw_chunks = _split_with_overlap(segments, max_tok, overlap)

    results: list[dict] = []
    out_idx = 0
    for text in raw_chunks:
        if is_malformed_chunk(text):
            continue
        results.append({
            "text": text,
            "chunk_index": out_idx,
            "chunk_strategy": strategy,
            "token_estimate": estimate_token_count(text),
            "section_number": sec_num,
            "catchline": catchline,
            "document_section_id": sec_id,
        })
        out_idx += 1
    return results
