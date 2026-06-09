"""
PDF text extraction and legal section parsing.

Extracts per-page text for page-number tracking, and uses multiple
regex strategies to split into legal sections.
"""
from __future__ import annotations

import re


def extract_pdf_text(path: str) -> dict:
    """
    Extract text from a PDF with per-page tracking.

    Returns:
        ok, text, page_count, extraction_mode, error, pages
        pages is a list of {"page": int, "text": str, "char_offset": int}
    """
    try:
        import fitz
    except ImportError:
        return {"ok": False, "text": "", "page_count": 0, "extraction_mode": "none",
                "error": "PyMuPDF (fitz) is not installed.", "pages": []}

    try:
        doc = fitz.open(path)
        pages: list[dict] = []
        full_parts: list[str] = []
        offset = 0
        for i, page in enumerate(doc):
            t = page.get_text("text") or ""
            pages.append({"page": i, "text": t, "char_offset": offset})
            full_parts.append(t)
            offset += len(t) + 1
        full_text = "\n".join(full_parts)
        page_count = len(doc)
        doc.close()
        return {"ok": True, "text": full_text, "page_count": page_count,
                "extraction_mode": "fitz_text", "error": None, "pages": pages}
    except Exception as exc:
        return {"ok": False, "text": "", "page_count": 0,
                "extraction_mode": "fitz_text", "error": str(exc), "pages": []}


def _char_offset_to_page(offset: int, pages: list[dict]) -> int:
    """Map a character offset in the full text to a 0-based page number."""
    for i in range(len(pages) - 1, -1, -1):
        if offset >= pages[i]["char_offset"]:
            return pages[i]["page"]
    return 0


# Primary pattern: Section 1-1-1 / Sec. 1-1-1 / § 1-1-1
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:Section|Sec\.?|§)\s*:?\s*(?P<number>[0-9]+(?:[-–.][0-9A-Za-z]+)*)",
    re.IGNORECASE | re.MULTILINE,
)

# Fallback: §1-1-1 (no space), or numbered-dash patterns at line start
_SECTION_FALLBACK_RE = re.compile(
    r"^\s*§(?P<number>[0-9]+(?:[-–.][0-9A-Za-z]+)+)"
    r"|^\s*(?P<number2>[0-9]{1,3}[-–][0-9]{1,4}[-–][0-9A-Za-z]+)",
    re.MULTILINE,
)

_CATCHLINE_RE = re.compile(
    r"(?:Catchline|Short\s*title)\s*:?\s*(?P<text>.+)",
    re.IGNORECASE,
)

_HISTORY_RE = re.compile(
    r"(?:History|Source|Acts?)\s*:?\s*(?P<text>.+)",
    re.IGNORECASE,
)

_LARGE_PAGE_THRESHOLD = 50


def parse_legal_sections(text: str, pages: list[dict] | None = None) -> list[dict]:
    """
    Split extracted text into legal sections with page tracking.

    Uses primary regex first; if too few matches on a large document,
    falls back to aggressive secondary patterns. Never returns a single
    monolithic section for a large document.
    """
    if not text or not text.strip():
        return []

    hits = list(_SECTION_HEADER_RE.finditer(text))

    page_count = len(pages) if pages else 0

    if len(hits) < 3 and page_count >= _LARGE_PAGE_THRESHOLD:
        fallback_hits = list(_SECTION_FALLBACK_RE.finditer(text))
        if len(fallback_hits) > len(hits):
            hits = fallback_hits

    if not hits:
        max_section_chars = 80000
        if len(text) <= max_section_chars:
            return [{"section_number": None, "catchline": None, "history": None,
                      "body_text": text.strip(), "section_order": 0,
                      "page_start": 0, "page_end": (page_count - 1) if page_count else None}]
        chunks: list[dict] = []
        paragraphs = re.split(r"\n{2,}", text)
        buf: list[str] = []
        buf_len = 0
        idx = 0
        for para in paragraphs:
            if buf and buf_len + len(para) > max_section_chars:
                body = "\n\n".join(buf)
                ps = _char_offset_to_page(text.index(buf[0]) if buf[0] in text else 0, pages or []) if pages else None
                chunks.append({"section_number": None, "catchline": None, "history": None,
                               "body_text": body, "section_order": idx,
                               "page_start": ps, "page_end": ps})
                buf = [para]
                buf_len = len(para)
                idx += 1
            else:
                buf.append(para)
                buf_len += len(para)
        if buf:
            body = "\n\n".join(buf)
            ps = _char_offset_to_page(text.index(buf[0]) if buf[0] in text else 0, pages or []) if pages else None
            chunks.append({"section_number": None, "catchline": None, "history": None,
                           "body_text": body, "section_order": idx,
                           "page_start": ps, "page_end": ps})
        return chunks

    sections: list[dict] = []
    for idx, match in enumerate(hits):
        start = match.start()
        end = hits[idx + 1].start() if idx + 1 < len(hits) else len(text)
        block = text[start:end]

        section_number = match.group("number") or (match.group("number2") if hasattr(match, "group") and match.lastgroup != "number" else None)
        if not section_number:
            try:
                section_number = match.group("number2")
            except IndexError:
                section_number = None

        catchline = None
        cm = _CATCHLINE_RE.search(block)
        if cm:
            catchline = cm.group("text").strip()
        else:
            first_line = block.split("\n", 1)[0]
            after_num = re.sub(r"^\s*(?:Section|Sec\.?|§)\s*:?\s*[\d\-–.A-Za-z]+\.?\s*", "", first_line, flags=re.IGNORECASE).strip()
            if after_num and len(after_num) < 120 and not after_num[0].isdigit():
                catchline = after_num.rstrip(".")

        history = None
        hm = _HISTORY_RE.search(block)
        if hm:
            history = hm.group("text").strip()

        body_lines: list[str] = []
        for line in block.splitlines():
            if _SECTION_HEADER_RE.match(line) or _SECTION_FALLBACK_RE.match(line):
                continue
            if _CATCHLINE_RE.match(line):
                continue
            if _HISTORY_RE.match(line):
                continue
            body_lines.append(line)
        body_text = "\n".join(body_lines).strip() or block.strip()

        page_start = _char_offset_to_page(start, pages) if pages else None
        page_end = _char_offset_to_page(end - 1, pages) if pages else None

        sections.append({
            "section_number": section_number,
            "catchline": catchline,
            "history": history,
            "body_text": body_text,
            "section_order": idx,
            "page_start": page_start,
            "page_end": page_end,
        })

    return sections
