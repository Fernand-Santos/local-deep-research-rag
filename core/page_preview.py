"""
PDF page preview renderer using PyMuPDF (fitz).

Renders individual PDF pages as PNG images for cited evidence previews.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path


def render_page_preview(pdf_path: str, page_number: int, dpi: int = 120) -> dict:
    """
    Render a single page from a PDF as a PNG image.

    Args:
        pdf_path: absolute path to the PDF
        page_number: 0-based page index
        dpi: resolution

    Returns:
        {"ok": bool, "image_bytes": bytes | None, "image_b64": str | None,
         "page_number": int, "error": str | None}
    """
    if not Path(pdf_path).is_file():
        return {"ok": False, "image_bytes": None, "image_b64": None,
                "page_number": page_number, "error": f"File not found: {pdf_path}"}

    try:
        import fitz
        doc = fitz.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            doc.close()
            return {"ok": False, "image_bytes": None, "image_b64": None,
                    "page_number": page_number, "error": f"Page {page_number} out of range (0-{len(doc)-1})"}

        page = doc[page_number]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        b64 = base64.b64encode(img_bytes).decode("ascii")
        return {"ok": True, "image_bytes": img_bytes, "image_b64": b64,
                "page_number": page_number, "error": None}
    except ImportError:
        return {"ok": False, "image_bytes": None, "image_b64": None,
                "page_number": page_number, "error": "PyMuPDF (fitz) not installed"}
    except Exception as exc:
        return {"ok": False, "image_bytes": None, "image_b64": None,
                "page_number": page_number, "error": str(exc)[:200]}


def _resolve_source_path(source_path: str, config=None) -> str:
    """Resolve a relative/filename-only source_path using the documents table."""
    if Path(source_path).is_file():
        return source_path
    if config is None:
        return source_path

    try:
        from core.db import connect_db
        with connect_db(config) as conn:
            row = conn.execute(
                "SELECT path FROM documents WHERE path LIKE ? LIMIT 1",
                (f"%{Path(source_path).name}",),
            ).fetchone()
            if row:
                return row[0]
    except Exception:
        pass
    return source_path


def get_evidence_page_previews(evidence_chunks: list[dict], max_previews: int = 3, config=None) -> list[dict]:
    """
    Render page previews for the top cited evidence chunks.

    Falls back to page 0 when page_start is unavailable.
    Resolves relative source paths via the documents table.
    """
    previews: list[dict] = []
    seen: set[str] = set()

    for ev in evidence_chunks[:max_previews * 3]:
        source_path = ev.get("source_path", "")
        if not source_path:
            continue

        resolved = _resolve_source_path(source_path, config)
        page_start = ev.get("page_start")
        page = page_start if (page_start is not None and page_start >= 0) else 0

        key = f"{resolved}:{page}"
        if key in seen:
            continue
        seen.add(key)

        result = render_page_preview(resolved, page)
        if result["ok"]:
            previews.append({
                "source_path": resolved,
                "page_number": page,
                "section_number": ev.get("section_number", ""),
                "catchline": ev.get("catchline", ""),
                "image_b64": result["image_b64"],
            })
            if len(previews) >= max_previews:
                break

    return previews
