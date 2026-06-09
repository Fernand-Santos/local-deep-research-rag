from __future__ import annotations

import re
from pathlib import Path


def parse_path_metadata(path: str) -> dict:
    """
    Derive corpus / jurisdiction metadata from folder hierarchy.

    Recognises patterns such as:
        Laws_Regulations/States/Alabama/...
        Laws_Regulations/Federal/...
    """
    parts = Path(path).parts

    result: dict = {
        "corpus": None,
        "jurisdiction_type": None,
        "jurisdiction": None,
    }

    normalised = [p.lower().replace(" ", "_") for p in parts]

    for i, token in enumerate(normalised):
        if token in {"laws_regulations", "laws", "regulations", "statutes", "codes"}:
            result["corpus"] = parts[i]
            remaining = normalised[i + 1 :]
            remaining_raw = parts[i + 1 :]

            if remaining and remaining[0] in {"states", "state"}:
                result["jurisdiction_type"] = "state"
                if len(remaining) > 1:
                    result["jurisdiction"] = remaining_raw[1]
            elif remaining and remaining[0] in {"federal", "fed"}:
                result["jurisdiction_type"] = "federal"
                result["jurisdiction"] = "Federal"
            elif remaining:
                result["jurisdiction_type"] = "unknown"
                result["jurisdiction"] = remaining_raw[0]
            break

    return result


_TITLE_RE = re.compile(
    r"^(?P<jurisdiction>[A-Za-z_]+?)_Title_(?P<number>[0-9A-Za-z]+)_(?P<name>.+)$",
    re.IGNORECASE,
)

_CHAPTER_RE = re.compile(
    r"^(?P<jurisdiction>[A-Za-z_]+?)_Chapter_(?P<number>[0-9A-Za-z]+)_(?P<name>.+)$",
    re.IGNORECASE,
)


def parse_filename_metadata(filename: str) -> dict:
    """
    Derive document_family / title / chapter info from filename stem.

    Handles patterns like:
        Alabama_Title_1_General_Provisions.pdf
        Alabama_Chapter_5_Notices.pdf
    """
    stem = Path(filename).stem

    result: dict = {
        "document_family": None,
        "title_number": None,
        "title_name": None,
    }

    m = _TITLE_RE.match(stem)
    if m:
        result["document_family"] = "title"
        result["title_number"] = m.group("number")
        result["title_name"] = m.group("name").replace("_", " ")
        return result

    m = _CHAPTER_RE.match(stem)
    if m:
        result["document_family"] = "chapter"
        result["title_number"] = m.group("number")
        result["title_name"] = m.group("name").replace("_", " ")
        return result

    # Fallback: use the stem as title_name.
    result["title_name"] = stem.replace("_", " ")
    return result
