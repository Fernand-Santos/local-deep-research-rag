"""
Automatic state-folder discovery under a canonical States root directory.

Scans immediate subdirectories to identify state jurisdictions and their PDFs.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def normalize_state_folder_name(name: str) -> str:
    """Convert a folder name like 'New_York' to 'new york'."""
    return re.sub(r"[_\-]+", " ", name).strip().lower()


def discover_state_folders(states_root: str) -> list[dict]:
    """
    Scan immediate child directories under *states_root* and return metadata
    for each state folder found.

    Returns a sorted list of dicts:
      {
        "folder_name": str,          # raw folder name
        "jurisdiction": str,          # normalized lowercase
        "path": str,                  # absolute path
        "pdf_count": int,
        "pdf_files": list[str],       # basenames
      }
    """
    root = Path(states_root)
    if not root.is_dir():
        return []

    results: list[dict] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue

        pdfs = [f.name for f in entry.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]

        results.append({
            "folder_name": entry.name,
            "jurisdiction": normalize_state_folder_name(entry.name),
            "path": str(entry),
            "pdf_count": len(pdfs),
            "pdf_files": sorted(pdfs),
        })

    return results
