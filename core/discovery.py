from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def classify_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".md", ".markdown"}:
        return "markdown"
    if ext in {".txt", ".text", ".log"}:
        return "text"
    if ext == ".docx":
        return "docx"
    if ext in {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sh",
        ".ps1",
        ".sql",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".xml",
        ".html",
        ".css",
    }:
        return "code"
    return "other"


def should_ignore(path: str, root_path: str, ignore_globs: list[str] | None = None) -> bool:
    """
    Decide whether to ignore `path` during discovery.

    - Always ignores common junk dirs.
    - Applies caller-provided ignore globs against:
      - posix-style relative path (e.g. foo/bar.txt)
      - posix-style absolute path
    """
    ignore_globs = ignore_globs or []
    root = Path(root_path)
    p = Path(path)

    # Quick ignore for common junk/hidden roots.
    junk_names = {".git", "node_modules", "__pycache__", ".venv", ".idea"}
    parts = set(p.parts)
    if parts & junk_names:
        return True

    try:
        rel = p.resolve().relative_to(root.resolve())
        rel_posix = rel.as_posix()
    except Exception:
        rel_posix = p.name

    abs_posix = p.resolve().as_posix() if p.exists() else p.absolute().as_posix()

    for g in ignore_globs:
        gg = (g or "").strip()
        if not gg:
            continue
        if fnmatch.fnmatch(rel_posix, gg) or fnmatch.fnmatch(abs_posix, gg):
            return True

    return False


def discover_files(root_path: str, ignore_globs: list[str] | None = None) -> list[dict]:
    """
    Recursively discover files under root_path.

    Returns list of:
      { path, rel_path, size_bytes, mtime_ns, file_type }
    """
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        return []

    results: list[dict] = []
    ignore_globs = ignore_globs or []

    # Use os.walk for speed and to prune ignored dirs.
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune junk dirs early.
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__", ".venv", ".idea"}]

        # Apply ignore globs to directories (so we don't descend).
        pruned: list[str] = []
        for d in list(dirnames):
            full = Path(dirpath) / d
            if should_ignore(str(full), str(root), ignore_globs=ignore_globs):
                pruned.append(d)
        for d in pruned:
            if d in dirnames:
                dirnames.remove(d)

        for fn in filenames:
            full_path = Path(dirpath) / fn
            if should_ignore(str(full_path), str(root), ignore_globs=ignore_globs):
                continue
            try:
                st = full_path.stat()
                rel_path = full_path.resolve().relative_to(root.resolve()).as_posix()
            except Exception:
                continue

            results.append(
                {
                    "path": os.path.normpath(str(full_path.resolve())),
                    "rel_path": rel_path,
                    "size_bytes": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                    "file_type": classify_file(str(full_path)),
                }
            )

    return results

