from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    OLLAMA_BASE_URL: str
    APP_DATA_DIR: str
    DEFAULT_CHAT_MODEL: str
    DEFAULT_EMBED_MODEL: str
    TRACE_DIR: str
    IGNORE_GLOBS: list[str]
    CHROMA_PERSIST_DIR: str
    CATALOG_COLLECTION_PREFIX: str
    TITLE_COLLECTION_PREFIX: str
    STATES_ROOT: str


def _default_app_data_dir() -> str:
    # Keep defaults deterministic and user-writable on Windows.
    return str(Path.home() / ".local_deep_research")


def load_config() -> AppConfig:
    """
    Load app configuration.

    Defaults can be overridden by environment variables:
    - OLLAMA_BASE_URL
    - APP_DATA_DIR
    - DEFAULT_CHAT_MODEL
    - DEFAULT_EMBED_MODEL
    - TRACE_DIR
    """
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    app_data_dir = os.getenv("APP_DATA_DIR", _default_app_data_dir()).strip()
    default_chat_model = os.getenv("DEFAULT_CHAT_MODEL", "qwen2.5").strip()
    default_embed_model = os.getenv("DEFAULT_EMBED_MODEL", "nomic-embed-text-v2-moe:latest").strip()

    trace_dir_env = os.getenv("TRACE_DIR")
    if trace_dir_env and trace_dir_env.strip():
        trace_dir = trace_dir_env.strip()
    else:
        trace_dir = str(Path(app_data_dir) / "traces")

    ignore_globs_env = os.getenv("IGNORE_GLOBS", "").strip()
    if ignore_globs_env:
        ignore_globs = [g.strip() for g in ignore_globs_env.split(",") if g.strip()]
    else:
        ignore_globs = [
            "**/.git/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/.idea/**",
        ]

    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "").strip()
    if not chroma_persist_dir:
        chroma_persist_dir = str(Path(app_data_dir) / "chroma")
    catalog_collection_prefix = os.getenv("CATALOG_COLLECTION_PREFIX", "catalog_").strip()
    title_collection_prefix = os.getenv("TITLE_COLLECTION_PREFIX", "title_").strip()

    default_states_root = str(Path(__file__).resolve().parent.parent / "Laws_Regulations" / "States")
    states_root = os.getenv("STATES_ROOT", "").strip() or default_states_root

    return AppConfig(
        OLLAMA_BASE_URL=ollama_base_url,
        APP_DATA_DIR=app_data_dir,
        DEFAULT_CHAT_MODEL=default_chat_model,
        DEFAULT_EMBED_MODEL=default_embed_model,
        TRACE_DIR=trace_dir,
        IGNORE_GLOBS=ignore_globs,
        CHROMA_PERSIST_DIR=chroma_persist_dir,
        CATALOG_COLLECTION_PREFIX=catalog_collection_prefix,
        TITLE_COLLECTION_PREFIX=title_collection_prefix,
        STATES_ROOT=states_root,
    )
