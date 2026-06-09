"""
Embedding via Ollama /api/embed.

Uses conservative per-text truncation and small batches. On HTTP 400 (context
length), falls back to single-item requests with progressively smaller caps.
"""
from __future__ import annotations

import logging
import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v2-moe:latest"
BATCH_SIZE = 8
MAX_EMBED_CHARS = 8192
_FALLBACK_LIMITS = (8192, 4096, 2048, 1024)


def _post_embed(url: str, model: str, inputs: list[str]) -> list[list[float]]:
    resp = requests.post(
        url,
        json={"model": model, "input": inputs},
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    embs = data.get("embeddings", [])
    if len(embs) != len(inputs):
        raise RuntimeError(f"Expected {len(inputs)} embeddings, got {len(embs)}")
    return embs


def _embed_one_text(url: str, model: str, text: str) -> list[float]:
    """Embed a single string; shrink on context errors."""
    t = text or ""
    last_err = ""
    for lim in _FALLBACK_LIMITS:
        chunk = t[:lim] if lim else t
        if not chunk.strip():
            raise RuntimeError("empty text after truncation")
        try:
            vecs = _post_embed(url, model, [chunk])
            return vecs[0]
        except RuntimeError as e:
            last_err = str(e)
            if "400" not in last_err and "context" not in last_err.lower():
                raise
            logger.debug("embed retry lim=%d: %s", lim, last_err[:120])
            continue
    raise RuntimeError(last_err or "embed failed after truncation")


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[list[float]]:
    """
    Embed texts. Each string is capped at MAX_EMBED_CHARS before batching.
    If a batch fails with context length, falls back to per-text embedding.
    """
    if not texts:
        return []
    if not model or not model.strip():
        raise ValueError("embed model name is required")

    url = f"{ollama_url.rstrip('/')}/api/embed"
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_raw = texts[start : start + BATCH_SIZE]
        batch = [(t or "")[:MAX_EMBED_CHARS] for t in batch_raw]
        try:
            all_embeddings.extend(_post_embed(url, model, batch))
        except RuntimeError as e:
            err_s = str(e)
            if "400" not in err_s and "context" not in err_s.lower():
                raise
            logger.warning("embed batch failed (%s), falling back to single-text mode", err_s[:100])
            for t in batch_raw:
                all_embeddings.append(_embed_one_text(url, model, t))

    return all_embeddings
