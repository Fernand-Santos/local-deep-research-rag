"""
Ollama chat wrapper with thinking-output capture.

Two modes:
- ollama_chat(): non-streaming, returns final result (for backend pipelines).
- ollama_chat_stream(): yields incremental tokens (for live UI rendering).

Supports models that emit a `thinking` field (e.g. DeepSeek-R1, Qwen3.5).
Automatically falls back to non-thinking mode if the model rejects `think: true`.
"""
from __future__ import annotations

import json
from typing import Generator

import requests


def _build_options(
    temperature: float,
    max_tokens: int,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    opts: dict = {"temperature": temperature, "num_predict": max_tokens}
    if top_p is not None:
        opts["top_p"] = top_p
    if top_k is not None:
        opts["top_k"] = top_k
    return opts


def ollama_chat(
    model: str,
    messages: list[dict],
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    enable_thinking: bool = True,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Non-streaming call. Returns complete result.

    Returns:
      {"ok": bool, "content": str, "thinking": str | None, "model": str, "error": str | None}
    """
    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": _build_options(temperature, max_tokens, top_p, top_k),
    }
    if enable_thinking:
        payload["think"] = True

    try:
        resp = requests.post(url, json=payload, timeout=300)

        if resp.status_code != 200 and enable_thinking:
            payload.pop("think", None)
            resp = requests.post(url, json=payload, timeout=300)

        if resp.status_code != 200:
            return {"ok": False, "content": "", "thinking": None, "model": model,
                    "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

        data = resp.json()
        msg = data.get("message", {})
        return {"ok": True, "content": msg.get("content", ""), "thinking": msg.get("thinking") or None,
                "model": model, "error": None}
    except requests.exceptions.Timeout:
        return {"ok": False, "content": "", "thinking": None, "model": model, "error": "Timeout (300s)"}
    except Exception as exc:
        return {"ok": False, "content": "", "thinking": None, "model": model, "error": str(exc)[:300]}


def ollama_chat_stream(
    model: str,
    messages: list[dict],
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    enable_thinking: bool = True,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Generator[dict, None, None]:
    """
    Streaming call. Yields incremental token dicts for live UI rendering.

    Each yielded dict:
      {"type": "thinking" | "content" | "done" | "error", "text": str}

    "thinking" chunks arrive first (if the model supports it),
    followed by "content" chunks, then a single "done".
    """
    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": _build_options(temperature, max_tokens, top_p, top_k),
    }
    if enable_thinking:
        payload["think"] = True

    try:
        resp = requests.post(url, json=payload, timeout=300, stream=True)

        if resp.status_code != 200 and enable_thinking:
            payload.pop("think", None)
            resp = requests.post(url, json=payload, timeout=300, stream=True)

        if resp.status_code != 200:
            yield {"type": "error", "text": f"HTTP {resp.status_code}: {resp.text[:300]}"}
            return

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = chunk.get("message", {})
            thinking_tok = msg.get("thinking", "")
            content_tok = msg.get("content", "")

            if thinking_tok:
                yield {"type": "thinking", "text": thinking_tok}
            if content_tok:
                yield {"type": "content", "text": content_tok}

            if chunk.get("done"):
                yield {"type": "done", "text": ""}
                return

    except requests.exceptions.Timeout:
        yield {"type": "error", "text": "Timeout (300s)"}
    except Exception as exc:
        yield {"type": "error", "text": str(exc)[:300]}
