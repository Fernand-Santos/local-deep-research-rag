from __future__ import annotations

from typing import Any

import requests


def check_ollama(base_url: str) -> dict[str, Any]:
    """
    Check Ollama connectivity and list available models.

    Return shape:
      {
        "ok": bool,
        "models": list[str],
        "error": str | None
      }
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return {"ok": False, "models": [], "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        models = [
            m.get("name", "")
            for m in (data.get("models", []) if isinstance(data, dict) else [])
            if isinstance(m, dict) and m.get("name")
        ]
        return {"ok": True, "models": models, "error": None}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "models": [], "error": "Connection failed (is Ollama running?)"}
    except requests.exceptions.Timeout:
        return {"ok": False, "models": [], "error": "Timeout"}
    except Exception:
        # Avoid leaking traceback details into the UI.
        return {"ok": False, "models": [], "error": "Unexpected error"}
