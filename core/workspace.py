from __future__ import annotations

import uuid
from pathlib import Path
from datetime import datetime, timezone


def ensure_app_dirs(config) -> None:
    """
    Ensure required application directories exist.

    Creates:
    - APP_DATA_DIR
    - APP_DATA_DIR/logs
    - TRACE_DIR
    """
    app_data_dir = Path(getattr(config, "APP_DATA_DIR"))
    trace_dir = Path(getattr(config, "TRACE_DIR"))

    app_data_dir.mkdir(parents=True, exist_ok=True)
    (app_data_dir / "logs").mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_workspace(config, name: str, description: str = "") -> dict:
    from core.db import connect_db

    ws_id = uuid.uuid4().hex
    ts = _utc_now_iso()
    name_clean = (name or "").strip()
    desc_clean = (description or "").strip()
    if not name_clean:
        return {"ok": False, "error": "Workspace name is required"}

    try:
        with connect_db(config) as conn:
            conn.execute(
                """
                INSERT INTO workspaces (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ws_id, name_clean, desc_clean, ts, ts),
            )
            conn.commit()
        return {
            "ok": True,
            "workspace": {
                "id": ws_id,
                "name": name_clean,
                "description": desc_clean,
                "created_at": ts,
                "updated_at": ts,
            },
        }
    except Exception as e:
        msg = str(e)
        if "UNIQUE" in msg.upper():
            return {"ok": False, "error": "Workspace name already exists"}
        return {"ok": False, "error": "Failed to create workspace"}


def list_workspaces(config) -> list[dict]:
    from core.db import connect_db

    with connect_db(config) as conn:
        rows = conn.execute(
            "SELECT id, name, description, created_at, updated_at FROM workspaces ORDER BY updated_at DESC, name ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_workspace(config, workspace_id: str) -> dict | None:
    from core.db import connect_db

    ws_id = (workspace_id or "").strip()
    if not ws_id:
        return None
    with connect_db(config) as conn:
        row = conn.execute(
            "SELECT id, name, description, created_at, updated_at FROM workspaces WHERE id = ?",
            (ws_id,),
        ).fetchone()
    return dict(row) if row else None
