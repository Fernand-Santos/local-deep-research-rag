from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class TraceEvent:
    ts_utc: str
    run_id: str
    agent_id: str
    event_type: str
    payload: dict


def emit_event(
    run_id: str,
    agent_id: str,
    event_type: str,
    payload: dict,
    trace_dir: str,
) -> str:
    """
    Append one JSONL trace event to <trace_dir>/<run_id>.jsonl.
    Returns the JSONL file path.
    """
    td = Path(trace_dir)
    td.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    evt = TraceEvent(
        ts_utc=ts,
        run_id=run_id,
        agent_id=agent_id,
        event_type=event_type,
        payload=payload if isinstance(payload, dict) else {"payload": payload},
    )

    out_path = td / f"{run_id}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")

    return str(out_path)
