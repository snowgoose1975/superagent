"""Best-effort append-only lifecycle events for a SuperAgent run session."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_session_event(
    path: str | Path,
    *,
    session_id: str,
    event_type: str,
    stage: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one non-authoritative event without interrupting the ML run.

    Detailed trial, API, and budget records remain in their existing files. This
    stream only provides a compact chronological view of session progress.
    """
    event = {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "session_id": session_id,
        "event_type": event_type,
        "stage": stage,
        "status": status,
        "details": details or {},
    }
    event_path = Path(path)
    try:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    except (OSError, TypeError, ValueError) as exc:
        # Lifecycle telemetry is supplementary; a logging problem must not
        # change the training result or the authoritative API/budget records.
        print(
            f"[Log] Could not write events.jsonl ({type(exc).__name__}). Continuing the run.",
            file=sys.stderr,
            flush=True,
        )
