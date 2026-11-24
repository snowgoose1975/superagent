from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evaluation_d.logger import append_session_event


def test_append_session_event_writes_ordered_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sessions" / "session_demo" / "events.jsonl"
    append_session_event(
        path,
        session_id="session_demo",
        event_type="session_started",
        stage="session",
        status="started",
        details={"planner_mode": "offline"},
    )
    append_session_event(
        path,
        session_id="session_demo",
        event_type="session_completed",
        stage="session",
        status="success",
    )

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == [
        "session_started",
        "session_completed",
    ]
    assert all(event["session_id"] == "session_demo" for event in events)
    assert all(event["schema_version"] == "1.0" for event in events)
    assert events[0]["details"] == {"planner_mode": "offline"}
    datetime.fromisoformat(events[0]["timestamp"].replace("Z", "+00:00"))


def test_event_log_write_failure_does_not_raise(tmp_path: Path, capsys) -> None:
    parent_file = tmp_path / "not_a_directory"
    parent_file.write_text("occupied", encoding="utf-8")

    append_session_event(
        parent_file / "events.jsonl",
        session_id="session_demo",
        event_type="session_completed",
        stage="session",
        status="success",
    )

    assert "Could not write events.jsonl" in capsys.readouterr().err
