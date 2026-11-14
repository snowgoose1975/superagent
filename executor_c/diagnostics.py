"""Structured, label-free diagnostics for one C worker experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .contracts import publish_json, read_json, write_json

SCHEMA_VERSION = "1.0"
TIMING_FIELDS = ("input", "fit", "predict", "serialize", "reload_check", "total")
_SAFE_TEXT = re.compile(r"[\r\n\t]+")


def _safe_message(value: Any, limit: int = 500) -> str:
    text = _SAFE_TEXT.sub(" ", str(value)).strip()
    return text[:limit]


def experiment_signature(plan: dict[str, Any], input_hashes: dict[str, str] | None) -> str:
    payload = {
        "plan": plan,
        "input_hashes": input_hashes or {},
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_diagnostics(
    *,
    experiment_id: str,
    status: str = "failed",
    phase: str = "input",
    timings_seconds: dict[str, float | None] | None = None,
    model_bytes: int | None = None,
    input_shape: dict[str, list[int] | None] | None = None,
    input_matrix_bytes: int | None = None,
    random_seed: int | None = None,
    effective_iterations: int | None = None,
    warnings: list[dict[str, str]] | None = None,
    warnings_truncated: bool = False,
    failure: dict[str, str] | None = None,
    signature: str | None = None,
) -> dict[str, Any]:
    times = {name: None for name in TIMING_FIELDS}
    for name, value in (timings_seconds or {}).items():
        if name in times and (value is None or math.isfinite(value) and value >= 0):
            times[name] = None if value is None else round(float(value), 6)
    warning_items = []
    for item in (warnings or [])[:20]:
        warning_items.append(
            {
                "category": _safe_message(item.get("category", "Warning"), 100),
                "phase": _safe_message(item.get("phase", "fit"), 40),
                "message": _safe_message(item.get("message", "")),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": status,
        "phase": _safe_message(phase, 40),
        "timings_seconds": times,
        "model_bytes": model_bytes if type(model_bytes) is int and model_bytes >= 0 else None,
        "input_shape": input_shape or {"train": None, "validation": None},
        "input_matrix_bytes": (
            input_matrix_bytes
            if type(input_matrix_bytes) is int and input_matrix_bytes >= 0
            else None
        ),
        "random_seed": random_seed if type(random_seed) is int else None,
        "effective_iterations": (
            effective_iterations
            if type(effective_iterations) is int and effective_iterations >= 0
            else None
        ),
        "warnings": warning_items,
        "warnings_truncated": bool(warnings_truncated or len(warnings or []) > 20),
        "failure": (
            {
                "code": _safe_message(failure.get("code", "UNKNOWN"), 100),
                "message": _safe_message(failure.get("message", "")),
            }
            if failure
            else None
        ),
        "experiment_signature": signature,
    }


def write_worker_snapshot(path: Path, diagnostics: dict[str, Any]) -> None:
    """Atomically refresh the worker's last known phase for timeout recovery."""
    pending = path.with_name(path.name + ".pending")
    content = json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
    pending.replace(path)


def publish_final(path: Path, diagnostics: dict[str, Any]) -> None:
    publish_json(path, diagnostics)


def read_worker_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path, "ARTIFACT_INVALID")
    except Exception:
        return None
    if value.get("schema_version") != SCHEMA_VERSION:
        return None
    return value


def safe_history_view(value: Any) -> dict[str, Any] | None:
    """Return only approved diagnostic fields for A's next planning prompt."""
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        return None
    times = value.get("timings_seconds")
    if not isinstance(times, dict):
        return None
    return {
        "status": value.get("status"),
        "phase": value.get("phase"),
        "timings_seconds": {key: times.get(key) for key in TIMING_FIELDS},
        "model_bytes": value.get("model_bytes"),
        "input_shape": value.get("input_shape"),
        "input_matrix_bytes": value.get("input_matrix_bytes"),
        "effective_iterations": value.get("effective_iterations"),
        "warnings": value.get("warnings", [])[:20]
        if isinstance(value.get("warnings"), list)
        else [],
        "failure": value.get("failure"),
    }
