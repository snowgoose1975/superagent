"""Validated, opt-in search-state configuration kept separate from task schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from executor_c.contracts import read_json

SEARCH_FIELDS = {
    "schema_version",
    "require_family_coverage",
    "min_exploration_trials",
    "patience",
    "min_delta",
    "max_consecutive_failures",
    "max_session_seconds",
    "finalize_reserve_seconds",
    "confirmation_seed_offsets",
}


def load_search_config(path: str | Path) -> dict[str, Any]:
    value = read_json(path, error_code="PLAN_INVALID")
    if set(value) != SEARCH_FIELDS or value.get("schema_version") != "1.0":
        raise ValueError("search config 字段或 schema_version 不受支持")
    if type(value["require_family_coverage"]) is not bool:
        raise ValueError("require_family_coverage 必须是布尔值")
    for field, low, high in (
        ("min_exploration_trials", 1, 100),
        ("patience", 1, 100),
        ("max_consecutive_failures", 1, 20),
        ("max_session_seconds", 60, 86400),
        ("finalize_reserve_seconds", 1, 3600),
    ):
        item = value[field]
        if type(item) is not int or not low <= item <= high:
            raise ValueError(f"{field} 必须是 {low}–{high} 的整数")
    if value["finalize_reserve_seconds"] >= value["max_session_seconds"]:
        raise ValueError("finalize_reserve_seconds 必须小于 max_session_seconds")
    delta = value["min_delta"]
    if type(delta) not in (int, float) or not 0 <= delta <= 1:
        raise ValueError("min_delta 必须是 [0, 1] 内的有限数")
    offsets = value["confirmation_seed_offsets"]
    if (
        not isinstance(offsets, list)
        or len(offsets) > 2
        or any(type(offset) is not int or not 1 <= offset <= 2**32 - 1 for offset in offsets)
    ):
        raise ValueError("confirmation_seed_offsets 最多含两个正整数偏移")
    return value
