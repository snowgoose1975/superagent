"""Synthetic-only tests for restart-safe task-wide accounting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from evaluation_d.task_budget import TaskBudgetError, TaskBudgetLedger

_PROFILE = "deepseek-v4.1-flash-peak-2026-10-21"


def _ledger(root: Path, budget: str = "20") -> TaskBudgetLedger:
    return TaskBudgetLedger(root, "course-task-1", budget_rmb=budget, profile=_PROFILE)


def test_decimal_settlement_and_restart_preserve_profile_and_history(tmp_path):
    ledger = _ledger(tmp_path)
    reserved = ledger.reserve(
        "session-a:1",
        session_id="session-a",
        max_input_tokens=1_000_000,
        max_output_tokens=100_000,
    )
    assert Decimal(reserved["reserved_rmb"]) == Decimal("2.822064")
    assert Decimal(ledger.snapshot()["remaining_rmb"]) == Decimal("17.177936")

    actual = ledger.settle("session-a:1", input_tokens=100, output_tokens=10)
    assert actual == {"actual_cost_rmb": "0.0002822064", "status": "settled"}

    restarted = _ledger(tmp_path)
    state = restarted.snapshot()
    assert state["settled_cost_rmb"] == "0.0002822064"
    assert state["reserved_rmb"] == "0"
    assert state["settled_count"] == 1
    assert state["price_profile"]["model_identity_verified_for_school_endpoint"] is False
    assert state["price_profile"]["input_cny_per_million"] == "2.015760"
    assert state["price_profile"]["output_cny_per_million"] == "8.063040"


def test_pending_and_unknown_keep_worst_case_reserved_after_restart(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(
        "session-a:1",
        session_id="session-a",
        max_input_tokens=1_000_000,
        max_output_tokens=100_000,
    )
    assert reservation["reserved_rmb"] == "2.822064"
    restarted = _ledger(tmp_path)
    assert restarted.snapshot()["pending_count"] == 1
    assert restarted.snapshot()["reserved_rmb"] == "2.822064"

    marked = restarted.mark_unknown("session-a:1", reason_code="TIMEOUT")
    assert marked == {"reserved_rmb": "2.822064", "status": "unknown"}
    reopened = _ledger(tmp_path)
    state = reopened.snapshot()
    assert state["unknown_count"] == 1
    assert state["remaining_rmb"] == "17.177936"
    with pytest.raises(TaskBudgetError, match="cannot be settled"):
        reopened.settle("session-a:1", input_tokens=100, output_tokens=10)
    with pytest.raises(TaskBudgetError, match="cannot be reused"):
        reopened.reserve(
            "session-a:1",
            session_id="session-a",
            max_input_tokens=1_000_000,
            max_output_tokens=100_000,
        )


def test_budget_id_isolation_and_limits_are_fail_closed(tmp_path):
    with pytest.raises(TaskBudgetError, match="safe identifier"):
        TaskBudgetLedger(
            tmp_path,
            "../escape",
            budget_rmb="20",
            profile=_PROFILE,
        )
    ledger = _ledger(tmp_path, budget="10")
    with pytest.raises(TaskBudgetError, match="cannot cover"):
        ledger.reserve(
            "too-large",
            session_id="session-a",
            max_input_tokens=5_000_000,
            max_output_tokens=0,
        )
    assert ledger.snapshot()["pending_count"] == 0
    with pytest.raises(TaskBudgetError, match="differ"):
        _ledger(tmp_path, budget="11")

    equivalent_root = tmp_path / "equivalent"
    _ledger(equivalent_root, budget="20.00")
    assert _ledger(equivalent_root, budget="20").snapshot()["budget_rmb"] == "20"


def test_concurrent_reservations_cannot_overdraw_task_budget(tmp_path):
    ledgers = [_ledger(tmp_path, budget="20") for _ in range(2)]

    def reserve(index: int) -> str:
        try:
            ledgers[index].reserve(
                f"session-{index}:1",
                session_id=f"session-{index}",
                max_input_tokens=5_000_000,
                max_output_tokens=0,
            )
            return "reserved"
        except TaskBudgetError:
            return "denied"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(reserve, [0, 1]))
    assert sorted(outcomes) == ["denied", "reserved"]
    state = _ledger(tmp_path).snapshot()
    assert state["pending_count"] == 1
    assert Decimal(state["reserved_rmb"]) == Decimal("10.0788")
    assert Decimal(state["remaining_rmb"]) == Decimal("9.9212")


def test_unknown_reservation_is_idempotently_marked_but_not_repriced(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.reserve(
        "session-a:1",
        session_id="session-a",
        max_input_tokens=100,
        max_output_tokens=20,
    )
    assert ledger.mark_unknown("session-a:1") == ledger.mark_unknown("session-a:1")
    assert _ledger(tmp_path).snapshot()["unknown_count"] == 1
    with pytest.raises(TaskBudgetError, match="differ"):
        TaskBudgetLedger(
            tmp_path,
            "course-task-1",
            budget_rmb="20",
            profile={
                **ledger.snapshot()["price_profile"],
                "input_cny_per_million": "0.50",
            },
        )

