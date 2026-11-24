"""Synthetic-only tests for restart-safe task-wide accounting."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from evaluation_d.task_budget import TaskBudgetError, TaskBudgetLedger

_PROFILE = "deepseek-v4.1-flash-offpeak-2026-10-21"


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
    assert Decimal(reserved["reserved_rmb"]) == Decimal("1.411032")
    assert reserved["pricing_profile_id"] == _PROFILE
    assert Decimal(ledger.snapshot()["remaining_rmb"]) == Decimal("18.588968")

    actual = ledger.settle("session-a:1", input_tokens=100, output_tokens=10)
    assert actual == {"actual_cost_rmb": "0.0001411032", "status": "settled"}

    restarted = _ledger(tmp_path)
    state = restarted.snapshot()
    assert state["settled_cost_rmb"] == "0.0001411032"
    assert state["reserved_rmb"] == "0"
    assert state["settled_count"] == 1
    assert state["price_profile"]["model_identity_verified_for_school_endpoint"] is False
    assert state["price_profile"]["profile_id"] == _PROFILE
    assert state["price_profile"]["input_cny_per_million"] == "1.007880"
    assert state["price_profile"]["output_cny_per_million"] == "4.031520"


def test_pending_and_unknown_keep_worst_case_reserved_after_restart(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(
        "session-a:1",
        session_id="session-a",
        max_input_tokens=1_000_000,
        max_output_tokens=100_000,
    )
    assert reservation["reserved_rmb"] == "1.411032"
    restarted = _ledger(tmp_path)
    assert restarted.snapshot()["pending_count"] == 1
    assert restarted.snapshot()["reserved_rmb"] == "1.411032"

    marked = restarted.mark_unknown("session-a:1", reason_code="TIMEOUT")
    assert marked == {"reserved_rmb": "1.411032", "status": "unknown"}
    reopened = _ledger(tmp_path)
    state = reopened.snapshot()
    assert state["unknown_count"] == 1
    assert state["remaining_rmb"] == "18.588968"
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
            max_input_tokens=12_000_000,
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
                max_input_tokens=15_000_000,
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
    assert Decimal(state["reserved_rmb"]) == Decimal("15.1182")
    assert Decimal(state["remaining_rmb"]) == Decimal("4.8818")


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


def test_peak_to_offpeak_transition_preserves_historical_charges_and_unknown_reserves(tmp_path):
    ledger = TaskBudgetLedger(
        tmp_path,
        "transition-task",
        budget_rmb="20",
        profile="deepseek-v4.1-flash-peak-2026-10-21",
    )
    ledger.reserve(
        "old-session:1",
        session_id="old-session",
        max_input_tokens=1_000_000,
        max_output_tokens=100_000,
    )
    ledger.settle("old-session:1", input_tokens=100, output_tokens=10)
    ledger.reserve(
        "old-session:2",
        session_id="old-session",
        max_input_tokens=1_000_000,
        max_output_tokens=100_000,
    )
    ledger.mark_unknown("old-session:2", reason_code="TIMEOUT")

    migrated = TaskBudgetLedger(
        tmp_path,
        "transition-task",
        budget_rmb="20",
        profile=_PROFILE,
    )
    state = migrated.snapshot()
    assert state["price_profile"]["profile_id"] == _PROFILE
    assert state["settled_cost_rmb"] == "0.0002822064"
    assert state["reserved_rmb"] == "2.822064"
    assert state["unknown_count"] == 1
    assert len(state["profile_migrations"]) == 1
    migration = state["profile_migrations"][0]
    assert migration["from_profile"]["profile_id"] == "deepseek-v4.1-flash-peak-2026-10-21"
    assert migration["to_profile"]["profile_id"] == _PROFILE
    raw = json.loads(migrated.path.read_text(encoding="utf-8"))
    assert raw["reservations"]["old-session:1"]["pricing_profile_id"] == migration[
        "from_profile"
    ]["profile_id"]
    assert raw["reservations"]["old-session:2"]["pricing_profile_id"] == migration[
        "from_profile"
    ]["profile_id"]

    new_reservation = migrated.reserve(
        "new-session:1",
        session_id="new-session",
        max_input_tokens=1_000_000,
        max_output_tokens=100_000,
    )
    assert new_reservation["reserved_rmb"] == "1.411032"
    assert new_reservation["pricing_profile_id"] == _PROFILE


def test_pending_peak_reservations_settle_at_their_original_tariff(tmp_path):
    peak = "deepseek-v4.1-flash-peak-2026-10-21"
    old = TaskBudgetLedger(tmp_path, "in-flight", budget_rmb="20", profile=peak)
    for attempt in (1, 2):
        old.reserve(
            f"old:{attempt}", session_id="old", max_input_tokens=1000, max_output_tokens=500,
        )
    migrated = TaskBudgetLedger(tmp_path, "in-flight", budget_rmb="20", profile=_PROFILE)
    assert migrated.settle("old:1", input_tokens=1000, output_tokens=500)["actual_cost_rmb"] == "0.00604728"
    assert old.settle("old:2", input_tokens=1000, output_tokens=500)["actual_cost_rmb"] == "0.00604728"
    with pytest.raises(TaskBudgetError, match="differ"):
        old.reserve("old:3", session_id="old", max_input_tokens=1000, max_output_tokens=500)
    state = migrated.snapshot()
    assert state["settled_cost_rmb"] == "0.01209456"
    assert state["course_offpeak_settled_cost_rmb"] == "0.00604728"
    assert state["settled_input_tokens"] == 2000
    assert state["settled_output_tokens"] == 1000
    assert state["pending_count"] == 0


@pytest.mark.parametrize("budget", ["-1", "NaN", "500"])
def test_rejects_out_of_range_course_budget(tmp_path, budget):
    with pytest.raises(TaskBudgetError):
        _ledger(tmp_path, budget)

