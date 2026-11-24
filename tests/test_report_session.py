"""Deterministic offline report generation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluation_d.task_budget import TaskBudgetLedger
from scripts.report_session import ReportError, build_session_report


def _make_project(root: Path) -> Path:
    session_dir = root / "sessions" / "session_synthetic"
    session_dir.mkdir(parents=True)
    (root / "runs" / "task" / "v1" / "exp_1").mkdir(parents=True)
    (root / "runs" / "task" / "v1" / "exp_2").mkdir(parents=True)
    summary = {
        "schema_version": "1.0",
        "task_id": "task",
        "data_version": "v1",
        "session_id": "session_synthetic",
        "status": "success",
        "attempted_trials": 2,
        "history_path": "sessions/session_synthetic/history.jsonl",
        "best_experiment_id": "exp_2",
        "stop_reason": "trial limit",
        "llm_gateway": {
            "usage_status": "known",
            "input_tokens": 100,
            "output_tokens": 20,
            "estimated_cost_rmb": "0.006",
        },
    }
    summary_path = session_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    entries = [
        {
            "trial": 1,
            "experiment_id": "exp_1",
            "plan": {"model": "random_forest", "params": {"n_estimators": 10}},
            "decision_source": "rules",
            "status": "success",
            "result_relpath": "task/v1/exp_1/result.json",
            "evaluation": {
                "metric": "log_loss",
                "value": 1.4,
                "greater_is_better": False,
            },
            "duration_seconds": 2.5,
        },
        {
            "trial": 2,
            "experiment_id": "exp_2",
            "plan": {"model": "extra_trees", "params": {"n_estimators": 20}},
            "decision_source": "llm",
            "status": "success",
            "result_relpath": "task/v1/exp_2/result.json",
            "evaluation": {
                "metric": "log_loss",
                "value": 1.0,
                "greater_is_better": False,
            },
            "duration_seconds": 3.0,
        },
    ]
    history = session_dir / "history.jsonl"
    history.write_text("".join(json.dumps(item) + "\n" for item in entries), encoding="utf-8")
    diag = {
        "schema_version": "1.0",
        "phase": "complete",
        "timings_seconds": {"fit": 1.2, "predict": 0.1, "total": 1.5},
        "model_bytes": 345,
        "warnings": [{"category": "UserWarning"}],
        "failure": None,
    }
    (root / "runs" / "task" / "v1" / "exp_2" / "diagnostics.json").write_text(
        json.dumps(diag), encoding="utf-8"
    )
    return summary_path


def test_report_contains_leaderboard_metric_curve_usage_and_optional_task_ledger(tmp_path):
    summary_path = _make_project(tmp_path)
    task_ledger = TaskBudgetLedger(
        tmp_path / "task-budgets",
        "course-task-1",
        budget_rmb="20",
        profile="deepseek-v4.1-flash-offpeak-2026-10-21",
    )
    task_ledger.reserve(
        "session_synthetic:1",
        session_id="session_synthetic",
        max_input_tokens=100,
        max_output_tokens=20,
    )
    output = build_session_report(
        summary_path,
        tmp_path / "report_1",
        task_ledger_path=task_ledger.path,
    )
    with (output / "leaderboard.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["best_so_far"] == "1.4"
    assert rows[1]["best_so_far"] == "1.0"
    assert rows[0]["diagnostics_status"] == "missing"
    assert rows[1]["diagnostics_status"] == "present"
    assert rows[1]["fit_seconds"] == "1.2"
    assert rows[1]["warnings_count"] == "1"

    curve = json.loads((output / "best_so_far.json").read_text(encoding="utf-8"))
    assert [entry["best_so_far"] for entry in curve] == [1.4, 1.0]
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "configured development prices" in markdown
    assert "Task-wide formal ledger" in markdown
    assert "model_identity_verified_for_school_endpoint" not in markdown
    assert "not confirmation of the school endpoint" in markdown
    assert "1.007880" in markdown and "4.031520" in markdown
    assert "quick_start/pricing" in markdown


def test_report_recalculates_course_offpeak_cost_without_rewriting_ledger(tmp_path):
    summary_path = _make_project(tmp_path)
    ledger = TaskBudgetLedger(
        tmp_path, "course-history", budget_rmb="20",
        profile="deepseek-v4.1-flash-peak-2026-10-21",
    )
    ledger.reserve(
        "historical:1", session_id="historical", max_input_tokens=1000, max_output_tokens=500,
    )
    ledger.settle("historical:1", input_tokens=1000, output_tokens=500)
    original = ledger.path.read_bytes()
    output = build_session_report(summary_path, tmp_path / "report", task_ledger_path=ledger.path)
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "Historically settled: `0.00604728` RMB" in markdown
    assert "off-peak rate: `0.00302364` RMB" in markdown
    assert "from `1000` input and `500` output tokens" in markdown
    assert ledger.path.read_bytes() == original


def test_report_refuses_existing_output_and_path_escape(tmp_path):
    summary_path = _make_project(tmp_path)
    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(ReportError, match="already exists"):
        build_session_report(summary_path, existing)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["history_path"] = "../../outside/history.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ReportError, match="escapes"):
        build_session_report(summary_path, tmp_path / "report_bad")


def test_report_shows_missing_history_and_does_not_fabricate_rows(tmp_path):
    summary_path = _make_project(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["history_path"] = "sessions/session_synthetic/missing.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = build_session_report(summary_path, tmp_path / "report_missing")
    with (output / "leaderboard.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert "history missing" in (output / "report.md").read_text(encoding="utf-8")

