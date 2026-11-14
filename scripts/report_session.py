"""Build deterministic, offline artifacts from one SuperAgent session summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


_CSV_FIELDS = [
    "trial",
    "experiment_id",
    "status",
    "model",
    "params",
    "metric",
    "score",
    "best_so_far",
    "duration_seconds",
    "decision_source",
    "diagnostics_status",
    "phase",
    "fit_seconds",
    "predict_seconds",
    "total_seconds",
    "model_bytes",
    "warnings_count",
    "failure_code",
]
class ReportError(ValueError):
    """Invalid or unsafe report input."""


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {what}: {exc.__class__.__name__}") from None
    if not isinstance(value, dict):
        raise ReportError(f"{what} must be a JSON object")
    return value


def _inside(root: Path, relative: Any, what: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ReportError(f"{what} must be a relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ReportError(f"{what} escapes the project root") from None
    return path


def _safe_text(value: Any, max_length: int = 400) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return text[:max_length]


def _metric_value(evaluation: Any) -> tuple[str, float | None, bool | None]:
    if not isinstance(evaluation, dict):
        return "", None, None
    name = evaluation.get("metric")
    score = evaluation.get("value")
    direction = evaluation.get("greater_is_better")
    if not isinstance(name, str):
        name = ""
    if type(score) not in (float, int) or not math.isfinite(score):
        score = None
    else:
        score = float(score)
    if type(direction) is not bool:
        direction = None
    if evaluation.get("status") not in {None, "success"}:
        score = None
    return name, score, direction


def _load_history(root: Path, summary: dict[str, Any], summary_path: Path) -> list[dict[str, Any]]:
    rel = summary.get("history_path")
    if rel is None:
        try:
            parent_rel = summary_path.parent.resolve().relative_to(root.resolve())
        except ValueError:
            raise ReportError("summary path is outside the project root") from None
        candidate = _inside(root, str(parent_rel / "history.jsonl"), "history_path")
    else:
        candidate = _inside(root, rel, "history_path")
    if not candidate.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with candidate.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    raise ReportError(f"history.jsonl has invalid JSON at line {line_number}") from None
                if not isinstance(value, dict):
                    raise ReportError(f"history.jsonl line {line_number} is not an object")
                records.append(value)
    except (OSError, UnicodeError) as exc:
        raise ReportError(f"cannot read history.jsonl: {exc.__class__.__name__}") from None
    return records


def _diagnostics(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    rel = record.get("result_relpath")
    if not isinstance(rel, str):
        return {"diagnostics_status": "missing"}
    result_path = _inside(root / "runs", rel, "result_relpath")
    path = result_path.parent / "diagnostics.json"
    if not path.is_file():
        return {"diagnostics_status": "missing"}
    try:
        value = _read_json(path, "diagnostics.json")
    except ReportError:
        return {"diagnostics_status": "invalid"}
    if value.get("schema_version") != "1.0":
        return {"diagnostics_status": "invalid"}
    timings = value.get("timings_seconds")
    timings = timings if isinstance(timings, dict) else {}
    warnings = value.get("warnings")
    failure = value.get("failure")
    return {
        "diagnostics_status": "present",
        "phase": _safe_text(value.get("phase"), 80),
        "fit_seconds": timings.get("fit") if _finite_nonnegative(timings.get("fit")) else None,
        "predict_seconds": (
            timings.get("predict") if _finite_nonnegative(timings.get("predict")) else None
        ),
        "total_seconds": timings.get("total") if _finite_nonnegative(timings.get("total")) else None,
        "model_bytes": value.get("model_bytes")
        if type(value.get("model_bytes")) is int and value["model_bytes"] >= 0
        else None,
        "warnings_count": len(warnings) if isinstance(warnings, list) else 0,
        "failure_code": _safe_text(failure.get("code"), 80)
        if isinstance(failure, dict)
        else "",
    }


def _finite_nonnegative(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _task_budget(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = _read_json(path, "task budget ledger")
    if data.get("schema_version") != "1.0" or not isinstance(data.get("reservations"), dict):
        raise ReportError("task budget ledger schema is unsupported")
    settled = Decimal("0")
    reserved = Decimal("0")
    counts = {"pending": 0, "unknown": 0, "settled": 0}
    for item in data["reservations"].values():
        if not isinstance(item, dict) or item.get("status") not in counts:
            raise ReportError("task budget ledger contains an invalid reservation")
        counts[item["status"]] += 1
        amount = item.get("actual_cost_rmb") if item["status"] == "settled" else item.get("reserved_rmb")
        try:
            number = Decimal(str(amount))
        except (InvalidOperation, TypeError, ValueError):
            raise ReportError("task budget ledger contains an invalid amount") from None
        if not number.is_finite() or number < 0:
            raise ReportError("task budget ledger contains an invalid amount")
        if item["status"] == "settled":
            settled += number
        else:
            reserved += number
    try:
        limit = Decimal(str(data["budget_rmb"]))
    except (InvalidOperation, TypeError, ValueError):
        raise ReportError("task budget ledger contains an invalid limit") from None
    if not limit.is_finite() or limit <= 0:
        raise ReportError("task budget ledger contains an invalid limit")
    profile = data.get("price_profile")
    return {
        "budget_id": _safe_text(data.get("budget_id"), 128),
        "budget_rmb": format(limit, "f"),
        "settled_cost_rmb": format(settled, "f"),
        "reserved_rmb": format(reserved, "f"),
        "remaining_rmb": format(limit - settled - reserved, "f"),
        "counts": counts,
        "price_profile": profile if isinstance(profile, dict) else {},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in _CSV_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _markdown(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    budget: dict[str, Any] | None,
) -> str:
    task_id = _safe_text(summary.get("task_id"), 160) or "unknown"
    session_id = _safe_text(summary.get("session_id"), 160) or "unknown"
    status = _safe_text(summary.get("status"), 80) or "unknown"
    attempts = summary.get("attempted_trials", len(rows))
    lines = [
        f"# Session report: {session_id}",
        "",
        f"- Task: `{task_id}`",
        f"- Status: `{status}`",
        f"- Data version: `{_safe_text(summary.get('data_version'), 160) or 'unknown'}`",
        f"- Attempts: {_safe_text(attempts, 32)}",
    ]
    metric = curve[0]["metric"] if curve else ""
    best = curve[-1]["best_so_far"] if curve else None
    best_experiment = summary.get("best_experiment_id")
    if metric:
        lines.append(f"- Selection metric: `{_safe_text(metric, 80)}`")
    if best is not None:
        lines.append(
            f"- Best validation score: `{best}` "
            f"({_safe_text(best_experiment, 160) or 'unknown experiment'})"
        )
    stop_reason = _safe_text(summary.get("stop_reason"), 500)
    if stop_reason:
        lines.append(f"- Stop reason: {stop_reason}")
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Trial | Experiment | Model | Status | Metric | Score | Best so far |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    if rows:
        for row in rows:
            lines.append(
                (
                    "| {trial} | `{experiment_id}` | `{model}` | {status} | "
                    "{metric} | {score} | {best_so_far} |"
                ).format(
                    trial=row.get("trial", ""),
                    experiment_id=_safe_text(row.get("experiment_id"), 160),
                    model=_safe_text(row.get("model"), 100),
                    status=_safe_text(row.get("status"), 80),
                    metric=_safe_text(row.get("metric"), 80),
                    score="" if row.get("score") is None else row["score"],
                    best_so_far="" if row.get("best_so_far") is None else row["best_so_far"],
                )
            )
    else:
        lines.append("| — | — | — | history missing | — | — | — |")
    gateway = summary.get("llm_gateway")
    lines += ["", "## API usage", ""]
    if isinstance(gateway, dict):
        lines += [
            f"- Usage status: `{_safe_text(gateway.get('usage_status'), 40) or 'unknown'}`",
            f"- Input/output tokens: `{gateway.get('input_tokens')}` / `{gateway.get('output_tokens')}`",
            f"- Session estimate: `{_safe_text(gateway.get('estimated_cost_rmb'), 40) or 'unknown'}` RMB",
            "- This session estimate uses its configured development prices; it is not "
            "formal school-endpoint billing evidence.",
        ]
    else:
        lines.append("No gateway usage metadata was attached to this summary.")
    if budget is not None:
        profile = budget["price_profile"]
        basis = profile.get("source_basis")
        basis = basis if isinstance(basis, dict) else {}
        sources = profile.get("sources")
        sources = sources if isinstance(sources, list) else []
        lines += [
            "",
            "## Task-wide formal ledger",
            "",
            f"- Budget ID: `{budget['budget_id']}`; limit: `{budget['budget_rmb']}` RMB",
            f"- Settled: `{budget['settled_cost_rmb']}` RMB; reserved (including unknown): "
            f"`{budget['reserved_rmb']}` RMB; remaining: `{budget['remaining_rmb']}` RMB",
            f"- Price profile: `{_safe_text(profile.get('profile_id'), 160)}` "
            f"({_safe_text(profile.get('effective_date'), 40)})",
            "- Rates per million tokens: "
            f"input `{_safe_text(profile.get('input_cny_per_million'), 40)}` CNY; "
            f"output `{_safe_text(profile.get('output_cny_per_million'), 40)}` CNY.",
            "- Source basis: peak/cache-miss USD "
            f"`{_safe_text(basis.get('usd_input_per_million'), 30)}` input / "
            f"`{_safe_text(basis.get('usd_output_per_million'), 30)}` output; FX "
            f"`{_safe_text(basis.get('fx_cny_per_usd'), 30)}` CNY/USD on "
            f"`{_safe_text(basis.get('fx_date'), 40)}`.",
            f"- Profile caveat: {_safe_text(profile.get('billing_caveat'), 400)}",
            f"- Reservation counts: {json.dumps(budget['counts'], sort_keys=True)}",
        ]
        if sources:
            source_links = ", ".join(
                f"[source {i}]({url})" for i, url in enumerate(sources, 1)
            )
            lines.append("- Price sources: " + source_links)
    lines += [
        "",
        "## Artifacts",
        "",
        "- `leaderboard.csv`: experiment rows and available C diagnostics.",
        "- `best_so_far.json`: deterministic validation-score curve.",
        "",
    ]
    return "\n".join(lines)


def build_session_report(
    summary_path: str | Path,
    output: str | Path,
    *,
    project_root: str | Path | None = None,
    task_ledger_path: str | Path | None = None,
) -> Path:
    """Write a unique directory with leaderboard, curve JSON, and Markdown."""
    summary_file = Path(summary_path).expanduser().resolve()
    if project_root is None:
        # Project summary convention: <project>/sessions/<id>/summary.json.
        if summary_file.parent.parent.name != "sessions":
            raise ReportError("cannot infer project root; supply --project-root")
        root = summary_file.parent.parent.parent.resolve()
    else:
        root = Path(project_root).expanduser().resolve()
    summary = _read_json(summary_file, "session summary")
    if not isinstance(summary.get("session_id"), str):
        raise ReportError("summary is missing session_id")
    history = _load_history(root, summary, summary_file)
    rows: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    best_score: float | None = None
    curve_metric = ""
    greater_is_better: bool | None = None
    for index, record in enumerate(history, 1):
        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
        evaluation = record.get("evaluation")
        metric, score, direction = _metric_value(evaluation)
        if record.get("status") != "success":
            score = None
        if metric:
            if not curve_metric:
                curve_metric, greater_is_better = metric, direction
            elif metric != curve_metric or direction != greater_is_better:
                raise ReportError("history contains inconsistent selection metrics")
        if record.get("status") == "success" and score is not None and direction is not None:
            if best_score is None or (score > best_score if direction else score < best_score):
                best_score = score
        result_relpath = record.get("result_relpath")
        diagnostic = _diagnostics(root, record) if result_relpath else {"diagnostics_status": "missing"}
        row = {
            "trial": record.get("trial", index),
            "experiment_id": _safe_text(record.get("experiment_id"), 160),
            "status": _safe_text(record.get("status"), 80),
            "model": _safe_text(plan.get("model", decision.get("model")), 100),
            "params": plan.get("params", decision.get("params", {})),
            "metric": metric,
            "score": score,
            "best_so_far": best_score,
            "duration_seconds": record.get("duration_seconds")
            if _finite_nonnegative(record.get("duration_seconds"))
            else None,
            "decision_source": _safe_text(record.get("decision_source"), 80),
            **diagnostic,
        }
        rows.append(row)
        if score is not None:
            curve.append(
                {
                    "trial": row["trial"],
                    "experiment_id": row["experiment_id"],
                    "metric": metric,
                    "value": score,
                    "best_so_far": best_score,
                    "greater_is_better": direction,
                }
            )
    budget = _task_budget(Path(task_ledger_path).expanduser().resolve()) if task_ledger_path else None
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists():
        raise ReportError("output directory already exists; choose a unique new path")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_csv(output_dir / "leaderboard.csv", rows)
        curve_path = output_dir / "best_so_far.json"
        with curve_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(curve, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        markdown_path = output_dir / "report.md"
        with markdown_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_markdown(summary, rows, curve, budget))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # Keep any partial output for inspection; never delete artifacts automatically.
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="path to session summary.json")
    parser.add_argument("--output", required=True, help="new unique report directory")
    parser.add_argument("--project-root", help="project root (inferred for sessions/<id>/summary.json)")
    parser.add_argument("--task-ledger", help="optional formal task-wide budget ledger JSON")
    args = parser.parse_args(argv)
    try:
        report_dir = build_session_report(
            args.summary,
            args.output,
            project_root=args.project_root,
            task_ledger_path=args.task_ledger,
        )
    except ReportError as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return 2
    print(report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

