"""Prepare data in B, plan in A, execute in C, and score in D."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from agent_a.controller import PackageAController, SchoolAController
from agent_a.search_config import load_search_config
from data_b.prepare import load_training_data, prepare_data
from evaluation_d.evaluate import evaluate, evaluate_training_majority_baseline
from evaluation_d.final_test import FinalTestError, evaluate_frozen_bundle, format_test_result
from evaluation_d.logger import append_session_event
from executor_c.text_encoder import (
    DOWNLOAD_TIMEOUT, ENCODER_ID, MODES as ENCODER_MODES, EncoderError,
    encoder_status, prepare_weights,
)
from evaluation_d.school_gateway import GatewayError
from evaluation_d.metric_contracts import metric_in_domain, validate_metric_spec
from executor_c.contracts import read_json, resolve_data_path, sha256_file, validate_plan, write_json
from executor_c.diagnostics import safe_history_view
from executor_c.executor import execute
from executor_c.model_registry import default_pipeline, runtime_capabilities

PROJECT_ROOT = Path(__file__).resolve().parent
TASK_FIELDS = {
    "schema_version",
    "task_id",
    "data_version",
    "source_format",
    "source_root",
    "feature_columns",
    "feature_types",
    "label_values",
    "label_names",
    "primary_metric",
    "partitions",
    "train_policy",
}
TASK_FIELDS_V2 = {
    "schema_version",
    "task_id",
    "data_version",
    "source_format",
    "source_root",
    "modality",
    "label_values",
    "label_names",
    "primary_metric",
    "partitions",
    "train_policy",
}


def load_task_spec(config_path: str | Path) -> dict[str, Any]:
    """Resolve source_root relative to the configuration file."""
    path = Path(config_path).resolve()
    spec = read_json(path, error_code="PLAN_INVALID")
    if spec.get("schema_version") == "1.0":
        if not TASK_FIELDS <= set(spec) <= TASK_FIELDS | {"preprocessing"}:
            raise ValueError("TaskSpec 1.0 fields do not match the table contract")
    elif spec.get("schema_version") == "2.0":
        if set(spec) != TASK_FIELDS_V2:
            raise ValueError("TaskSpec 2.0 fields do not match the text or image contract")
    else:
        raise ValueError("Unsupported TaskSpec schema_version")
    if not isinstance(spec["source_root"], str) or not spec["source_root"]:
        raise ValueError("source_root must be a non-empty path string")
    spec["primary_metric"] = validate_metric_spec(spec["primary_metric"])
    spec["source_root"] = str((path.parent / spec["source_root"]).resolve())
    policy = spec["train_policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "max_trials",
        "random_seed",
        "timeout_seconds",
    }:
        raise ValueError("train_policy requires max_trials, random_seed, and timeout_seconds")
    if type(policy["max_trials"]) is not int or policy["max_trials"] < 1:
        raise ValueError("max_trials must be a positive integer")
    # Use C to validate seeds, time limits, and identifiers.
    validate_plan(
        {
            "schema_version": "1.0",
            "task_id": spec["task_id"],
            "data_version": spec["data_version"],
            "experiment_id": "preflight",
            "model": "logistic_regression",
            "params": {},
            "random_seed": policy["random_seed"],
            "timeout_seconds": policy["timeout_seconds"],
        }
    )
    return spec


def _append_history(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def _copy_exclusive(source: Path, target: Path) -> None:
    with source.open("rb") as input_stream, target.open("xb") as output_stream:
        for block in iter(lambda: input_stream.read(1024 * 1024), b""):
            output_stream.write(block)


def _fingerprint_from_digests(digests: list[str]) -> str | None:
    if not digests:
        return None
    encoded = json.dumps(sorted(digests), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prepared_budget_fingerprint(package: dict[str, Any]) -> str | None:
    train = package.get("source_file_sha256", {}).get("train", {})
    return _fingerprint_from_digests([
        item["sha256"] for item in train.values()
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    ])


def _automatic_budget_id(root: Path, discovery: dict[str, Any]) -> str:
    """Retain a task's previous ledger when moving from manual to automatic input."""
    fingerprint = discovery["budget_fingerprint"]
    default_id = "auto_" + fingerprint[:40]
    ledger_root = root / "sessions" / "task-budgets"
    candidates = {default_id} if (ledger_root / f"{default_id}.json").is_file() else set()

    def artifact(relative: Any, expected_name: str) -> Path:
        if (
            not isinstance(relative, str) or Path(relative).name != expected_name
            or Path(relative).is_absolute() or "\\" in relative
            or any(part.startswith(".") for part in Path(relative).parts)
        ):
            raise ValueError("Invalid historical task artifact path")
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError("Historical task artifact path escapes the project directory")
        return candidate

    for summary_path in (root / "sessions").glob("session_*/summary.json"):
        historical = read_json(summary_path)
        accounting = (historical.get("llm_gateway") or {}).get("task_budget")
        if not isinstance(accounting, dict) or not accounting.get("budget_id"):
            continue
        previous_fingerprint = historical.get("budget_fingerprint")
        if previous_fingerprint is None and historical.get("discovery_path"):
            report = artifact(historical["discovery_path"], "task.discovery.json")
            if report.is_file():
                previous_fingerprint = read_json(report).get("budget_fingerprint")
        if previous_fingerprint is None and historical.get("data_manifest_path"):
            manifest = artifact(historical["data_manifest_path"], "data_manifest.json")
            package = artifact(
                (manifest.parent / "package_result.json").relative_to(root.resolve()).as_posix(),
                "package_result.json",
            )
            if package.is_file():
                previous_fingerprint = _prepared_budget_fingerprint(read_json(package))
            else:
                # Old manual sessions kept the raw task spec separately from prepared data.
                snapshot_path = summary_path.with_name("task.snapshot.json")
                if snapshot_path.is_file():
                    snapshot = read_json(snapshot_path)
                    source_root = snapshot.get("source_root")
                    train = snapshot.get("partitions", {}).get("train", {})
                    try:
                        digests = [
                            sha256_file(resolve_data_path(source_root, train[key]))
                            for key in ("features_path", "labels_path")
                        ] if isinstance(source_root, str) else []
                    except (KeyError, OSError, ValueError):
                        digests = []
                    previous_fingerprint = _fingerprint_from_digests(digests)
        same_task_without_fingerprint = (
            previous_fingerprint is None and bool(discovery.get("task_id"))
            and historical.get("task_id") == discovery["task_id"]
        )
        if previous_fingerprint == fingerprint or same_task_without_fingerprint:
            candidates.add(accounting["budget_id"])
    if len(candidates) > 1:
        raise ValueError("Multiple historical ledgers match the same task")
    if candidates:
        budget_id = next(iter(candidates))
        if not (ledger_root / f"{budget_id}.json").is_file():
            raise ValueError("The existing task ledger is missing")
        return budget_id
    return default_id


def freeze_best(
    spec: dict[str, Any],
    prepared: dict[str, Any],
    best: dict[str, Any],
    output_root: Path,
    session_id: str,
    *,
    partial: bool = False,
    destination: Path | None = None,
    members: list[dict[str, Any]] | None = None,
) -> Path:
    """Freeze the selected fitted model and publish bundle.json last."""
    run_dir = (output_root / "runs" / best["result_relpath"]).parent
    worker = read_json(run_dir / "worker_metadata.json")
    manifest = read_json(prepared["data_manifest_path"])
    is_v1 = manifest["schema_version"] == "1.0"
    preprocessor: Path | None = None
    if is_v1:
        preprocessor = Path(prepared["preprocessor_path"])
        if worker["input_hashes"].get("preprocessor") != sha256_file(preprocessor):
            raise ValueError("The B preprocessor changed after the selected trial")
    final_id = ("partial_" if partial else "final_") + session_id[-8:]
    frozen = destination or output_root / "submission" / spec["task_id"] / final_id
    frozen.mkdir(parents=True, exist_ok=False)
    sources = {
        "model.joblib": run_dir / "model.joblib",
        "label_mapping.json": Path(prepared["label_mapping_path"]),
        "plan.json": run_dir / "plan.json",
        "environment.json": run_dir / "environment.json",
    }
    input_rules = Path(spec["source_root"]) / "reader_rules.json"
    if input_rules.is_file():
        sources["reader_rules.json"] = input_rules
    if preprocessor is not None:
        sources["preprocessor.joblib"] = preprocessor
    if "text_encoder" in best["plan"].get("pipeline", {}):
        sources["encoder.json"] = run_dir / "encoder.json"
    hashes = {}
    for name, source in sources.items():
        _copy_exclusive(source, frozen / name)
        hashes[name] = sha256_file(frozen / name)
        if hashes[name] != sha256_file(source):
            raise ValueError("An artifact changed while the bundle was being copied")
    bundle = {
        "schema_version": "1.0",
        "task_id": spec["task_id"],
        "data_version": spec["data_version"],
        "session_id": session_id,
        "final_id": final_id,
        "experiment_id": best["experiment_id"],
        "selection": {
            "partition": "validation",
            "metric": spec["primary_metric"]["name"],
            "value": best["evaluation"]["value"],
            "greater_is_better": spec["primary_metric"]["greater_is_better"],
        },
        "classes": manifest["classes"],
        "hashes": hashes,
        "training_policy": {"fit_partition": "train", "refit_on_validation": False},
    }
    if is_v1:
        bundle.update(
            feature_columns=spec["feature_columns"],
            feature_types=spec["feature_types"],
            artifacts={
                "model_path": "model.joblib",
                "preprocessor_path": "preprocessor.joblib",
                "label_mapping_path": "label_mapping.json",
            },
        )
    else:
        bundle.update(
            schema_version="2.0",
            representation=manifest["representation"],
            modality=manifest["modality"],
            output_label_mode="encoded",
            artifacts={
                "model_path": "model.joblib",
                "label_mapping_path": "label_mapping.json",
            },
        )
    if is_v1 and best["plan"]["model"] == "catboost":
        bundle["input_representation"] = "native_table"
        bundle["preprocessing"] = spec.get("preprocessing", {})
    if members:
        bundle["ensemble"] = []
        for index, member in enumerate(members):
            member_path = freeze_best(
                spec, prepared, member, output_root, session_id,
                destination=frozen / f"member_{index + 1}",
            )
            relative = member_path.relative_to(frozen).as_posix()
            bundle["ensemble"].append(relative)
            bundle["hashes"][relative] = sha256_file(member_path)
            member_bundle = read_json(member_path)
            if "encoder.json" in member_bundle["hashes"] and "encoder.json" not in bundle["hashes"]:
                encoder_marker = frozen / "encoder.json"
                _copy_exclusive(member_path.parent / "encoder.json", encoder_marker)
                bundle["hashes"]["encoder.json"] = sha256_file(encoder_marker)
    write_json(frozen / "bundle.json", bundle)
    return frozen / "bundle.json"


def _check_score(
    score: dict[str, Any],
    plan: dict[str, Any],
    expected_rows: int,
    metric_spec: dict[str, Any],
) -> None:
    if (
        not isinstance(score, dict)
        or score.get("status") != "success"
        or score.get("schema_version") != "1.0"
    ):
        raise ValueError("D did not return a successful score")
    if any(score.get(key) != plan[key] for key in ("task_id", "data_version", "experiment_id")):
        raise ValueError("The D score identity does not match this trial")
    value = score.get("value")
    if (
        score.get("metric") != metric_spec["name"]
        or score.get("greater_is_better") is not metric_spec["greater_is_better"]
        or not metric_in_domain(metric_spec["name"], value)
    ):
        raise ValueError("The D score has the wrong metric or falls outside its domain")
    if type(score.get("n_samples")) is not int or score["n_samples"] != expected_rows:
        raise ValueError("The D score must cover every row of the fixed validation split")


def _best_scored_trial(
    history: list[dict[str, Any]], metric_spec: dict[str, Any]
) -> dict[str, Any] | None:
    successful = [
        item
        for item in history
        if item.get("status") == "success" and item.get("evaluation") is not None
    ]
    if not successful:
        return None
    chooser = max if metric_spec["greater_is_better"] else min
    return chooser(successful, key=lambda item: item["evaluation"]["value"])


def _execute_trial(
    *,
    trial: int,
    decision: dict[str, Any],
    plan: dict[str, Any],
    decision_source: str,
    display_source: str | None = None,
    root: Path,
    prepared: dict[str, Any],
    session_dir: Path,
    spec: dict[str, Any],
    scorer: Any,
    pretrained: str = "disabled",
    download_timeout: float = DOWNLOAD_TIMEOUT,
    encoder_available: bool = False,
    encoder_no_time_limit: bool = False,
    performance: bool = False,
) -> dict[str, Any]:
    experiment_id = plan["experiment_id"]
    write_json(
        session_dir / f"decision_{trial:03d}.json",
        {**decision, "resolved_plan": plan, "decision_source": decision_source},
    )
    print(
        f"[A:{display_source or decision_source}] Trial {trial} | {plan['model']} | {decision['reason']}",
        flush=True,
    )
    print(f"[A->C] params = {json.dumps(plan['params'], ensure_ascii=False)}", flush=True)
    entry: dict[str, Any] = {
        "trial": trial,
        "experiment_id": experiment_id,
        "decision": decision,
        "decision_source": decision_source,
        "search_phase": decision.get("phase", "search"),
        "plan": plan,
        "status": "failed",
        "result_relpath": None,
        "evaluation": None,
        "error": None,
        "duration_seconds": None,
    }
    trial_started = time.monotonic()
    try:
        if "text_encoder" in plan.get("pipeline", {}):
            if not encoder_available:
                raise EncoderError("ENCODER_UNAVAILABLE", "The local encoder is unavailable for this trial")
            weights = prepare_weights(pretrained, timeout=download_timeout)
            entry["encoder"] = {"weights": weights}
            relative = execute(
                plan, prepared["data_manifest_path"], root / "runs",
                **({"no_time_limit": True} if performance else {}),
                encoder_runtime={
                    "enabled": True,
                    "feature_cache": str(root / ".cache" / "development-embeddings"),
                    "weights": weights, "no_time_limit": encoder_no_time_limit,
                },
            )
            encoder_path = (root / "runs" / relative).parent / "encoder.json"
            if encoder_path.is_file():
                entry["encoder"] = read_json(encoder_path)
        else:
            relative = execute(
                plan, prepared["data_manifest_path"], root / "runs",
                **({"no_time_limit": True} if performance else {}),
            )
        execution = read_json(root / "runs" / relative)
        entry.update(
            status=execution["status"],
            result_relpath=relative,
            error=execution["error"],
            duration_seconds=execution.get("duration_seconds"),
        )
        diagnostics_path = (root / "runs" / relative).parent / "diagnostics.json"
        if diagnostics_path.is_file():
            entry["diagnostics"] = safe_history_view(read_json(diagnostics_path))
        print(f"[C] {experiment_id}: {execution['status']}", flush=True)
        if execution["status"] == "success":
            score = scorer(
                root / "runs" / relative,
                Path(prepared["evaluation_manifest_path"]),
                session_dir / f"evaluation_{trial:03d}.json",
            )
            _check_score(
                score,
                plan,
                spec["partitions"]["validation"]["n_samples"],
                spec["primary_metric"],
            )
            entry["evaluation"] = score
            print(
                f"[D] validation {score['metric']} = {score['value']:.6f}",
                flush=True,
            )
    except EncoderError as exc:
        entry.update(
            error={"code": exc.code, "message": str(exc)},
            duration_seconds=round(time.monotonic() - trial_started, 3),
        )
        print(f"[C] {experiment_id}: {exc.code}, {exc}", flush=True)
    except Exception as exc:
        entry.update(status="failed", error={"code": "TRIAL_ERROR", "message": str(exc)})
        if entry["result_relpath"]:
            diagnostics_path = (
                root / "runs" / entry["result_relpath"]
            ).parent / "diagnostics.json"
            if diagnostics_path.is_file():
                entry["diagnostics"] = safe_history_view(read_json(diagnostics_path))
        print(f"[Search] Trial failed: {type(exc).__name__}", flush=True)
    return entry


def run_project(
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    controller: Any = None,
    evaluator: Any = None,
    *,
    planner_mode: str = "offline",
    gateway: Any = None,
    search_config_path: str | Path | None = None,
    test_config_path: str | Path | None = None,
    task_directory: str | Path | None = None,
    gateway_factory: Callable[[dict[str, Any] | None, Path], Any] | None = None,
    max_trials: int = 20,
    pretrained: str = "disabled",
    encoder_no_time_limit: bool = False,
    performance: bool = False,
    require_complete: bool = False,
    b_analysis: str = "rules",
    task_documents: list[str] | None = None,
) -> Path:
    """Run discovery, validation search, freezing, and independent test evaluation."""
    if pretrained not in ENCODER_MODES:
        raise ValueError("pretrained must be disabled, cache-only, or download")
    if encoder_no_time_limit and pretrained == "disabled":
        raise ValueError("--encoder-no-time-limit requires --pretrained cache-only or download")
    if planner_mode not in {"offline", "school"}:
        raise ValueError("planner_mode must be offline or school")
    if b_analysis not in {"rules", "llm"}:
        raise ValueError("b_analysis must be rules or llm")
    if b_analysis == "llm" and (task_directory is None or planner_mode != "school"):
        raise ValueError("B LLM analysis requires --task and --planner school with a shared D gateway")
    if task_documents is not None and b_analysis != "llm":
        raise ValueError("--task-doc requires B LLM analysis")
    if task_directory is not None and (config_path is not None or test_config_path is not None):
        raise ValueError("--task is incompatible with --config and --test-config")
    if gateway is not None and gateway_factory is not None:
        raise ValueError("Provide only one of gateway and gateway_factory")
    if controller is not None and (
        gateway is not None or gateway_factory is not None or planner_mode != "offline"
    ):
        raise ValueError("A custom controller cannot be combined with planner or gateway configuration")
    if controller is None and planner_mode == "school" and gateway is None and gateway_factory is None:
        raise ValueError("School mode requires a school gateway")
    spec = (
        load_task_spec(config_path or PROJECT_ROOT / "configs" / "coupon.json")
        if task_directory is None else None
    )
    if (task_directory is not None or performance) and search_config_path is None:
        search_config_path = PROJECT_ROOT / "configs" / "search-defaults.json"
    search_config = load_search_config(search_config_path) if search_config_path else None
    if performance and search_config is not None:
        search_config["patience"] = max(search_config["patience"], 20)
        search_config["min_exploration_trials"] = max(search_config["min_exploration_trials"], 20)
    search_profile = "expanded" if search_config is not None else "legacy"
    session_started = time.monotonic()
    root = Path(output_root).resolve() if output_root is not None else PROJECT_ROOT
    session_id = (
        datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    )
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    history_path = session_dir / "history.jsonl"
    events_path = session_dir / "events.jsonl"
    history_path.open("x", encoding="utf-8").close()
    planner = controller
    scorer = evaluator if evaluator is not None else evaluate
    mode = (
        getattr(planner, "mode", "injected_controller")
        if planner is not None
        else "school_llm"
        if planner_mode == "school"
        else "offline_rules"
    )
    history: list[dict[str, Any]] = []

    def log_event(
        event_type: str,
        *,
        stage: str,
        status: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        append_session_event(
            events_path,
            session_id=session_id,
            event_type=event_type,
            stage=stage,
            status=status,
            details=details,
        )

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "task_id": spec["task_id"] if spec is not None else None,
        "data_version": spec["data_version"] if spec is not None else None,
        "session_id": session_id,
        "final_id": None,
        "status": "failed",
        "planner_mode": mode,
        "strategy_source": getattr(planner, "strategy_source", None),
        "real_llm_calls": 0 if mode == "offline_rules" else None,
        "history_path": history_path.relative_to(root).as_posix(),
        "events_path": events_path.relative_to(root).as_posix(),
        "best_experiment_id": None,
        "best_validation_accuracy": None,
        "best_validation_metric": None,
        "bundle_path": None,
        "partial_bundle_path": None,
        "stop_reason": None,
        "error": None,
        "input_mode": "task_directory" if task_directory is not None else "config",
        "test_evaluation": {
            "status": "not_requested"
            if test_config_path is None and task_directory is None else "not_run"
        },
        "validation_baseline": None,
        "search_profile": search_profile,
        "search_config": search_config,
        "b_analysis": b_analysis,
    }
    prepared: dict[str, Any] | None = None
    training_data = None
    task_context: dict[str, Any] = {}
    stage = "task_discovery" if task_directory is not None else "preparation"
    log_event(
        "session_started",
        stage="session",
        status="started",
        details={"input_mode": summary["input_mode"], "planner_mode": mode},
    )
    try:
        if task_directory is not None:
            from data_b.task_directory import TaskDirectoryError, discover_task_directory

            log_event("task_discovery_started", stage="task_discovery", status="started")
            print(f"[Task] Discovering instructions and development data: {Path(task_directory).resolve()}", flush=True)
            resolution = None
            preflight = None
            if b_analysis == "llm":
                from data_b.llm_analysis import EmptyAnalysisReply, resolve_layout, review_quality

                def record_b_fallback(phase: str) -> None:
                    details = {"phase": phase, "error_code": "INVALID_RESPONSE", "mode": "rules"}
                    summary.setdefault("b_fallbacks", []).append(details)
                    log_event("b_analysis_fallback", stage="task_discovery", details=details)
                    print(f"[B] School {phase} reply was empty. Continuing with the validated default.", flush=True)

                try:
                    preflight = discover_task_directory(
                        Path(task_directory), session_dir / "budget_input", max_trials=max_trials,
                    )
                except TaskDirectoryError:
                    pass
                if preflight is not None:
                    report = preflight["discovery_report"]
                    write_json(session_dir / "budget.discovery.json", report)
                    summary["budget_fingerprint"] = report["budget_fingerprint"]
                if gateway is None:
                    stage = "gateway_init"
                    gateway = gateway_factory(None, session_dir)
                    stage = "task_discovery"
                if not isinstance(gateway.metadata().get("task_budget"), dict):
                    raise ValueError("B LLM analysis requires a persistent D task budget ledger")
                try:
                    resolution = resolve_layout(task_directory, gateway, session_dir / "b_analysis", task_documents)
                except EmptyAnalysisReply:
                    if preflight is None or not gateway.recover_empty_reply():
                        raise
                    record_b_fallback("layout")
            discovered = preflight if b_analysis == "llm" and resolution is None else discover_task_directory(
                Path(task_directory), session_dir / "task_input", max_trials=max_trials,
                resolution=resolution,
            )
            spec = discovered["task_spec"]
            if resolution is not None:
                report = discovered["discovery_report"]
                preflight_path = session_dir / "budget.discovery.json"
                if preflight_path.is_file() and (
                    report["budget_fingerprint"] != read_json(preflight_path)["budget_fingerprint"]
                ):
                    raise ValueError("B resolved different training inputs from the budget preflight")
                budget_root = PROJECT_ROOT if gateway_factory is not None else root
                expected_budget = _automatic_budget_id(budget_root, report)
                current_budget = gateway.metadata()["task_budget"]["budget_id"]
                if (
                    (budget_root / "sessions" / "task-budgets" / f"{expected_budget}.json").is_file()
                    and current_budget != expected_budget
                ):
                    # Keep the billable layout attempt in its original ledger without
                    # claiming that two IDs cover the same training data.
                    summary.pop("budget_fingerprint", None)
                    raise ValueError("These training inputs already have a different task budget ledger")
                summary["budget_fingerprint"] = report["budget_fingerprint"]
                try:
                    training_data = load_training_data(spec)
                    spec, quality = review_quality(spec, gateway, session_dir / "b_analysis", resolution["documents"],
                                                   training_data=training_data)
                except EmptyAnalysisReply:
                    if not gateway.recover_empty_reply():
                        raise
                    record_b_fallback("quality")
                else:
                    discovered["discovery_report"]["data_version"] = spec["data_version"]
                    summary["b_quality_report"] = (session_dir / "b_analysis" / "quality.json").relative_to(root).as_posix()
                    summary["b_quality_findings"] = quality["findings"]
            test_config_path = discovered["test_descriptor_path"]
            if test_config_path is None:
                summary["test_evaluation"] = {"status": "not_requested"}
            task_context = discovered["task_context"]
            write_json(session_dir / "task.discovery.json", discovered["discovery_report"])
            write_json(session_dir / "task_context.json", task_context)
            summary.update(
                task_id=spec["task_id"],
                data_version=spec["data_version"],
                discovery_path=(session_dir / "task.discovery.json").relative_to(root).as_posix(),
                test_descriptor_path=(Path(test_config_path).relative_to(root).as_posix()
                                      if test_config_path is not None else None),
            )
            kind = spec.get("modality", {}).get("kind", "tabular")
            log_event(
                "task_discovery_completed",
                stage="task_discovery",
                details={
                    "task_id": spec["task_id"],
                    "modality": kind,
                    "metric": spec["primary_metric"]["name"],
                },
            )
            print(
                f"[Task] Detected {kind} classification with {spec['primary_metric']['name']}. "
                + ("The task contract is ready. Test data is read after freezing." if test_config_path is not None
                   else "The task contract is ready. No test split was supplied."),
                flush=True,
            )
        write_json(session_dir / "task.snapshot.json", spec)
        spec = load_task_spec(session_dir / "task.snapshot.json")
        if task_directory is None:
            log_event(
                "task_contract_loaded",
                stage="task_discovery",
                details={
                    "task_id": spec["task_id"],
                    "modality": spec.get("modality", {}).get("kind", "tabular"),
                    "metric": spec["primary_metric"]["name"],
                },
            )
        if gateway_factory is not None and gateway is None:
            stage = "gateway_init"
            gateway = gateway_factory(spec, session_dir)
        stage = "preparation"
        log_event("data_preparation_started", stage="preparation", status="started")
        print("[B] Preparing train and validation data.", flush=True)
        prepared = prepare_data(spec, root / "data" / "prepared", training_data=training_data)
        training_data = None
        summary["budget_fingerprint"] = (
            discovered["discovery_report"]["budget_fingerprint"] if task_directory is not None
            else _prepared_budget_fingerprint(prepared)
        )
        if prepared.get("status") != "success":
            detail = prepared.get("error") or {}
            raise ValueError("B data preparation failed: " + str(detail.get("message", "no successful package")))
        print(f"[B] Data version ready. Cached package reused: {prepared['reused']}", flush=True)
        log_event(
            "data_preparation_completed",
            stage="preparation",
            details={"task_id": spec["task_id"], "reused": bool(prepared["reused"])},
        )
        summary["data_manifest_path"] = (
            Path(prepared["data_manifest_path"]).relative_to(root).as_posix()
        )
        summary["evaluation_manifest_path"] = (
            Path(prepared["evaluation_manifest_path"]).relative_to(root).as_posix()
        )
        training_profile = read_json(prepared["profile_path"])
        manifest = read_json(prepared["data_manifest_path"])
        class_counts = list(training_profile.get("train_label_counts", {}).values())
        positive_counts = [count for count in class_counts if type(count) is int and count > 0]
        if len(positive_counts) != len(class_counts) or not positive_counts:
            raise ValueError("The B training profile lacks valid class counts")
        training_profile["n_classes"] = len(class_counts)
        training_profile["majority_share"] = max(class_counts) / sum(class_counts)
        training_profile["class_imbalance_ratio"] = max(class_counts) / min(class_counts)
        stage = "baseline"
        log_event("validation_baseline_started", stage="baseline", status="started")
        validation_baseline = evaluate_training_majority_baseline(
            Path(prepared["evaluation_manifest_path"]),
            training_profile.get("train_label_counts", {}),
        )
        summary["validation_baseline"] = validation_baseline
        log_event(
            "validation_baseline_scored",
            stage="baseline",
            details={
                "metric": validation_baseline["metric"],
                "value": validation_baseline["value"],
            },
        )
        print(
            f"[Baseline] validation {validation_baseline['metric']} = "
            f"{validation_baseline['value']:.6f} "
            f"(training-majority, class {validation_baseline['class_id']})",
            flush=True,
        )
        policy = spec["train_policy"]
        if planner is None:
            planner_kwargs = {
                "task_id": spec["task_id"],
                "data_version": spec["data_version"],
                "n_features": manifest["n_features"],
                "max_trials": policy["max_trials"],
                "seed": policy["random_seed"],
                "timeout": policy["timeout_seconds"],
                "search_profile": search_profile,
                "search_config": search_config,
            }
            planner = (
                SchoolAController(**planner_kwargs, gateway=gateway, local_after_paid_limit=performance)
                if planner_mode == "school"
                else PackageAController(**planner_kwargs)
            )
        summary["strategy_source"] = getattr(planner, "strategy_source", None)
        stage = "search"
        local_encoder = "not_text"
        if manifest.get("representation") == "raw_text":
            local_encoder = encoder_status(
                pretrained,
                n_rows=training_profile["n_train"] + spec["partitions"]["validation"]["n_samples"],
                n_fields=manifest["n_features"],
            )
        long_encoder_run = (
            (encoder_no_time_limit or performance or pretrained == "download")
            and local_encoder in {"cached", "downloadable"}
        )
        session_deadline_enabled = search_config is not None and not (long_encoder_run or performance)
        summary["pretrained"] = {
            "mode": pretrained, "status": local_encoder, "no_time_limit": long_encoder_run,
        }
        if local_encoder in {"cached", "downloadable"} and session_deadline_enabled:
            search_config["finalize_reserve_seconds"] = max(
                search_config["finalize_reserve_seconds"],
                min(300, search_config["max_session_seconds"] // 3),
            )
        if pretrained != "disabled" and local_encoder != "not_text":
            print(f"[C] Local encoder: {local_encoder}", flush=True)
        if long_encoder_run:
            print("[C] Encoder training and final inference have no compute deadline. Progress is reported periodically.", flush=True)
        log_event("encoder_availability", stage=stage, details=summary["pretrained"])
        log_event(
            "search_started",
            stage="search",
            status="started",
            details={"max_trials": policy["max_trials"]},
        )
        summary["performance_mode"] = performance
        representation = manifest.get("representation", "dense_numeric")
        pairwise_text = (
            representation == "raw_text" and manifest["n_features"] == 2
            and len(class_counts) == 2 and task_context.get("pairwise_preference") is True
        )
        catalogue = runtime_capabilities(
            representation,
            pairwise_text=pairwise_text,
            n_train=training_profile["n_train"],
            n_features=manifest["n_features"],
            image_shape=manifest.get("modality", {}).get("image_shape"),
            min_class_count=min(class_counts, default=0),
            native_table=(
                "native_train.json" in prepared.get("artifact_sha256", {})
                and "text" not in spec.get("feature_types", {}).values()
            ),
            performance=performance,
        )
        summary["available_models"] = sorted(catalogue["models"])
        for trial in range(1, policy["max_trials"] + 1):
            elapsed = time.monotonic() - session_started
            if session_deadline_enabled:
                if elapsed + search_config["finalize_reserve_seconds"] >= search_config[
                    "max_session_seconds"
                ]:
                    summary["stop_reason"] = "session_time_finalize_reserve"
                    break
                time_remaining = max(0.0, search_config["max_session_seconds"] - elapsed)
            else:
                time_remaining = None
            trial_catalogue = copy.deepcopy(catalogue)
            encoder_available = local_encoder in {"cached", "downloadable"}
            if encoder_available:
                trial_catalogue["pipeline_components"]["text_encoder"][ENCODER_ID][
                    "availability"
                ] = local_encoder
            else:
                trial_catalogue["pipeline_components"].pop("text_encoder", None)
            context = {
                "task_id": spec["task_id"],
                "data_version": spec["data_version"],
                "training_profile": training_profile,
                "task_context": task_context,
                "primary_metric": spec["primary_metric"],
                "validation_baseline": validation_baseline,
                "data_overview": {
                    "n_train": training_profile["n_train"],
                    "n_features": manifest["n_features"],
                    "representations": [manifest.get("representation", "dense_numeric")],
                },
                "capabilities": trial_catalogue,
                "history": history,
                "remaining_trials": policy["max_trials"] - len(history),
                "search": {
                    "performance": performance,
                    "elapsed_seconds": round(elapsed, 3),
                    "time_remaining_seconds": (
                        None if time_remaining is None else round(time_remaining, 3)
                    ),
                    "max_session_seconds": (
                        search_config["max_session_seconds"] if session_deadline_enabled else None
                    ),
                    "finalize_reserve_seconds": (
                        search_config["finalize_reserve_seconds"] if session_deadline_enabled else None
                    ),
                },
            }
            log_event(
                "planner_decision_started",
                stage="search",
                status="started",
                details={"trial": trial},
            )
            decision = planner.decide(copy.deepcopy(context))
            round_fallback = getattr(planner, "round_fallback_reason", None)
            if round_fallback:
                detail = {"trial": trial, "error_code": round_fallback, "mode": "offline_rules"}
                summary.setdefault("planner_round_fallbacks", []).append(detail)
                log_event("planner_round_fallback", stage="search", details=detail)
                print(
                    f"[A] School planning fallback ({round_fallback}). "
                    "Local rules plan this round. School planning resumes next round.",
                    flush=True,
                )
            local_only_reason = getattr(planner, "local_only_reason", None)
            if local_only_reason and "planner_fallback" not in summary:
                summary["planner_fallback"] = {
                    "trial": trial,
                    "error_code": local_only_reason,
                    "mode": "offline_rules",
                }
                log_event("planner_fallback", stage="search", details=summary["planner_fallback"])
                print(
                    f"[A] School planning ended ({local_only_reason})."
                    " Remaining trials use local rules.",
                    flush=True,
                )
            if not isinstance(decision, dict) or decision.get("action") not in ("run", "stop"):
                raise ValueError("A must return a run or stop decision")
            if not isinstance(decision.get("reason"), str) or len(decision["reason"]) > 2000:
                raise ValueError("The A decision reason must be a string of at most 2000 characters")
            log_event(
                "planner_decision",
                stage="search",
                details={
                    "trial": trial,
                    "action": decision["action"],
                    "model": decision.get("model"),
                    "decision_source": getattr(planner, "last_decision_source", mode),
                },
            )
            if decision["action"] == "stop":
                if set(decision) != {"action", "reason"}:
                    raise ValueError("The stop decision contains unknown fields")
                summary["stop_reason"] = decision["reason"]
                break
            required_decision = {"action", "reason", "model", "params"}
            if set(decision) not in (required_decision, required_decision | {"pipeline"}):
                raise ValueError("The run decision fields do not match the A interface")
            if spec["schema_version"] == "1.0" and "pipeline" in decision:
                raise ValueError("TaskSpec 1.0 does not accept a Plan 2.0 pipeline")
            timeout_seconds = policy["timeout_seconds"]
            if session_deadline_enabled:
                remaining_training = (
                    search_config["max_session_seconds"]
                    - (time.monotonic() - session_started)
                    - search_config["finalize_reserve_seconds"]
                )
                if remaining_training < 1:
                    summary["stop_reason"] = "session_time_finalize_reserve"
                    break
                timeout_seconds = min(timeout_seconds, int(remaining_training))
            experiment_id = f"exp_{session_id[-8:]}_{trial:03d}"
            plan_version = "2.0" if manifest["schema_version"] == "2.0" else "1.0"
            plan_value = {
                "schema_version": plan_version,
                "task_id": spec["task_id"],
                "data_version": spec["data_version"],
                "experiment_id": experiment_id,
                "model": decision["model"],
                "params": decision["params"],
                "random_seed": policy["random_seed"],
                "timeout_seconds": timeout_seconds,
            }
            if plan_version == "2.0":
                representation = manifest["representation"]
                plan_value.update(
                    representation=representation,
                    pipeline=decision.get("pipeline", default_pipeline(representation)),
                )
            if "text_encoder" in plan_value.get("pipeline", {}) and encoder_available:
                encoder_window = (
                    remaining_training if session_deadline_enabled
                    else DOWNLOAD_TIMEOUT + max(30, timeout_seconds)
                )
                download_timeout = min(
                    DOWNLOAD_TIMEOUT, max(0, encoder_window - max(30, timeout_seconds)),
                )
                if local_encoder == "cached":
                    download_timeout = 0
                plan_value["timeout_seconds"] = min(600, max(1, int(encoder_window - download_timeout)))
            else:
                download_timeout = 0
            plan = validate_plan(plan_value)
            log_event(
                "trial_started",
                stage="search",
                status="started",
                details={
                    "trial": trial,
                    "experiment_id": experiment_id,
                    "phase": decision.get("phase", "search"),
                    "model": plan["model"],
                    "decision_source": getattr(planner, "last_decision_source", mode),
                },
            )
            entry = _execute_trial(
                trial=trial,
                decision=decision,
                plan=plan,
                decision_source=getattr(planner, "last_decision_source", mode),
                display_source="offline_rules" if local_only_reason or round_fallback else (
                    "school_llm" if getattr(planner, "last_decision_source", None) == "llm"
                    else "offline_rules" if planner_mode == "school" else mode
                ),
                root=root,
                prepared=prepared,
                session_dir=session_dir,
                spec=spec,
                scorer=scorer,
                pretrained=pretrained,
                encoder_available=encoder_available,
                download_timeout=download_timeout,
                encoder_no_time_limit=long_encoder_run,
                performance=performance,
            )
            if "text_encoder" in plan.get("pipeline", {}):
                if entry["status"] == "success":
                    local_encoder = "cached"
                else:
                    local_encoder = "trial_failed"
                summary["pretrained"]["status"] = local_encoder
            value = (entry.get("evaluation") or {}).get("value")
            if type(value) in (int, float):
                entry["baseline_delta"] = (
                    value - validation_baseline["value"]
                    if spec["primary_metric"]["greater_is_better"]
                    else validation_baseline["value"] - value
                )
            history.append(entry)
            _append_history(history_path, entry)
            evaluation = entry.get("evaluation") or {}
            trial_error = entry.get("error") or {}
            log_event(
                "trial_completed",
                stage="search",
                status=entry["status"],
                details={
                    "trial": trial,
                    "experiment_id": experiment_id,
                    "phase": decision.get("phase", "search"),
                    "metric": evaluation.get("metric"),
                    "value": evaluation.get("value"),
                    "error_code": trial_error.get("code")
                    if isinstance(trial_error, dict)
                    else None,
                },
            )
        else:
            summary["stop_reason"] = "trial_limit_reached"
        offsets = search_config["confirmation_seed_offsets"] if search_config else []
        if offsets:
            remaining_trials = policy["max_trials"] - len(history)
            active_gateway = gateway if gateway is not None else getattr(planner, "gateway", None)
            usage_status = (
                active_gateway.metadata().get("usage_status")
                if active_gateway is not None
                else None
            )
            if remaining_trials <= 0:
                summary["confirmation_skipped_reason"] = "trial_limit_reached"
            elif usage_status in {"unknown", "pending"}:
                summary["confirmation_skipped_reason"] = "gateway_usage_unknown"
            elif getattr(planner, "last_decision_source", None) == "gateway_stop":
                summary["confirmation_skipped_reason"] = "gateway_stop"
            else:
                log_event(
                    "confirmation_started",
                    stage="confirmation",
                    status="started",
                    details={"requested_trials": min(len(offsets), remaining_trials)},
                )
                successful = [
                    item
                    for item in history
                    if item["status"] == "success" and item["evaluation"] is not None
                ]
                ranked = sorted(
                    enumerate(successful),
                    key=lambda pair: (
                        -pair[1]["evaluation"]["value"]
                        if spec["primary_metric"]["greater_is_better"]
                        else pair[1]["evaluation"]["value"],
                        pair[0],
                    ),
                )
                candidates = []
                seen_recipes: set[str] = set()
                for _, candidate in ranked:
                    recipe = json.dumps(
                        {
                            "model": candidate["plan"]["model"],
                            "params": candidate["plan"]["params"],
                            "pipeline": candidate["plan"].get("pipeline"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    if recipe not in seen_recipes:
                        seen_recipes.add(recipe)
                        candidates.append(candidate)
                requested = min(len(offsets), remaining_trials)
                completed = 0
                if not candidates:
                    summary["confirmation_skipped_reason"] = "no_successful_candidate"
                for index, offset in enumerate(offsets[:requested]):
                    if not candidates:
                        break
                    if session_deadline_enabled:
                        remaining_training = (
                            search_config["max_session_seconds"]
                            - (time.monotonic() - session_started)
                            - search_config["finalize_reserve_seconds"]
                        )
                        if remaining_training < 1:
                            summary["confirmation_skipped_reason"] = (
                                "session_time_finalize_reserve"
                            )
                            break
                        timeout_seconds = min(policy["timeout_seconds"], int(remaining_training))
                    else:
                        timeout_seconds = policy["timeout_seconds"]
                    candidate = candidates[min(index, len(candidates) - 1)]
                    candidate_plan = candidate["plan"]
                    trial = len(history) + 1
                    experiment_id = f"exp_{session_id[-8:]}_{trial:03d}"
                    decision = {
                        "action": "run",
                        "model": candidate_plan["model"],
                        "params": copy.deepcopy(candidate_plan["params"]),
                        "reason": (
                            f"Confirm pre-ranked candidate {candidate['experiment_id']} with another seed"
                        ),
                        "phase": "confirmation",
                        "seed_offset": offset,
                    }
                    plan_value = {
                        "schema_version": candidate_plan["schema_version"],
                        "task_id": spec["task_id"],
                        "data_version": spec["data_version"],
                        "experiment_id": experiment_id,
                        "model": candidate_plan["model"],
                        "params": copy.deepcopy(candidate_plan["params"]),
                        "random_seed": (policy["random_seed"] + offset) % 2**32,
                        "timeout_seconds": timeout_seconds,
                    }
                    if candidate_plan["schema_version"] == "2.0":
                        plan_value.update(
                            representation=candidate_plan["representation"],
                            pipeline=copy.deepcopy(candidate_plan["pipeline"]),
                        )
                    plan = validate_plan(plan_value)
                    log_event(
                        "trial_started",
                        stage="confirmation",
                        status="started",
                        details={
                            "trial": trial,
                            "experiment_id": experiment_id,
                            "phase": "confirmation",
                            "model": plan["model"],
                            "decision_source": "confirmation",
                        },
                    )
                    entry = _execute_trial(
                        trial=trial,
                        decision=decision,
                        plan=plan,
                        decision_source="confirmation",
                        root=root,
                        prepared=prepared,
                        session_dir=session_dir,
                        spec=spec,
                        scorer=scorer,
                        pretrained=pretrained,
                        encoder_available="text_encoder" in plan.get("pipeline", {}),
                        download_timeout=0,
                        encoder_no_time_limit=long_encoder_run,
                        performance=performance,
                    )
                    value = (entry.get("evaluation") or {}).get("value")
                    if type(value) in (int, float):
                        entry["baseline_delta"] = (
                            value - validation_baseline["value"]
                            if spec["primary_metric"]["greater_is_better"]
                            else validation_baseline["value"] - value
                        )
                    history.append(entry)
                    _append_history(history_path, entry)
                    evaluation = entry.get("evaluation") or {}
                    trial_error = entry.get("error") or {}
                    log_event(
                        "trial_completed",
                        stage="confirmation",
                        status=entry["status"],
                        details={
                            "trial": trial,
                            "experiment_id": experiment_id,
                            "phase": "confirmation",
                            "metric": evaluation.get("metric"),
                            "value": evaluation.get("value"),
                            "error_code": trial_error.get("code")
                            if isinstance(trial_error, dict)
                            else None,
                        },
                    )
                    completed += 1
                summary["confirmation_trials"] = completed
                if completed < requested and "confirmation_skipped_reason" not in summary:
                    summary["confirmation_skipped_reason"] = "not_all_requested_runs_completed"
            if "confirmation_skipped_reason" in summary:
                log_event(
                    "confirmation_skipped",
                    stage="confirmation",
                    status="skipped",
                    details={"reason": summary["confirmation_skipped_reason"]},
                )
            else:
                log_event(
                    "confirmation_completed",
                    stage="confirmation",
                    status="completed",
                    details={"completed_trials": summary.get("confirmation_trials", 0)},
                )
        log_event(
            "search_completed",
            stage="search",
            status="completed",
            details={
                "attempted_trials": len(history),
                "stop_reason": summary["stop_reason"],
            },
        )
        best = _best_scored_trial(history, spec["primary_metric"])
        if best is not None:
            stage = "freeze"
            ensemble = None
            if performance and evaluator is None:
                from evaluation_d.ensemble import compare_ensembles

                ensemble, comparisons = compare_ensembles(
                    history, root / "runs", Path(prepared["evaluation_manifest_path"]),
                    spec["primary_metric"],
                )
                summary["ensemble_comparisons"] = comparisons
                if ensemble:
                    members, score = ensemble
                    with_score = {**best, "evaluation": score}
                    bundle = freeze_best(
                        spec, prepared, with_score, root, session_id, members=members,
                    )
                    best = with_score
                    summary["ensemble_members"] = [member["experiment_id"] for member in members]
                    print(f"[Ensemble] validation {score['metric']} = {score['value']:.6f}", flush=True)
            if ensemble is None:
                bundle = freeze_best(spec, prepared, best, root, session_id)
            log_event(
                "bundle_frozen",
                stage="freeze",
                details={
                    "final_id": bundle.parent.name,
                    "experiment_id": best["experiment_id"],
                    "metric": best["evaluation"]["metric"],
                    "value": best["evaluation"]["value"],
                },
            )
            summary.update(
                status="success",
                best_experiment_id=best["experiment_id"],
                best_validation_metric={
                    "name": spec["primary_metric"]["name"],
                    "value": best["evaluation"]["value"],
                    "greater_is_better": spec["primary_metric"]["greater_is_better"],
                },
                best_validation_accuracy=(
                    best["evaluation"]["value"]
                    if spec["primary_metric"]["name"] == "accuracy"
                    else None
                ),
                bundle_path=bundle.relative_to(root).as_posix(),
                final_id=bundle.parent.name,
            )
            print(f"[Final plan] {best['experiment_id']}", flush=True)
            print(
                f"[Validation] {spec['primary_metric']['name']} = "
                f"{best['evaluation']['value']:.6f}",
                flush=True,
            )
            validation_gain = (
                best["evaluation"]["value"] - validation_baseline["value"]
                if spec["primary_metric"]["greater_is_better"]
                else validation_baseline["value"] - best["evaluation"]["value"]
            )
            summary["validation_gain_over_baseline"] = validation_gain
            summary["validation_quality"] = (
                "above_training_majority_baseline"
                if validation_gain > 0
                else "at_or_below_training_majority_baseline"
            )
            print(
                f"[Quality] Gain over the training-majority baseline {validation_gain:+.6f} "
                f"({summary['validation_quality']})",
                flush=True,
            )
            print(f"[Bundle] {bundle.relative_to(root).as_posix()}", flush=True)
            if test_config_path is not None:
                stage = "final_test"
                log_event("final_test_started", stage="final_test", status="started")
                print("[Test] Evaluating the frozen bundle on the independent test split.", flush=True)
                encoder_test_options = {}
                if (bundle.parent / "encoder.json").is_file():
                    if long_encoder_run:
                        encoder_test_options["encoder_timeout"] = None
                    elif session_deadline_enabled:
                        encoder_test_options["encoder_timeout"] = max(
                            1, search_config["max_session_seconds"] - (time.monotonic() - session_started)
                        )
                test_report = evaluate_frozen_bundle(
                    bundle, test_config_path, session_dir, **encoder_test_options,
                )
                summary["test_evaluation"] = {
                    "status": "success",
                    "metric": test_report["metric"],
                    "value": test_report["value"],
                    "accuracy": test_report["accuracy"],
                    "n_samples": test_report["n_samples"],
                    "report_path": (session_dir / "test_evaluation.json")
                    .relative_to(root)
                    .as_posix(),
                    "predictions_path": (session_dir / "test_predictions.csv")
                    .relative_to(root)
                    .as_posix(),
                }
                log_event(
                    "final_test_completed",
                    stage="final_test",
                    details={
                        "metric": test_report["metric"],
                        "value": test_report["value"],
                        "n_samples": test_report["n_samples"],
                        "report_path": summary["test_evaluation"]["report_path"],
                    },
                )
                for line in format_test_result(test_report):
                    print(line, flush=True)
                print(
                    f"[Report] {(session_dir / 'test_evaluation.json').relative_to(root).as_posix()}",
                    flush=True,
                )
                print(
                    f"[Predictions] {(session_dir / 'test_predictions.csv').relative_to(root).as_posix()}",
                    flush=True,
                )
            else:
                log_event(
                    "final_test_not_requested",
                    stage="final_test",
                    status="not_requested",
                )
        else:
            summary["error"] = {
                "code": "NO_SUCCESSFUL_TRIAL",
                "message": "No successfully scored trial is available to freeze",
            }
    except Exception as exc:
        summary["status"] = "failed"
        if isinstance(exc, FinalTestError):
            summary["error"] = {
                "code": FinalTestError.code,
                "stage": "final_test",
                "message": str(exc),
            }
            print(f"[Test] Evaluation of the frozen bundle failed: {exc}", flush=True)
            report_path = session_dir / "test_evaluation.json"
            if report_path.is_file():
                report = read_json(report_path)
                summary["test_evaluation"] = {
                    "status": "failed",
                    "error": report.get("error"),
                    "report_path": report_path.relative_to(root).as_posix(),
                    "predictions_path": (
                        (session_dir / "test_predictions.csv").relative_to(root).as_posix()
                        if (session_dir / "test_predictions.csv").is_file()
                        else None
                    ),
                }
            log_event(
                "stage_failed",
                stage="final_test",
                status="failed",
                details={"error_code": FinalTestError.code},
            )
        else:
            summary["error"] = {
                "code": getattr(
                    exc, "code", "TASK_DISCOVERY_FAILED" if stage == "task_discovery" else "SESSION_ERROR"
                ),
                "stage": stage,
                "message": str(exc),
            }
            log_event(
                "stage_failed",
                stage=summary["error"]["stage"],
                status="failed",
                details={"error_code": summary["error"]["code"]},
            )
    if (
        summary["status"] != "success" and prepared is not None
        and spec is not None and summary["bundle_path"] is None
    ):
        best = _best_scored_trial(history, spec["primary_metric"])
        if best is not None:
            try:
                partial_bundle = freeze_best(
                    spec, prepared, best, root, session_id, partial=True
                )
                summary.update(
                    partial_bundle_path=partial_bundle.relative_to(root).as_posix(),
                    partial_id=partial_bundle.parent.name,
                    partial_best_experiment_id=best["experiment_id"],
                    partial_validation_metric={
                        "name": spec["primary_metric"]["name"],
                        "value": best["evaluation"]["value"],
                        "greater_is_better": spec["primary_metric"]["greater_is_better"],
                    },
                )
                log_event(
                    "partial_bundle_frozen",
                    stage="freeze",
                    status="partial",
                    details={
                        "final_id": partial_bundle.parent.name,
                        "experiment_id": best["experiment_id"],
                        "metric": best["evaluation"]["metric"],
                        "value": best["evaluation"]["value"],
                    },
                )
            except Exception as partial_exc:
                summary["partial_bundle_error"] = {
                    "code": getattr(partial_exc, "code", "PARTIAL_FREEZE_ERROR"),
                    "message": str(partial_exc),
                }
                log_event(
                    "partial_freeze_failed",
                    stage="freeze",
                    status="failed",
                    details={"error_code": summary["partial_bundle_error"]["code"]},
                )
    summary["trial_failures"] = [
        {"experiment_id": entry["experiment_id"], "model": entry["plan"]["model"],
         "status": entry["status"], "error": entry.get("error")}
        for entry in history if entry["status"] != "success"
    ]
    encoder = summary.get("pretrained", {})
    unavailable = []
    if encoder.get("mode") == "download" and encoder.get("status") in {
        "disk_limit", "missing_dependencies",
    }:
        unavailable.append({"resource": ENCODER_ID, "reason": encoder["status"]})
    summary["unavailable_resources"] = unavailable
    if summary["status"] == "success" and require_complete and (
        summary["trial_failures"] or unavailable
    ):
        summary["status"] = "degraded"
        print(
            f"[Degraded] {len(summary['trial_failures'])} selected trials failed. "
            f"{len(unavailable)} resources were unavailable. The successful result was saved.",
            flush=True,
        )
    summary["attempted_trials"] = len(history)
    summary["elapsed_seconds"] = round(time.monotonic() - session_started, 3)
    summary["last_decision_source"] = getattr(planner, "last_decision_source", None)
    summary["decision_sources"] = [entry["decision_source"] for entry in history]
    active_gateway = gateway if gateway is not None else getattr(planner, "gateway", None)
    if active_gateway is not None:
        summary["llm_gateway"] = active_gateway.metadata()
        summary["real_llm_calls"] = summary["llm_gateway"]["real_llm_calls"]
    if summary["status"] == "failed" and summary.get("error"):
        error = summary["error"]
        if error.get("stage") != "final_test":
            print(f"[Failure:{error['code']}] {error['message']}", flush=True)
    write_json(session_dir / "summary.json", summary)
    active_api_session = (summary.get("llm_gateway") or {}).get("session_id")
    log_event(
        "session_completed",
        stage="session",
        status=summary["status"],
        details={
            "attempted_trials": summary["attempted_trials"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "stop_reason": summary["stop_reason"],
            "error_code": (summary.get("error") or {}).get("code"),
            "test_status": (summary.get("test_evaluation") or {}).get("status"),
            "api_session_id": active_api_session,
        },
    )
    print(f"[Done] {summary['status']} | summary: {session_dir / 'summary.json'}", flush=True)
    print(f"[Events] {events_path.relative_to(root).as_posix()}", flush=True)
    return session_dir / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SuperAgent v2 classification, planning, training, and evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="Search on train and validation and evaluate test after freezing"
    )
    task_input = run_parser.add_mutually_exclusive_group()
    task_input.add_argument("--task", type=Path, help="Discover the task directory and evaluate the frozen model")
    task_input.add_argument("--config", type=Path, help="Use a manual TaskSpec for development")
    run_parser.add_argument("--output-root", type=Path)
    run_parser.add_argument("--planner", choices=("offline", "school"), default="offline")
    run_parser.add_argument("--pretrained", choices=ENCODER_MODES,
                            help="Local text encoder policy. Task-directory runs default to on-demand download")
    run_parser.add_argument("--encoder-no-time-limit", action="store_true",
                            help="Remove local encoder training and final prediction deadlines")
    run_parser.add_argument("--b-analysis", choices=("rules", "llm"), default="rules")
    run_parser.add_argument("--task-doc", action="append", help="Relative task document path. Repeat for additional documents")
    run_parser.add_argument(
        "--llm-config", type=Path, default=PROJECT_ROOT / "configs" / "school-api.json"
    )
    run_parser.add_argument("--llm-max-calls", type=int, help="Defaults to 20 for --task and 1 for --config")
    run_parser.add_argument("--llm-budget-rmb", help="Defaults to CNY 20 for --task and CNY 0.10 for --config")
    run_parser.add_argument("--require-complete", action="store_true",
                            help="Return degraded status if a selected trial fails")
    run_parser.add_argument("--max-trials", type=int, help="Local trial limit for task-directory mode")
    run_parser.add_argument("--performance", action="store_true",
                            help="Expand local search and continue locally after the school quota is exhausted")
    run_parser.add_argument(
        "--billing-profile",
        choices=(
            "deepseek-v4.1-flash-offpeak-2026-10-21",
            "deepseek-v4.1-flash-peak-2026-10-21",
        ),
        help="Select the reference tariff for the task ledger",
    )
    run_parser.add_argument("--task-budget-id", help="Stable budget ledger identifier across runs")
    run_parser.add_argument("--task-budget-rmb", default="20")
    run_parser.add_argument(
        "--search-config",
        type=Path,
        help="Override search settings. Task-directory runs default to expanded search",
    )
    run_parser.add_argument(
        "--test-config",
        type=Path,
        help="Configure independent test evaluation of the selected frozen bundle",
    )
    test_parser = subparsers.add_parser(
        "evaluate-test", help="Evaluate an existing final bundle on the independent test split"
    )
    test_parser.add_argument("--bundle", type=Path, required=True)
    test_parser.add_argument("--test-config", type=Path, required=True)
    test_parser.add_argument("--output-root", type=Path)
    test_parser.add_argument("--pretrained", choices=("cache-only", "download"), default="cache-only")
    test_parser.add_argument("--encoder-no-time-limit", action="store_true",
                             help="Remove the local encoder deadline for final prediction")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate-test":
            root = args.output_root.resolve() if args.output_root else PROJECT_ROOT
            output_dir = root / "sessions" / (
                "test_evaluation_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                + uuid.uuid4().hex[:8]
            )
            output_dir.mkdir(parents=True, exist_ok=False)
            try:
                report = evaluate_frozen_bundle(
                    args.bundle, args.test_config, output_dir, pretrained=args.pretrained,
                    **({"encoder_timeout": None} if args.encoder_no_time_limit else {}),
                )
            except FinalTestError as exc:
                report_path = output_dir / "test_evaluation.json"
                if report_path.is_file():
                    report = read_json(report_path)
                    for line in format_test_result(report):
                        print(line, flush=True)
                    print(f"[Report] {report_path}", flush=True)
                else:
                    print(f"[Test] Failed: {exc}", file=sys.stderr, flush=True)
                return 1
            for line in format_test_result(report):
                print(line, flush=True)
            print(f"[Report] {output_dir / 'test_evaluation.json'}", flush=True)
            print(f"[Predictions] {output_dir / 'test_predictions.csv'}", flush=True)
            return 0
        if args.task is not None and args.test_config is not None:
            raise ValueError("--task discovers the test split. Omit --test-config")
        automatic = args.task is not None
        if args.b_analysis == "llm" and (not automatic or args.planner != "school"):
            raise ValueError("--b-analysis llm requires --task and --planner school")
        billing_profile = args.billing_profile
        if args.performance and args.planner == "school" and not automatic:
            raise ValueError("School performance mode requires --task for the task budget ledger")
        if automatic and args.planner == "school":
            billing_profile = billing_profile or "deepseek-v4.1-flash-offpeak-2026-10-21"
        elif bool(billing_profile) != bool(args.task_budget_id):
            raise ValueError("Provide --billing-profile and --task-budget-id together")
        if billing_profile and args.planner != "school":
            raise ValueError("Task budget ledgers require --planner school")
        gateway_factory = None
        if args.planner == "school":
            from evaluation_d.env_config import load_api_key
            from evaluation_d.school_gateway import SchoolGateway

            root = args.output_root.resolve() if args.output_root is not None else PROJECT_ROOT
            max_calls = args.llm_max_calls if args.llm_max_calls is not None else (20 if automatic else 1)
            budget_rmb = args.llm_budget_rmb if args.llm_budget_rmb is not None else ("20" if automatic else "0.10")

            if automatic:
                for field, amount in (
                    ("--task-budget-rmb", args.task_budget_rmb), ("--llm-budget-rmb", budget_rmb),
                ):
                    try:
                        parsed = Decimal(amount)
                    except InvalidOperation:
                        raise ValueError(f"{field} requires a valid amount") from None
                    if not parsed.is_finite() or not 0 < parsed <= 20:
                        raise ValueError(f"{field} must be greater than zero and at most CNY 20")

            def make_gateway(spec: dict[str, Any] | None, session_dir: Path) -> Any:
                config = read_json(args.llm_config, error_code="API_CONFIG_INVALID")
                if billing_profile:
                    print(f"[Task budget tariff] {billing_profile}", flush=True)
                if config.get("pricing_profile"):
                    print(f"[Gateway cost estimate] {config['pricing_profile']}", flush=True)
                # Load settings only from this project's .env after budget routing.
                task_budget = None
                if billing_profile:
                    from evaluation_d.task_budget import TaskBudgetLedger

                    ledger_root = PROJECT_ROOT if automatic else root
                    budget_id = args.task_budget_id
                    if automatic:
                        report_path = session_dir / (
                            "budget.discovery.json" if spec is None else "task.discovery.json"
                        )
                        if report_path.is_file():
                            canonical_id = _automatic_budget_id(ledger_root, read_json(report_path))
                            if budget_id is not None and budget_id != canonical_id:
                                raise ValueError("Use the existing ledger for these training inputs")
                            budget_id = canonical_id
                        elif budget_id is None:
                            raise ValueError(
                                "Unrecognized task layout requires --task-budget-id before billable discovery"
                            )
                    task_budget = TaskBudgetLedger(
                        ledger_root / "sessions" / "task-budgets",
                        budget_id,
                        budget_rmb=args.task_budget_rmb,
                        profile=billing_profile,
                    )
                    print(
                        f"[Budget] Task ledger {budget_id}. Limit CNY {args.task_budget_rmb}. "
                        "Later runs of this task share the ledger.",
                        flush=True,
                    )
                api_key = load_api_key(PROJECT_ROOT / ".env")
                print("School API mode. Credentials loaded from the project .env.", flush=True)
                return SchoolGateway(
                    config,
                    api_key,
                    root / "sessions",
                    max_calls=max_calls,
                    budget_rmb=budget_rmb,
                    task_budget_ledger=task_budget,
                )

            gateway_factory = make_gateway
        path = run_project(
            args.config,
            args.output_root,
            planner_mode=args.planner,
            gateway_factory=gateway_factory,
            search_config_path=args.search_config,
            test_config_path=args.test_config,
            task_directory=args.task,
            max_trials=args.max_trials if args.max_trials is not None else (80 if args.performance else 20),
            performance=args.performance,
            require_complete=args.require_complete,
            pretrained=args.pretrained or ("download" if automatic else "disabled"),
            encoder_no_time_limit=args.encoder_no_time_limit,
            b_analysis=args.b_analysis,
            task_documents=args.task_doc,
        )
        return 0 if read_json(path)["status"] == "success" else 1
    except (EOFError, KeyboardInterrupt):
        print("Run stopped.", file=sys.stderr)
        return 2
    except (OSError, ValueError, GatewayError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
