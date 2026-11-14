"""阅读入口：B 准备数据 -> A 决定方案 -> C 执行 -> D 评分 -> A 读取反馈。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_a.controller import PackageAController, SchoolAController
from agent_a.search_config import load_search_config
from data_b.prepare import prepare_data
from evaluation_d.evaluate import evaluate
from evaluation_d.final_test import FinalTestError, evaluate_frozen_bundle, format_test_result
from evaluation_d.school_gateway import GatewayError
from evaluation_d.metric_contracts import metric_in_domain, validate_metric_spec
from executor_c.contracts import capabilities, read_json, resolve_data_path, sha256_file, validate_plan, write_json
from executor_c.diagnostics import safe_history_view
from executor_c.executor import execute
from executor_c.model_registry import default_pipeline

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
    """配置中的 source_root 相对配置文件解析，迁移项目不依赖机器绝对路径。"""
    path = Path(config_path).resolve()
    spec = read_json(path, error_code="PLAN_INVALID")
    if spec.get("schema_version") == "1.0":
        if set(spec) != TASK_FIELDS:
            raise ValueError("TaskSpec 1.0 字段不符合表格任务约定")
    elif spec.get("schema_version") == "2.0":
        if set(spec) != TASK_FIELDS_V2:
            raise ValueError("TaskSpec 2.0 字段不符合文本/图像任务约定")
    else:
        raise ValueError("不支持此 TaskSpec schema_version")
    if not isinstance(spec["source_root"], str) or not spec["source_root"]:
        raise ValueError("source_root 必须是非空路径字符串")
    spec["primary_metric"] = validate_metric_spec(spec["primary_metric"])
    spec["source_root"] = str((path.parent / spec["source_root"]).resolve())
    policy = spec["train_policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "max_trials",
        "random_seed",
        "timeout_seconds",
    }:
        raise ValueError("train_policy 需要 max_trials/random_seed/timeout_seconds")
    if type(policy["max_trials"]) is not int or not 1 <= policy["max_trials"] <= 20:
        raise ValueError("max_trials 必须是 1–20 的整数")
    # 借用 C 的严格检查核对种子、时间限制和安全身份字段。
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


def _automatic_budget_id(root: Path, discovery: dict[str, Any]) -> str:
    """Retain a task's previous ledger when moving from manual to automatic input."""
    fingerprint = discovery["budget_fingerprint"]
    default_id = "auto_" + fingerprint[:40]
    ledger_root = root / "sessions" / "task-budgets"
    candidates = {default_id} if (ledger_root / f"{default_id}.json").is_file() else set()

    def artifact(relative: Any, expected_name: str) -> Path:
        if (
            not isinstance(relative, str) or Path(relative).name != expected_name
            or any(part.startswith(".") for part in Path(relative).parts)
        ):
            raise ValueError("历史任务工件路径不符合约定")
        return resolve_data_path(root, relative)

    for summary_path in (root / "sessions").glob("session_*/summary.json"):
        historical = read_json(summary_path)
        accounting = (historical.get("llm_gateway") or {}).get("task_budget")
        if not isinstance(accounting, dict) or not accounting.get("budget_id"):
            continue
        previous_fingerprint = None
        if historical.get("discovery_path"):
            previous_fingerprint = read_json(
                artifact(historical["discovery_path"], "task.discovery.json")
            ).get("budget_fingerprint")
        elif historical.get("data_manifest_path"):
            manifest_path = artifact(historical["data_manifest_path"], "data_manifest.json")
            package_path = artifact(
                (manifest_path.parent / "package_result.json").relative_to(root).as_posix(),
                "package_result.json",
            )
            train = read_json(package_path).get("source_file_sha256", {}).get("train", {})
            digests = [
                item["sha256"] for item in train.values()
                if isinstance(item, dict) and isinstance(item.get("sha256"), str)
            ]
            if digests:
                content = json.dumps(sorted(digests), ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, indent=2) + "\n"
                previous_fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if previous_fingerprint == fingerprint:
            candidates.add(accounting["budget_id"])
    if len(candidates) > 1:
        raise ValueError("同一任务发现多个历史预算账本，拒绝另建或重置任务预算")
    if candidates:
        budget_id = next(iter(candidates))
        if not (ledger_root / f"{budget_id}.json").is_file():
            raise ValueError("同任务的历史预算账本已缺失，拒绝从零创建")
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
) -> Path:
    """冻结选定模型，不重新使用 validation 拟合；bundle.json 最后发布。"""
    run_dir = (output_root / "runs" / best["result_relpath"]).parent
    worker = read_json(run_dir / "worker_metadata.json")
    manifest = read_json(prepared["data_manifest_path"])
    is_v1 = manifest["schema_version"] == "1.0"
    preprocessor: Path | None = None
    if is_v1:
        preprocessor = Path(prepared["preprocessor_path"])
        if worker["input_hashes"].get("preprocessor") != sha256_file(preprocessor):
            raise ValueError("选定实验的 B 预处理器已改变，拒绝混合版本冻结")
    final_id = ("partial_" if partial else "final_") + session_id[-8:]
    frozen = output_root / "submission" / spec["task_id"] / final_id
    frozen.mkdir(parents=True, exist_ok=False)
    sources = {
        "model.joblib": run_dir / "model.joblib",
        "label_mapping.json": Path(prepared["label_mapping_path"]),
        "plan.json": run_dir / "plan.json",
        "environment.json": run_dir / "environment.json",
    }
    if preprocessor is not None:
        sources["preprocessor.joblib"] = preprocessor
    hashes = {}
    for name, source in sources.items():
        _copy_exclusive(source, frozen / name)
        hashes[name] = sha256_file(frozen / name)
        if hashes[name] != sha256_file(source):
            raise ValueError("冻结复制过程中产物发生变化")
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
        raise ValueError("D 未返回成功评分")
    if any(score.get(key) != plan[key] for key in ("task_id", "data_version", "experiment_id")):
        raise ValueError("D 评分身份与本次实验不一致")
    value = score.get("value")
    if (
        score.get("metric") != metric_spec["name"]
        or score.get("greater_is_better") is not metric_spec["greater_is_better"]
        or not metric_in_domain(metric_spec["name"], value)
    ):
        raise ValueError("D 评分与任务 primary_metric 不匹配或超出指标定义域")
    if type(score.get("n_samples")) is not int or score["n_samples"] != expected_rows:
        raise ValueError("D 的评分样本数必须等于固定验证集行数")


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
) -> dict[str, Any]:
    experiment_id = plan["experiment_id"]
    write_json(
        session_dir / f"decision_{trial:03d}.json",
        {**decision, "resolved_plan": plan, "decision_source": decision_source},
    )
    print(
        f"[A:{display_source or decision_source}] 第 {trial} 轮：{plan['model']}，{decision['reason']}",
        flush=True,
    )
    print(f"[A→C] params = {json.dumps(plan['params'], ensure_ascii=False)}", flush=True)
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
    try:
        relative = execute(plan, prepared["data_manifest_path"], root / "runs")
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
        print(f"[C] {experiment_id}：{execution['status']}", flush=True)
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
    except Exception as exc:
        entry.update(status="failed", error={"code": "TRIAL_ERROR", "message": str(exc)})
        if entry["result_relpath"]:
            diagnostics_path = (
                root / "runs" / entry["result_relpath"]
            ).parent / "diagnostics.json"
            if diagnostics_path.is_file():
                entry["diagnostics"] = safe_history_view(read_json(diagnostics_path))
        print(f"[循环] 记录本轮失败：{type(exc).__name__}", flush=True)
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
    gateway_factory: Callable[[dict[str, Any], Path], Any] | None = None,
    max_trials: int = 20,
) -> Path:
    """从任务目录自动接入或读取开发配置；冻结 validation 最优方案后评估 test。"""
    if planner_mode not in {"offline", "school"}:
        raise ValueError("planner_mode 必须是 offline 或 school")
    if task_directory is not None and (config_path is not None or test_config_path is not None):
        raise ValueError("--task 自动接入不接受人工 --config/--test-config")
    if gateway is not None and gateway_factory is not None:
        raise ValueError("gateway 与 gateway_factory 不可同时提供")
    if controller is not None and (
        gateway is not None or gateway_factory is not None or planner_mode != "offline"
    ):
        raise ValueError("自定义 controller 与主程序的 planner/gateway 配置不可同时使用")
    if controller is None and planner_mode == "school" and gateway is None and gateway_factory is None:
        raise ValueError("school 模式必须提供运行时学校网关")
    spec = (
        load_task_spec(config_path or PROJECT_ROOT / "configs" / "coupon.json")
        if task_directory is None else None
    )
    if task_directory is not None and search_config_path is None:
        search_config_path = PROJECT_ROOT / "configs" / "search-defaults.json"
    search_config = load_search_config(search_config_path) if search_config_path else None
    search_profile = "expanded" if search_config is not None else "legacy"
    session_started = time.monotonic()
    root = Path(output_root).resolve() if output_root is not None else PROJECT_ROOT
    session_id = (
        datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    )
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    history_path = session_dir / "history.jsonl"
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
        "search_profile": search_profile,
        "search_config": search_config,
    }
    prepared: dict[str, Any] | None = None
    task_context: dict[str, Any] = {}
    stage = "task_discovery" if task_directory is not None else "preparation"
    try:
        if task_directory is not None:
            from data_b.task_directory import discover_task_directory

            print(f"[任务] 自动读取任务说明和开发数据：{Path(task_directory).resolve()}", flush=True)
            discovered = discover_task_directory(
                Path(task_directory), session_dir / "task_input", max_trials=max_trials
            )
            spec = discovered["task_spec"]
            test_config_path = discovered["test_descriptor_path"]
            task_context = discovered["task_context"]
            write_json(session_dir / "task.discovery.json", discovered["discovery_report"])
            write_json(session_dir / "task_context.json", task_context)
            summary.update(
                task_id=spec["task_id"],
                data_version=spec["data_version"],
                discovery_path=(session_dir / "task.discovery.json").relative_to(root).as_posix(),
                test_descriptor_path=Path(test_config_path).relative_to(root).as_posix(),
            )
            kind = spec.get("modality", {}).get("kind", "tabular")
            print(
                f"[任务] 已识别 {kind} 分类，指标 {spec['primary_metric']['name']}；"
                "自动生成内部任务契约，test 将在冻结后读取。",
                flush=True,
            )
        write_json(session_dir / "task.snapshot.json", spec)
        spec = load_task_spec(session_dir / "task.snapshot.json")
        if gateway_factory is not None:
            stage = "gateway_init"
            gateway = gateway_factory(spec, session_dir)
        stage = "preparation"
        print("[B] 读取任务并准备训练/验证数据。", flush=True)
        prepared = prepare_data(spec, root / "data" / "prepared")
        if prepared.get("status") != "success":
            detail = prepared.get("error") or {}
            raise ValueError("B 数据准备失败：" + str(detail.get("message", "未返回成功数据包")))
        print(f"[B] 数据版本就绪，复用已有版本：{prepared['reused']}", flush=True)
        summary["data_manifest_path"] = (
            Path(prepared["data_manifest_path"]).relative_to(root).as_posix()
        )
        summary["evaluation_manifest_path"] = (
            Path(prepared["evaluation_manifest_path"]).relative_to(root).as_posix()
        )
        training_profile = read_json(prepared["profile_path"])
        manifest = read_json(prepared["data_manifest_path"])
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
                SchoolAController(**planner_kwargs, gateway=gateway)
                if planner_mode == "school"
                else PackageAController(**planner_kwargs)
            )
        summary["strategy_source"] = getattr(planner, "strategy_source", None)
        stage = "search"
        for trial in range(1, policy["max_trials"] + 1):
            elapsed = time.monotonic() - session_started
            if search_config is not None:
                if elapsed + search_config["finalize_reserve_seconds"] >= search_config[
                    "max_session_seconds"
                ]:
                    summary["stop_reason"] = "session_time_finalize_reserve"
                    break
                time_remaining = max(0.0, search_config["max_session_seconds"] - elapsed)
            else:
                time_remaining = None
            context = {
                "task_id": spec["task_id"],
                "data_version": spec["data_version"],
                "training_profile": training_profile,
                "task_context": task_context,
                "primary_metric": spec["primary_metric"],
                "data_overview": {
                    "n_train": training_profile["n_train"],
                    "n_features": manifest["n_features"],
                    "representations": [manifest.get("representation", "dense_numeric")],
                },
                "capabilities": capabilities(),
                "history": history,
                "remaining_trials": policy["max_trials"] - len(history),
                "search": {
                    "elapsed_seconds": round(elapsed, 3),
                    "time_remaining_seconds": (
                        None if time_remaining is None else round(time_remaining, 3)
                    ),
                    "max_session_seconds": (
                        search_config["max_session_seconds"] if search_config else None
                    ),
                    "finalize_reserve_seconds": (
                        search_config["finalize_reserve_seconds"] if search_config else None
                    ),
                },
            }
            decision = planner.decide(copy.deepcopy(context))
            if not isinstance(decision, dict) or decision.get("action") not in ("run", "stop"):
                raise ValueError("A 必须返回 run 或 stop 决定")
            if not isinstance(decision.get("reason"), str) or len(decision["reason"]) > 2000:
                raise ValueError("A 的 reason 必须是长度不超过 2000 的字符串")
            if decision["action"] == "stop":
                if set(decision) != {"action", "reason"}:
                    raise ValueError("stop 决定含未知字段")
                summary["stop_reason"] = decision["reason"]
                break
            required_decision = {"action", "reason", "model", "params"}
            if set(decision) not in (required_decision, required_decision | {"pipeline"}):
                raise ValueError("run 决定字段不符合 A 的接口")
            if spec["schema_version"] == "1.0" and "pipeline" in decision:
                raise ValueError("TaskSpec 1.0 不接受 Plan 2.0 pipeline")
            timeout_seconds = policy["timeout_seconds"]
            if search_config is not None:
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
            plan = validate_plan(plan_value)
            entry = _execute_trial(
                trial=trial,
                decision=decision,
                plan=plan,
                decision_source=getattr(planner, "last_decision_source", mode),
                display_source=mode,
                root=root,
                prepared=prepared,
                session_dir=session_dir,
                spec=spec,
                scorer=scorer,
            )
            history.append(entry)
            _append_history(history_path, entry)
        else:
            summary["stop_reason"] = "达到主程序的最大实验次数"
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
                    if search_config is not None:
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
                            f"对预先排序候选 {candidate['experiment_id']} 进行 seed 复核"
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
                    )
                    history.append(entry)
                    _append_history(history_path, entry)
                    completed += 1
                summary["confirmation_trials"] = completed
                if completed < requested and "confirmation_skipped_reason" not in summary:
                    summary["confirmation_skipped_reason"] = "not_all_requested_runs_completed"
        best = _best_scored_trial(history, spec["primary_metric"])
        if best is not None:
            stage = "freeze"
            bundle = freeze_best(spec, prepared, best, root, session_id)
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
            print(f"[最终方案] {best['experiment_id']}", flush=True)
            print(
                f"[Validation] {spec['primary_metric']['name']} = "
                f"{best['evaluation']['value']:.6f}",
                flush=True,
            )
            print(f"[工件] {bundle.relative_to(root).as_posix()}", flush=True)
            if test_config_path is not None:
                stage = "final_test"
                print("[Test] 最终方案已冻结，开始独立评估 test。", flush=True)
                test_report = evaluate_frozen_bundle(
                    bundle, test_config_path, session_dir
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
                for line in format_test_result(test_report):
                    print(line, flush=True)
                print(
                    f"[报告] {(session_dir / 'test_evaluation.json').relative_to(root).as_posix()}",
                    flush=True,
                )
                print(
                    f"[预测] {(session_dir / 'test_predictions.csv').relative_to(root).as_posix()}",
                    flush=True,
                )
        else:
            summary["error"] = {
                "code": "NO_SUCCESSFUL_TRIAL",
                "message": "没有可冻结的成功评分实验",
            }
    except Exception as exc:
        summary["status"] = "failed"
        if isinstance(exc, FinalTestError):
            summary["error"] = {
                "code": FinalTestError.code,
                "stage": "final_test",
                "message": str(exc),
            }
            print(f"[Test] final bundle 已冻结，但 test 评估失败：{exc}", flush=True)
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
        else:
            summary["error"] = {
                "code": getattr(
                    exc, "code", "TASK_DISCOVERY_FAILED" if stage == "task_discovery" else "SESSION_ERROR"
                ),
                "stage": stage,
                "message": str(exc),
            }
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
            except Exception as partial_exc:
                summary["partial_bundle_error"] = {
                    "code": getattr(partial_exc, "code", "PARTIAL_FREEZE_ERROR"),
                    "message": str(partial_exc),
                }
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
            print(f"[失败:{error['code']}] {error['message']}", flush=True)
    write_json(session_dir / "summary.json", summary)
    print(f"[完成] {summary['status']}，summary：{session_dir / 'summary.json'}", flush=True)
    return session_dir / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="四人模块整合 v2：同学 A 规划、B 准备数据、C 训练、D 评分及学校 API 网关"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="运行 train/validation 搜索，可选在冻结后评估 test"
    )
    task_input = run_parser.add_mutually_exclusive_group()
    task_input.add_argument("--task", type=Path, help="自动理解任务目录并运行至最终 test，无需数据集配置")
    task_input.add_argument("--config", type=Path, help="开发兼容入口：使用人工 TaskSpec")
    run_parser.add_argument("--output-root", type=Path)
    run_parser.add_argument("--planner", choices=("offline", "school"), default="offline")
    run_parser.add_argument(
        "--llm-config", type=Path, default=PROJECT_ROOT / "configs" / "school-api.json"
    )
    run_parser.add_argument("--llm-max-calls", type=int, help="--task 默认 20，旧配置入口默认 1")
    run_parser.add_argument("--llm-budget-rmb", help="--task 默认 20 元，旧配置入口默认 0.10 元")
    run_parser.add_argument("--max-trials", type=int, default=20, help="自动任务模式最多实验数，默认 20")
    run_parser.add_argument(
        "--billing-profile",
        choices=("deepseek-v4.1-flash-peak-2026-10-21",),
        help="显式启用课程级参考价格台账；不会改变网关端点或其开发价格配置",
    )
    run_parser.add_argument("--task-budget-id", help="跨会话复用的安全预算编号")
    run_parser.add_argument("--task-budget-rmb", default="20")
    run_parser.add_argument(
        "--search-config",
        type=Path,
        help="覆盖搜索设置；--task 默认增强搜索，旧配置入口省略时保持 legacy",
    )
    run_parser.add_argument(
        "--test-config",
        type=Path,
        help="在最终 validation 最优 bundle 冻结后独立评估 test",
    )
    test_parser = subparsers.add_parser(
        "evaluate-test", help="对已有 final bundle 补跑一次独立 test 评估"
    )
    test_parser.add_argument("--bundle", type=Path, required=True)
    test_parser.add_argument("--test-config", type=Path, required=True)
    test_parser.add_argument("--output-root", type=Path)
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
                report = evaluate_frozen_bundle(args.bundle, args.test_config, output_dir)
            except FinalTestError as exc:
                report_path = output_dir / "test_evaluation.json"
                if report_path.is_file():
                    report = read_json(report_path)
                    for line in format_test_result(report):
                        print(line, flush=True)
                    print(f"[报告] {report_path}", flush=True)
                else:
                    print(f"[Test] 失败：{exc}", file=sys.stderr, flush=True)
                return 1
            for line in format_test_result(report):
                print(line, flush=True)
            print(f"[报告] {output_dir / 'test_evaluation.json'}", flush=True)
            print(f"[预测] {output_dir / 'test_predictions.csv'}", flush=True)
            return 0
        if args.task is not None and args.test_config is not None:
            raise ValueError("--task 自动发现最终 test，无需 --test-config")
        automatic = args.task is not None
        billing_profile = args.billing_profile
        if automatic and args.planner == "school":
            billing_profile = billing_profile or "deepseek-v4.1-flash-peak-2026-10-21"
        elif bool(billing_profile) != bool(args.task_budget_id):
            raise ValueError("--billing-profile 与 --task-budget-id 必须同时提供")
        if billing_profile and args.planner != "school":
            raise ValueError("课程预算台账只适用于 --planner school")
        gateway_factory = None
        if args.planner == "school":
            from evaluation_d.env_config import load_api_key
            from evaluation_d.school_gateway import SchoolGateway

            root = args.output_root.resolve() if args.output_root is not None else PROJECT_ROOT
            max_calls = args.llm_max_calls if args.llm_max_calls is not None else (20 if automatic else 1)
            budget_rmb = args.llm_budget_rmb if args.llm_budget_rmb is not None else ("20" if automatic else "0.10")

            def make_gateway(spec: dict[str, Any], session_dir: Path) -> Any:
                config = read_json(args.llm_config, error_code="API_CONFIG_INVALID")
                # 任务识别成功后才加载本项目 .env；不搜索父目录。
                api_key = load_api_key(PROJECT_ROOT / ".env")
                print("学校 API 模式：已从项目根 .env 加载 key。", flush=True)
                task_budget = None
                if billing_profile:
                    from evaluation_d.task_budget import TaskBudgetLedger

                    budget_id = args.task_budget_id
                    if budget_id is None:
                        budget_id = _automatic_budget_id(
                            root, read_json(session_dir / "task.discovery.json")
                        )
                    task_budget = TaskBudgetLedger(
                        root / "sessions" / "task-budgets",
                        budget_id,
                        budget_rmb=args.task_budget_rmb,
                        profile=billing_profile,
                    )
                    print(
                        f"[预算] 任务账本 {budget_id}，上限 {args.task_budget_rmb} 元；"
                        "同任务续跑复用账本。",
                        flush=True,
                    )
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
            max_trials=args.max_trials,
        )
        return 0 if read_json(path)["status"] == "success" else 1
    except (EOFError, KeyboardInterrupt):
        print("已停止本次运行。", file=sys.stderr)
        return 2
    except (OSError, ValueError, GatewayError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
