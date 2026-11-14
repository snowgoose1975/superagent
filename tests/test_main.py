"""端到端验证合成任务、可替换 A/D 和冻结模型推理。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import main as team_main
import numpy as np
import pytest
from data_b.prepare import prepare_data
from evaluation_d.evaluate import evaluate
from executor_c.contracts import read_json
from scripts.create_demo import create_demo

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _history(root: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / summary["history_path"]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_score(score: dict[str, Any]) -> None:
    assert score["status"] == "success"
    assert score["metric"] == "accuracy"
    assert score["greater_is_better"] is True
    assert type(score["value"]) is float and 0.0 <= score["value"] <= 1.0
    assert score["n_samples"] == 40


def test_demo_runs_three_real_trials_freezes_and_predicts_in_new_process(tmp_path, monkeypatch):
    output_root = tmp_path / "integrated_demo"
    config_path = create_demo(output_root)
    assert config_path.is_absolute()
    spec = read_json(config_path)
    assert spec["task_id"] == "synthetic_coupon"
    assert spec["data_version"] == "demo_v1"
    assert len(spec["feature_columns"]) == 21
    assert set(spec["partitions"]) == {"train", "validation"}
    assert spec["partitions"]["train"]["n_samples"] == 120
    assert spec["partitions"]["validation"]["n_samples"] == 40

    raw_dir = output_root / "data" / "raw" / "synthetic_coupon"
    # 内容故意不是合法 NPY；仅 train/validation 被配置引用，主循环不得触碰它。
    with (raw_dir / "C_test_sentinel.npy").open("xb") as stream:
        stream.write(b"test partition must remain unread")
    original_open = Path.open
    attempted_test_reads: list[Path] = []

    def audit_open(path: Path, *args: Any, **kwargs: Any):
        resolved = Path(path)
        if "test" in resolved.name.lower():
            attempted_test_reads.append(resolved)
            raise AssertionError(f"主循环尝试读取测试分区：{resolved.name}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audit_open)
    summary_path = team_main.run_project(config_path, output_root=output_root)
    assert attempted_test_reads == []

    summary = read_json(summary_path)
    assert summary["status"] == "success", summary
    assert summary["planner_mode"] == "offline_rules"
    assert summary["real_llm_calls"] == 0
    assert summary["attempted_trials"] == 3
    assert summary["best_experiment_id"]
    assert summary["best_validation_accuracy"] is not None
    assert summary["test_evaluation"]["status"] == "not_requested"
    bundle_path = output_root / summary["bundle_path"]
    assert bundle_path.is_file()
    history = _history(output_root, summary)
    assert len(history) == 3
    assert {entry["plan"]["model"] for entry in history} == {
        "random_forest",
        "logistic_regression",
    }
    for entry in history:
        _assert_score(entry["evaluation"])
        assert entry["status"] == "success"

    # 相同源摘要只复用 B 版本；验证复用前后清单字节完全一致。
    prepared_dir = output_root / "data" / "prepared" / "synthetic_coupon" / "demo_v1"
    package_result_path = prepared_dir / "package_result.json"
    previous_package_result = package_result_path.read_bytes()
    prepared = prepare_data(
        team_main.load_task_spec(config_path), output_root / "data" / "prepared"
    )
    assert prepared["status"] == "success" and prepared["reused"] is True
    assert package_result_path.read_bytes() == previous_package_result

    best = next(item for item in history if item["experiment_id"] == summary["best_experiment_id"])
    result_path = output_root / "runs" / best["result_relpath"]
    best_predictions = result_path.parent / "predictions.csv"
    predicted_path = output_root / "prediction-new-process.csv"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")) if part
        ),
    }
    process = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--bundle",
            str(bundle_path),
            "--features",
            str(raw_dir / "C_val.npy"),
            "--output",
            str(predicted_path),
            "--id-prefix",
            "validation",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    with best_predictions.open("r", encoding="utf-8", newline="") as stream:
        best_rows = list(csv.DictReader(stream))
    with predicted_path.open("r", encoding="utf-8", newline="") as stream:
        predicted_rows = list(csv.DictReader(stream))
    assert [row["sample_id"] for row in predicted_rows] == [row["sample_id"] for row in best_rows]
    assert [row["y_pred"] for row in predicted_rows] == [row["y_pred"] for row in best_rows]
    for expected, actual in zip(best_rows, predicted_rows, strict=True):
        for class_id in range(2):
            assert float(actual[f"p_{class_id}"]) == pytest.approx(
                float(expected[f"p_{class_id}"]), abs=1e-12, rel=0
            )

    # 修改源数据后，冻结 B 版本拒绝复用并保留原 package_result。
    train_path = raw_dir / "C_train.npy"
    changed = np.load(train_path, allow_pickle=True)
    changed[0, 0] = int(changed[0, 0]) + 1
    with train_path.open("wb") as stream:
        np.save(stream, changed, allow_pickle=True)
    rejected = prepare_data(
        team_main.load_task_spec(config_path), output_root / "data" / "prepared"
    )
    assert rejected["status"] == "failed"
    assert "变化" in rejected["error"]["message"] or "已变化" in rejected["error"]["message"]
    assert package_result_path.read_bytes() == previous_package_result


class OneShotController:
    """小型替换 A：执行一次逻辑回归后停止。"""

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "action": "run",
                "model": "logistic_regression",
                "params": {"C": 0.5, "max_iter": 500},
                "reason": "可替换控制器执行一次基线。",
            }
        return {"action": "stop", "reason": "替换控制器主动停止。"}


def test_controller_and_real_evaluator_are_replaceable(tmp_path):
    output_root = tmp_path / "replaceable_demo"
    config_path = create_demo(output_root, n_train=72, n_valid=24)
    controller = OneShotController()
    captured_scores: list[dict[str, Any]] = []

    def scoring_wrapper(result_path: Path, evaluation_manifest_path: Path, output_path: Path):
        score = evaluate(result_path, evaluation_manifest_path, output_path)
        captured_scores.append(score)
        return score

    summary_path = team_main.run_project(
        config_path,
        output_root=output_root,
        controller=controller,
        evaluator=scoring_wrapper,
    )
    summary = read_json(summary_path)
    assert summary["status"] == "success"
    assert summary["planner_mode"] == "injected_controller"
    assert summary["real_llm_calls"] is None
    assert controller.calls == 2
    assert len(captured_scores) == 1
    score = captured_scores[0]
    assert score["status"] == "success" and score["metric"] == "accuracy"
    assert type(score["value"]) is float and 0 <= score["value"] <= 1
    assert summary["best_experiment_id"] == score["experiment_id"]


def _write_test_config(config_path: Path, source_root: Path, n_samples: int) -> Path:
    test_config_path = config_path.parent / "test.json"
    test_config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "task_id": "synthetic_coupon",
                "source_root": os.path.relpath(source_root, test_config_path.parent),
                "features_path": "C_test.npy",
                "labels_path": "y_test.npy",
                "n_samples": n_samples,
            }
        ),
        encoding="utf-8",
    )
    return test_config_path


def test_final_test_runs_only_after_freeze_and_standalone_reuses_final_bundle(
    tmp_path, monkeypatch, capsys
):
    output_root = tmp_path / "final_test_demo"
    config_path = create_demo(output_root, n_train=72, n_valid=24)
    raw_root = output_root / "data" / "raw" / "synthetic_coupon"
    X_valid = np.load(raw_root / "C_val.npy", allow_pickle=True)
    y_valid = np.load(raw_root / "y_val.npy", allow_pickle=False)
    test_features = raw_root / "C_test.npy"
    test_labels = raw_root / "y_test.npy"
    np.save(test_features, X_valid[:9], allow_pickle=True)
    np.save(test_labels, y_valid[:9], allow_pickle=False)
    test_config = _write_test_config(config_path, raw_root, 9)

    class StopAfterOne:
        def __init__(self):
            self.calls = 0

        def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                return {
                    "action": "run",
                    "model": "logistic_regression",
                    "params": {"C": 1.0, "max_iter": 500},
                    "reason": "synthetic final-test flow",
                }
            return {"action": "stop", "reason": "freeze and evaluate"}

    controller = StopAfterOne()
    original_open = Path.open
    test_reads: list[Path] = []
    frozen_bundle: list[Path] = []

    def audit_test_reads(path: Path, *args: Any, **kwargs: Any):
        candidate = Path(path).resolve()
        if candidate in {test_features.resolve(), test_labels.resolve()}:
            test_reads.append(candidate)
            assert frozen_bundle and frozen_bundle[0].is_file()
            frozen = read_json(frozen_bundle[0])
            assert frozen["selection"]["partition"] == "validation"
            assert frozen["final_id"].startswith("final_")
        return original_open(path, *args, **kwargs)

    original_freeze = team_main.freeze_best

    def record_freeze(*args, **kwargs):
        result = original_freeze(*args, **kwargs)
        frozen_bundle.append(result.resolve())
        return result

    monkeypatch.setattr(team_main, "freeze_best", record_freeze)
    monkeypatch.setattr(Path, "open", audit_test_reads)
    summary_path = team_main.run_project(
        config_path,
        output_root=output_root,
        controller=controller,
        test_config_path=test_config,
    )
    summary = read_json(summary_path)
    assert summary["status"] == "success", summary
    assert summary["test_evaluation"]["status"] == "success"
    assert summary["test_evaluation"]["metric"] == "accuracy"
    assert summary["test_evaluation"]["n_samples"] == 9
    assert len(test_reads) >= 2
    assert controller.calls == 2
    bundle_path = output_root / summary["bundle_path"]
    bundle_before = bundle_path.read_bytes()
    report_path = output_root / summary["test_evaluation"]["report_path"]
    report = read_json(report_path)
    assert report["status"] == "success"
    assert report["value"] == report["accuracy"]
    assert report["selection_partition"] == "validation"
    assert (output_root / summary["test_evaluation"]["predictions_path"]).is_file()
    terminal = capsys.readouterr().out
    assert "[最终方案]" in terminal
    assert "[Validation] accuracy =" in terminal
    assert "[Test] accuracy =" in terminal

    standalone_root = tmp_path / "standalone_test"
    assert team_main.main(
        [
            "evaluate-test",
            "--bundle",
            str(bundle_path),
            "--test-config",
            str(test_config),
            "--output-root",
            str(standalone_root),
        ]
    ) == 0
    standalone_output = capsys.readouterr().out
    assert "[Test] accuracy =" in standalone_output
    assert bundle_path.read_bytes() == bundle_before


def test_final_test_failure_keeps_final_bundle_without_partial_refreeze(tmp_path):
    output_root = tmp_path / "final_test_failure"
    config_path = create_demo(output_root, n_train=72, n_valid=24)
    raw_root = output_root / "data" / "raw" / "synthetic_coupon"
    X_valid = np.load(raw_root / "C_val.npy", allow_pickle=True)
    y_valid = np.load(raw_root / "y_val.npy", allow_pickle=False)
    test_features = raw_root / "C_test.npy"
    test_labels = raw_root / "y_test.npy"
    np.save(test_features, X_valid[:6], allow_pickle=True)
    np.save(test_labels, np.full(6, 99, dtype=np.int64), allow_pickle=False)
    test_config = _write_test_config(config_path, raw_root, 6)

    summary_path = team_main.run_project(
        config_path,
        output_root=output_root,
        controller=OneShotController(),
        test_config_path=test_config,
    )
    summary = read_json(summary_path)
    assert summary["status"] == "failed"
    assert summary["error"]["code"] == "FINAL_TEST_FAILED"
    assert summary["error"]["stage"] == "final_test"
    assert summary["test_evaluation"]["status"] == "failed"
    assert summary["bundle_path"]
    assert (output_root / summary["bundle_path"]).is_file()
    assert summary["partial_bundle_path"] is None
    frozen_dirs = list((output_root / "submission" / "synthetic_coupon").iterdir())
    assert len(frozen_dirs) == 1
    assert frozen_dirs[0].name.startswith("final_")
    report = read_json(output_root / summary["test_evaluation"]["report_path"])
    assert report["status"] == "failed"


def test_configured_confirmation_trials_use_recorded_offset_seeds(tmp_path):
    output_root = tmp_path / "confirmation_demo"
    config_path = create_demo(output_root, n_train=72, n_valid=24)
    task = read_json(config_path)
    task["train_policy"]["max_trials"] = 3
    config_path.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
    search_config = output_root / "search.json"
    search_config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "require_family_coverage": True,
                "min_exploration_trials": 1,
                "patience": 1,
                "min_delta": 0.0,
                "max_consecutive_failures": 1,
                "max_session_seconds": 600,
                "finalize_reserve_seconds": 60,
                "confirmation_seed_offsets": [1, 2],
            }
        ),
        encoding="utf-8",
    )

    class StopAfterOne:
        calls = 0

        def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                return {
                    "action": "run",
                    "model": "logistic_regression",
                    "params": {"C": 1.0},
                    "reason": "one baseline before configured confirmation",
                }
            return {"action": "stop", "reason": "search complete"}

    summary_path = team_main.run_project(
        config_path,
        output_root=output_root,
        controller=StopAfterOne(),
        search_config_path=search_config,
    )
    summary = read_json(summary_path)
    history = _history(output_root, summary)
    assert summary["status"] == "success"
    assert summary["confirmation_trials"] == 2
    assert [entry["plan"]["random_seed"] for entry in history] == [42, 43, 44]
    assert [entry["search_phase"] for entry in history] == [
        "search",
        "confirmation",
        "confirmation",
    ]


class TwoTrialController:
    def __init__(self) -> None:
        self.decisions = [
            ("logistic_regression", {"C": 1.0, "max_iter": 500}),
            ("random_forest", {"n_estimators": 12, "max_depth": 4}),
        ]

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.decisions:
            return {"action": "stop", "reason": "两轮候选已结束。"}
        model, params = self.decisions.pop(0)
        return {
            "action": "run",
            "model": model,
            "params": params,
            "reason": "验证评分异常隔离检查。",
        }


def test_evaluator_failure_does_not_select_failed_trial(tmp_path):
    output_root = tmp_path / "scoring_failure_demo"
    config_path = create_demo(output_root, n_train=72, n_valid=24)
    real_scores: list[dict[str, Any]] = []
    call_count = 0

    def fail_once_then_score(result_path: Path, evaluation_manifest_path: Path, output_path: Path):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("injected evaluator outage")
        score = evaluate(result_path, evaluation_manifest_path, output_path)
        real_scores.append(score)
        return score

    summary_path = team_main.run_project(
        config_path,
        output_root=output_root,
        controller=TwoTrialController(),
        evaluator=fail_once_then_score,
    )
    summary = read_json(summary_path)
    assert summary["status"] == "success"
    history = _history(output_root, summary)
    assert len(history) == 2
    assert history[0]["status"] == "failed"
    assert history[0]["evaluation"] is None
    assert history[0]["error"]["code"] == "TRIAL_ERROR"
    assert history[1]["status"] == "success"
    assert summary["best_experiment_id"] == history[1]["experiment_id"]
    assert (
        len(real_scores) == 1 and real_scores[0]["experiment_id"] == summary["best_experiment_id"]
    )


@pytest.mark.parametrize(
    ("metric", "greater"),
    [("f1_macro", True), ("log_loss", False)],
)
def test_non_accuracy_primary_metric_scores_and_selects_in_declared_direction(
    tmp_path, metric, greater
):
    output_root = tmp_path / f"metric_{metric}"
    config_path = create_demo(output_root, n_train=72, n_valid=24)
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    spec["data_version"] = f"{metric}_v1"
    spec["primary_metric"] = {"name": metric, "greater_is_better": greater}
    spec["train_policy"]["max_trials"] = 3
    config_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    class MetricController:
        def __init__(self):
            self.decisions = [
                ("logistic_regression", {"C": 0.1, "max_iter": 1000}),
                ("random_forest", {"n_estimators": 12, "max_depth": 4}),
                ("extra_trees", {"n_estimators": 12, "max_depth": 4}),
            ]

        def decide(self, context):
            if not self.decisions:
                return {"action": "stop", "reason": "three candidates completed"}
            model, params = self.decisions.pop(0)
            return {
                "action": "run",
                "model": model,
                "params": params,
                "reason": "test declared metric direction",
            }

    summary_path = team_main.run_project(
        config_path, output_root=output_root, controller=MetricController()
    )
    summary = read_json(summary_path)
    history = _history(output_root, summary)
    values = [entry["evaluation"]["value"] for entry in history]
    expected = (max if greater else min)(values)
    assert summary["status"] == "success"
    assert summary["best_validation_accuracy"] is None
    assert summary["best_validation_metric"] == {
        "name": metric,
        "value": expected,
        "greater_is_better": greater,
    }
    assert all(entry["evaluation"]["metric"] == metric for entry in history)
    bundle = read_json(output_root / summary["bundle_path"])
    assert bundle["selection"]["metric"] == metric
    assert bundle["selection"]["value"] == expected
