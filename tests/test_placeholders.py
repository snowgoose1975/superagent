"""Meaningful checks for the replaceable A/D starter modules."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

TEAM_ROOT = Path(__file__).resolve().parents[1]
if str(TEAM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEAM_ROOT))

OfflineController = importlib.import_module("agent_a").OfflineController
evaluate = importlib.import_module("evaluation_d.evaluate").evaluate


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _evaluation_case(root: Path) -> tuple[Path, Path, Path]:
    task_id, version, experiment = "coupon_task", "v1", "exp_001"
    package = root / "data" / task_id / version
    package.mkdir(parents=True)
    _write_json(
        package / "data_manifest.json",
        {
            "schema_version": "1.0",
            "task_id": task_id,
            "data_version": version,
            "classes": [{"id": 0, "name": "no"}, {"id": 1, "name": "yes"}],
        },
    )
    evaluation_manifest_path = package / "evaluation_manifest.json"
    _write_json(
        evaluation_manifest_path,
        {
            "schema_version": "1.0",
            "task_id": task_id,
            "data_version": version,
            "data_manifest_path": "data_manifest.json",
            "labels_path": "validation_labels.csv",
            "primary_metric": {"name": "accuracy", "greater_is_better": True},
            "positive_class_id": 1,
        },
    )
    with (package / "validation_labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "y_true"])
        # Deliberately reverse order relative to predictions to prove ID alignment.
        writer.writerows([["sample-b", 1], ["sample-a", 0]])

    run_dir = root / "runs" / task_id / version / experiment
    run_dir.mkdir(parents=True)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "y_pred", "p_0", "p_1"])
        writer.writerows([["sample-a", 0, 0.9, 0.1], ["sample-b", 1, 0.1, 0.9]])
    result_path = run_dir / "result.json"
    _write_json(
        result_path,
        {
            "schema_version": "1.0",
            "task_id": task_id,
            "data_version": version,
            "experiment_id": experiment,
            "status": "success",
            "n_predictions": 2,
            "class_ids": [0, 1],
            "artifacts": {"predictions_path": "predictions.csv"},
        },
    )
    return result_path, evaluation_manifest_path, package / "validation_labels.csv"


def test_controller_uses_real_history_feedback_and_stops_after_three() -> None:
    controller = OfflineController()
    assert controller.mode == "offline_rules"
    first = controller.decide({"remaining_trials": 3, "history": []})
    assert first["model"] == "logistic_regression"
    second = controller.decide({"remaining_trials": 2, "history": [{"status": "failed"}]})
    assert second["model"] == "random_forest"

    third = controller.decide(
        {
            "remaining_trials": 1,
            "history": [
                {
                    "status": "success",
                    "plan": {"model": "logistic_regression"},
                    "evaluation": {"value": 0.7},
                },
                {
                    "status": "success",
                    "plan": {"model": "random_forest"},
                    "evaluation": {"value": 0.8},
                },
            ],
        }
    )
    assert third["model"] == "random_forest"
    assert third["params"]["n_estimators"] == 200
    third_lr = controller.decide(
        {
            "remaining_trials": 1,
            "history": [
                {
                    "status": "success",
                    "plan": {"model": "logistic_regression"},
                    "evaluation": {"value": 0.8},
                },
                {
                    "status": "success",
                    "plan": {"model": "random_forest"},
                    "evaluation": {"value": 0.7},
                },
            ],
        }
    )
    assert third_lr["model"] == "logistic_regression"
    assert third_lr["params"]["C"] == 0.1
    stopped = controller.decide({"remaining_trials": 0, "history": []})
    assert stopped["action"] == "stop"
    exhausted = controller.decide({"remaining_trials": 1, "history": [{}, {}, {}]})
    assert exhausted["action"] == "stop"


def test_evaluator_aligns_shuffled_labels_by_sample_id(tmp_path: Path) -> None:
    result_path, eval_path, _ = _evaluation_case(tmp_path)
    output = tmp_path / "score.json"
    score = evaluate(result_path, eval_path, output)
    assert score["value"] == 1.0
    assert score["metric"] == "accuracy"
    assert score["n_samples"] == 2
    assert json.loads(output.read_text(encoding="utf-8")) == score


def test_evaluator_rejects_duplicate_or_missing_label_ids(tmp_path: Path) -> None:
    result_path, eval_path, labels_path = _evaluation_case(tmp_path)
    labels_path.write_text("sample_id,y_true\nsample-a,0\nsample-a,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        evaluate(result_path, eval_path)

    labels_path.write_text("sample_id,y_true\nsample-a,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="集合不一致"):
        evaluate(result_path, eval_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("task", "task_id/data_version"),
        ("metric", "只接受 primary_metric=accuracy"),
    ],
)
def test_evaluator_rejects_wrong_task_or_metric(
    tmp_path: Path, mutation: str, message: str
) -> None:
    result_path, eval_path, _ = _evaluation_case(tmp_path)
    manifest = json.loads(eval_path.read_text(encoding="utf-8"))
    if mutation == "task":
        manifest["task_id"] = "another_task"
    else:
        manifest["primary_metric"] = {"name": "f1", "greater_is_better": True}
    eval_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        evaluate(result_path, eval_path)
