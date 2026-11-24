"""Synthetic text/image C→D→freeze→predict routes; never use school credentials."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import main as team_main
from executor_c.contracts import ContractError, validate_plan
from executor_c.model_registry import default_pipeline
from predict import predict_bundle


def _task(
    root: Path,
    *,
    task_id: str,
    source_format: str,
    modality: dict,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> Path:
    source = root / "source"
    source.mkdir(parents=True)
    for name, value in (
        ("X_train.npy", X_train),
        ("X_valid.npy", X_valid),
        ("y_train.npy", y_train),
        ("y_valid.npy", y_valid),
    ):
        np.save(source / name, value, allow_pickle=False)
    spec = {
        "schema_version": "2.0",
        "task_id": task_id,
        "data_version": "synthetic_v1",
        "source_format": source_format,
        "source_root": str(source),
        "modality": modality,
        "label_values": [0, 1],
        "label_names": ["negative", "positive"],
        "primary_metric": {"name": "accuracy", "greater_is_better": True},
        "partitions": {
            "train": {
                "features_path": "X_train.npy",
                "labels_path": "y_train.npy",
                "n_samples": len(y_train),
            },
            "validation": {
                "features_path": "X_valid.npy",
                "labels_path": "y_valid.npy",
                "n_samples": len(y_valid),
            },
        },
        "train_policy": {"max_trials": 1, "random_seed": 7, "timeout_seconds": 60},
    }
    config = root / "task.json"
    config.write_text(json.dumps(spec), encoding="utf-8")
    return config


def _run_and_predict(
    config: Path,
    output_root: Path,
    prediction_input: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    task = json.loads(config.read_text(encoding="utf-8"))
    source_root = Path(task["source_root"])
    np.save(source_root / "X_test.npy", prediction_input, allow_pickle=False)
    np.save(source_root / "y_test.npy", test_labels[: len(prediction_input)], allow_pickle=False)
    test_config = config.parent / "test.json"
    test_config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "task_id": task["task_id"],
                "source_root": "source",
                "features_path": "X_test.npy",
                "labels_path": "y_test.npy",
                "n_samples": len(prediction_input),
            }
        ),
        encoding="utf-8",
    )
    summary_path = team_main.run_project(config, output_root, test_config_path=test_config)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["test_evaluation"]["status"] == "success"
    assert summary["test_evaluation"]["n_samples"] == len(prediction_input)
    bundle = output_root / summary["bundle_path"]
    prediction_path = output_root / "prediction-input.npy"
    np.save(prediction_path, prediction_input, allow_pickle=False)
    output = predict_bundle(bundle, prediction_path, output_root / "prediction.csv")
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(prediction_input) + 1
    assert rows[0] == "sample_id,y_pred,p_0,p_1"


def test_text_task_runs_freezes_and_predicts_with_train_only_tfidf(tmp_path: Path) -> None:
    X_train = np.array([["good offer", "coupon"], ["bad deal", "refund"]] * 12, dtype="U20")
    X_valid = np.array([["good coupon", "offer"], ["bad refund", "deal"]] * 3, dtype="U20")
    y_train = np.array([1, 0] * 12, dtype=np.int64)
    y_valid = np.array([1, 0] * 3, dtype=np.int64)
    config = _task(
        tmp_path,
        task_id="synthetic_text",
        source_format="text_array_package",
        modality={"kind": "text", "text_columns": ["title", "body"], "max_characters": 1000},
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )
    _run_and_predict(config, tmp_path / "run", X_valid[:2], y_valid[:2])


def test_image_task_runs_freezes_and_predicts_with_declared_range(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    X_train = rng.integers(0, 256, size=(24, 6, 6), dtype=np.uint8)
    X_train[:12, :3, :3] = 240
    X_train[12:, 3:, 3:] = 240
    X_valid = rng.integers(0, 256, size=(6, 6, 6), dtype=np.uint8)
    X_valid[:3, :3, :3] = 240
    X_valid[3:, 3:, 3:] = 240
    y_train = np.array([1, 0] * 12, dtype=np.int64)
    y_valid = np.array([1, 0] * 3, dtype=np.int64)
    config = _task(
        tmp_path,
        task_id="synthetic_image",
        source_format="image_array_package",
        modality={
            "kind": "image",
            "image_shape": [6, 6],
            "channel_layout": "HW",
            "value_range": [0, 255],
        },
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )
    _run_and_predict(config, tmp_path / "run", X_valid[:2], y_valid[:2])


def test_plan_v2_rejects_incompatible_model_and_unknown_pipeline_component() -> None:
    base = {
        "schema_version": "2.0",
        "task_id": "text_task",
        "data_version": "v1",
        "experiment_id": "exp_001",
        "model": "random_forest",
        "params": {},
        "random_seed": 1,
        "timeout_seconds": 30,
        "representation": "raw_text",
        "pipeline": default_pipeline("raw_text"),
    }
    with pytest.raises(ContractError, match="does not support"):
        validate_plan(base)
    base["model"] = "logistic_regression"
    base["pipeline"] = {**default_pipeline("raw_text"), "python": {"name": "unsafe"}}
    with pytest.raises(ContractError, match="unsupported"):
        validate_plan(base)
