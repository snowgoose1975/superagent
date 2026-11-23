"""Score a frozen final bundle once on a separately configured test partition."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from data_b.npy_reader import read_npy_array
from data_b.raw_reader import typed_category_token
from evaluation_d.metric_contracts import score_predictions, validate_metric_spec
from executor_c.contracts import read_json, resolve_data_path, sha256_file, write_json
from predict import predict_bundle

_TEST_CONFIG_FIELDS = {
    "schema_version",
    "task_id",
    "source_root",
    "features_path",
    "labels_path",
    "n_samples",
}


class FinalTestError(ValueError):
    """A test evaluation failed after the final bundle was already frozen."""

    code = "FINAL_TEST_FAILED"


def _load_test_config(config_path: str | Path, expected_task_id: str) -> tuple[dict[str, Any], Path, Path]:
    path = Path(config_path).expanduser().resolve(strict=True)
    config = read_json(path, error_code="FINAL_TEST_CONFIG_INVALID")
    if set(config) != _TEST_CONFIG_FIELDS or config.get("schema_version") != "1.0":
        raise ValueError("The test config must contain all schema 1.0 fields")
    if config.get("task_id") != expected_task_id:
        raise ValueError("The test task_id does not match the frozen bundle")
    if (
        not isinstance(config.get("source_root"), str)
        or not config["source_root"]
        or type(config.get("n_samples")) is not int
        or config["n_samples"] < 1
    ):
        raise ValueError("The test source_root or n_samples is invalid")
    root = (path.parent / config["source_root"]).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The test source_root must be an existing directory")
    features = resolve_data_path(root, config["features_path"])
    labels = resolve_data_path(root, config["labels_path"])
    if features == labels:
        raise ValueError("Test features and labels must be in separate files")
    return config, features, labels


def _load_encoded_labels(labels_path: Path, mapping: dict[str, Any], class_ids: list[int]) -> list[int]:
    label_values = mapping.get("label_values")
    classes = mapping.get("classes")
    if (
        not isinstance(label_values, list)
        or len(label_values) != len(class_ids)
        or not isinstance(classes, list)
        or [item.get("id") if isinstance(item, dict) else None for item in classes]
        != class_ids
    ):
        raise ValueError("The frozen label mapping and bundle define different classes")
    tokens = [typed_category_token(value) for value in label_values]
    if any(not isinstance(token, str) for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("The frozen label mapping contains invalid or duplicate source class values")
    label_to_id = {token: class_id for token, class_id in zip(tokens, class_ids, strict=True)}
    raw = read_npy_array(labels_path)
    if raw.ndim != 1:
        raise ValueError("Test labels must be a one-dimensional array")
    encoded: list[int] = []
    for value in raw:
        token = typed_category_token(value)
        if not isinstance(token, str) or token not in label_to_id:
            raise ValueError("Test labels contain a class absent from the frozen label mapping")
        encoded.append(label_to_id[token])
    return encoded


def _read_predictions(path: Path, class_ids: list[int]) -> tuple[list[int], list[list[float]]]:
    expected_header = ["sample_id", "y_pred", *[f"p_{class_id}" for class_id in class_ids]]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != expected_header:
                raise ValueError("The test prediction CSV header does not match the frozen classes")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("Cannot read the test prediction CSV") from exc
    if not rows:
        raise ValueError("Test predictions must be nonempty")
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    for index, row in enumerate(rows):
        if len(row) != len(expected_header) or row[0] != f"test_{index:09d}":
            raise ValueError("Test prediction rows or ordered sample IDs are invalid")
        try:
            label = int(row[1])
            probability = [float(value) for value in row[2:]]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Test predictions contain an invalid class number or probability") from exc
        if str(label) != row[1] or label not in class_ids:
            raise ValueError("Test predictions contain an invalid class")
        y_pred.append(label)
        probabilities.append(probability)
    return y_pred, probabilities


def evaluate_frozen_bundle(
    bundle_path: str | Path,
    test_config_path: str | Path,
    output_dir: str | Path,
    *,
    pretrained: str = "cache-only",
    encoder_timeout: float | None = 600,
) -> dict[str, Any]:
    """Predict and score test rows without fitting or changing the frozen model."""
    destination = Path(output_dir).expanduser().resolve()
    if not destination.is_dir():
        raise ValueError("The test report directory must already exist")
    report_path = destination / "test_evaluation.json"
    predictions_path = destination / "test_predictions.csv"
    bundle_file = Path(bundle_path).expanduser().resolve()
    bundle: dict[str, Any] = {}
    try:
        bundle_file = bundle_file.resolve(strict=True)
        bundle = read_json(bundle_file, error_code="FINAL_BUNDLE_INVALID")
        if (
            bundle.get("schema_version") not in {"1.0", "2.0"}
            or not isinstance(bundle.get("final_id"), str)
            or not bundle["final_id"].startswith("final_")
            or bundle.get("selection", {}).get("partition") != "validation"
            or bundle.get("training_policy", {}).get("refit_on_validation") is not False
        ):
            raise ValueError("Test evaluation requires a final bundle frozen after validation selection")
        classes = bundle.get("classes")
        class_ids = [item.get("id") if isinstance(item, dict) else None for item in classes]
        if class_ids != list(range(len(class_ids))) or len(class_ids) < 2:
            raise ValueError("Frozen class IDs must be the complete sequence 0..K-1")
        metric = validate_metric_spec(
            {
                "name": bundle["selection"].get("metric"),
                "greater_is_better": bundle["selection"].get("greater_is_better"),
            }
        )
        source_inputs = None
        requested_test_config = read_json(test_config_path, error_code="FINAL_TEST_CONFIG_INVALID")
        if requested_test_config.get("schema_version") == "2.0":
            from data_b.task_directory import materialize_final_test

            if (
                requested_test_config.get("kind") != "task_directory"
                or requested_test_config.get("task_id") != bundle["task_id"]
            ):
                raise ValueError("The automatic test descriptor does not match the frozen bundle")
            # Materialize CSV or multi-array test data only after freezing the final bundle.
            test_input_dir = destination / "test_input"
            test_config_path = materialize_final_test(test_config_path, test_input_dir)
            source_inputs = read_json(test_input_dir / "test_source.json")
        config, features_path, labels_path = _load_test_config(
            test_config_path, bundle["task_id"]
        )
        config_hash = sha256_file(Path(test_config_path).expanduser().resolve(strict=True))
        feature_hash = sha256_file(features_path)
        labels_hash = sha256_file(labels_path)
        mapping_path = resolve_data_path(bundle_file.parent, bundle["artifacts"]["label_mapping_path"])
        mapping = read_json(mapping_path, error_code="FINAL_BUNDLE_INVALID")

        if "encoder.json" in bundle["hashes"]:
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve().parents[1] / "predict.py"),
                     "--bundle", str(bundle_file), "--features", str(features_path),
                     "--output", str(predictions_path), "--id-prefix", "test",
                     "--pretrained", pretrained],
                    timeout=encoder_timeout, capture_output=encoder_timeout is not None, check=False,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError("Local encoder inference timed out on the final test partition") from exc
            if completed.returncode != 0:
                raise ValueError("Final test encoder inference failed. Check dependencies, the cache, and the frozen manifest.")
        else:
            predict_bundle(bundle_file, features_path, predictions_path, id_prefix="test")
        y_true = _load_encoded_labels(labels_path, mapping, class_ids)
        y_pred, probabilities = _read_predictions(predictions_path, class_ids)
        if (
            len(y_true) != config["n_samples"]
            or len(y_pred) != config["n_samples"]
            or len(y_true) != len(y_pred)
        ):
            raise ValueError("Test feature or label row count differs from n_samples")
        metric_value = score_predictions(metric["name"], y_true, y_pred, probabilities, class_ids)
        accuracy = score_predictions("accuracy", y_true, y_pred, probabilities, class_ids)
        if (
            sha256_file(features_path) != feature_hash
            or sha256_file(labels_path) != labels_hash
            or sha256_file(Path(test_config_path).expanduser().resolve(strict=True)) != config_hash
        ):
            raise ValueError("Test inputs or configuration changed during evaluation")
        report = {
            "schema_version": "1.0",
            "task_id": bundle["task_id"],
            "data_version": bundle["data_version"],
            "session_id": bundle["session_id"],
            "final_id": bundle["final_id"],
            "experiment_id": bundle["experiment_id"],
            "status": "success",
            "selection_partition": "validation",
            "metric": metric["name"],
            "value": metric_value,
            "greater_is_better": metric["greater_is_better"],
            "accuracy": accuracy,
            "n_samples": len(y_true),
            "artifacts": {
                "predictions_path": predictions_path.name,
                "bundle_sha256": sha256_file(bundle_file),
                "test_config_sha256": config_hash,
                "features_sha256": feature_hash,
                "labels_sha256": labels_hash,
                "predictions_sha256": sha256_file(predictions_path),
            },
        }
        if source_inputs is not None:
            report["artifacts"]["source_inputs"] = source_inputs
        write_json(report_path, report)
        return report
    except Exception as exc:
        report = {
            "schema_version": "1.0",
            "task_id": bundle.get("task_id"),
            "data_version": bundle.get("data_version"),
            "session_id": bundle.get("session_id"),
            "final_id": bundle.get("final_id"),
            "experiment_id": bundle.get("experiment_id"),
            "status": "failed",
            "artifacts": {
                "predictions_path": predictions_path.name if predictions_path.exists() else None,
            },
            "error": {"code": getattr(exc, "code", "FINAL_TEST_FAILED"), "message": str(exc)},
        }
        try:
            write_json(report_path, report)
        except FileExistsError:
            pass
        except OSError as report_exc:
            raise FinalTestError(
                f"{exc}. Could not write the test failure report: {report_exc}"
            ) from exc
        raise FinalTestError(str(exc)) from exc


def format_test_result(report: dict[str, Any]) -> list[str]:
    """Terminal lines for the configured metric and the familiar accuracy score."""
    if report.get("status") != "success":
        return [f"[Test] Failed: {report.get('error', {}).get('message', 'Unknown error')}"]
    n_samples = report["n_samples"]
    metric = report["metric"]
    lines = []
    if metric == "accuracy":
        lines.append(f"[Test] accuracy = {report['value']:.6f} ({report['value']:.2%})，n = {n_samples}")
    else:
        lines.append(f"[Test] {metric} = {report['value']:.6f}，n = {n_samples}")
        lines.append(f"[Test] accuracy = {report['accuracy']:.6f} ({report['accuracy']:.2%})，n = {n_samples}")
    return lines

