"""Evaluate predictions against validation labels aligned by sample_id."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from evaluation_d.metric_contracts import score_predictions, validate_metric_spec
from executor_c.consumer import load_successful_predictions
from executor_c.contracts import read_json, resolve_data_path, write_json


def _read_csv(path: Path, expected_header: list[str]) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != expected_header:
                raise ValueError(f"{path.name} must have the header {expected_header!r}.")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Cannot read {path.name}.") from exc
    if any(len(row) != len(expected_header) for row in rows):
        raise ValueError(f"{path.name} contains a row with the wrong number of columns.")
    return rows


def _resolve_manifest_file(manifest_path: Path, relative: Any) -> Path:
    """Resolve paths relative to the manifest and reject paths outside its directory."""
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in PurePosixPath(relative).parts
    ):
        raise ValueError("The evaluation manifest must use relative POSIX paths within its directory.")
    return resolve_data_path(manifest_path.parent, relative)


def evaluate(
    result_path: Path,
    evaluation_manifest_path: Path,
    output_path: Path | None = None,
    *,
    runs_root: Path | None = None,
) -> dict[str, Any]:
    """Score completed C artifacts while leaving result.json intact."""
    result_path = Path(result_path).expanduser().resolve(strict=True)
    evaluation_manifest_path = Path(evaluation_manifest_path).expanduser().resolve(strict=True)
    if result_path.name != "result.json" or not result_path.is_file():
        raise ValueError("result_path must point to C's result.json.")

    # The consumer validates status, prediction path, probabilities, and row structure.
    if runs_root is None:
        if len(result_path.parents) < 4:
            raise ValueError("Expected a result inside <runs>/<task>/<version>/<experiment>")
        runs_root = result_path.parents[3]
    else:
        runs_root = Path(runs_root).expanduser().resolve(strict=True)
    if not result_path.is_relative_to(runs_root):
        raise ValueError("The result must be inside runs_root")
    relpath = result_path.relative_to(runs_root).as_posix()
    result, predictions = load_successful_predictions(relpath, runs_root)

    return _score_validation(result, predictions, evaluation_manifest_path, output_path)


def _score_validation(result, predictions, evaluation_manifest_path, output_path=None):
    evaluation_manifest = read_json(evaluation_manifest_path, "DATA_INVALID")
    expected_fields = {
        "schema_version",
        "task_id",
        "data_version",
        "data_manifest_path",
        "labels_path",
        "primary_metric",
        "positive_class_id",
    }
    if set(evaluation_manifest) != expected_fields:
        raise ValueError("The evaluation manifest contains missing or unknown fields.")
    if evaluation_manifest["schema_version"] != "1.0":
        raise ValueError("The evaluation manifest schema_version is unsupported.")
    identity = (result["task_id"], result["data_version"])
    if (evaluation_manifest["task_id"], evaluation_manifest["data_version"]) != identity:
        raise ValueError("The evaluation manifest and C result must have matching task_id/data_version.")
    metric = validate_metric_spec(evaluation_manifest["primary_metric"])

    data_manifest_path = _resolve_manifest_file(
        evaluation_manifest_path, evaluation_manifest["data_manifest_path"]
    )
    data_manifest = read_json(data_manifest_path, "DATA_INVALID")
    if (data_manifest.get("task_id"), data_manifest.get("data_version")) != identity:
        raise ValueError("The B data manifest and C result must have matching task_id/data_version.")
    classes = data_manifest.get("classes")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or [item.get("id") if isinstance(item, dict) else None for item in classes]
        != list(range(len(classes)))
        or result.get("class_ids") != list(range(len(classes)))
    ):
        raise ValueError("B classes and C class_ids must use the same 0..K-1 sequence.")

    labels_path = _resolve_manifest_file(
        evaluation_manifest_path, evaluation_manifest["labels_path"]
    )
    label_rows = _read_csv(labels_path, ["sample_id", "y_true"])
    labels: dict[str, int] = {}
    class_ids = result["class_ids"]
    for line_number, row in enumerate(label_rows, start=2):
        sample_id, label_text = row
        if not sample_id.strip() or sample_id in labels:
            raise ValueError(f"Label file line {line_number} contains an empty or duplicate sample_id.")
        try:
            label = int(label_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Label file line {line_number} must have integer y_true.") from exc
        if str(label) != label_text or label not in class_ids:
            raise ValueError(f"Label file line {line_number} has an invalid class ID in y_true.")
        labels[sample_id] = label

    prediction_ids = [row["sample_id"] for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("C predictions contain duplicate sample IDs.")
    if set(prediction_ids) != set(labels):
        missing = len(set(prediction_ids) - set(labels))
        extra = len(set(labels) - set(prediction_ids))
        raise ValueError(
            f"C predictions and D labels have mismatched sample ID sets ({missing} missing labels, {extra} extra labels)."
        )
    n_samples = len(predictions)
    if n_samples == 0:
        raise ValueError("Validation predictions and labels must be nonempty.")
    y_true = [labels[row["sample_id"]] for row in predictions]
    y_pred = [int(row["y_pred"]) for row in predictions]
    probabilities = [
        [float(row[f"p_{class_id}"]) for class_id in class_ids]
        for row in predictions
    ]
    value = score_predictions(metric["name"], y_true, y_pred, probabilities, class_ids)

    evaluation = {
        "schema_version": "1.0",
        "task_id": result["task_id"],
        "data_version": result["data_version"],
        "experiment_id": result["experiment_id"],
        "status": "success",
        "metric": metric["name"],
        "value": value,
        "greater_is_better": metric["greater_is_better"],
        "n_samples": n_samples,
    }
    if output_path is not None:
        write_json(Path(output_path), evaluation, exclusive=True)
    return evaluation


def evaluate_training_majority_baseline(
    evaluation_manifest_path: Path,
    training_label_counts: dict[str, int],
) -> dict[str, Any]:
    """Score a training-derived majority/probability baseline on validation only."""
    evaluation_manifest_path = Path(evaluation_manifest_path).expanduser().resolve(strict=True)
    evaluation_manifest = read_json(evaluation_manifest_path, "DATA_INVALID")
    expected_fields = {
        "schema_version", "task_id", "data_version", "data_manifest_path",
        "labels_path", "primary_metric", "positive_class_id",
    }
    if set(evaluation_manifest) != expected_fields or evaluation_manifest.get("schema_version") != "1.0":
        raise ValueError("The evaluation manifest contains missing or unknown fields, or an unsupported schema version.")
    metric = validate_metric_spec(evaluation_manifest["primary_metric"])
    data_manifest_path = _resolve_manifest_file(
        evaluation_manifest_path, evaluation_manifest["data_manifest_path"]
    )
    data_manifest = read_json(data_manifest_path, "DATA_INVALID")
    if (
        data_manifest.get("task_id") != evaluation_manifest.get("task_id")
        or data_manifest.get("data_version") != evaluation_manifest.get("data_version")
    ):
        raise ValueError("The evaluation and B data manifests have a task identity mismatch.")
    classes = data_manifest.get("classes")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or [item.get("id") if isinstance(item, dict) else None for item in classes]
        != list(range(len(classes)))
        or not isinstance(training_label_counts, dict)
    ):
        raise ValueError("Cannot derive the majority baseline from the training class statistics.")
    counts = [training_label_counts.get(item.get("name")) for item in classes]
    if any(type(count) is not int or count < 0 for count in counts) or sum(counts) <= 0:
        raise ValueError("Training class counts are missing or invalid.")
    probabilities_row = [count / sum(counts) for count in counts]
    class_id = max(range(len(counts)), key=lambda index: counts[index])
    labels_path = _resolve_manifest_file(
        evaluation_manifest_path, evaluation_manifest["labels_path"]
    )
    label_rows = _read_csv(labels_path, ["sample_id", "y_true"])
    labels: list[int] = []
    seen: set[str] = set()
    for row in label_rows:
        sample_id, label_text = row
        if not sample_id.strip() or sample_id in seen:
            raise ValueError("Validation labels contain an empty or duplicate sample_id.")
        seen.add(sample_id)
        try:
            label = int(label_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Validation y_true must be an integer class ID.") from exc
        if str(label) != label_text or label not in range(len(classes)):
            raise ValueError("Validation y_true contains an unknown class.")
        labels.append(label)
    if not labels:
        raise ValueError("Validation labels must be nonempty.")
    value = score_predictions(
        metric["name"],
        labels,
        [class_id] * len(labels),
        [probabilities_row] * len(labels),
        list(range(len(classes))),
    )
    return {
        "type": "training_majority",
        "metric": metric["name"],
        "value": value,
        "class_id": class_id,
        "n_samples": len(labels),
    }
