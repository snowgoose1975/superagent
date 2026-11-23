"""Read and structurally validate a successful C prediction artifact.

This module deliberately does not read validation labels or compute metrics.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from executor_c.contracts import read_json, validate_identity


def load_successful_predictions(
    result_relpath: str | Path, runs_root: str | Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Load a successful result and its predictions without scoring them."""
    root = Path(runs_root).expanduser().resolve(strict=True)
    rel_text = str(result_relpath)
    rel = PurePosixPath(rel_text)
    if (
        "\\" in rel_text
        or rel.is_absolute()
        or PureWindowsPath(rel_text).drive
        or ".." in rel.parts
    ):
        raise ValueError("result_relpath must be relative to runs_root")
    if rel.name != "result.json" or len(rel.parts) != 4:
        raise ValueError(
            "result_relpath must be <task_id>/<data_version>/<experiment_id>/result.json"
        )
    result_path = (root / Path(*rel.parts)).resolve(strict=True)
    if not result_path.is_relative_to(root) or not result_path.is_file():
        raise ValueError("result.json must be a file within runs_root")
    result = read_json(result_path, error_code="DATA_INVALID")
    if not isinstance(result, dict) or result.get("schema_version") != "1.0":
        raise ValueError("Unsupported result.json structure or schema_version")
    if result.get("status") != "success":
        raise ValueError(f"Expected a successful result, got {result.get('status')!r}")
    identity = {key: result.get(key) for key in ("task_id", "data_version", "experiment_id")}
    if validate_identity(identity) != tuple(rel.parts[:3]):
        raise ValueError("result_relpath does not match the run identity in result.json")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("predictions_path"), str):
        raise ValueError("The successful result lacks artifacts.predictions_path")
    prediction_text = artifacts["predictions_path"]
    prediction_rel = PurePosixPath(prediction_text)
    if (
        "\\" in prediction_text
        or prediction_rel.is_absolute()
        or PureWindowsPath(prediction_text).drive
        or ".." in prediction_rel.parts
    ):
        raise ValueError("predictions_path must be relative to and contained within the result.json directory")
    prediction_path = (result_path.parent / Path(*prediction_rel.parts)).resolve(strict=True)
    if (
        not prediction_path.is_relative_to(result_path.parent.resolve())
        or not prediction_path.is_file()
    ):
        raise ValueError("Predictions must be a file within this experiment directory")

    class_ids = result.get("class_ids")
    if (
        not isinstance(class_ids, list)
        or len(class_ids) < 2
        or any(type(value) is not int for value in class_ids)
        or class_ids != list(range(len(class_ids)))
    ):
        raise ValueError("class_ids in result.json must be consecutive integers starting at 0")
    expected_header = ["sample_id", "y_pred", *(f"p_{class_id}" for class_id in class_ids)]
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_header:
            raise ValueError(f"Prediction CSV columns must be {expected_header!r}")
        rows = list(reader)
    if not rows:
        raise ValueError("The predictions CSV must contain data rows")
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "")
        if not sample_id or not sample_id.strip() or sample_id in seen:
            raise ValueError(f"Row {line_number} has an empty or duplicate sample_id")
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Row {line_number} has the wrong column count")
        seen.add(sample_id)
        try:
            prediction_text = row["y_pred"]
            prediction = int(prediction_text)
            if str(prediction) != prediction_text:
                raise ValueError("y_pred must use canonical integer text")
            probabilities = [float(row[f"p_{class_id}"]) for class_id in class_ids]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Row {line_number} contains a nonnumeric prediction") from exc
        if prediction not in class_ids or not all(
            math.isfinite(p) and 0 <= p <= 1 for p in probabilities
        ):
            raise ValueError(f"Row {line_number} contains a class or probability outside the allowed range")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError(f"Probabilities in row {line_number} must sum to 1")
    if type(result.get("n_predictions")) is not int or result["n_predictions"] <= 0:
        raise ValueError("n_predictions in result.json must be a positive integer")
    if result["n_predictions"] != len(rows):
        raise ValueError("The prediction row count differs from n_predictions in result.json")
    return result, rows
