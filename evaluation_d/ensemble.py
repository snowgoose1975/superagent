"""Compare a fixed portfolio of equal-weight ensembles on validation only."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation_d.evaluate import _score_validation
from executor_c.consumer import load_successful_predictions


def compare_ensembles(history: list, runs: Path, manifest: Path, metric: dict):
    greater = metric["greater_is_better"]
    ranking = sorted(
        (item for item in history if item["status"] == "success" and item["evaluation"]),
        key=lambda item: item["evaluation"]["value"], reverse=greater,
    )
    if len(ranking) < 2:
        return None, []
    # Prefer one representative per model and representation recipe.
    members = []
    keys = set()
    for item in ranking:
        plan = item["plan"]
        key = (plan["model"], str(plan.get("pipeline")))
        if key not in keys:
            keys.add(key)
            members.append(item)
        if len(members) == 3:
            break

    best = None
    comparisons = []
    incumbent = ranking[0]["evaluation"]["value"]
    combined = None
    for index, member in enumerate(members, start=1):
        result, predictions = load_successful_predictions(member["result_relpath"], runs)
        class_ids = result["class_ids"]
        row_ids = [row["sample_id"] for row in predictions]
        matrix = np.asarray(
            [[float(row[f"p_{label}"]) for label in class_ids] for row in predictions],
            dtype=np.float64,
        )
        if combined is None:
            combined = matrix
            expected_ids, expected_classes = row_ids, class_ids
            continue
        if row_ids != expected_ids or class_ids != expected_classes:
            raise ValueError("ensemble validation members do not share ordered samples and classes")
        combined += matrix
        mean = combined / index
        rows = [
            {"sample_id": sample_id, "y_pred": class_ids[int(np.argmax(row))],
             **{f"p_{label}": value for label, value in zip(class_ids, row, strict=True)}}
            for sample_id, row in zip(row_ids, mean, strict=True)
        ]
        score = _score_validation(result, rows, manifest)
        selected = members[:index]
        comparisons.append({
            "members": [item["experiment_id"] for item in selected],
            "metric": score["metric"], "value": score["value"],
        })
        if (score["value"] > incumbent if greater else score["value"] < incumbent):
            incumbent = score["value"]
            best = (selected, {**score, "experiment_id": ranking[0]["experiment_id"]})
    return best, comparisons
