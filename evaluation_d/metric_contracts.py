"""Supported primary metrics and their shared validation/scoring semantics."""

from __future__ import annotations

import math
from typing import Any

SUPPORTED_METRICS = {
    "accuracy": True,
    "f1_macro": True,
    "log_loss": False,
}


def validate_metric_spec(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"name", "greater_is_better"}
        or value.get("name") not in SUPPORTED_METRICS
        or type(value.get("greater_is_better")) is not bool
        or value["greater_is_better"] is not SUPPORTED_METRICS[value["name"]]
    ):
        raise ValueError(
            "primary_metric 必须是 accuracy/f1_macro（越大越好）或 log_loss（越小越好）"
        )
    return {"name": value["name"], "greater_is_better": value["greater_is_better"]}


def metric_in_domain(name: str, value: Any) -> bool:
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    if name in {"accuracy", "f1_macro"}:
        return 0.0 <= value <= 1.0
    return name == "log_loss" and value >= 0.0


def score_predictions(
    name: str,
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_ids: list[int],
) -> float:
    if not y_true or len(y_true) != len(y_pred) or len(y_true) != len(probabilities):
        raise ValueError("标签、预测类别和概率行数必须相同且非空")
    if class_ids != list(range(len(class_ids))) or len(class_ids) < 2:
        raise ValueError("class_ids 必须是完整的 0..K-1 类别序列")
    if any(label not in class_ids for label in y_true + y_pred):
        raise ValueError("标签或预测包含未知类别")
    if any(
        len(row) != len(class_ids)
        or any(not math.isfinite(value) or value < 0 or value > 1 for value in row)
        or abs(math.fsum(row) - 1.0) > 1e-6
        for row in probabilities
    ):
        raise ValueError("概率矩阵的列数、范围、有限性或行和不合法")

    if name == "accuracy":
        return sum(actual == predicted for actual, predicted in zip(y_true, y_pred, strict=True)) / len(y_true)
    if name == "f1_macro":
        per_class = []
        for class_id in class_ids:
            tp = sum(actual == class_id and predicted == class_id for actual, predicted in zip(y_true, y_pred, strict=True))
            fp = sum(actual != class_id and predicted == class_id for actual, predicted in zip(y_true, y_pred, strict=True))
            fn = sum(actual == class_id and predicted != class_id for actual, predicted in zip(y_true, y_pred, strict=True))
            denominator = 2 * tp + fp + fn
            per_class.append(2 * tp / denominator if denominator else 0.0)
        return math.fsum(per_class) / len(class_ids)
    if name == "log_loss":
        epsilon = 2.220446049250313e-16
        return -math.fsum(
            math.log(max(probabilities[index][actual], epsilon))
            for index, actual in enumerate(y_true)
        ) / len(y_true)
    raise ValueError("不支持的 primary metric")
