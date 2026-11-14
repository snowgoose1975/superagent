"""Metric-domain and missing-class semantics are fixed by the task contract."""

from __future__ import annotations

import pytest

from evaluation_d.metric_contracts import (
    metric_in_domain,
    score_predictions,
    validate_metric_spec,
)


def test_accuracy_and_macro_f1_use_declared_complete_class_order():
    y_true = [0, 0]
    y_pred = [0, 1]
    probabilities = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
    assert score_predictions("accuracy", y_true, y_pred, probabilities, [0, 1, 2]) == 0.5
    assert score_predictions("f1_macro", y_true, y_pred, probabilities, [0, 1, 2]) == pytest.approx(
        2 / 9
    )


def test_log_loss_can_exceed_one_and_is_lower_is_better():
    value = score_predictions(
        "log_loss",
        [0, 1],
        [1, 0],
        [[0.05, 0.95], [0.05, 0.95]],
        [0, 1],
    )
    assert value > 1.0
    assert metric_in_domain("log_loss", value)
    assert not metric_in_domain("log_loss", -0.1)


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "accuracy", "greater_is_better": 1},
        {"name": "log_loss", "greater_is_better": True},
        {"name": "unknown", "greater_is_better": True},
    ],
)
def test_metric_spec_rejects_wrong_direction_or_unknown_name(spec):
    with pytest.raises(ValueError):
        validate_metric_spec(spec)


def test_score_rejects_unknown_true_label_but_accepts_missing_validation_class():
    assert score_predictions("f1_macro", [0, 0], [0, 0], [[0.9, 0.1], [0.8, 0.2]], [0, 1]) == 0.5
    with pytest.raises(ValueError, match="未知类别"):
        score_predictions("accuracy", [2], [0], [[1.0, 0.0]], [0, 1])
