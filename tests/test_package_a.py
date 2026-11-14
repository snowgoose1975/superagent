"""Synthetic checks for adapting the upstream rule planner to the v2 contract."""

from __future__ import annotations

from agent_a.controller import PackageAController
from executor_c.contracts import capabilities


def make_context(history=None, remaining=3, n_train=120):
    return {
        "training_profile": {"n_train": n_train},
        "capabilities": {},
        "history": history or [],
        "remaining_trials": remaining,
    }


def entry(controller, model, params, *, status="success", value=0.5):
    return {
        "experiment_id": "exp_session_001",
        "plan": {
            "task_id": controller.task_id,
            "data_version": controller.data_version,
            "model": model,
            "params": params,
        },
        "status": status,
        "evaluation": {"status": "success", "value": value} if status == "success" else None,
        "duration_seconds": 1.0,
        "result_relpath": None,
        "error": None,
    }


def test_package_planner_uses_encoded_width_and_returns_main_owned_fields():
    low = PackageAController("task", "v1", n_features=111, seed=42)
    wide = PackageAController("task", "v1", n_features=600, seed=42)
    low_decision = low.decide(make_context())
    wide_decision = wide.decide(make_context())
    assert low_decision["action"] == wide_decision["action"] == "run"
    assert low_decision["model"] == "random_forest"
    assert wide_decision["model"] == "logistic_regression"
    assert set(low_decision) == {"action", "model", "params", "reason"}
    assert low.last_decision_source == "rules"


def test_history_default_completion_and_numeric_normalization_avoid_duplicate():
    controller = PackageAController("task", "v1", n_features=10)
    tried = entry(
        controller,
        "random_forest",
        {"n_estimators": 100},
        status="failed",
    )
    decision = controller.decide(make_context([tried]))
    assert decision["action"] == "run"
    assert decision["model"] == "logistic_regression"

    lr = entry(controller, "logistic_regression", {"C": 1, "max_iter": 1000})
    after_both = controller.decide(make_context([tried, lr]))
    assert after_both["action"] == "run"
    assert after_both["model"] == "logistic_regression"
    assert after_both["params"]["C"] == 0.1


def test_no_improvement_counts_failures_and_respects_main_trial_cap():
    controller = PackageAController("task", "v1", n_features=10, patience=2, max_trials=8)
    history = [
        entry(controller, "random_forest", {"n_estimators": 100}, value=0.8),
        entry(controller, "logistic_regression", {"C": 1, "max_iter": 1000}, status="failed"),
        entry(controller, "logistic_regression", {"C": 0.1, "max_iter": 1000}, status="failed"),
    ]
    assert controller.decide(make_context(history))["action"] == "stop"
    assert controller.decide(make_context(remaining=0))["action"] == "stop"


def test_remaining_trial_limit_stops_before_upstream_planner():
    controller = PackageAController("task", "v1", n_features=10, max_trials=8)
    decision = controller.decide(make_context(remaining=0))
    assert decision == {"action": "stop", "reason": "main_trial_limit_reached"}
    assert controller.last_decision_source == "trial_limit"


def test_expanded_profile_covers_each_supported_dense_model_family():
    controller = PackageAController(
        "task",
        "v1",
        n_features=10,
        max_trials=12,
        search_profile="expanded",
        search_config={
            "require_family_coverage": True,
            "min_exploration_trials": 8,
            "patience": 8,
            "min_delta": 0.0,
            "max_consecutive_failures": 10,
        },
    )
    history = []
    selected = []
    for trial in range(4):
        decision = controller.decide(
            {
                **make_context(history, remaining=12 - trial),
                "capabilities": capabilities(),
            }
        )
        assert decision["action"] == "run"
        selected.append(decision["model"])
        history.append(
            entry(controller, decision["model"], decision["params"], value=0.6)
        )
    assert set(selected) == set(capabilities()["models"])


def test_expanded_patience_starts_after_minimum_exploration_and_ignores_failures():
    controller = PackageAController(
        "task",
        "v1",
        n_features=10,
        max_trials=20,
        search_profile="expanded",
        search_config={
            "require_family_coverage": False,
            "min_exploration_trials": 8,
            "patience": 2,
            "min_delta": 0.01,
            "max_consecutive_failures": 10,
        },
    )
    history = [
        entry(controller, "random_forest", {"n_estimators": 100}, value=0.8),
        *[
            entry(
                controller,
                "logistic_regression",
                {"C": 1, "max_iter": 1000},
                status="failed",
            )
            for _ in range(7)
        ],
    ]
    context = {
        **make_context(history, remaining=12),
        "capabilities": capabilities(),
    }
    assert controller.decide(context)["action"] == "run"
    history.append(
        entry(controller, "extra_trees", {"n_estimators": 100}, value=0.79)
    )
    assert controller.decide({**context, "history": history})["action"] == "run"
    history.append(
        entry(controller, "hist_gradient_boosting", {"max_iter": 100}, value=0.78)
    )
    assert controller.decide({**context, "history": history})["action"] == "stop"
