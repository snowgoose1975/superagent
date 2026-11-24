"""Mock-only checks for school A gateway handling; no HTTP or API keys."""

from __future__ import annotations

import json

import pytest
from agent_a.controller import PackageAController, SchoolAController
from agent_a.upstream.llm_planner import LLMPlanner, _compact_task_context, _system_message
from agent_a.upstream.planner import Planner
from executor_c.model_registry import capabilities


class FakeGateway:
    def __init__(self, reply=None, *, can_call=True, error=None):
        self.reply = reply or {
            "text": '{"model":"random_forest","params":{"n_estimators":100,"max_depth":null,"min_samples_leaf":1}}',
            "input_tokens": 10,
            "output_tokens": 12,
            "cost_rmb": 0.001,
        }
        self.can_call = can_call
        self.error = error
        self.remaining_budget_rmb = 0.1
        self.calls = 0
        self.last_messages = None

    def metadata(self):
        return {"provider": "mock", "is_mock": True}

    def set_request_timeout_cap(self, seconds):
        self.request_timeout_cap = seconds

    def call_llm(self, messages):
        self.calls += 1
        self.last_messages = messages
        assert {"role": "system", "content": messages[0]["content"]} == messages[0]
        assert "capabilities" in messages[1]["content"]
        if self.error:
            raise self.error
        return self.reply


def context():
    return {
        "training_profile": {"n_train": 100, "n_features_raw": 4},
        "capabilities": capabilities(),
        "history": [],
        "remaining_trials": 3,
    }


def test_school_controller_accepts_valid_llm_plan_and_records_source():
    gateway = FakeGateway()
    controller = SchoolAController("task", "v1", 10, gateway=gateway)
    decision = controller.decide(context())
    assert decision["action"] == "run"
    assert decision["model"] == "random_forest"
    assert controller.last_decision_source == "llm"
    assert controller.decision_sources == ["llm"]
    assert gateway.calls == 1


def test_invalid_llm_reply_falls_back_to_rules_without_logging_reply():
    gateway = FakeGateway(
        reply={
            "text": '{"model":"random_forest","params":{"n_estimators":501,"max_depth":null,"min_samples_leaf":1}}',
            "input_tokens": 10,
            "output_tokens": 12,
            "cost_rmb": 0.001,
        }
    )
    controller = SchoolAController("task", "v1", 10, gateway=gateway)
    decision = controller.decide(context())
    assert decision["action"] == "run"
    assert controller.last_decision_source == "rules_fallback"
    assert "text" not in str(controller._events.names)


def test_missing_usage_stops_via_exception_instead_of_rule_fallback():
    gateway = FakeGateway(reply={"text": '{"action":"stop"}'})
    controller = SchoolAController("task", "v1", 10, gateway=gateway)
    with pytest.raises(RuntimeError, match="missing usage"):
        controller.decide(context())
    assert gateway.calls == 1


def test_gateway_unavailable_stops_without_call_and_gateway_error_propagates():
    unavailable = FakeGateway(can_call=False)
    controller = SchoolAController("task", "v1", 10, gateway=unavailable)
    assert controller.decide(context())["action"] == "stop"
    assert unavailable.calls == 0

    broken = SchoolAController(
        "task", "v1", 10, gateway=FakeGateway(error=RuntimeError("gateway failed"))
    )
    with pytest.raises(RuntimeError, match="gateway failed"):
        broken.decide(context())


def test_remaining_trial_limit_stops_without_school_api_call():
    gateway = FakeGateway()
    controller = SchoolAController("task", "v1", 10, gateway=gateway)
    assert controller.decide({**context(), "remaining_trials": 0})["action"] == "stop"
    assert gateway.calls == 0
    assert controller.last_decision_source == "trial_limit"


def test_llm_stop_is_respected_during_coverage_phase():
    gateway = FakeGateway(
        reply={
            "text": '{"action":"stop","reason":"validation plateau"}',
            "input_tokens": 10,
            "output_tokens": 12,
            "cost_rmb": 0.001,
        }
    )
    controller = SchoolAController(
        "task",
        "v1",
        10,
        gateway=gateway,
        search_profile="expanded",
        search_config={"require_family_coverage": True, "min_exploration_trials": 8},
    )
    decision = controller.decide(
        {
            **context(),
            "remaining_trials": 5,
        }
    )
    assert decision == {"action": "stop", "reason": "validation plateau"}
    assert controller.last_decision_source == "planner_stop"


def test_second_llm_prompt_receives_only_safe_diagnostics_summary():
    gateway = FakeGateway()
    controller = SchoolAController("task", "v1", 10, gateway=gateway)
    first = controller.decide(context())
    history = [
        {
            "experiment_id": "exp_001",
            "plan": {
                "task_id": "task",
                "data_version": "v1",
                "model": first["model"],
                "params": first["params"],
            },
            "status": "success",
            "evaluation": {"status": "success", "value": 0.7},
            "duration_seconds": 1.0,
            "diagnostics": {
                "schema_version": "1.0",
                "status": "success",
                "phase": "complete",
                "timings_seconds": {"fit": 0.4, "total": 0.8},
                "input_shape": {"train": [80, 10], "validation": [20, 10]},
                "warnings": [{"category": "ConvergenceWarning", "phase": "fit", "message": "test"}],
                "failure": None,
            },
        }
    ]
    controller.decide({**context(), "history": history})
    prompt = gateway.last_messages[1]["content"]
    assert json.loads(prompt)["history"][0]["diagnostics"]["timings_seconds"]["fit"] == 0.4
    assert "validation_labels" not in prompt and "y_true" not in prompt


def test_planner_compacts_long_history_before_school_gateway_limit():
    history = [
        {
            "experiment_id": f"exp_{index:03d}",
            "model": "random_forest",
            "params": {
                "n_estimators": 100 + (index % 4) * 100,
                "max_depth": None,
                "min_samples_leaf": 1,
            },
            "status": "success",
            "metric_value": 0.7 + index / 1000,
            "diagnostics": {
                "status": "success",
                "phase": "complete",
                "timings_seconds": {"fit": 1.0, "total": 1.2},
                "warnings": [
                    {
                        "category": "ConvergenceWarning",
                        "phase": "fit",
                        "message": "x" * 1000,
                    }
                    for _ in range(20)
                ],
            },
        }
        for index in range(20)
    ]
    context_value = {
        "history": history,
        "data_overview": {
            "n_train": 8117,
            "n_features": 111,
            "representations": ["dense_numeric"],
            "training_profile": {
                "n_train": 8117,
                "n_features_raw": 111,
                "feature_names": [f"feature_{index}" for index in range(111)],
                "feature_types": {f"feature_{index}": "categorical" for index in range(111)},
                "train_label_counts": {"decline": 4500, "accept": 3617},
            },
        },
        "capabilities": capabilities(),
        "search": {"phase": "refinement", "pending_model_ids": []},
        "remaining_trials": 0,
    }
    planner = LLMPlanner(lambda _messages: None, Planner(), max_input_bytes=16_384)
    system = _system_message(context_value)
    user = planner._user_message(context_value, system_message=system)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    serialized_size = len(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    payload = json.loads(user)
    assert serialized_size + 256 + 512 <= 16_384
    assert len(payload["history"]) == 20
    assert payload["history"][0]["params"]["n_estimators"] == 100
    assert "diagnostics" not in payload["history"][0]
    assert payload["data_overview"]["training_profile"]["feature_names_omitted"] == 95


def test_task_documentation_is_bounded_and_reaches_planner_with_metric_name():
    documentation = {
        "metric": {"name": "f1_macro"},
        "objective": "\u9884\u6d4b\u5206\u7c7b\u6807\u7b7e\u3002" * 1000,
        "description": '\u8bf4\u660e\uff1a\\\n"' * 5000,
        "input_columns": [f"\u4e2d\u6587\u7279\u5f81_{index}" * 40 for index in range(100)],
        "target_column": "target",
        "task_type": "classification",
        "label_semantics": {"0": "\u62d2\u7edd", "1": "\u63a5\u53d7"},
        "raw_rows": ["must not be included"],
        "test_labels": ["must not be included"],
    }
    controller = PackageAController("task", "v1", 10)
    upstream = controller._upstream_context(
        {
            **context(),
            "primary_metric": {"name": "f1_macro", "greater_is_better": True},
            "task_context": documentation,
        },
        remaining_budget=1.0,
    )
    assert upstream["task_context"] == documentation
    assert upstream["task_context"] is not documentation
    planner = LLMPlanner(lambda _messages: None, Planner())
    payload = json.loads(planner._user_message(upstream))
    bounded = payload["task_context"]
    bounded_bytes = json.dumps(bounded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(bounded_bytes) <= 1600
    assert bounded["metric"] == "f1_macro"
    assert bounded["objective"] and bounded["description"] and bounded["input_columns"]
    assert "raw_rows" not in bounded and "test_labels" not in bounded
    assert payload["primary_metric"]["name"] == "f1_macro"
    assert "untrusted dataset documentation" in _system_message(upstream)


def test_task_documentation_yields_to_full_experiment_history_near_input_limit():
    history = [
        {
            "model": "random_forest",
            "params": {"n_estimators": 100 + index, "max_depth": None},
            "status": "success",
            "metric_value": 0.7,
        }
        for index in range(20)
    ]
    base_context = {
        "history": history,
        "primary_metric": {"name": "accuracy", "greater_is_better": True},
        "capabilities": capabilities(),
        "data_overview": {},
    }
    planner = LLMPlanner(lambda _messages: None, Planner())
    system = _system_message(base_context)
    baseline = planner._user_message(base_context, system_message=system)
    baseline_size = len(json.dumps(
        [{"role": "system", "content": system}, {"role": "user", "content": baseline}],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8"))
    max_bytes = baseline_size + 256 + 512 + 8
    planner = LLMPlanner(lambda _messages: None, Planner(), max_input_bytes=max_bytes)
    user = planner._user_message({
        **base_context,
        "task_context": {"objective": "\u5206\u7c7b" * 5000, "metric": "accuracy"},
    }, system_message=system)
    payload = json.loads(user)
    assert "task_context" not in payload
    assert len(payload["history"]) == 20
    assert payload["history"][-1]["params"] == history[-1]["params"]
    assert payload["primary_metric"]["name"] == "accuracy"
    assert _compact_task_context({"description": "a" * 1000}, max_bytes=24) == {}
