"""Exercise planning, training, and scoring with mock HTTP responses."""

from __future__ import annotations

import json

import main as team_main
from evaluation_d.school_gateway import SchoolGateway
from executor_c.contracts import read_json
from scripts.create_demo import create_demo


def _config():
    return read_json(team_main.PROJECT_ROOT / "configs" / "school-api.json")


def _response(text, *, usage=True):
    body = {"choices": [{"message": {"content": text}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110}
    return 200, {}, json.dumps(body).encode("utf-8")


def _plan():
    return json.dumps(
        {
            "model": "logistic_regression",
            "params": {"C": 0.5, "max_iter": 500},
        }
    )


def test_school_single_mock_call_runs_real_c_d_and_marks_source(tmp_path):
    root = tmp_path / "school_flow"
    config_path = create_demo(root, n_train=72, n_valid=24)
    payloads = []

    def transport(request, timeout):
        payloads.append(json.loads(request.data))
        assert timeout == 90
        return _response(_plan())

    gateway = SchoolGateway(
        _config(), "synthetic-key-not-a-secret", root / "sessions", transport=transport
    )
    summary = read_json(
        team_main.run_project(config_path, root, planner_mode="school", gateway=gateway)
    )
    assert summary["status"] == "success", summary
    assert summary["planner_mode"] == "school_llm"
    assert summary["real_llm_calls"] == 0  # Mock usage is counted separately.
    assert summary["attempted_trials"] == 1
    assert len(payloads) == 1
    prompt = json.dumps(payloads[0]["messages"])
    assert "capabilities" in prompt
    assert "y_val" not in prompt and "validation_labels" not in prompt
    history = [
        json.loads(line) for line in (root / summary["history_path"]).read_text().splitlines()
    ]
    assert len(history) == 1
    assert history[0]["plan"]["model"] == "logistic_regression"
    assert history[0]["plan"]["random_seed"] == 42
    assert history[0]["plan"]["params"]["C"] == 0.5
    assert "llm" in history[0]["decision_source"]
    assert "fallback" not in history[0]["decision_source"]
    assert history[0]["evaluation"]["n_samples"] == 24
    assert (root / summary["bundle_path"]).is_file()


def test_school_unknown_usage_stops_before_any_training(tmp_path):
    root = tmp_path / "unknown_usage"
    config_path = create_demo(root, n_train=72, n_valid=24)
    attempts = []

    def transport(request, timeout):
        attempts.append(1)
        return _response(_plan(), usage=False)

    gateway = SchoolGateway(
        _config(),
        "synthetic-key-not-a-secret",
        root / "sessions",
        transport=transport,
    )
    summary = read_json(
        team_main.run_project(config_path, root, planner_mode="school", gateway=gateway)
    )
    assert summary["status"] == "failed"
    assert summary["attempted_trials"] == 0
    assert summary["bundle_path"] is None
    assert len(attempts) == 1 and gateway.can_call is False
    assert summary["llm_gateway"]["usage_status"] == "unknown"
    assert summary["llm_gateway"]["estimated_cost_rmb"] is None
    assert not (root / "runs").exists()


def test_school_known_usage_invalid_plan_falls_back_with_explicit_source(tmp_path):
    root = tmp_path / "invalid_plan"
    config_path = create_demo(root, n_train=72, n_valid=24)

    def transport(request, timeout):
        return _response("This response is deliberately not a JSON plan.")

    gateway = SchoolGateway(
        _config(), "synthetic-key-not-a-secret", root / "sessions", transport=transport
    )
    summary = read_json(
        team_main.run_project(config_path, root, planner_mode="school", gateway=gateway)
    )
    assert summary["status"] == "success", summary
    assert summary["attempted_trials"] == 1
    history = [
        json.loads(line) for line in (root / summary["history_path"]).read_text().splitlines()
    ]
    assert "fallback" in history[0]["decision_source"]
    assert summary["llm_gateway"]["usage_status"] == "known"


def test_school_network_failure_has_no_fallback_or_secret_in_logs(tmp_path):
    root = tmp_path / "network_failure"
    config_path = create_demo(root, n_train=72, n_valid=24)
    sentinel = "synthetic-sensitive-exception-text"

    def transport(request, timeout):
        raise TimeoutError(sentinel)

    gateway = SchoolGateway(_config(), sentinel, root / "sessions", transport=transport)
    summary = read_json(
        team_main.run_project(config_path, root, planner_mode="school", gateway=gateway)
    )
    assert summary["status"] == "failed" and summary["attempted_trials"] == 0
    for path in (root / "sessions").rglob("*"):
        if path.is_file():
            assert sentinel not in path.read_text(encoding="utf-8")
    assert gateway.can_call is False


def test_later_unknown_school_usage_preserves_best_partial_bundle(tmp_path):
    root = tmp_path / "partial_after_unknown_usage"
    config_path = create_demo(root, n_train=72, n_valid=24)
    task = read_json(config_path)
    task["train_policy"]["max_trials"] = 3
    config_path.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
    attempts = []

    def transport(request, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            return _response(_plan())
        raise TimeoutError("synthetic later timeout")

    gateway = SchoolGateway(
        _config(),
        "synthetic-key-not-a-secret",
        root / "sessions",
        transport=transport,
        max_calls=2,
    )
    summary = read_json(
        team_main.run_project(config_path, root, planner_mode="school", gateway=gateway)
    )
    assert summary["status"] == "failed"
    assert summary["attempted_trials"] == 1
    assert summary["bundle_path"] is None
    assert summary["partial_best_experiment_id"]
    partial_path = root / summary["partial_bundle_path"]
    bundle = read_json(partial_path)
    assert bundle["final_id"].startswith("partial_")
    assert bundle["experiment_id"] == summary["partial_best_experiment_id"]
    assert summary["llm_gateway"]["usage_status"] == "unknown"
    assert len(attempts) == 2
