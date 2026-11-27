"""Regression checks for the local portfolio, budget handoff and frozen output."""
import json

import numpy as np

from evaluation_d.school_gateway import SchoolGateway
from executor_c.model_factory import build_estimator
from executor_c.model_registry import runtime_capabilities
from main import PROJECT_ROOT, run_project
from predict import predict_bundle


def _image_task(source):
    source.mkdir()
    for split, rows in (("train", 48), ("val", 12), ("test", 10)):
        labels = np.arange(rows) % 2
        images = np.zeros((rows, 8, 8), dtype=np.uint8)
        images[labels == 0, 2:5, :] = 255
        images[labels == 1, :, 2:5] = 255
        np.save(source / f"X_{split}.npy", images, allow_pickle=False)
        np.save(source / f"y_{split}.npy", labels, allow_pickle=False)
    (source / "README.md").write_text("Image classification. Report test accuracy.", encoding="utf-8")


def test_performance_school_cap_hands_off_and_runs_image_gradient(tmp_path):
    source = tmp_path / "images"
    _image_task(source)
    root = tmp_path / "work"
    calls = []

    def transport(request, timeout):
        calls.append((request, timeout))
        body = {"choices": [{"message": {"content": '{"model":"random_forest","params":{}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
        return 200, {}, json.dumps(body).encode()

    config = json.loads((PROJECT_ROOT / "configs" / "school-api.json").read_text())
    gateway = SchoolGateway(config, "synthetic", root / "sessions", max_calls=1, transport=transport)
    summary_path = run_project(
        task_directory=source, output_root=root, planner_mode="school", gateway=gateway,
        max_trials=6, performance=True,
    )
    summary = json.loads(summary_path.read_text())
    history = [json.loads(line) for line in (root / summary["history_path"]).read_text().splitlines()]
    assert summary["status"] == "success", summary
    assert len(calls) == 1
    assert summary["planner_fallback"]["error_code"] == "CALL_LIMIT"
    assert len(history) == 6
    assert any(
        item["status"] == "success" and
        item["plan"]["pipeline"]["image_transform"]["name"] == "gradient_cells"
        for item in history
    )
    bundle = next((root / "submission").glob("*/*/bundle.json"))
    prediction = predict_bundle(bundle, source / "X_test.npy", tmp_path / "new-predictions.csv")
    assert len(prediction.read_text().splitlines()) == 11


def test_rare_class_removes_calibrated_svm_and_hashing_handles_new_words():
    catalogue = runtime_capabilities(
        "raw_text", min_class_count=2, native_table=False, performance=True,
    )
    assert "linear_svc" not in catalogue["models"]
    assert "catboost" not in catalogue["models"]
    pipeline = {"text_vectorizer": {"name": "hashing", "params": {}}}
    model = build_estimator("sgd_logistic", {}, 42, representation="raw_text", pipeline=pipeline)
    model.fit(["two cats", "two dogs", "many cats", "many dogs"], [0, 1, 0, 1])
    probability = model.predict_proba(["strange cat and a dog"])
    np.testing.assert_allclose(probability.sum(axis=1), [1.0])


def test_insufficient_shared_budget_keeps_search_local(tmp_path):
    from evaluation_d.task_budget import TaskBudgetLedger

    source = tmp_path / "images"
    _image_task(source)
    root = tmp_path / "work"
    ledger = TaskBudgetLedger(
        tmp_path / "ledger", "synthetic", budget_rmb="0.001",
        profile="deepseek-v4.1-flash-offpeak-2026-10-21",
    )
    config = json.loads((PROJECT_ROOT / "configs" / "school-api.json").read_text())

    def transport(request, timeout):
        raise AssertionError("a request without a full budget reservation must not be sent")

    gateway = SchoolGateway(
        config, "synthetic", root / "sessions", transport=transport, task_budget_ledger=ledger,
    )
    summary = json.loads(run_project(
        task_directory=source, output_root=root, planner_mode="school", gateway=gateway,
        max_trials=3, performance=True,
    ).read_text())
    assert summary["status"] == "success", summary
    assert summary["llm_gateway"]["call_attempts"] == 0
    assert summary["planner_fallback"]["error_code"] == "TASK_BALANCE_EXHAUSTED"
    assert summary["attempted_trials"] == 3
    assert ledger.snapshot()["settled_count"] == 0


def test_frequency_fallback_retains_known_and_unknown_categories():
    import pandas as pd

    from data_b.preprocessing import TablePreprocessor
    from executor_c.native_table import native_features

    kinds = {"category": "categorical", "value": "numeric"}
    training = pd.DataFrame({"category": ["a", "a", "b", "c"], "value": [1, 2, 3, 4]})
    new_rows = pd.DataFrame({"category": ["a", "d", None], "value": [5, 6, None]})
    preprocessor = TablePreprocessor(kinds, max_features=3, allow_frequency_fallback=True)
    preprocessor.fit(training)
    transformed = preprocessor.transform(new_rows)
    np.testing.assert_allclose(transformed[:, 1], [0.5, 0.0, 0.0])
    native = native_features(new_rows, kinds)
    assert native["category"].tolist() == ["value:a", "value:d", "missing"]
    assert np.isnan(native["value"].iloc[2])


def test_frozen_catboost_predicts_unseen_categories(tmp_path):
    import pytest

    pytest.importorskip("catboost")
    source = tmp_path / "native_table"
    source.mkdir()
    for split, rows in (("train", 60), ("val", 12), ("test", 10)):
        labels = np.arange(rows) % 2
        values = np.where(labels == 0, "low", "high")
        if split == "test":
            values[0] = "new"
        np.save(source / f"C_{split}.npy", values.reshape(-1, 1), allow_pickle=False)
        np.save(source / f"y_{split}.npy", labels, allow_pickle=False)
    (source / "README.md").write_text(
        "Use category arrays C_*.npy and labels y_*.npy; report test accuracy.",
        encoding="utf-8",
    )

    class CatBoostController:
        def decide(self, context):
            return {"action": "run", "model": "catboost",
                    "params": {"iterations": 25}, "reason": "native classifier"}

    root = tmp_path / "work"
    summary = json.loads(run_project(
        task_directory=source, output_root=root, controller=CatBoostController(),
        performance=True, max_trials=1,
    ).read_text())
    assert summary["status"] == "success", summary
    bundle = next((root / "submission").glob("*/*/bundle.json"))
    assert json.loads(bundle.read_text())["input_representation"] == "native_table"
    prediction = predict_bundle(bundle, source / "C_test.npy", tmp_path / "standalone-catboost.csv")
    assert len(prediction.read_text().splitlines()) == 11
