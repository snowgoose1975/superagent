"""Synthetic automatic-task flows, runnable with unittest and without pytest.

All inputs, API credentials and API responses are generated inside these tests.
Temporary artifacts are retained under /tmp for inspection.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from agent_a.upstream.llm_planner import LLMPlanner
from agent_a.upstream.planner import Planner
from data_b.modality_readers import combine_text_columns
from evaluation_d.evaluate import evaluate_training_majority_baseline
from executor_c.model_factory import build_estimator
from executor_c.model_registry import (
    capabilities, default_pipeline, normalize_pipeline, runtime_capabilities,
)

import main as team_main
from evaluation_d.school_gateway import SchoolGateway


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class _OneTrialController:
    def __init__(self, events: list[str], pipeline: dict[str, Any] | None = None) -> None:
        self.events = events
        self.pipeline = pipeline
        self.calls = 0
        self.contexts: list[dict[str, Any]] = []

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        if "frozen" in self.events or "test_started" in self.events:
            raise AssertionError("planner called after the final model was frozen")
        self.events.append("decide")
        self.calls += 1
        self.contexts.append(context)
        decision = {
            "action": "run",
            "model": "logistic_regression",
            "params": {"C": 1.0, "max_iter": 200},
            "reason": "Synthetic automatic-task integration check.",
        }
        if self.pipeline is not None:
            decision["pipeline"] = self.pipeline
        return decision


class AutomaticTaskRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="superagent-task-run-"))

    def _npy_task(
        self,
        name: str,
        *,
        numeric: bool = True,
        categorical: bool = True,
        metric: str = "accuracy",
        invalid_test_labels: bool = False,
    ) -> tuple[Path, set[Path]]:
        source = self.root / name
        source.mkdir()
        sizes = {"train": 48, "val": 12, "test": 8}
        test_paths: set[Path] = set()
        for split, count in sizes.items():
            labels = np.asarray([0, 1] * (count // 2), dtype=np.int64)
            if numeric:
                values = np.column_stack((labels * 4 - 2, np.arange(count) % 3)).astype(float)
                feature_path = source / f"N_{split}.npy"
                np.save(feature_path, values, allow_pickle=False)
                if split == "test":
                    test_paths.add(feature_path.resolve())
            if categorical:
                values = np.asarray([["low" if label == 0 else "high"] for label in labels])
                feature_path = source / f"C_{split}.npy"
                np.save(feature_path, values, allow_pickle=False)
                if split == "test":
                    test_paths.add(feature_path.resolve())
            label_path = source / f"y_{split}.npy"
            if split == "test" and invalid_test_labels:
                labels = np.full(count, 99, dtype=np.int64)
            np.save(label_path, labels, allow_pickle=False)
            if split == "test":
                test_paths.add(label_path.resolve())
        (source / "Readme.md").write_text(
            "Develop a binary classifier. Use N_*.npy as numeric features and "
            "C_*.npy as categorical features when present; y_*.npy contains labels. "
            "Use train and val for development and test only for final evaluation. "
            f"Report test {metric}.\n",
            encoding="utf-8",
        )
        (source / "info.json").write_text(
            json.dumps(
                {
                    "task_type": "binclass",
                    "num_classes": 2,
                    "n_num_features": 2 if numeric else 0,
                    "n_cat_features": 1 if categorical else 0,
                    "train_size": sizes["train"],
                    "val_size": sizes["val"],
                    "test_size": sizes["test"],
                }
            ),
            encoding="utf-8",
        )
        return source, test_paths

    def _csv_text_task(self, name: str) -> tuple[Path, set[Path]]:
        source = self.root / name
        source.mkdir()
        for split, count in (("train", 40), ("val", 10), ("test", 8)):
            with (source / f"{split}.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["body_a", "body_b", "label_pairwise"])
                for index in range(count):
                    label = index % 2
                    first = (
                        "wonderful exciting good news makes readers delighted today"
                        if label else
                        "terrible disappointing bad news leaves readers unhappy today"
                    )
                    second = "another detailed article compares the latest regional events today"
                    writer.writerow([first, second, label])
        (source / "README.md").write_text(
            "Predict which document has higher engagement. Use `body_a` and `body_b` "
            "as input. The target `label_pairwise` is 1 if A wins and 0 if B wins. "
            "Use train.csv and val.csv for model development. Use labeled test.csv "
            "only for final evaluation and report test accuracy.\n",
            encoding="utf-8",
        )
        return source, {(source / "test.csv").resolve()}

    def _run_with_read_audit(
        self,
        source: Path,
        test_paths: set[Path],
        output_name: str,
        *,
        pipeline: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Path, _OneTrialController, list[str], str]:
        output_root = self.root / output_name
        events: list[str] = []
        controller = _OneTrialController(events, pipeline)
        original_open = Path.open
        original_freeze = team_main.freeze_best
        original_evaluate = team_main.evaluate_frozen_bundle
        stdout = io.StringIO()

        def audit_open(path: Path, *args: Any, **kwargs: Any):
            if Path(path).resolve() in test_paths:
                self.assertIn("frozen", events, f"test opened before freeze returned: {path}")
                events.append("test_read")
            return original_open(path, *args, **kwargs)

        def record_freeze(*args: Any, **kwargs: Any) -> Path:
            bundle = original_freeze(*args, **kwargs)
            self.assertTrue(bundle.is_file())
            self.assertFalse(kwargs.get("partial", False))
            events.append("frozen")
            return bundle

        def record_evaluation(*args: Any, **kwargs: Any):
            self.assertIn("frozen", events)
            events.append("test_started")
            return original_evaluate(*args, **kwargs)

        with (
            mock.patch.object(Path, "open", audit_open),
            mock.patch.object(team_main, "freeze_best", record_freeze),
            mock.patch.object(team_main, "evaluate_frozen_bundle", record_evaluation),
            contextlib.redirect_stdout(stdout),
        ):
            summary_path = team_main.run_project(
                output_root=output_root,
                task_directory=source,
                controller=controller,
                max_trials=1,
            )
        return _read_json(summary_path), output_root, controller, events, stdout.getvalue()

    def _assert_success(
        self, summary: dict[str, Any], output_root: Path,
        controller: _OneTrialController, events: list[str], terminal: str,
    ) -> None:
        self.assertEqual(summary["status"], "success", summary)
        self.assertEqual(summary["input_mode"], "task_directory")
        self.assertEqual(summary["attempted_trials"], 1)
        self.assertEqual(summary["search_profile"], "expanded")
        self.assertEqual(summary["test_evaluation"]["status"], "success")
        self.assertEqual(summary["test_evaluation"]["n_samples"], 8)
        self.assertEqual(controller.calls, 1)
        self.assertIn("test_read", events)
        self.assertLess(events.index("decide"), events.index("frozen"))
        self.assertNotIn("decide", events[events.index("test_started") + 1:])
        bundle = _read_json(output_root / summary["bundle_path"])
        self.assertEqual(bundle["selection"]["partition"], "validation")
        self.assertFalse(bundle["training_policy"]["refit_on_validation"])
        self.assertIsNone(summary["partial_bundle_path"])
        report = _read_json(output_root / summary["test_evaluation"]["report_path"])
        self.assertEqual(report["value"], summary["test_evaluation"]["value"])
        self.assertEqual(report["accuracy"], report["value"])
        self.assertTrue((output_root / summary["test_evaluation"]["predictions_path"]).is_file())
        self.assertIn("[Test] accuracy =", terminal)
        self.assertIn(f"{report['value']:.6f}", terminal)
        history = (output_root / summary["history_path"]).read_text(encoding="utf-8")
        self.assertNotIn("test_evaluation", history)
        session_dir = (output_root / summary["history_path"]).parent
        snapshot = _read_json(session_dir / "task.snapshot.json")
        self.assertEqual(set(snapshot["partitions"]), {"train", "validation"})
        self.assertEqual(snapshot["train_policy"]["max_trials"], 1)
        for context in controller.contexts:
            self.assertNotIn("test_evaluation", context)
            self.assertNotIn("test_labels", context)

    def test_renamed_mixed_numeric_categorical_npy_runs_through_final_test(self) -> None:
        source, test_paths = self._npy_task("unseen_arbitrary_name")
        self._assert_success(*self._run_with_read_audit(source, test_paths, "mixed_run"))

    def test_categorical_only_npy_needs_no_hand_written_config(self) -> None:
        source, test_paths = self._npy_task("another_new_dataset", numeric=False)
        self._assert_success(*self._run_with_read_audit(source, test_paths, "categorical_run"))

    def test_csv_text_with_readme_target_runs_and_does_not_open_test_early(self) -> None:
        source, test_paths = self._csv_text_task("unseen_text_pair_task")
        pipeline = default_pipeline("raw_text")
        pipeline["text_vectorizer"]["params"]["field_mode"] = "separate"
        result = self._run_with_read_audit(
            source, test_paths, "text_run", pipeline=pipeline
        )
        self._assert_success(*result)
        summary, output_root = result[:2]
        bundle = _read_json(output_root / summary["bundle_path"])
        self.assertEqual(bundle["schema_version"], "2.0")
        self.assertEqual(bundle["representation"], "raw_text")
        self.assertEqual(bundle["modality"]["text_columns"], ["body_a", "body_b"])
        history = (output_root / summary["history_path"]).read_text(encoding="utf-8")
        self.assertIn('"field_mode": "separate"', history)
        self.assertEqual(summary["validation_baseline"]["metric"], "accuracy")
        self.assertIn("[Baseline] validation accuracy =", result[4])

    def test_final_test_failure_preserves_final_without_partial_or_planner_reentry(self) -> None:
        source, test_paths = self._npy_task(
            "invalid_final_labels", categorical=False, invalid_test_labels=True
        )
        summary, output_root, controller, events, _terminal = self._run_with_read_audit(
            source, test_paths, "failed_test_run"
        )
        self.assertEqual(summary["status"], "failed", summary)
        self.assertEqual(summary["error"]["code"], "FINAL_TEST_FAILED")
        self.assertEqual(summary["error"]["stage"], "final_test")
        self.assertEqual(summary["test_evaluation"]["status"], "failed")
        self.assertTrue((output_root / summary["bundle_path"]).is_file())
        self.assertIsNone(summary["partial_bundle_path"])
        self.assertEqual(list((output_root / "submission").rglob("partial_*")), [])
        self.assertEqual(controller.calls, 1)
        self.assertEqual(events.count("frozen"), 1)
        self.assertNotIn("decide", events[events.index("test_started") + 1:])

    def test_unsupported_task_and_metric_fail_before_gateway_or_training(self) -> None:
        unknown = self.root / "unknown_format"
        unknown.mkdir()
        (unknown / "README.md").write_text("Develop a classifier; report accuracy.", encoding="utf-8")
        unknown_metric, _ = self._npy_task(
            "unknown_metric", categorical=False, metric="ROC-AUC"
        )
        for source in (unknown, unknown_metric):
            with self.subTest(task=source.name):
                output_root = self.root / f"failure_{source.name}"
                factory = mock.Mock(side_effect=AssertionError("gateway initialized before validation"))
                with contextlib.redirect_stdout(io.StringIO()):
                    summary_path = team_main.run_project(
                        output_root=output_root, task_directory=source,
                        planner_mode="school", gateway_factory=factory, max_trials=1,
                    )
                summary = _read_json(summary_path)
                factory.assert_not_called()
                self.assertEqual(summary["status"], "failed", summary)
                self.assertEqual(summary["error"]["stage"], "task_discovery")
                self.assertTrue(summary["error"]["code"])
                self.assertTrue(summary["error"]["message"])
                self.assertEqual(summary["attempted_trials"], 0)
                self.assertIsNone(summary["bundle_path"])
                self.assertFalse((output_root / "runs").exists())

    def test_school_cli_uses_mock_transport_and_automatic_budget_defaults(self) -> None:
        source, _ = self._npy_task("synthetic_school_cli", categorical=False)
        output_root = self.root / "school_run"
        payloads: list[dict[str, Any]] = []
        constructor_kwargs: list[dict[str, Any]] = []

        def transport(request: Any, _timeout: float):
            payloads.append(json.loads(request.data))
            content = json.dumps(
                {"model": "logistic_regression", "params": {"C": 1.0, "max_iter": 200},
                 "reason": "Synthetic mocked school planner."}
            )
            body = {"choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110}}
            return 200, {}, json.dumps(body).encode("utf-8")

        def make_gateway(*args: Any, **kwargs: Any) -> SchoolGateway:
            constructor_kwargs.append(dict(kwargs))
            return SchoolGateway(*args, **kwargs, transport=transport)

        with (
            mock.patch("evaluation_d.env_config.load_api_key", return_value="synthetic-test-key") as key_loader,
            mock.patch("evaluation_d.school_gateway.SchoolGateway", side_effect=make_gateway),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = team_main.main(
                ["run", "--task", str(source), "--planner", "school",
                 "--max-trials", "1", "--output-root", str(output_root)]
            )
        self.assertEqual(result, 0)
        key_loader.assert_called_once()
        self.assertEqual(len(constructor_kwargs), 1)
        self.assertEqual(constructor_kwargs[0]["max_calls"], 20)
        self.assertEqual(str(constructor_kwargs[0]["budget_rmb"]), "20")
        self.assertIsNotNone(constructor_kwargs[0]["task_budget_ledger"])
        self.assertEqual(len(payloads), 1)
        summaries = list((output_root / "sessions").glob("session_*/summary.json"))
        self.assertEqual(len(summaries), 1)
        summary = _read_json(summaries[0])
        self.assertEqual(summary["status"], "success", summary)
        self.assertEqual(summary["test_evaluation"]["status"], "success")
        self.assertEqual(summary["llm_gateway"]["transport"], "injected")
        self.assertEqual(summary["llm_gateway"]["real_llm_calls"], 0)
        self.assertEqual(summary["llm_gateway"]["mock_calls"], 1)
        self.assertEqual(summary["llm_gateway"]["task_budget"]["budget_id"][:5], "auto_")
        serialized_prompt = json.dumps(payloads[0]["messages"])
        self.assertNotIn("y_test.npy", serialized_prompt)
        self.assertNotIn("synthetic-test-key", serialized_prompt)

    def test_legacy_budget_is_reused_and_conflicting_ledgers_do_not_reset_budget(self) -> None:
        output_root = self.root / "budget_history"
        prepared = output_root / "data/prepared/old_task/v1"
        prepared.mkdir(parents=True)
        (prepared / "data_manifest.json").write_text("{}", encoding="utf-8")
        digests = [hashlib.sha256(b"synthetic-features").hexdigest(),
                   hashlib.sha256(b"synthetic-labels").hexdigest()]
        (prepared / "package_result.json").write_text(json.dumps({
            "source_file_sha256": {"train": {
                "features": {"sha256": digests[0]}, "labels": {"sha256": digests[1]},
            }},
        }), encoding="utf-8")
        session = output_root / "sessions/session_previous"
        session.mkdir(parents=True)
        (session / "summary.json").write_text(json.dumps({
            "data_manifest_path": "data/prepared/old_task/v1/data_manifest.json",
            "llm_gateway": {"task_budget": {"budget_id": "legacy_task_budget"}},
        }), encoding="utf-8")
        ledger_root = output_root / "sessions/task-budgets"
        ledger_root.mkdir()
        (ledger_root / "legacy_task_budget.json").write_text("{}", encoding="utf-8")
        content = json.dumps(sorted(digests), ensure_ascii=False, sort_keys=True,
                             allow_nan=False, indent=2) + "\n"
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        discovery = {"budget_fingerprint": fingerprint}
        self.assertEqual(team_main._automatic_budget_id(output_root, discovery), "legacy_task_budget")
        auto_id = "auto_" + fingerprint[:40]
        (ledger_root / f"{auto_id}.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Multiple historical ledgers"):
            team_main._automatic_budget_id(output_root, discovery)

    def test_unverified_legacy_identity_cannot_create_or_adopt_ledger(self) -> None:
        root = self.root / "deleted_prepared"
        session = root / "sessions" / "session_previous"
        ledger = root / "sessions" / "task-budgets"
        session.mkdir(parents=True)
        ledger.mkdir()
        (ledger / "legacy.json").write_text("{}", encoding="utf-8")
        (session / "summary.json").write_text(json.dumps({
            "task_id": "legacy_table",
            "data_manifest_path": "data/prepared/legacy_table/v1/data_manifest.json",
            "llm_gateway": {"task_budget": {"budget_id": "legacy"}},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot verify historical ledgers"):
            team_main._automatic_budget_id(root, {
                "budget_fingerprint": "a" * 64, "task_id": "legacy_table",
            })
        self.assertEqual(team_main._automatic_budget_id(root, {
            "budget_fingerprint": "b" * 64, "task_id": "unrelated",
        }), "auto_" + "b" * 40)

    def test_manual_task_spec_entry_remains_compatible(self) -> None:
        from scripts.create_demo import create_demo

        output_root = self.root / "legacy_demo"
        config = create_demo(output_root, n_train=48, n_valid=12)
        task = _read_json(config)
        task["train_policy"]["max_trials"] = 1
        config.write_text(json.dumps(task), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            summary = _read_json(team_main.run_project(
                config, output_root, controller=_OneTrialController([])
            ))
        self.assertEqual(summary["status"], "success", summary)
        self.assertEqual(summary["input_mode"], "config")
        self.assertEqual(summary["test_evaluation"]["status"], "not_requested")

    def test_manual_school_budget_cap_is_checked_before_key_loading(self) -> None:
        for amount in ("-1", "NaN", "500"):
            with self.subTest(amount=amount), mock.patch("evaluation_d.env_config.load_api_key") as key_loader:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = team_main.main([
                        "run", "--config", str(self.root / "unused.json"), "--planner", "school",
                        "--llm-budget-rmb", amount,
                    ])
                self.assertEqual(code, 2)
                key_loader.assert_not_called()

    def test_malformed_decision_leaves_room_for_valid_trial(self) -> None:
        from scripts.create_demo import create_demo

        config = create_demo(self.root, n_train=48, n_valid=12)
        spec = _read_json(config)
        spec["train_policy"]["max_trials"] = 2
        config.write_text(json.dumps(spec), encoding="utf-8")
        controller = _OneTrialController([])
        good = controller.decide({})
        with (
            mock.patch.object(controller, "decide", side_effect=[{"action": "run"}, good]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            summary = _read_json(team_main.run_project(config, self.root, controller=controller))
        self.assertEqual(summary["status"], "success", summary.get("error"))
        self.assertEqual(summary["attempted_trials"], 1)
        self.assertEqual(summary["rejected_decisions"][0]["trial"], 1)
        self.assertTrue(summary["best_experiment_id"].endswith("_002"))

    def test_automatic_npy_images_run_to_final_test(self) -> None:
        source = self.root / "unknown_image_collection"
        source.mkdir()
        test_paths = set()
        rng = np.random.default_rng(73)
        for split, count in (("train", 32), ("val", 12), ("test", 8)):
            y = np.arange(count) % 2
            X = rng.integers(0, 256, size=(count, 4, 4), dtype=np.uint8)
            X[:, :2, :2] = (y[:, None, None] * 200).astype(np.uint8)
            feature, label = source / f"images_{split}.npy", source / f"y_{split}.npy"
            np.save(feature, X, allow_pickle=False)
            np.save(label, y, allow_pickle=False)
            if split == "test":
                test_paths.update({feature.resolve(), label.resolve()})
        (source / "README.md").write_text(
            "# New image task\n\nClassify image arrays. Report test accuracy.", encoding="utf-8"
        )
        result = self._run_with_read_audit(source, test_paths, "image_run")
        self._assert_success(*result)
        summary, root = result[:2]
        bundle = _read_json(root / summary["bundle_path"])
        self.assertEqual(bundle["representation"], "image_tensor")
        context = _read_json((root / summary["history_path"]).parent / "task_context.json")
        self.assertIn("Classify image", context["description"])

    def test_primary_f1_and_logloss_survive_auto_final_evaluation(self) -> None:
        for name in ("f1_macro", "log_loss"):
            with self.subTest(metric=name):
                source, test_paths = self._npy_task(
                    f"declared_{name}", categorical=False, metric=name
                )
                result = self._run_with_read_audit(source, test_paths, f"run_{name}")
                summary, root = result[:2]
                self.assertEqual(summary["status"], "success", summary)
                report = _read_json(root / summary["test_evaluation"]["report_path"])
                self.assertEqual(report["metric"], name)
                self.assertEqual(report["greater_is_better"], name != "log_loss")
                self.assertTrue(0 <= report["accuracy"] <= 1)


class GeneralRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="superagent-general-recipe-"))

    def test_separate_field_text_pipeline_preserves_where_a_signal_occurs(self) -> None:
        y = np.asarray([0, 1] * 20, dtype=np.int64)
        X = np.asarray(
            [["signal", "common"] if label == 0 else ["common", "signal"] for label in y],
            dtype=object,
        )
        modality = {
            "kind": "text",
            "text_columns": ["left", "right"],
            "max_characters": 100,
        }
        accuracies = {}
        for field_mode in ("shared", "separate"):
            pipeline = default_pipeline("raw_text")
            pipeline["text_vectorizer"]["params"]["field_mode"] = field_mode
            model = build_estimator(
                "logistic_regression",
                {"C": 1.0, "max_iter": 200},
                42,
                representation="raw_text",
                pipeline=pipeline,
                modality=modality,
            )
            values = combine_text_columns(X, modality["text_columns"]) if field_mode == "shared" else X
            model.fit(values, y)
            accuracies[field_mode] = float(np.mean(model.predict(values) == y))
        self.assertGreater(accuracies["separate"], 0.99)
        self.assertLess(accuracies["shared"], 0.60)

    def test_default_text_search_covers_model_families_and_field_recipes(self) -> None:
        legacy_pipeline = default_pipeline("raw_text")
        legacy_pipeline["text_vectorizer"]["params"].pop("field_mode")
        self.assertEqual(
            normalize_pipeline("raw_text", legacy_pipeline), default_pipeline("raw_text")
        )
        planner = Planner()
        context: dict[str, Any] = {
            "task_id": "task",
            "data_version": "v1",
            "history": [],
            "greater_is_better": True,
            "timeout_seconds": 120,
            "search_profile": "expanded",
            "data_overview": {"n_features": 2, "representations": ["raw_text"]},
            "capabilities": runtime_capabilities(
                "raw_text", min_class_count=10, native_table=False, performance=False,
            ),
        }
        recipes = set()
        for _ in range(12):
            proposal = planner.next_plan(context)
            self.assertIsNotNone(proposal)
            plan = proposal["plan"]
            self.assertEqual(plan["schema_version"], "2.0")
            strategy = plan["pipeline"]["text_vectorizer"]["params"]
            recipes.add(
                (
                    plan["model"],
                    strategy["field_mode"],
                    strategy["analyzer"],
                    strategy["ngram_min"],
                    strategy["ngram_max"],
                )
            )
            context["history"].append(
                {
                    "model": plan["model"],
                    "params": plan["params"],
                    "pipeline": plan["pipeline"],
                    "representation": "raw_text",
                    "status": "success",
                    "metric_value": None,
                }
            )
        self.assertEqual(len(recipes), 12)
        self.assertEqual(
            {recipe[0] for recipe in recipes},
            {"logistic_regression", "multinomial_nb", "complement_nb"},
        )
        self.assertTrue({recipe[2] for recipe in recipes} == {"word", "char"})

    def test_llm_can_choose_and_deduplicate_on_the_full_text_recipe(self) -> None:
        shared = default_pipeline("raw_text")
        separate = default_pipeline("raw_text")
        separate["text_vectorizer"]["params"]["field_mode"] = "separate"
        reply = {
            "text": json.dumps(
                {
                    "model": "logistic_regression",
                    "params": {"C": 1.0},
                    "pipeline": separate,
                }
            ),
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_rmb": 0.0,
        }
        planner = LLMPlanner(lambda _messages: reply, Planner())
        context: dict[str, Any] = {
            "task_id": "task",
            "data_version": "v1",
            "history": [
                {
                    "model": "logistic_regression",
                    "params": {"C": 1.0},
                    "pipeline": shared,
                    "representation": "raw_text",
                    "status": "success",
                    "metric_value": 0.51,
                }
            ],
            "greater_is_better": True,
            "timeout_seconds": 120,
            "remaining_trials": 5,
            "data_overview": {"n_train": 30, "n_features": 2, "representations": ["raw_text"]},
            "capabilities": capabilities(),
            "search_profile": "expanded",
            "search": {
                "phase": "coverage",
                "pending_model_ids": ["logistic_regression"],
                "pending_text_recipes": [
                    {
                        "model": "logistic_regression",
                        "field_mode": "separate",
                        "analyzer": "word",
                        "ngram_min": 1,
                        "ngram_max": 1,
                    }
                ],
            },
        }
        selected = planner.next_plan(context)
        self.assertEqual(selected["plan"]["schema_version"], "2.0")
        self.assertEqual(selected["plan"]["pipeline"], separate)

    def test_training_majority_baseline_is_scored_on_validation(self) -> None:
        task = self.root / "baseline"
        task.mkdir()
        (task / "data_manifest.json").write_text(
            json.dumps(
                {
                    "task_id": "baseline",
                    "data_version": "v1",
                    "classes": [
                        {"id": 0, "name": "minor"},
                        {"id": 1, "name": "major"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (task / "evaluation_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "task_id": "baseline",
                    "data_version": "v1",
                    "data_manifest_path": "data_manifest.json",
                    "labels_path": "validation_labels.csv",
                    "primary_metric": {"name": "accuracy", "greater_is_better": True},
                    "positive_class_id": None,
                }
            ),
            encoding="utf-8",
        )
        (task / "validation_labels.csv").write_text(
            "sample_id,y_true\nv1,0\nv2,0\nv3,1\nv4,1\nv5,1\n", encoding="utf-8"
        )
        baseline = evaluate_training_majority_baseline(
            task / "evaluation_manifest.json", {"minor": 2, "major": 8}
        )
        self.assertEqual(baseline["class_id"], 1)
        self.assertEqual(baseline["metric"], "accuracy")
        self.assertEqual(baseline["value"], 0.6)
        self.assertEqual(baseline["n_samples"], 5)
        evaluation_manifest_path = task / "evaluation_manifest.json"
        evaluation_manifest = _read_json(evaluation_manifest_path)
        evaluation_manifest["primary_metric"] = {
            "name": "log_loss", "greater_is_better": False,
        }
        evaluation_manifest_path.write_text(
            json.dumps(evaluation_manifest), encoding="utf-8"
        )
        logloss_baseline = evaluate_training_majority_baseline(
            evaluation_manifest_path, {"minor": 2, "major": 8}
        )
        expected_logloss = -(2 * math.log(0.2) + 3 * math.log(0.8)) / 5
        self.assertTrue(math.isclose(logloss_baseline["value"], expected_logloss))
        self.assertEqual(logloss_baseline["metric"], "log_loss")
        manifest_path = task / "data_manifest.json"
        manifest = _read_json(manifest_path)
        manifest["data_version"] = "different-version"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluate_training_majority_baseline(
                task / "evaluation_manifest.json", {"minor": 2, "major": 8}
            )


if __name__ == "__main__":
    unittest.main()
