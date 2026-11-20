"""Three B upgrades with synthetic documents/data and the real D gateway + fake HTTP."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

import main as team_main
from data_b.llm_analysis import AnalysisError, resolve_layout, review_quality
from data_b.prepare import prepare_data
from data_b.preprocessing import TablePreprocessor, validate_preprocessing
from data_b.task_directory import discover_task_directory
from evaluation_d.school_gateway import GatewayError, SchoolGateway
from evaluation_d.task_budget import TaskBudgetLedger

ROOT = Path(__file__).resolve().parents[1]


class BLLMTests(unittest.TestCase):
    def setUp(self):
        self.area = ROOT / ".cache" / "test-temp" / ("b-llm-" + uuid.uuid4().hex)
        self.source = self.area / "source"
        (self.source / "records").mkdir(parents=True)
        self.docs = {
            "train": "Training data is records/development.csv.",
            "validation": "Validation data is records/holdout.csv.",
            "test": "Test data is records/sealed.csv; never open before freezing.",
            "target_column": "The target column is clicked.",
            "metric": "The primary metric is accuracy.",
            "city_code": "city_code is a categorical city identifier, not a continuous measurement.",
            "age": "For age, -999 means missing, not a real age.",
        }
        (self.source / "使用说明.md").write_text("\n".join(self.docs.values()), encoding="utf-8")
        for role, count in (("development", 60), ("holdout", 12)):
            with (self.source / "records" / (role + ".csv")).open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["person", "city_code", "age", "clicked"])
                for i in range(count):
                    writer.writerow([f"{role}-{i:05d}", i % 3 if i < count - 1 else 99,
                                     -999 if i % 7 == 0 else 20 + i, i % 2])
        (self.source / "records/sealed.csv").write_bytes(b"must stay unread in B")

    def evidence(self, field):
        return {"field": field, "document": "使用说明.md", "quote": self.docs[field]}

    def layout(self):
        return {"format": "csv", "feature_prefixes": [],
                "partitions": {"train": {"features": ["records/development.csv"], "labels": None},
                               "validation": {"features": ["records/holdout.csv"], "labels": None},
                               "test": {"features": ["records/sealed.csv"], "labels": None}},
                "target_column": "clicked", "id_columns": ["person"], "metric": "accuracy",
                "feature_columns": [], "feature_types": {},
                "evidence": [self.evidence(field) for field in ("train", "validation", "test", "target_column", "metric")],
                "unresolved_questions": []}

    def quality(self):
        return {"findings": ["age uses a documented missing marker; city_code is categorical."],
                "type_overrides": {"city_code": "categorical"},
                "preprocessing": {"numeric_imputation": "median", "add_missing_indicator": True,
                                  "categorical_min_frequency": 3, "missing_values": {"age": [-999]}},
                "evidence": [self.evidence("age"), self.evidence("city_code")], "unresolved_questions": []}

    def gateway(self, replies, max_calls=8):
        self.requests = []
        def transport(request, timeout):
            body = json.loads(request.data)
            self.requests.append(body)
            item = replies[len(self.requests) - 1]
            if isinstance(item, Exception):
                raise item
            payload = {"choices": [{"message": {"content": json.dumps(item, ensure_ascii=False)}}],
                       "usage": {"prompt_tokens": 500, "completion_tokens": 200}}
            return 200, {}, json.dumps(payload).encode("utf-8")
        self.budget = TaskBudgetLedger(self.area / "budgets", "synthetic-task", budget_rmb="20",
                                     profile="deepseek-v4.1-flash-peak-2026-10-21")
        self.gw = SchoolGateway({"url": "https://example.invalid/v1/chat/completions", "model": "test-model",
                                 "input_cny_per_million": "1", "output_cny_per_million": "1",
                                 "max_input_tokens": 16384, "max_tokens": 1024},
                                "synthetic-only-not-real-key", self.area / "api", max_calls=max_calls,
                                budget_rmb="20", transport=transport, task_budget_ledger=self.budget)
        return self.gw

    def discover(self, gateway, name="analysis"):
        resolution = resolve_layout(self.source, gateway, self.area / name)
        discovered = discover_task_directory(self.source, self.area / (name + "_staging"), resolution=resolution, max_trials=1)
        return resolution, discovered

    def test_nonstandard_csv_names_and_quality_produce_c_package(self):
        gateway = self.gateway([self.layout(), self.quality()])
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotEqual(path.name, "sealed.csv")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            resolution, discovered = self.discover(gateway)
            spec, report = review_quality(discovered["task_spec"], gateway, self.area / "analysis", resolution["documents"])
            result = prepare_data(spec, self.area / "prepared")
        self.assertEqual(result["status"], "success", result.get("error"))
        processor = joblib.load(result["preprocessor_path"])
        self.assertEqual(processor.categorical_, ["city_code"])
        self.assertIn("missingindicator_age", " ".join(processor.feature_names_))
        self.assertGreater(processor.transformer_.named_transformers_["num"].statistics_[0], 0)
        self.assertTrue(np.isfinite(np.load(Path(result["data_manifest_path"]).parent / "X_valid.npy")).all())
        self.assertEqual(gateway.metadata()["call_attempts"], 2)
        self.assertEqual(gateway.metadata()["real_llm_calls"], 0)
        self.assertEqual(self.budget.snapshot()["settled_count"], 2)
        self.assertEqual(report["type_overrides"]["city_code"], "categorical")

    def test_unknown_usage_stops_without_retry(self):
        gateway = self.gateway([TimeoutError("synthetic transport timeout")])
        with self.assertRaises(GatewayError):
            resolve_layout(self.source, gateway, self.area / "analysis")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(gateway.metadata()["usage_status"], "unknown")

    def test_bad_json_schema_gets_one_known_usage_correction(self):
        gateway = self.gateway([{"wrong": True}, self.layout()])
        self.discover(gateway)
        self.assertEqual(gateway.metadata()["call_attempts"], 2)

    def test_fabricated_evidence_is_rejected(self):
        value = self.layout()
        value["evidence"][0]["quote"] = "invented statement"
        gateway = self.gateway([value, value])
        with self.assertRaises(AnalysisError):
            resolve_layout(self.source, gateway, self.area / "analysis")
        self.assertFalse((self.area / "analysis_staging").exists())

    def test_train_test_overlap_is_rejected(self):
        value = self.layout()
        value["partitions"]["test"]["features"] = ["records/development.csv"]
        gateway = self.gateway([value, value])
        with self.assertRaises(AnalysisError):
            resolve_layout(self.source, gateway, self.area / "analysis")

    def test_missing_metric_is_not_defaulted(self):
        value = self.layout()
        value["metric"] = None
        gateway = self.gateway([value, value])
        with self.assertRaises(AnalysisError):
            resolve_layout(self.source, gateway, self.area / "analysis")

    def test_unsupported_format_is_not_silently_converted(self):
        value = self.layout()
        value["format"] = "parquet"
        gateway = self.gateway([value, value])
        with self.assertRaises(AnalysisError):
            resolve_layout(self.source, gateway, self.area / "analysis")

    def test_quality_cannot_change_labels_or_metric(self):
        value = self.quality()
        value["target_column"] = "age"
        gateway = self.gateway([self.layout(), value, value])
        resolution, discovered = self.discover(gateway)
        with self.assertRaises(AnalysisError):
            review_quality(discovered["task_spec"], gateway, self.area / "analysis", resolution["documents"])

    def test_unproven_missing_marker_is_rejected(self):
        value = self.quality()
        value["preprocessing"]["missing_values"]["age"] = [-123]
        gateway = self.gateway([self.layout(), value, value])
        resolution, discovered = self.discover(gateway)
        with self.assertRaises(AnalysisError):
            review_quality(discovered["task_spec"], gateway, self.area / "analysis", resolution["documents"])

    def test_negative_threshold_and_code_component_rejected(self):
        for options in ({"categorical_min_frequency": -1}, {"python_code": "print('no')"},
                        {"missing_values": {"label": [-999]}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_preprocessing(options, {"city": "categorical"})

    def test_rare_categories_are_merged_from_training_only(self):
        import pandas as pd
        processor = TablePreprocessor({"city": "categorical", "age": "numeric"}, options={
            "categorical_min_frequency": 2, "missing_values": {"age": [-999]}, "add_missing_indicator": True})
        train = pd.DataFrame({"city": ["a", "a", "a", "b", "c"], "age": [-999., 10., 20., 30., 40.]})
        processor.fit(train)
        output = processor.transform(pd.DataFrame({"city": ["b", "c", "new"], "age": [-999., -999., -999.]}))
        np.testing.assert_array_equal(output[0], output[1])
        np.testing.assert_array_equal(output[1], output[2])
        self.assertEqual(output[0, 0], 25.)
        self.assertEqual(output[0, 1], 1.)

    def test_nonstandard_npy_paths(self):
        for name, array in (("feature_blob.npy", np.arange(80).reshape(40, 2)),
                            ("class_blob.npy", np.asarray([0, 1] * 20))):
            np.save(self.source / name, array)
        doc = "Train feature file feature_blob.npy and labels class_blob.npy. Primary metric accuracy."
        (self.source / "数组说明.md").write_text(doc, encoding="utf-8")
        value = self.layout()
        value.update(format="npy", feature_prefixes=["n"], target_column=None, id_columns=[])
        value["partitions"] = {"train": {"features": ["feature_blob.npy"], "labels": "class_blob.npy"}, "validation": None, "test": None}
        value["evidence"] = [{"field": field, "document": "数组说明.md", "quote": doc} for field in ("train", "metric")]
        gateway = self.gateway([value])
        _, discovered = self.discover(gateway)
        self.assertEqual(discovered["task_spec"]["source_format"], "npy_array_package")

    def test_explicit_arbitrary_txt_document(self):
        (self.source / "说明文件.txt").write_text("\n".join(self.docs.values()), encoding="utf-8")
        value = self.layout()
        for evidence in value["evidence"]:
            evidence["document"] = "说明文件.txt"
        gateway = self.gateway([value])
        resolve_layout(self.source, gateway, self.area / "analysis", ["说明文件.txt"])

    def test_numeric_looking_categories_retain_lexical_identity(self):
        path = self.source / "records/development.csv"
        text = path.read_text(encoding="utf-8").replace(",0,", ",000,").replace(",1,", ",001,")
        path.write_text(text, encoding="utf-8")
        gateway = self.gateway([self.layout(), self.quality()])
        resolution, discovered = self.discover(gateway)
        spec, _ = review_quality(discovered["task_spec"], gateway, self.area / "analysis", resolution["documents"])
        result = prepare_data(spec, self.area / "prepared")
        self.assertEqual(result["status"], "success", result.get("error"))
        processor = joblib.load(result["preprocessor_path"])
        categories = processor.transformer_.named_transformers_["cat"].named_steps["onehotencoder"].categories_[0]
        self.assertTrue(any('000' in value for value in categories))

    def test_full_flow_shared_gateway_and_frozen_preprocessor(self):
        # Exclude synthetic test from this development run; final-test routing has separate coverage.
        value = self.layout()
        value["partitions"]["test"] = None
        gateway = self.gateway([value, self.quality(), {"model": "logistic_regression", "params": {}}])
        summary_path = team_main.run_project(task_directory=self.source, output_root=self.area / "output",
                                            planner_mode="school", gateway=gateway, max_trials=1, b_analysis="llm")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "success", summary.get("error"))
        self.assertEqual(summary["llm_gateway"]["call_attempts"], 3)
        self.assertNotIn("sealed.csv", json.dumps(self.requests[2]["messages"]))
        spec = json.loads((summary_path.parent / "task.snapshot.json").read_text(encoding="utf-8"))
        features = Path(spec["source_root"]) / spec["partitions"]["validation"]["features_path"]
        predicted = self.area / "frozen_predictions.csv"
        process = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--bundle",
                                  str(self.area / "output" / summary["bundle_path"]), "--features", str(features),
                                  "--output", str(predicted)], capture_output=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stderr.decode(errors="replace"))
        with predicted.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 13)

    def test_cli_requires_stable_budget_before_loading_key(self):
        with patch("evaluation_d.env_config.load_api_key") as key_loader:
            code = team_main.main(["run", "--task", str(self.source), "--planner", "school", "--b-analysis", "llm"])
        self.assertEqual(code, 2)
        key_loader.assert_not_called()

    def test_cli_initializes_one_gateway_for_all_three_stages(self):
        value = self.layout()
        value["partitions"]["test"] = None
        replies = [value, self.quality(), {"model": "logistic_regression", "params": {}}]
        calls = []
        def transport(request, timeout):
            calls.append(json.loads(request.data))
            payload = {"choices": [{"message": {"content": json.dumps(replies[len(calls)-1], ensure_ascii=False)}}],
                       "usage": {"prompt_tokens": 500, "completion_tokens": 200}}
            return 200, {}, json.dumps(payload).encode()
        config = self.area / "api-config.json"
        config.write_text(json.dumps({"url": "https://example.invalid/v1/chat/completions", "model": "mock-only",
                                      "input_cny_per_million": "1", "output_cny_per_million": "1"}), encoding="utf-8")
        with patch("evaluation_d.env_config.load_api_key", return_value="synthetic-key") as key_loader, patch(
                "evaluation_d.school_gateway._urllib_transport", side_effect=transport):
            code = team_main.main(["run", "--task", str(self.source), "--output-root", str(self.area / "cli"),
                                   "--planner", "school", "--b-analysis", "llm", "--task-budget-id", "course-task",
                                   "--llm-config", str(config), "--max-trials", "1", "--llm-max-calls", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        key_loader.assert_called_once()

    def test_main_records_failed_discovery_usage_without_training(self):
        gateway = self.gateway([TimeoutError("synthetic timeout")])
        path = team_main.run_project(task_directory=self.source, output_root=self.area / "failure",
                                    planner_mode="school", gateway=gateway, b_analysis="llm", max_trials=1)
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["attempted_trials"], 0)
        self.assertEqual(summary["llm_gateway"]["usage_status"], "unknown")
        self.assertEqual(summary["llm_gateway"]["call_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
