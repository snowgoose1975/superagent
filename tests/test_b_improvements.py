"""B regression checks using retained synthetic inputs; no network or credentials."""
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

from agent_a.upstream.llm_planner import _compact_profile
from data_b.npy_reader import read_npy_array
from data_b.prepare import prepare_data
from data_b.preprocessing import TablePreprocessor
from data_b.task_directory import TaskDirectoryError, discover_task_directory, materialize_final_test
from executor_c.contracts import load_prepared_data, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, header, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class BImprovementsTests(unittest.TestCase):
    def setUp(self):
        self.area = ROOT / ".cache" / "test-temp" / f"b-improvements-{uuid.uuid4().hex}"
        self.source = self.area / "source"
        self.source.mkdir(parents=True)
        (self.source / "README.md").write_text(
            "Predict a class. Target `label`. Primary metric: accuracy.", encoding="utf-8")

    def csv_task(self, *, validation=True, test=False, mixed=False, many=False):
        names = ["amount", "category"]
        if mixed:
            names.append("review")
        if many:
            names += [f"number_{i}" for i in range(8)]
        for split, count in (("train", 40), ("val", 12)):
            if split == "val" and not validation:
                continue
            rows = []
            for i in range(count):
                label = i % 2
                values = [label * 10 + 1 if i else "", "a" if label else "b"]
                if mixed:
                    values.append("A long customer review with positive feedback and good service" if label else
                                  "A long customer review with negative feedback and bad service")
                if many:
                    values += [i + j for j in range(8)]
                rows.append([f"{split}-{i:05d}", *values, label])
            write_csv(self.source / f"{split}.csv", ["id", *names, "label"], rows)
        if test:
            (self.source / "test.csv").write_bytes(b"test contents must stay unread")

    def discover(self, name="staged"):
        return discover_task_directory(self.source, self.area / name, max_trials=1)

    def prepare(self, discovered):
        result = prepare_data(discovered["task_spec"], self.area / "prepared")
        self.assertEqual(result["status"], "success", result.get("error"))
        return result

    def test_csv_without_test_prepares_and_has_no_descriptor(self):
        self.csv_task()
        discovered = self.discover()
        self.assertIsNone(discovered["test_descriptor_path"])
        self.assertFalse((self.area / "staged/test_descriptor.json").exists())
        self.prepare(discovered)

    def test_train_only_csv_holdout_preserves_source_rows(self):
        self.csv_task(validation=False)
        discovered = self.discover()
        result = self.prepare(discovered)
        with Path(result["sample_index_path"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 40)
        self.assertEqual({r["source_file"] for r in rows}, {"train.csv"})
        self.assertEqual(len({r["sample_id"] for r in rows}), 40)
        for row in rows:
            self.assertEqual(row["sample_id"], f"train-{int(row['source_row']):05d}")
        self.assertEqual({r["split"] for r in rows}, {"train", "validation"})

    def test_npy_train_only_without_test(self):
        np.save(self.source / "N_train.npy", np.arange(80).reshape(40, 2))
        np.save(self.source / "y_train.npy", np.asarray([0, 1] * 20))
        result = self.prepare(self.discover())
        manifest = json.loads(Path(result["data_manifest_path"]).read_text())
        self.assertEqual(manifest["train"]["n_samples"] + manifest["validation"]["n_samples"], 40)

    def test_csv_original_id_survives_b_and_c_contract(self):
        self.csv_task()
        result = self.prepare(self.discover())
        plan = validate_plan({"schema_version": "1.0", "task_id": result["task_id"],
                              "data_version": result["data_version"], "experiment_id": "probe",
                              "model": "logistic_regression", "params": {},
                              "random_seed": 42, "timeout_seconds": 30})
        data = load_prepared_data(result["data_manifest_path"], plan)
        self.assertEqual(data.sample_ids, [f"val-{i:05d}" for i in range(12)])

    def test_duplicate_original_ids_are_rejected(self):
        write_csv(self.source / "train.csv", ["id", "x", "label"], [["0001", i, i % 2] for i in range(8)])
        with self.assertRaisesRegex(TaskDirectoryError, "Duplicate source sample IDs"):
            self.discover()

    def test_multi_column_ids_are_reversible_and_unique(self):
        (self.source / "info.json").write_text(json.dumps({"id_columns": ["site", "record"]}), encoding="utf-8")
        write_csv(self.source / "train.csv", ["site", "record", "x", "label"],
                  [["s" + str(i % 2), f"{i // 2:04d}", i, i % 2] for i in range(20)])
        result = self.prepare(self.discover())
        with Path(result["sample_index_path"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len({r["sample_id"] for r in rows}), 20)
        self.assertTrue(all(len(json.loads(r["original_id"])) == 2 for r in rows))

    def test_mixed_columns_have_independent_train_only_transforms(self):
        self.csv_task(mixed=True, many=True)
        result = self.prepare(self.discover())
        package = Path(result["data_manifest_path"]).parent
        processor = joblib.load(package / "preprocessor.joblib")
        self.assertIn("amount", processor.numeric_)
        self.assertEqual(processor.text_, ["review"])
        self.assertIn("category", processor.categorical_)
        self.assertTrue(np.isfinite(np.load(package / "X_valid.npy")).all())
        vocabulary = processor.transformer_.named_transformers_["text_0"].named_steps["tfidfvectorizer"].vocabulary_
        self.assertIn("positive", vocabulary)
        self.assertNotIn("id", processor.original_feature_names_)
        self.assertNotIn("label", processor.original_feature_names_)

    def test_validation_words_and_numeric_extreme_do_not_fit(self):
        header = ["id", "amount", "review", "label"]
        train = [[f"t{i}", "" if i == 0 else i, "An ordinary customer review about local service quality", i % 2] for i in range(10)]
        valid = [[f"v{i}", 100000, "An unprecedentedword review with distinct unseen lexical material", i % 2] for i in range(4)]
        write_csv(self.source / "train.csv", header, train)
        write_csv(self.source / "val.csv", header, valid)
        result = self.prepare(self.discover())
        processor = joblib.load(Path(result["preprocessor_path"]))
        vocabulary = processor.transformer_.named_transformers_["text_0"].named_steps["tfidfvectorizer"].vocabulary_
        self.assertNotIn("unprecedentedword", vocabulary)
        self.assertEqual(processor.transformer_.named_transformers_["num"].statistics_[0], 5)

    def test_explicit_feature_types_override_heuristic(self):
        self.csv_task(mixed=True)
        (self.source / "info.json").write_text(json.dumps({"feature_types": {
            "amount": "numeric", "category": "categorical", "review": "text"}}), encoding="utf-8")
        self.prepare(self.discover())

    def test_sample_index_changes_invalidate_cache(self):
        self.csv_task()
        discovered = self.discover()
        first = self.prepare(discovered)
        source_index = self.area / "staged/sample_index_validation.csv"
        source_index.write_text(source_index.read_text().replace("val-00000", "changed-00000"), encoding="utf-8")
        second = prepare_data(discovered["task_spec"], self.area / "prepared")
        self.assertEqual(second["status"], "failed")
        self.assertTrue(Path(first["data_manifest_path"]).is_file())

    def test_profile_is_training_only_and_reaches_bounded_llm_context(self):
        self.csv_task(mixed=True)
        result = self.prepare(self.discover())
        profile = json.loads(Path(result["profile_path"]).read_text())
        self.assertEqual(profile["n_train"], 40)
        self.assertEqual(profile["quality_summary"]["columns_with_missing"], 1)
        self.assertNotIn("validation_label_counts", profile)
        compact = _compact_profile(profile)
        self.assertIn("quality_summary", compact)
        self.assertIn("column_statistics_sample", compact)
        self.assertNotIn("source_file", json.dumps(compact))

    def test_preparation_code_change_gets_new_automatic_version(self):
        self.csv_task()
        first = self.discover("first")
        from data_b import task_directory
        original_hash = task_directory._sha256
        def changed_hash(path):
            return "0" * 64 if Path(path).name == "preprocessing.py" else original_hash(path)
        with patch.object(task_directory, "_sha256", changed_hash):
            second = self.discover("second")
        self.assertNotEqual(first["task_spec"]["data_version"], second["task_spec"]["data_version"])
        self.assertEqual(first["discovery_report"]["budget_fingerprint"], second["discovery_report"]["budget_fingerprint"])

    def test_many_text_fields_share_the_dense_feature_budget(self):
        import pandas as pd
        frame = pd.DataFrame({**{f"text_{i}": ["common alpha", "common beta"] for i in range(8)},
                              "number": [1.0, 2.0]})
        kinds = {name: "text" if name.startswith("text_") else "numeric" for name in frame}
        processor = TablePreprocessor(kinds)
        output = processor.fit_transform(frame)
        self.assertLessEqual(output.shape[1], 4096)
        self.assertEqual(len(processor.text_), 8)

    def test_pure_text_keeps_sparse_route_and_provenance(self):
        for split, count in (("train", 12), ("val", 4)):
            write_csv(self.source / f"{split}.csv", ["id", "review", "label"],
                      [[f"{split}-{i:05d}", "This is a sufficiently long review about a service experience", i % 2] for i in range(count)])
        discovered = self.discover()
        self.assertEqual(discovered["task_spec"]["schema_version"], "2.0")
        result = self.prepare(discovered)
        base = Path(result["data_manifest_path"]).parent
        self.assertTrue((base / "X_train.data").is_file())
        with (base / "valid_ids.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(list(csv.reader(stream))[1][0], "val-00000")
        profile = json.loads(Path(result["profile_path"]).read_text())
        self.assertIn("length", profile["column_statistics"][0])

    def test_image_without_test_has_train_statistics(self):
        for split, size in (("train", 8), ("val", 4)):
            np.save(self.source / f"X_{split}.npy", np.arange(size * 4, dtype=np.uint8).reshape(size, 2, 2))
            np.save(self.source / f"y_{split}.npy", np.asarray([0, 1] * (size // 2)))
        result = self.prepare(self.discover())
        profile = json.loads(Path(result["profile_path"]).read_text())
        self.assertEqual(profile["quality_summary"]["pixel_max"], 31)

    def test_test_stays_unread_and_mixed_reader_is_frozen(self):
        self.csv_task(mixed=True, test=True)
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == "test.csv":
                raise AssertionError("test read before freeze")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            discovered = self.discover()
            self.prepare(discovered)
        # Release synthetic test only after the discovery/preparation boundary.
        with (self.source / "val.csv").open(encoding="utf-8") as stream:
            content = stream.read()
        (self.source / "test.csv").write_text(content, encoding="utf-8")
        path = materialize_final_test(discovered["test_descriptor_path"], self.area / "test")
        features = read_npy_array(path.parent / "X_test.npy")
        self.assertEqual(features.shape, (12, 3))
        self.assertIsInstance(features[0, 2], str)
        with (path.parent / "sample_index_test.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(next(csv.DictReader(stream))["original_id"], "val-00000")

    def test_mixed_end_to_end_freeze_and_fresh_process_prediction(self):
        import main as team_main
        self.csv_task(mixed=True)
        class Controller:
            def decide(self, context):
                return {"action": "run", "model": "logistic_regression", "params": {}, "reason": "synthetic regression"}
        summary_path = team_main.run_project(task_directory=self.source, output_root=self.area / "output", controller=Controller(), max_trials=1)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "success", summary.get("error"))
        self.assertIsNone(summary["test_descriptor_path"])
        self.assertEqual(summary["test_evaluation"]["status"], "not_requested")
        bundle = self.area / "output" / summary["bundle_path"]
        snapshot = json.loads((summary_path.parent / "task.snapshot.json").read_text(encoding="utf-8"))
        validation_path = Path(snapshot["source_root"]) / snapshot["partitions"]["validation"]["features_path"]
        completed = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--bundle", str(bundle),
                                    "--features", str(validation_path), "--output", str(self.area / "predictions.csv")],
                                   capture_output=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
        with (self.area / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            fresh_rows = list(csv.reader(stream))
        self.assertEqual(len(fresh_rows), 13)
        c_predictions = (self.area / "output" / "runs" / summary["task_id"] /
                         summary["data_version"] / summary["best_experiment_id"] / "predictions.csv")
        with c_predictions.open(encoding="utf-8", newline="") as stream:
            c_rows = list(csv.reader(stream))
        np.testing.assert_allclose(
            np.asarray([row[1:] for row in fresh_rows[1:]], dtype=float),
            np.asarray([row[1:] for row in c_rows[1:]], dtype=float), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
