"""Row-shard alignment, bounded composition, deferred test and frozen prediction."""
from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main as team_main
from data_b.input_readers import normalize_prediction_input
from data_b.llm_analysis import AnalysisError, resolve_layout
from data_b.prepare import prepare_data
from data_b.task_directory import discover_task_directory, materialize_final_test

ROOT = Path(__file__).resolve().parents[1]


class Replies:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def call_llm(self, messages):
        self.calls += 1
        return {"text": json.dumps(self.value)}


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.area = ROOT / ".cache/test-temp" / ("shards-" + uuid.uuid4().hex)
        self.raw = self.area / "raw"
        self.raw.mkdir(parents=True)

    def metadata(self, **options):
        (self.raw / "info.json").write_text(json.dumps({"metric": "accuracy", **options}), encoding="utf-8")

    def csv(self, name, rows, header=("id", "value", "label"), delimiter=","):
        with (self.raw / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)

    def simple_csv(self):
        self.csv("train_10.csv", [["b0", 10, 0], ["b1", 11, 1]])
        self.csv("train_2.csv", [["a0", 2, 0], ["a1", 3, 1]])
        self.csv("val.csv", [["v0", 4, 0], ["v1", 5, 1]])
        self.metadata()

    def discover(self, name="staged"):
        return discover_task_directory(self.raw, self.area / name, max_trials=1)

    def contract(self, fmt, parts, **options):
        self.metadata(reader_format=fmt, partitions=parts, **options)

    def pair(self, feature, labels=None):
        return {"features": [feature], "labels": labels}

    def shards(self, *pairs):
        return {"shards": list(pairs)}

    def test_numbered_csv_order_provenance_and_single_file_validation(self):
        self.simple_csv()
        result = self.discover()
        spec = result["task_spec"]
        X = np.load(Path(spec["source_root"]) / "X_train.npy", allow_pickle=True)
        self.assertEqual(X[:, 0].astype(int).tolist(), [2, 3, 10, 11])
        with (Path(spec["source_root"]) / "sample_index_train.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["source_file"] for r in rows], ["train_2.csv"] * 2 + ["train_10.csv"] * 2)
        self.assertEqual([r["source_row"] for r in rows], ["0", "1", "0", "1"])
        self.assertEqual([r["sample_id"] for r in rows], ["a0", "a1", "b0", "b1"])
        package = prepare_data(spec, self.area / "prepared")
        self.assertEqual(package["status"], "success", package.get("error"))

    def test_tsv_shards_preserve_multiline_text(self):
        for n in (1, 2):
            self.csv(f"train_{n}.tsv", [[f"id{n}a", "good\nproduct", 0], [f"id{n}b", "bad product", 1]],
                     header=("id", "review", "label"), delimiter="\t")
        self.metadata(feature_columns=["review"], feature_types={"review": "text"})
        result = self.discover()
        self.assertEqual(result["task_spec"]["source_format"], "text_array_package")
        self.assertEqual(prepare_data(result["task_spec"], self.area / "prepared")["status"], "success")

    def test_all_training_shards_contribute_to_type_inference(self):
        self.csv("train_1.csv", [["a0", "001", 0], ["a1", "002", 1]])
        self.csv("train_2.csv", [["b0", "code-A", 0], ["b1", "code-B", 1]])
        self.csv("val.csv", [["v0", "003", 0], ["v1", "code-C", 1]])
        self.metadata()
        result = self.discover()
        self.assertEqual(result["task_spec"]["feature_types"], {"value": "categorical"})
        values = np.load(Path(result["task_spec"]["source_root"]) / "X_train.npy", allow_pickle=True)
        self.assertEqual(values[0, 0], "001")

    def test_missing_text_in_a_shard_keeps_original_rows_and_labels(self):
        self.csv("train_1.csv", [["a0", "", 0], ["a1", "good item", 1]], header=("id", "review", "label"))
        self.csv("train_2.csv", [["b0", "bad item", 0], ["b1", "good item", 1]], header=("id", "review", "label"))
        self.csv("val.csv", [["v0", "good item", 1], ["v1", "bad item", 0]], header=("id", "review", "label"))
        self.metadata(feature_columns=["review"], feature_types={"review": "text"})
        result = self.discover()
        filtered = result["discovery_report"]["sample_filtering"]["train"]
        self.assertEqual((filtered["input_count"], filtered["kept_count"]), (4, 3))
        self.assertEqual(filtered["dropped_samples"][0]["source_file"], "train_1.csv")
        self.assertEqual(filtered["dropped_samples"][0]["source_row"], "0")
        y = np.load(Path(result["task_spec"]["source_root"]) / "y_train.npy", allow_pickle=True)
        self.assertEqual(y.tolist(), ["1", "0", "1"])

    def test_duplicate_ids_across_shards_rejected(self):
        self.simple_csv()
        self.csv("train_10.csv", [["a0", 10, 0], ["b1", 11, 1]])
        with self.assertRaisesRegex(ValueError, "unique|Duplicate"):
            self.discover()

    def test_columns_mismatch_and_duplicate_number_rejected(self):
        self.simple_csv()
        self.csv("train_10.csv", [["b0", 10, 0]], header=("id", "other", "label"))
        with self.assertRaisesRegex(ValueError, "columns"):
            self.discover()
        self.csv("train_02.csv", [["z0", 20, 0]])
        with self.assertRaisesRegex(ValueError, "Duplicate shard"):
            self.discover("other")

    def test_mixing_monolithic_and_numbered_split_rejected(self):
        self.simple_csv()
        self.csv("train.csv", [["z", 20, 0]])
        with self.assertRaisesRegex(ValueError, "monolithic"):
            self.discover()

    def test_npys_pair_labels_per_shard_and_preserve_images(self):
        parts = {}
        expected = []
        for role in ("train", "validation"):
            items = []
            for n in (1, 2):
                features = f"{role}-{n}.npy"
                labels = f"{role}-labels-{n}.npy"
                X = np.full((2, 2, 3, 3), n, dtype=np.uint8)
                np.save(self.raw / features, X)
                np.save(self.raw / labels, np.asarray([[0], [1]], dtype=np.uint8))
                items.append(self.pair(features, labels))
                if role == "train":
                    expected.append(X)
            parts[role] = self.shards(*items)
        self.contract("npy", parts, feature_prefixes=["images"])
        result = self.discover()
        spec = result["task_spec"]
        actual = np.load(Path(spec["source_root"]) / "X_train.npy")
        np.testing.assert_array_equal(actual, np.concatenate(expected))
        self.assertEqual(actual.dtype, np.uint8)
        self.assertEqual(prepare_data(spec, self.area / "prepared")["status"], "success")

    def test_npy_shard_row_mismatch_not_hidden_by_total_counts(self):
        for name, count in (("a", 2), ("b", 3)):
            np.save(self.raw / f"{name}.npy", np.zeros((count, 2)))
            np.save(self.raw / f"y{name}.npy", [0, 1, 0] if name == "a" else [0, 1])
        self.contract("npy", {"train": self.shards(self.pair("a.npy", "ya.npy"), self.pair("b.npy", "yb.npy"))})
        with self.assertRaisesRegex(ValueError, "row counts"):
            self.discover()

    def test_npy_column_blocks_within_each_row_shard(self):
        items = []
        for n in (1, 2):
            np.save(self.raw / f"n{n}.npy", np.asarray([[n], [n + 1]], dtype=float))
            np.save(self.raw / f"c{n}.npy", np.asarray([["001"], ["002"]]))
            np.save(self.raw / f"y{n}.npy", [0, 1])
            items.append({"features": [f"n{n}.npy", f"c{n}.npy"], "labels": f"y{n}.npy"})
        self.contract("npy", {"train": self.shards(*items)}, feature_prefixes=["n", "c"])
        spec = self.discover()["task_spec"]
        self.assertEqual(spec["feature_types"], {"n_0": "numeric", "c_0": "categorical"})
        self.assertEqual(prepare_data(spec, self.area / "prepared")["status"], "success")

    def test_single_and_sharded_npy_infer_byte_strings_as_text(self):
        values = np.asarray([b"good", b"bad", b"yes", b"no"])
        labels = np.asarray([0, 1, 0, 1])
        np.save(self.raw / "X_train.npy", values)
        np.save(self.raw / "y_train.npy", labels)
        np.save(self.raw / "X_val.npy", values[:2])
        np.save(self.raw / "y_val.npy", labels[:2])
        single = self.discover("single")["task_spec"]

        np.save(self.raw / "a.npy", values[:2])
        np.save(self.raw / "b.npy", values[2:])
        np.save(self.raw / "ya.npy", labels[:2])
        np.save(self.raw / "yb.npy", labels[2:])
        self.contract("npy", {"train": self.shards(self.pair("a.npy", "ya.npy"), self.pair("b.npy", "yb.npy")),
                              "validation": self.pair("X_val.npy", "y_val.npy")})
        sharded = self.discover()["task_spec"]

        self.assertEqual(sharded["modality"], single["modality"])
        for spec in (single, sharded):
            self.assertEqual(spec["source_format"], "text_array_package")
            np.testing.assert_array_equal(np.load(Path(spec["source_root"]) / "X_train.npy", allow_pickle=True),
                                          [["good"], ["bad"], ["yes"], ["no"]])

    def test_npy_shard_shapes_and_dtypes_must_match(self):
        for name, shape in (("a", (2, 2, 2, 3)), ("b", (2, 3, 2, 3))):
            np.save(self.raw / f"{name}.npy", np.zeros(shape, dtype=np.uint8))
            np.save(self.raw / f"y{name}.npy", [0, 1])
        self.contract("npy", {"train": self.shards(self.pair("a.npy", "ya.npy"), self.pair("b.npy", "yb.npy"))})
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.discover()
        np.save(self.raw / "b.npy", np.zeros((2, 2, 2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.discover("different-dtype")

    def test_flat_json_shards_use_one_frozen_field_mapping(self):
        for name in ("a", "b", "v"):
            (self.raw / f"{name}.json").write_text(json.dumps([
                {"body": "bad item", "class": 0}, {"body": "good item", "class": 1}]), encoding="utf-8")
        self.contract("convert", {"train": self.shards(self.pair("a.json"), self.pair("b.json")),
                                   "validation": self.pair("v.json")},
                      reader_options={"source_format": "json", "field_paths": {"review": ["body"], "label": ["class"]}},
                      target_column="label", feature_columns=["review"], feature_types={"review": "text"})
        spec = self.discover()["task_spec"]
        self.assertEqual(spec["partitions"]["train"]["n_samples"], 4)
        self.assertEqual(prepare_data(spec, self.area / "prepared")["status"], "success")

    def test_shard_count_and_reused_paths_are_rejected(self):
        self.contract("csv", {"train": self.shards(*[self.pair(f"part{i}.csv") for i in range(129)])}, target_column="label")
        with self.assertRaisesRegex(ValueError, "128"):
            self.discover()
        self.contract("csv", {"train": self.shards(self.pair("same.csv"), self.pair("same.csv"))}, target_column="label")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.discover("reused")

    def test_text_line_shards_drop_blank_globally_and_keep_pairing(self):
        (self.raw / "a.txt").write_text("\ngood item\n", encoding="utf-8")
        (self.raw / "ya.txt").write_text("0\n1\n", encoding="utf-8")
        (self.raw / "b.txt").write_text("bad item\ngood item\n", encoding="utf-8")
        (self.raw / "yb.txt").write_text("0\n1\n", encoding="utf-8")
        (self.raw / "v.txt").write_text("bad item\ngood item\n", encoding="utf-8")
        (self.raw / "yv.txt").write_text("0\n1\n", encoding="utf-8")
        self.contract("text_lines", {"train": self.shards(self.pair("a.txt", "ya.txt"), self.pair("b.txt", "yb.txt")),
                                     "validation": self.pair("v.txt", "yv.txt")})
        result = self.discover()
        np.testing.assert_array_equal(np.load(Path(result["task_spec"]["source_root"]) / "y_train.npy", allow_pickle=True), ["1", "0", "1"])
        self.assertEqual(prepare_data(result["task_spec"], self.area / "prepared")["status"], "success")

    def test_aggregate_budget_cannot_be_bypassed_by_sharding(self):
        self.simple_csv()
        with patch("data_b.input_readers.MAX_ROWS", 3), self.assertRaisesRegex(ValueError, "budget"):
            self.discover()

    def test_lazy_test_shards_and_frozen_unlabeled_prediction(self):
        self.simple_csv()
        (self.raw / "test_1.csv").write_text("DO NOT OPEN", encoding="utf-8")
        (self.raw / "test_2.csv").write_text("DO NOT OPEN", encoding="utf-8")
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            self.assertFalse(path.parent == self.raw and path.name.startswith("test_"))
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            result = self.discover()
        self.csv("test_1.csv", [["t1", 2, 0]])
        self.csv("test_2.csv", [["t2", 3, 1]])
        final = materialize_final_test(result["test_descriptor_path"], self.area / "final")
        self.assertEqual(json.loads(final.read_text(encoding="utf-8"))["n_samples"], 2)
        self.csv("new.csv", [["new", 8]], header=("id", "value"))
        normalized = normalize_prediction_input(self.area / "staged/reader_rules.json", self.raw, "new.csv", self.area / "new/X.npy")
        self.assertEqual(np.load(normalized, allow_pickle=True).shape, (1, 1))

    def test_llm_arbitrary_shard_names_and_complete_path_evidence(self):
        self.csv("observations-z.csv", [["b0", 10, 0], ["b1", 11, 1]])
        self.csv("observations-a.csv", [["a0", 2, 0], ["a1", 3, 1]])
        self.csv("holdout-z.csv", [["v0", 4, 0], ["v1", 5, 1]])
        train_quote = "Training shards in order: observations-z.csv then observations-a.csv."
        text = train_quote + "\nValidation is holdout-z.csv.\nTarget column is label.\nPrimary metric accuracy."
        (self.raw / "notes.md").write_text(text, encoding="utf-8")
        layout = {"partitions": {"train": self.shards(self.pair("observations-z.csv"), self.pair("observations-a.csv")),
                                  "validation": self.pair("holdout-z.csv"), "test": None},
                  "target_column": "label", "id_columns": ["id"], "metric": "accuracy",
                  "evidence": [{"field": role, "document": "notes.md", "quote": quote} for role, quote in
                               (("train", train_quote), ("validation", "Validation is holdout-z.csv."),
                                ("target_column", "Target column is label."), ("metric", "Primary metric accuracy."))]}
        gateway = Replies(layout)
        resolution = resolve_layout(self.raw, gateway, self.area / "analysis", ["notes.md"])
        result = discover_task_directory(self.raw, self.area / "staged", resolution=resolution)
        values = np.load(Path(result["task_spec"]["source_root"]) / "X_train.npy", allow_pickle=True)
        self.assertEqual(values[:, 0].astype(int).tolist(), [10, 11, 2, 3])
        self.assertEqual(gateway.calls, 1)
        bad = copy.deepcopy(layout)
        bad["partitions"]["train"]["shards"][1]["features"] = ["missing.csv"]
        with self.assertRaises(AnalysisError):
            resolve_layout(self.raw, Replies(bad), self.area / "bad-analysis", ["notes.md"])

    def test_llm_column_blocks_and_row_shards_have_distinct_meanings(self):
        for n in (1, 2, 3):
            np.save(self.raw / f"numbers{n}.npy", np.asarray([[n], [n + 1]], dtype=float))
            np.save(self.raw / f"categories{n}.npy", np.asarray([["001"], ["002"]]))
            np.save(self.raw / f"targets{n}.npy", [0, 1])
        def block(n):
            return {"features": [f"categories{n}.npy", f"numbers{n}.npy"], "labels": f"targets{n}.npy"}
        train_quote = "Training shards: categories1.npy numbers1.npy targets1.npy; categories2.npy numbers2.npy targets2.npy."
        valid_quote = "Validation: categories3.npy numbers3.npy targets3.npy."
        doc = train_quote + "\n" + valid_quote + "\nPrimary metric accuracy."
        (self.raw / "notes.md").write_text(doc, encoding="utf-8")
        layout = {"feature_kinds": ["categorical", "numeric"], "metric": "accuracy",
                  "partitions": {"train": self.shards(block(1), block(2)), "validation": block(3)},
                  "evidence": [{"field": field, "document": "notes.md", "quote": quote} for field, quote in
                               (("train", train_quote), ("validation", valid_quote), ("metric", "Primary metric accuracy."))]}
        resolution = resolve_layout(self.raw, Replies(layout), self.area / "analysis", ["notes.md"])
        result = discover_task_directory(self.raw, self.area / "staged", resolution=resolution)
        spec = result["task_spec"]
        values = np.load(Path(spec["source_root"]) / "X_train.npy", allow_pickle=True)
        self.assertEqual(values.tolist(), [[1.0, "001"], [2.0, "002"], [2.0, "001"], [3.0, "002"]])
        self.assertEqual(prepare_data(spec, self.area / "prepared")["status"], "success")

    def test_npy_aggregate_element_budget_is_enforced(self):
        from data_b.npy_reader import ReaderLimits

        for name in ("a", "b"):
            np.save(self.raw / f"{name}.npy", np.zeros((2, 2)))
            np.save(self.raw / f"y{name}.npy", [0, 1])
        self.contract("npy", {"train": self.shards(self.pair("a.npy", "ya.npy"), self.pair("b.npy", "yb.npy"))})
        with (patch("data_b.npy_reader.ReaderLimits", return_value=ReaderLimits(max_elements=6)),
              self.assertRaisesRegex(ValueError, "budget")):
            self.discover()

    def test_sharded_csv_trains_freezes_and_predicts_in_fresh_process(self):
        self.simple_csv()
        output = self.area / "output"
        summary_path = team_main.run_project(task_directory=self.raw, output_root=output,
                                             max_trials=1, pretrained="disabled")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "success", summary.get("error"))
        self.csv("fresh.csv", [["new", 8]], header=("id", "value"))
        result = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--bundle", str(output / summary["bundle_path"]),
                                 "--features", str(self.raw / "fresh.csv"), "--raw-input", "--source-root", str(self.raw),
                                 "--output", str(self.area / "predictions.csv")],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.area / "predictions.csv").read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
