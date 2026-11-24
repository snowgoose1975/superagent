"""Raw-format, declarative conversion and fused-feature compatibility tests."""
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
from PIL import Image

import main as team_main
from data_b.input_readers import (
    InputReaderError,
    normalize_prediction_input,
    validate_options,
)
from data_b.llm_analysis import AnalysisError, _path_evidence_matches, resolve_layout
from data_b.prepare import prepare_data
from data_b.task_directory import discover_task_directory, materialize_final_test

ROOT = Path(__file__).resolve().parents[1]


class Replies:
    def __init__(self, values):
        self.values = iter(values)
        self.messages = []

    def call_llm(self, messages):
        self.messages.append(copy.deepcopy(messages))
        return {"text": json.dumps(next(self.values))}


class InputReaderTests(unittest.TestCase):
    def test_documented_split_wildcards_preserve_role_and_original_evidence(self):
        quote = "For each split (train, val, test), use C_*.npy and y_*.npy."
        self.assertTrue(_path_evidence_matches("C_train.npy", quote, "train"))
        self.assertTrue(_path_evidence_matches("y_val.npy", quote, "validation"))
        self.assertFalse(_path_evidence_matches("C_test.npy", quote, "train"))
        self.assertFalse(_path_evidence_matches("elsewhere/C_train.npy", quote, "train"))

    def setUp(self):
        self.area = ROOT / ".cache" / "test-temp" / ("readers-" + uuid.uuid4().hex)
        self.source = self.area / "raw"
        self.source.mkdir(parents=True)

    def metadata(self, **kwargs):
        value = {"metric": "accuracy", **kwargs}
        (self.source / "info.json").write_text(json.dumps(value), encoding="utf-8")

    def records(self, name="train.tsv", *, count=12):
        rows = [{"id": f"{name}-{i}", "review": "good product" if i % 2 else "bad product", "label": i % 2}
                for i in range(count)]
        path = self.source / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "review", "label"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def discover(self):
        return discover_task_directory(self.source, self.area / "staged", max_trials=1)

    def test_native_tsv_text_and_lazy_final_test(self):
        self.records()
        self.records("val.tsv", count=4)
        (self.source / "test.tsv").write_text("UNREAD TEST", encoding="utf-8")
        self.metadata(target_column="label", id_columns=["id"], feature_columns=["review"],
                      feature_types={"review": "text"})
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.tsv")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            discovered = self.discover()
        self.assertEqual(discovered["task_spec"]["source_format"], "text_array_package")
        result = prepare_data(discovered["task_spec"], self.area / "prepared")
        self.assertEqual(result["status"], "success", result.get("error"))
        self.records("test.tsv", count=4)
        final = materialize_final_test(discovered["test_descriptor_path"], self.area / "final_input")
        self.assertEqual(json.loads(final.read_text(encoding="utf-8"))["n_samples"], 4)

    def test_flat_json_conversion_and_immutable_rules(self):
        rows = [{"body": "good" if i % 2 else "bad", "target": i % 2} for i in range(12)]
        (self.source / "source.json").write_text(json.dumps(rows), encoding="utf-8")
        self.metadata(reader_format="convert", reader_options={"source_format": "json",
                      "field_paths": {"text": ["body"], "label": ["target"]}},
                      partitions={"train": {"features": ["source.json"], "labels": None}},
                      target_column="label", feature_columns=["text"], feature_types={"text": "text"})
        discovered = self.discover()
        self.assertEqual(discovered["task_spec"]["modality"]["kind"], "text")
        reader = discovered["discovery_report"]["reader"]
        self.assertEqual(reader["options"]["field_paths"]["text"], ["body"])
        self.assertEqual(prepare_data(discovered["task_spec"], self.area / "prepared")["status"], "success")

    def test_nested_conversion_options_are_rejected(self):
        for options in (
            {"source_format": "json", "field_paths": {"text": ["body", "text"]}},
            {"source_format": "json", "records_path": ["rows"], "field_paths": {"text": ["body"]}},
        ):
            with self.assertRaises(InputReaderError):
                validate_options("convert", options)

    def test_nested_json_is_explicitly_rejected(self):
        (self.source / "train.json").write_text(json.dumps([{"text": {"content": "review"}, "label": 0}]), encoding="utf-8")
        self.metadata(reader_format="convert", reader_options={"source_format": "json",
                      "field_paths": {"text": ["text"], "label": ["label"]}}, target_column="label",
                      partitions={"train": {"features": ["train.json"], "labels": None}})
        with self.assertRaisesRegex(InputReaderError, "nested JSON is unsupported"):
            self.discover()

    def test_external_jsonl_reader_and_conversion_are_rejected(self):
        with self.assertRaises(InputReaderError):
            validate_options("jsonl", {})
        with self.assertRaises(InputReaderError):
            validate_options("convert", {"source_format": "jsonl", "field_paths": {"text": ["text"]}})

    def test_tsv_handles_quoted_multiline_and_numeric_columns(self):
        with (self.source / "train.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["age", "review", "label"])
            writer.writerows([[i, "a quoted\nreview", i % 2] for i in range(12)])
        self.metadata(target_column="label", feature_columns=["age", "review"],
                      feature_types={"age": "numeric", "review": "text"})
        discovered = self.discover()
        self.assertEqual(discovered["task_spec"]["schema_version"], "1.0")
        self.assertEqual(prepare_data(discovered["task_spec"], self.area / "prepared")["status"], "success")

    def test_custom_delimiter_conversion(self):
        (self.source / "unusual.dat").write_text("body|class\n" + "\n".join(
            f"{'good' if i % 2 else 'bad'}|{i % 2}" for i in range(12)), encoding="utf-8")
        self.metadata(reader_format="convert", reader_options={"source_format": "delimited", "delimiter": "|",
                      "field_paths": {"text": ["body"], "label": ["class"]}},
                      partitions={"train": {"features": ["unusual.dat"], "labels": None}},
                      target_column="label", feature_columns=["text"], feature_types={"text": "text"})
        self.assertEqual(self.discover()["task_spec"]["modality"]["kind"], "text")

    def test_text_lines_with_separate_labels(self):
        (self.source / "train.txt").write_text("\n".join(["good", "bad"] * 6), encoding="utf-8")
        (self.source / "y_train.txt").write_text("\n".join(["positive", "negative"] * 6), encoding="utf-8")
        self.metadata(reader_format="text_lines")
        discovered = self.discover()
        self.assertEqual(discovered["task_spec"]["label_names"], ["negative", "positive"])

    def test_mismatched_text_labels_are_rejected(self):
        (self.source / "train.txt").write_text("good\nbad", encoding="utf-8")
        (self.source / "y_train.txt").write_text("positive", encoding="utf-8")
        self.metadata(reader_format="text_lines")
        with self.assertRaisesRegex(InputReaderError, "counts must match"):
            self.discover()

    def class_directory(self, split, *, image, size=None):
        for label in ("positive", "negative"):
            directory = self.source / split / label
            directory.mkdir(parents=True)
            for i in range(6):
                if image:
                    mode = "RGBA" if i % 2 else "L"
                    value = (255, 0, 0, 100) if mode == "RGBA" else 180
                    Image.new(mode, size or (7 + i, 5 + i), value).save(directory / f"{i}.png")
                else:
                    (directory / f"{i}.txt").write_text(f"{label} review {i}", encoding="utf-8")

    def test_text_directory(self):
        self.class_directory("train", image=False)
        self.metadata(reader_format="text_directory")
        self.assertEqual(self.discover()["task_spec"]["modality"]["kind"], "text")

    def test_image_directory_normalizes_mode_and_defers_test(self):
        self.class_directory("train", image=True, size=(9, 8))
        self.class_directory("val", image=True, size=(9, 8))
        (self.source / "test").mkdir()
        (self.source / "test/sealed.png").write_bytes(b"unread")
        self.metadata(reader_format="image_directory")
        discovered = self.discover()
        self.assertEqual(discovered["task_spec"]["modality"]["image_shape"], [8, 9, 3])
        self.assertEqual(prepare_data(discovered["task_spec"], self.area / "prepared")["status"], "success")

    def test_image_readers_do_not_accept_resize_options(self):
        for fmt, options in (("image_directory", {"image_size": [64, 64]}),
                             ("image_csv", {"image_column": "photo", "image_size": [64, 64]})):
            with self.subTest(fmt=fmt), self.assertRaises(InputReaderError):
                validate_options(fmt, options)

    def test_image_directory_preserves_native_dimensions_and_pixels(self):
        self.class_directory("train", image=True, size=(96, 80))
        self.class_directory("val", image=True, size=(96, 80))
        self.metadata(reader_format="image_directory")
        discovered = self.discover()
        spec = discovered["task_spec"]
        self.assertEqual(spec["modality"]["image_shape"], [80, 96, 3])
        values = np.load(Path(spec["source_root"]) / "X_train.npy")
        with Image.open(self.source / "train/negative/0.png") as image:
            expected = np.asarray(image.convert("RGB"))
        np.testing.assert_array_equal(values[0], expected)
        rules = json.loads((Path(spec["source_root"]) / "reader_rules.json").read_text(encoding="utf-8"))
        self.assertEqual(rules["reader"]["decoded_image_shape"], [80, 96, 3])
        self.assertNotIn("image_size", rules["reader"]["options"])
        self.assertEqual(prepare_data(spec, self.area / "prepared")["status"], "success")

    def test_native_image_directory_rejects_different_training_sizes(self):
        self.class_directory("train", image=True)
        self.metadata(reader_format="image_directory")
        with self.assertRaisesRegex(InputReaderError, "Image size differs from training"):
            self.discover()

    def image_csv(self, *, mixed=False, with_test=True):
        (self.source / "pictures").mkdir()
        for split in (["train", "val", "test"] if with_test else ["train", "val"]):
            with (self.source / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "photo", "review", "price", "label"])
                for i in range(12 if split == "train" else 4):
                    name = f"pictures/{split}-{i}.png"
                    Image.new("RGB", (8, 7), (240 if i % 2 else 10, 0, 0)).save(self.source / name)
                    writer.writerow([f"{split}-{i}", name, "good product" if i % 2 else "bad product", i + 10, i % 2])
        options = {"image_column": "photo"}
        if mixed:
            options["auxiliary_types"] = {"review": "text", "price": "numeric"}
        self.metadata(reader_format="image_csv", reader_options=options, target_column="label", id_columns=["id"])

    def test_image_path_table(self):
        self.image_csv()
        self.assertEqual(self.discover()["task_spec"]["modality"]["kind"], "image")

    def native_image_csv(self, *, mixed=False):
        self.image_csv(mixed=mixed)
        options = {"image_column": "photo"}
        if mixed:
            options["auxiliary_types"] = {"review": "text", "price": "numeric"}
        self.metadata(reader_format="image_csv", reader_options=options, target_column="label", id_columns=["id"])

    def test_native_image_index_rejects_validation_size_mismatch(self):
        self.native_image_csv()
        Image.new("RGB", (9, 7)).save(self.source / "pictures/val-0.png")
        with self.assertRaisesRegex(InputReaderError, "Image size differs from training"):
            self.discover()

    def test_native_mixed_image_index_freezes_dimensions_without_resize_option(self):
        self.native_image_csv(mixed=True)
        discovered = self.discover()
        spec = discovered["task_spec"]
        rules = Path(spec["source_root"]) / "reader_rules.json"
        reader = json.loads(rules.read_text(encoding="utf-8"))["reader"]
        self.assertEqual(reader["decoded_image_shape"], [7, 8, 3])
        self.assertNotIn("image_size", reader["options"])
        self.assertEqual(spec["schema_version"], "1.0")
        self.assertEqual(prepare_data(spec, self.area / "prepared")["status"], "success")
        Image.new("RGB", (9, 7)).save(self.source / "pictures/new.png")
        (self.source / "new.csv").write_text(
            "id,photo,review,price\nnew,pictures/new.png,good,10\n", encoding="utf-8")
        with self.assertRaisesRegex(InputReaderError, "Image size differs from training"):
            normalize_prediction_input(rules, self.source, "new.csv", self.area / "invalid.npy")

    def test_native_image_index_runs_c_freezes_and_predicts_raw_input(self):
        self.native_image_csv()
        output = self.area / "native_output"
        summary_path = team_main.run_project(task_directory=self.source, output_root=output, max_trials=1)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "success", summary.get("error"))
        self.assertEqual(summary["test_evaluation"]["status"], "success")
        manifest = json.loads((output / summary["data_manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["modality"]["image_shape"], [7, 8, 3])
        (self.source / "new.csv").write_text("id,photo\nnew,pictures/test-0.png\n", encoding="utf-8")
        command = [sys.executable, str(ROOT / "predict.py"), "--bundle", str(output / summary["bundle_path"]),
                   "--features", str(self.source / "new.csv"), "--raw-input", "--source-root", str(self.source),
                   "--output", str(self.area / "native_prediction.csv")]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len((self.area / "native_prediction.csv").read_text(encoding="utf-8").splitlines()), 2)

    def test_missing_mixed_content_drops_whole_rows_and_preserves_labels_ids_and_counts(self):
        self.native_image_csv(mixed=True)
        for split in ("train", "val", "test"):
            path = self.source / f"{split}.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if split == "train":
                rows[0]["photo"] = ""
                rows[1]["review"] = "  "
                rows[2]["photo"] = "pictures/not-present.png"
                rows[3]["price"] = ""
            elif split == "val":
                rows[0]["review"] = ""
            else:
                rows[0]["photo"] = ""
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "photo", "review", "price", "label"])
                writer.writeheader()
                writer.writerows(rows)
        info_path = self.source / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.update(train_size=12, val_size=4, test_size=4)
        info_path.write_text(json.dumps(info), encoding="utf-8")
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.csv")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            discovered = self.discover()
            spec = discovered["task_spec"]
            self.assertEqual(spec["partitions"]["train"]["n_samples"], 9)
            self.assertEqual(spec["partitions"]["validation"]["n_samples"], 3)
            labels = np.load(Path(spec["source_root"]) / "y_train.npy", allow_pickle=True)
            np.testing.assert_array_equal(labels, [str(i % 2) for i in range(3, 12)])
            with (Path(spec["source_root"]) / "sample_index_train.csv").open(encoding="utf-8", newline="") as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual([row["source_row"] for row in samples], [str(i) for i in range(3, 12)])
            self.assertEqual([row["sample_id"] for row in samples], [f"train-{i}" for i in range(3, 12)])
            result = prepare_data(spec, self.area / "prepared")
            self.assertEqual(result["status"], "success", result.get("error"))
        filters = discovered["discovery_report"]["sample_filtering"]
        self.assertEqual(filters["train"]["input_count"], 12)
        self.assertEqual(filters["train"]["dropped_count"], 3)
        self.assertEqual(filters["train"]["reason_counts"], {
            "missing_image:photo": 1, "missing_text:review": 1, "missing_image_file:photo": 1})
        with self.assertRaisesRegex(ValueError, "Final test requires all 4 rows"):
            materialize_final_test(discovered["test_descriptor_path"], self.area / "filtered_test")
        self.assertFalse((self.area / "filtered_test" / "test_config.json").exists())
        (self.source / "new.csv").write_text(
            "id,photo,review,price\nkeep,pictures/test-1.png,good,10\n"
            "no-photo,,good,20\nno-text,pictures/test-2.png,   ,30\n"
            "no-file,pictures/not-present.png,good,40\n", encoding="utf-8")
        normalized = self.area / "raw_prediction/X.npy"
        normalize_prediction_input(Path(spec["source_root"]) / "reader_rules.json", self.source, "new.csv", normalized)
        self.assertEqual(len(np.load(normalized, allow_pickle=True)), 1)
        report = json.loads((normalized.parent / "filtering_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["input_count"], 4)
        self.assertEqual(report["dropped_count"], 3)

    def test_text_csv_missing_content_is_removed_while_numeric_missing_is_kept(self):
        for split, count in (("train", 12), ("val", 4)):
            with (self.source / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "review", "price", "label"])
                for i in range(count):
                    writer.writerow([f"{split}-{i}", " " if split == "train" and i in (0, 1) else "good product",
                                     "" if i == 2 else i + 10, i % 2])
        self.metadata(target_column="label", id_columns=["id"],
                      feature_types={"review": "text", "price": "numeric"})
        discovered = self.discover()
        spec = discovered["task_spec"]
        self.assertEqual(spec["partitions"]["train"]["n_samples"], 10)
        self.assertEqual(discovered["discovery_report"]["n_source_train"], 12)
        self.assertEqual(discovered["discovery_report"]["sample_filtering"]["train"]["dropped_count"], 2)
        np.testing.assert_array_equal(np.load(Path(spec["source_root"]) / "y_train.npy", allow_pickle=True), [str(i % 2) for i in range(2, 12)])
        self.assertEqual(prepare_data(spec, self.area / "prepared")["status"], "success")

    def test_text_lines_drop_empty_text_and_keep_original_line_numbers(self):
        texts = ["" if i == 0 else " " if i == 1 else f"review {i}" for i in range(12)]
        (self.source / "train.txt").write_text("\n".join(texts), encoding="utf-8")
        (self.source / "y_train.txt").write_text("\n".join(str(i % 2) for i in range(12)), encoding="utf-8")
        self.metadata(reader_format="text_lines")
        discovered = self.discover()
        source = Path(discovered["task_spec"]["source_root"])
        with (source / "sample_index_train.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (source / "sample_index_validation.csv").open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
        self.assertEqual(sorted(int(row["source_row"]) for row in rows), list(range(2, 12)))
        self.assertEqual(discovered["discovery_report"]["sample_filtering"]["train"]["dropped_count"], 2)

    def test_text_npy_drops_none_and_blank_rows(self):
        np.save(self.source / "text_train.npy", np.asarray([[None if i == 0 else " " if i == 1 else f"review {i}"]
                                                           for i in range(12)], dtype=object))
        np.save(self.source / "y_train.npy", np.asarray([i % 2 for i in range(12)]))
        self.metadata()
        discovered = self.discover()
        self.assertEqual(discovered["task_spec"]["modality"]["kind"], "text")
        self.assertEqual(discovered["discovery_report"]["sample_filtering"]["train"]["dropped_count"], 2)
        self.assertEqual(prepare_data(discovered["task_spec"], self.area / "prepared")["status"], "success")

    def test_text_directory_drops_empty_files(self):
        self.class_directory("train", image=False)
        (self.source / "train/negative/0.txt").write_text("", encoding="utf-8")
        (self.source / "train/positive/1.txt").write_text("  ", encoding="utf-8")
        self.metadata(reader_format="text_directory")
        discovered = self.discover()
        filtering = discovered["discovery_report"]["sample_filtering"]["train"]
        self.assertEqual(filtering["input_count"], 12)
        self.assertEqual(filtering["kept_count"], 10)
        self.assertEqual(filtering["dropped_count"], 2)
        self.assertEqual(prepare_data(discovered["task_spec"], self.area / "prepared")["status"], "success")

    def test_missing_image_array_is_dropped_but_black_images_are_valid(self):
        values = np.zeros((12, 4, 4, 3), dtype=float)
        values[0] = np.nan
        np.save(self.source / "images_train.npy", values)
        np.save(self.source / "y_train.npy", np.asarray([i % 2 for i in range(12)]))
        self.metadata()
        discovered = self.discover()
        filtering = discovered["discovery_report"]["sample_filtering"]["train"]
        self.assertEqual(filtering["dropped_count"], 1)
        self.assertEqual(filtering["kept_count"], 11)

    def test_all_missing_text_is_an_explicit_error(self):
        (self.source / "train.txt").write_text(" \n \n \n ", encoding="utf-8")
        (self.source / "y_train.txt").write_text("0\n1\n0\n1", encoding="utf-8")
        self.metadata(reader_format="text_lines")
        with self.assertRaisesRegex(ValueError, "No usable train samples"):
            self.discover()

    def test_mixed_reader_preserves_all_pixels_and_checks_element_limit(self):
        self.image_csv(mixed=True)
        Image.fromarray(np.arange(7 * 8 * 3, dtype=np.uint8).reshape(7, 8, 3)).save(self.source / "pictures/train-0.png")
        with patch("data_b.input_readers.MAX_ELEMENTS", 2000), self.assertRaisesRegex(InputReaderError, "Full-resolution mixed input"):
            discover_task_directory(self.source, self.area / "too_large")
        discovered = self.discover()
        spec = discovered["task_spec"]
        pixel_columns = [name for name in spec["feature_columns"] if name.startswith("__image_pixel_")]
        self.assertEqual(len(pixel_columns), 7 * 8 * 3)
        self.assertEqual(len(spec["feature_columns"]), 7 * 8 * 3 + 2)
        pixels = np.load(Path(spec["source_root"]) / "X_train.npy", allow_pickle=True)[0, 2:].astype(float)
        with Image.open(self.source / "pictures/train-0.png") as image:
            expected = np.asarray(image.convert("RGB")).reshape(-1) / 255
        np.testing.assert_array_equal(pixels, expected)

    def test_image_reused_across_partitions_is_rejected(self):
        self.image_csv()
        path = self.source / "val.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("pictures/val-0.png", "pictures/train-0.png"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reuse the same source"):
            self.discover()

    def test_corrupt_image_is_rejected_without_dropping_rows(self):
        self.image_csv()
        (self.source / "pictures/train-0.png").write_bytes(b"broken")
        with self.assertRaisesRegex(InputReaderError, "Cannot decode"):
            self.discover()

    def test_mixed_image_text_table_uses_existing_c_and_fresh_process_predict(self):
        self.image_csv(mixed=True)
        # More than 4096 pixel columns verifies that the native image is neither
        # resized nor capped by the ordinary table feature budget.
        for path in (self.source / "pictures").glob("*.png"):
            with Image.open(path) as image:
                color = image.getpixel((0, 0))
            Image.new("RGB", (48, 32), color).save(path)
        for split, field in (("train", "photo"), ("val", "review")):
            path = self.source / f"{split}.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[-1][field] = ""
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "photo", "review", "price", "label"])
                writer.writeheader()
                writer.writerows(rows)
        output = self.area / "output"
        summary_path = team_main.run_project(task_directory=self.source, output_root=output, max_trials=1)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "success", summary.get("error"))
        self.assertEqual(summary["test_evaluation"]["status"], "success")
        self.assertEqual(summary["test_evaluation"]["n_samples"], 4)
        manifest = json.loads((output / summary["data_manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertGreaterEqual(manifest["n_features"], 48 * 32 * 3)
        session = summary_path.parent
        command = [sys.executable, str(ROOT / "predict.py"), "--bundle", str(output / summary["bundle_path"]),
                   "--features", str(session / "test_input/X_test.npy"), "--output", str(self.area / "fresh.csv")]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        bundle = json.loads((output / summary["bundle_path"]).read_text(encoding="utf-8"))
        self.assertIn("reader_rules.json", bundle["hashes"])
        # Fresh unlabeled raw inputs must pass through the same frozen B adapter.
        with (self.source / "new.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "photo", "review", "price"])
            writer.writerow(["new-0", "pictures/test-0.png", "bad product", 10])
        raw_command = [sys.executable, str(ROOT / "predict.py"), "--bundle", str(output / summary["bundle_path"]),
                       "--features", str(self.source / "new.csv"), "--raw-input", "--source-root", str(self.source),
                       "--output", str(self.area / "raw_prediction.csv")]
        raw = subprocess.run(raw_command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
        self.assertEqual(raw.returncode, 0, raw.stderr)
        self.assertEqual(len((self.area / "raw_prediction.csv").read_text(encoding="utf-8").splitlines()), 2)

    def test_path_escape_and_conversion_code_are_rejected(self):
        for options in ({"source_format": "json", "field_paths": [], "code": "import os"},
                        {"source_format": "json", "field_paths": []}):
            with self.assertRaises(InputReaderError):
                validate_options("convert", options)
        self.image_csv()
        path = self.source / "train.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("pictures/train-0.png", "../outside.png"), encoding="utf-8")
        with self.assertRaises(InputReaderError):
            self.discover()

    def test_duplicate_record_ids_are_rejected(self):
        self.records()
        path = self.source / "train.tsv"
        path.write_text(path.read_text(encoding="utf-8").replace("train.tsv-1", "train.tsv-0"), encoding="utf-8")
        self.metadata(target_column="label", id_columns=["id"])
        with self.assertRaisesRegex(InputReaderError, "unique"):
            self.discover()

    def test_program_selects_conversion_without_another_llm_round(self):
        (self.source / "development.tsv").write_text("body|class\n" + "\n".join(
            f"{'good' if i % 2 else 'bad'}|{i % 2}" for i in range(12)), encoding="utf-8")
        (self.source / "sealed.tsv").write_bytes(b"MUST NOT READ")
        doc = ("Train is development.tsv. Test is sealed.tsv. "
               "The target column is label, stored as class. Primary metric accuracy.")
        (self.source / "instructions.md").write_text(doc, encoding="utf-8")
        layout = {"format": "tsv", "feature_prefixes": [], "target_column": "label", "id_columns": [],
                  "partitions": {"train": {"features": ["development.tsv"], "labels": None},
                                 "validation": None, "test": {"features": ["sealed.tsv"], "labels": None}},
                  "feature_columns": [], "feature_types": {}, "reader_options": {}, "metric": "accuracy",
                  "evidence": [{"field": field, "document": "instructions.md", "quote": doc}
                               for field in ("train", "test", "target_column", "metric")], "unresolved_questions": []}
        layout.pop("format")
        layout["column_mapping"] = {"text": "body", "label": "class"}
        layout["feature_types"] = {"text": "text"}
        gateway = Replies([{"document_paths": ["instructions.md"], "unresolved_questions": []},
                           layout])
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotEqual(path.name, "sealed.tsv")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            resolution = resolve_layout(self.source, gateway, self.area / "analysis")
            discovered = discover_task_directory(self.source, self.area / "staged", resolution=resolution)
        self.assertEqual(discovered["task_spec"]["modality"]["kind"], "text")
        options = resolution["layout"]["reader_options"]
        self.assertEqual(options["delimiter"], "|")
        self.assertEqual(options["field_paths"], {"text": ["body"], "label": ["class"]})
        self.assertNotIn("MUST NOT READ", json.dumps(gateway.messages))
        self.assertEqual(len(gateway.messages), 2)
        self.assertFalse((self.area / "analysis/training_probe.json").exists())
        self.assertFalse((self.area / "analysis/reader.json").exists())

    def test_program_does_not_guess_a_different_label_column(self):
        self.records("development.tsv")
        doc = "Train development.tsv. Target label. Primary metric accuracy."
        (self.source / "instructions.md").write_text(doc, encoding="utf-8")
        layout = {"format": "tsv", "feature_prefixes": [], "target_column": "label", "id_columns": [],
                  "partitions": {"train": {"features": ["development.tsv"], "labels": None},
                                 "validation": None, "test": None}, "feature_columns": [], "feature_types": {},
                  "metric": "accuracy", "evidence": [{"field": field, "document": "instructions.md", "quote": doc}
                  for field in ("train", "target_column", "metric")], "unresolved_questions": []}
        layout["target_column"] = "other"
        with self.assertRaises(AnalysisError):
            resolve_layout(self.source, Replies([layout] * 4), self.area / "analysis",
                           document_paths=["instructions.md"])

    def test_program_selects_flat_json_conversion_from_semantics_and_leaves_test_unread(self):
        rows = [{"body": "good product" if i % 2 else "bad product", "label": i % 2} for i in range(12)]
        (self.source / "dataset.json").write_text(json.dumps(rows), encoding="utf-8")
        (self.source / "sealed.json").write_bytes(b"MUST STAY UNREAD")
        doc = "Train dataset.json. Test sealed.json. Target label. Primary metric accuracy."
        (self.source / "instructions.md").write_text(doc, encoding="utf-8")
        layout = {"data_type": "text", "target_column": "label", "metric": "accuracy",
                  "partitions": {"train": {"features": "dataset.json"}, "test": {"features": "sealed.json"}},
                  "feature_types": {"body": "text"},
                  "evidence": [{"field": field, "document": "instructions.md", "quote": doc}
                               for field in ("train", "test", "target_column", "metric")]}
        gateway = Replies([layout])
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotEqual(path.name, "sealed.json")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            resolution = resolve_layout(self.source, gateway, self.area / "analysis", document_paths=["instructions.md"])
            discovered = discover_task_directory(self.source, self.area / "staged", resolution=resolution)
            result = prepare_data(discovered["task_spec"], self.area / "prepared")
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(resolution["layout"]["format"], "convert")
        self.assertEqual(resolution["layout"]["reader_options"]["source_format"], "json")
        self.assertEqual(discovered["task_spec"]["modality"]["kind"], "text")
        self.assertEqual(len(gateway.messages), 1)


if __name__ == "__main__":
    unittest.main()
