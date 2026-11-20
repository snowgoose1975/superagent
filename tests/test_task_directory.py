"""Synthetic task discovery checks; temporary inputs and outputs are preserved."""

from __future__ import annotations

import csv
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data_b.npy_reader import read_npy_array
from data_b import task_directory
from data_b.prepare import _validate_spec as validate_table_spec
from data_b.prepare_modalities import _validate_spec as validate_modality_spec
from data_b.task_directory import (
    TaskDirectoryError,
    discover_task_directory,
    materialize_final_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _array(path: Path, values: object) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=True)


def _csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class TaskDirectoryTests(unittest.TestCase):
    def _symlink(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows 未授予符号链接权限；不修改系统设置")
            raise

    def setUp(self) -> None:
        self.area = PROJECT_ROOT / ".cache" / "test-temp" / f"task-directory-{uuid.uuid4().hex}"
        self.area.mkdir(parents=True)
        self.source = self.area / "teacher_task"
        self.source.mkdir()

    def _npy_task(self, *, prefix: str = "N", holdout: bool = False) -> None:
        _array(self.source / f"{prefix}_train.npy", np.arange(24).reshape(12, 2))
        _array(self.source / "y_train.npy", np.asarray([0, 1] * 6))
        if not holdout:
            _array(self.source / f"{prefix}_valid.npy", np.arange(8).reshape(4, 2))
            _array(self.source / "y_valid.npy", np.asarray([0, 1, 0, 1]))
        (self.source / f"{prefix}_test.npy").write_bytes(b"must not open during discovery")
        (self.source / "y_test.npy").write_bytes(b"must not open during discovery")
        (self.source / "ReadMe.MD").write_text("Develop a classifier. Report test accuracy.", encoding="utf-8")

    def test_npy_discovers_without_opening_test_and_matches_b_contract(self) -> None:
        self._npy_task()
        original_open = Path.open

        def guarded_open(path: Path, *args: object, **kwargs: object):
            self.assertNotIn(path.name, {"N_test.npy", "y_test.npy"})
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            result = discover_task_directory(self.source, self.area / "staged")
        validate_table_spec(result["task_spec"])
        self.assertEqual(result["task_spec"]["label_values"], [0, 1])
        self.assertEqual(set(result["task_spec"]["partitions"]), {"train", "validation"})
        self.assertFalse(result["discovery_report"]["test_read_during_discovery"])
        self.assertNotIn("N_test.npy", result["discovery_report"]["source_files_sha256"])

    def test_mixed_n_c_has_stable_order_and_readme_column_names(self) -> None:
        self._npy_task()
        for split, count in (("train", 12), ("valid", 4)):
            _array(self.source / f"C_{split}.npy", np.asarray([["red"], ["blue"]] * (count // 2), dtype=object))
        (self.source / "C_test.npy").write_bytes(b"unread")
        (self.source / "ReadMe.MD").write_text(
            "Report test accuracy.\n\nThe columns, from left to right, are `a`, `b`, `color`.",
            encoding="utf-8")
        result = discover_task_directory(self.source, self.area / "staged")
        spec = result["task_spec"]
        self.assertEqual(spec["feature_columns"], ["a", "b", "color"])
        self.assertEqual(spec["feature_types"], {"a": "numeric", "b": "numeric", "color": "categorical"})
        values = read_npy_array(Path(spec["source_root"]) / "X_train.npy")
        self.assertEqual(list(values[0]), [0.0, 1.0, "red"])

    def test_csv_text_labels_and_id_are_inferred_and_final_materialization_is_consistent(self) -> None:
        (self.source / "README.md").write_text(
            "Predict the better headline. The target `label_pairwise` is 1 if A wins. Report test accuracy.",
            encoding="utf-8")
        header = ["id", "headline_A", "headline_B", "label_pairwise"]
        rows = [[i, f"The first headline about topic {i}", f"Another headline about topic {i}", i % 2] for i in range(8)]
        _csv(self.source / "train.csv", header, rows)
        _csv(self.source / "val.csv", header, rows[:4])
        (self.source / "test.csv").write_bytes(b"no test header until freeze")
        original_open = Path.open

        def guarded_open(path: Path, *args: object, **kwargs: object):
            self.assertNotEqual(path.name, "test.csv")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            result = discover_task_directory(self.source, self.area / "staged")
        validate_modality_spec(result["task_spec"])
        self.assertEqual(result["task_spec"]["modality"]["text_columns"], ["headline_A", "headline_B"])
        self.assertEqual(result["discovery_report"]["dropped_id_columns"], ["id"])
        self.assertEqual(result["task_spec"]["label_values"], ["0", "1"])
        # The test is now released after the synthetic final-freeze boundary.
        with (self.source / "test.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows[:3])
        config_path = materialize_final_test(result["test_descriptor_path"], self.area / "test_input")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["n_samples"], 3)
        values = read_npy_array(config_path.parent / "X_test.npy")
        labels = read_npy_array(config_path.parent / "y_test.npy")
        self.assertEqual(values.shape, (3, 2))
        self.assertEqual(list(labels), ["0", "1", "0"])
        sources = json.loads((config_path.parent / "test_source.json").read_text(encoding="utf-8"))
        self.assertEqual(set(sources["source_files_sha256"]), {"test.csv"})

    def test_csv_numeric_conversion_and_missing_values(self) -> None:
        (self.source / "info.json").write_text(json.dumps({"target_column": "outcome", "metric": "f1_macro"}), encoding="utf-8")
        header = ["amount", "group", "outcome"]
        _csv(self.source / "train.csv", header, [[1.5, "a", 0], [2.5, "b", 1], [3.5, "a", 0], [4.5, "b", 1]])
        _csv(self.source / "validation.csv", header, [["", "new", 0], [2, "a", 1]])
        _csv(self.source / "test.csv", header, [["", "new", 0], [3, "b", 1]])
        result = discover_task_directory(self.source, self.area / "staged")
        self.assertEqual(result["task_spec"]["primary_metric"]["name"], "f1_macro")
        self.assertEqual(result["task_spec"]["feature_types"], {"amount": "numeric", "group": "categorical"})
        config_path = materialize_final_test(result["test_descriptor_path"], self.area / "test_input")
        values = read_npy_array(config_path.parent / "X_test.npy")
        self.assertEqual(list(values[0]), [None, "new"])

    def test_csv_declared_inputs_exclude_extra_result_columns(self) -> None:
        (self.source / "README.md").write_text(
            "Use `a` and `b` as input. The target `label` is a class. Report test accuracy.",
            encoding="utf-8")
        header = ["a", "b", "future_result", "label"]
        for split in ("train", "val", "test"):
            _csv(self.source / f"{split}.csv", header, [[1, 2, 0, 0], [3, 4, 1, 1]])
        result = discover_task_directory(self.source, self.area / "staged")
        self.assertEqual(result["task_spec"]["feature_columns"], ["a", "b"])
        self.assertEqual(result["discovery_report"]["reader"]["ignored_columns"], ["future_result"])
        config_path = materialize_final_test(result["test_descriptor_path"], self.area / "test_input")
        values = read_npy_array(config_path.parent / "X_test.npy")
        self.assertEqual(values.shape, (2, 2))
        self.assertEqual(list(values[0]), [1.0, 2.0])

    def test_csv_input_declaration_cannot_include_target(self) -> None:
        (self.source / "info.json").write_text(
            json.dumps({"feature_columns": ["x", "label"]}), encoding="utf-8")
        for split in ("train", "val", "test"):
            _csv(self.source / f"{split}.csv", ["x", "label"], [[1, 0], [2, 1]])
        with self.assertRaisesRegex(TaskDirectoryError, "目标标签列不能作为输入"):
            discover_task_directory(self.source, self.area / "staged")

    def test_one_dimensional_strings_are_raw_text(self) -> None:
        for split in ("train", "val"):
            _array(self.source / f"X_{split}.npy", np.asarray(["hello", "world", "good", "bad"]))
            _array(self.source / f"y_{split}.npy", np.asarray([0, 1, 0, 1]))
        (self.source / "X_test.npy").write_bytes(b"unread")
        (self.source / "y_test.npy").write_bytes(b"unread")
        result = discover_task_directory(self.source, self.area / "staged")
        self.assertEqual(result["task_spec"]["source_format"], "text_array_package")
        validate_modality_spec(result["task_spec"])

    def test_image_contract_comes_from_train(self) -> None:
        for split, values in (("train", np.arange(16, dtype=np.uint8).reshape(4, 2, 2)),
                              ("val", np.full((4, 2, 2), 255, dtype=np.uint8))):
            _array(self.source / f"X_{split}.npy", values)
            _array(self.source / f"y_{split}.npy", np.asarray([0, 1, 0, 1]))
        (self.source / "X_test.npy").write_bytes(b"unread")
        (self.source / "y_test.npy").write_bytes(b"unread")
        result = discover_task_directory(self.source, self.area / "staged")
        self.assertEqual(result["task_spec"]["modality"], {"kind": "image", "image_shape": [2, 2], "channel_layout": "HW", "value_range": [0.0, 255.0]})
        validate_modality_spec(result["task_spec"])

    def test_missing_validation_reserves_train_rows_only_and_is_reproducible(self) -> None:
        self._npy_task(holdout=True)
        first = discover_task_directory(self.source, self.area / "first", random_seed=13)
        second = discover_task_directory(self.source, self.area / "second", random_seed=13)
        self.assertEqual(first["task_spec"]["data_version"], second["task_spec"]["data_version"])
        self.assertEqual(first["discovery_report"]["validation_split"]["kind"], "stratified_train_holdout")
        train_values = read_npy_array(self.area / "first" / "X_train.npy")
        valid_values = read_npy_array(self.area / "first" / "X_validation.npy")
        self.assertEqual(len(train_values) + len(valid_values), 12)
        self.assertFalse(set(train_values[:, 0]) & set(valid_values[:, 0]))
        for split in ("train", "validation"):
            labels = read_npy_array(self.area / "first" / f"y_{split}.npy")
            self.assertEqual(set(labels), {0, 1})

    def test_content_versions_change_and_budget_identity_survives_rename(self) -> None:
        self._npy_task()
        first = discover_task_directory(self.source, self.area / "first")
        renamed = self.area / "new_task_name"
        self.source.rename(renamed)
        second = discover_task_directory(renamed, self.area / "second")
        self.assertEqual(first["discovery_report"]["budget_fingerprint"], second["discovery_report"]["budget_fingerprint"])
        with (renamed / "N_train.npy").open("wb") as stream:
            np.save(stream, np.arange(24).reshape(12, 2) + 1)
        third = discover_task_directory(renamed, self.area / "third")
        self.assertNotEqual(first["task_spec"]["data_version"], third["task_spec"]["data_version"])
        self.assertNotEqual(first["discovery_report"]["budget_fingerprint"], third["discovery_report"]["budget_fingerprint"])

    def test_training_policy_versions_avoid_modal_cache_collisions(self) -> None:
        self._npy_task()
        first = discover_task_directory(self.source, self.area / "first", max_trials=1)
        second = discover_task_directory(self.source, self.area / "second", max_trials=20)
        self.assertNotEqual(first["task_spec"]["data_version"], second["task_spec"]["data_version"])
        self.assertEqual(first["discovery_report"]["budget_fingerprint"], second["discovery_report"]["budget_fingerprint"])

    def test_staging_cannot_write_inside_original_task(self) -> None:
        self._npy_task()
        destination = self.source / "generated"
        with self.assertRaisesRegex(TaskDirectoryError, "原始任务目录之外"):
            discover_task_directory(self.source, destination)
        self.assertFalse(destination.exists())

    def test_unknown_metric_is_an_explicit_error(self) -> None:
        self._npy_task()
        (self.source / "info.json").write_text('{"metric": "auc"}', encoding="utf-8")
        with self.assertRaisesRegex(TaskDirectoryError, "不支持的指标"):
            discover_task_directory(self.source, self.area / "staged")

    def test_primary_metric_outranks_diagnostic_mentions(self) -> None:
        self._npy_task()
        (self.source / "ReadMe.MD").write_text(
            "The primary metric is macro-F1. Also print accuracy as a diagnostic.", encoding="utf-8")
        result = discover_task_directory(self.source, self.area / "supported")
        self.assertEqual(result["task_spec"]["primary_metric"]["name"], "f1_macro")
        (self.source / "ReadMe.MD").write_text(
            "The primary metric is ROC-AUC. Also print accuracy as a diagnostic.", encoding="utf-8")
        with self.assertRaisesRegex(TaskDirectoryError, "不支持的主指标"):
            discover_task_directory(self.source, self.area / "unsupported")

    def test_explicit_metadata_counts_are_checked(self) -> None:
        self._npy_task()
        fields = {"train_size": 12, "val_size": 4, "num_classes": 2,
                  "n_num_features": 2, "n_cat_features": 0}
        info = self.source / "info.json"
        info.write_text(json.dumps(fields), encoding="utf-8")
        discover_task_directory(self.source, self.area / "valid_metadata")
        for field in fields:
            mismatch = {**fields, field: fields[field] + 1}
            info.write_text(json.dumps(mismatch), encoding="utf-8")
            with self.subTest(field=field), self.assertRaisesRegex(TaskDirectoryError, field):
                discover_task_directory(self.source, self.area / f"invalid_{field}")

    def test_ambiguous_target_is_an_explicit_error(self) -> None:
        for split in ("train", "val", "test"):
            _csv(self.source / f"{split}.csv", ["x", "label", "target"], [[1, 0, 1], [2, 1, 0]])
        with self.assertRaisesRegex(TaskDirectoryError, "唯一目标列"):
            discover_task_directory(self.source, self.area / "staged")

    def test_input_symlink_cannot_escape_task_root(self) -> None:
        _array(self.area / "outside.npy", np.arange(24).reshape(12, 2))
        self._symlink(self.source / "N_train.npy", self.area / "outside.npy")
        for name in ("N_val.npy", "N_test.npy", "y_train.npy", "y_val.npy", "y_test.npy"):
            (self.source / name).write_bytes(b"unread")
        with self.assertRaisesRegex(TaskDirectoryError, "越出任务目录"):
            discover_task_directory(self.source, self.area / "staged")

    def test_test_path_is_checked_only_at_final_materialization(self) -> None:
        self._npy_task()
        result = discover_task_directory(self.source, self.area / "staged")
        outside = self.area / "outside_test.npy"
        _array(outside, np.arange(8).reshape(4, 2))
        # Repoint the deliberately unread placeholder without deleting history.
        original = self.source / "N_test.npy"
        original.rename(self.source / "N_test.original")
        self._symlink(original, outside)
        with self.assertRaisesRegex(TaskDirectoryError, "越出任务目录"):
            materialize_final_test(result["test_descriptor_path"], self.area / "test_input")

    def test_test_source_mutation_is_detected_before_writing_outputs(self) -> None:
        self._npy_task()
        result = discover_task_directory(self.source, self.area / "staged")
        with (self.source / "N_test.npy").open("wb") as stream:
            np.save(stream, np.arange(8).reshape(4, 2))
        with (self.source / "y_test.npy").open("wb") as stream:
            np.save(stream, np.asarray([0, 1, 0, 1]))
        original_read = task_directory.read_npy_array

        def mutating_read(path: Path):
            values = original_read(path)
            if Path(path).name == "N_test.npy":
                with Path(path).open("wb") as stream:
                    np.save(stream, values + 1)
            return values

        with patch.object(task_directory, "read_npy_array", mutating_read):
            with self.assertRaisesRegex(TaskDirectoryError, "发生变化"):
                materialize_final_test(result["test_descriptor_path"], self.area / "test_input")
        self.assertFalse((self.area / "test_input" / "test_config.json").exists())


if __name__ == "__main__":
    unittest.main()
