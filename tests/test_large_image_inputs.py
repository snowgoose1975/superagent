"""Large primitive NPY, column labels and directory-table task descriptions."""
from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data_b.llm_analysis import _path_evidence_matches, resolve_layout
from data_b.modality_readers import load_image_tensor
from data_b.npy_reader import (
    ReaderLimits,
    normalize_classification_labels,
    read_npy_array,
)
from data_b.prepare import prepare_data
from data_b.task_directory import discover_task_directory, materialize_final_test
from executor_c.contracts import load_prepared_data, validate_plan
from executor_c.model_registry import default_pipeline

ROOT = Path(__file__).resolve().parents[1]


class Reply:
    def __init__(self, layout):
        self.layout = layout

    def call_llm(self, _messages):
        return {"text": json.dumps(self.layout)}


class LargeImageInputTests(unittest.TestCase):
    def setUp(self):
        self.area = ROOT / ".cache/test-temp" / ("large-images-" + uuid.uuid4().hex)
        self.area.mkdir(parents=True)

    def test_large_primitive_defaults_and_readonly_mapping(self):
        limits = ReaderLimits()
        self.assertEqual(limits.max_payload_bytes, 1024**3)
        self.assertEqual(limits.max_elements, 256_000_000)
        original = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        path = self.area / "pixels.npy"
        np.save(path, original)
        output = read_npy_array(path, ReaderLimits(mmap_threshold_bytes=1))
        self.assertIsInstance(output, np.memmap)
        self.assertFalse(output.flags.writeable)
        np.testing.assert_array_equal(output, original)
        with self.assertRaises(ValueError):
            output[0, 0, 0] = 100

    def test_fortran_layout_and_exact_payload_validation(self):
        original = np.asfortranarray(np.arange(24).reshape(6, 4))
        path = self.area / "fortran.npy"
        np.save(path, original)
        np.testing.assert_array_equal(read_npy_array(path, ReaderLimits(mmap_threshold_bytes=1)), original)
        truncated = self.area / "truncated.npy"
        truncated.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "payload size"):
            read_npy_array(truncated)
        with self.assertRaisesRegex(ValueError, "element count"):
            read_npy_array(path, ReaderLimits(max_elements=2))
        with self.assertRaisesRegex(ValueError, "payload exceeds"):
            read_npy_array(path, ReaderLimits(max_payload_bytes=2))

    def test_object_limits_remain_small_and_small_arrays_still_load(self):
        path = self.area / "categories.npy"
        original = np.asarray(["001", "002", None], dtype=object)
        np.save(path, original, allow_pickle=True)
        np.testing.assert_array_equal(read_npy_array(path), original)
        self.assertEqual(ReaderLimits().max_object_payload_bytes, 64 * 1024**2)
        self.assertEqual(ReaderLimits().max_object_elements, 8_000_000)
        with self.assertRaisesRegex(ValueError, "Object .npy element"):
            read_npy_array(path, ReaderLimits(max_object_elements=2))
        with self.assertRaisesRegex(ValueError, "payload exceeds"):
            read_npy_array(path, ReaderLimits(max_object_payload_bytes=1))

    def test_column_labels_normalize_without_flattening_multitarget(self):
        original = np.asarray([[2], [0], [1]], dtype=np.uint8)
        normalized = normalize_classification_labels(original)
        np.testing.assert_array_equal(normalized, [2, 0, 1])
        self.assertEqual(normalized.dtype, original.dtype)
        self.assertTrue(np.shares_memory(normalized, original))
        for shape in ((3, 2), (1, 3), (3, 1, 1)):
            with self.assertRaises(ValueError):
                normalize_classification_labels(np.zeros(shape))

    def test_directory_and_filename_must_be_in_same_table_row(self):
        quote = ("| `data/` | `images.npy`, `labels.npy` |\n"
                 "| `validation/` | `other.npy`, `targets.npy` |")
        self.assertTrue(_path_evidence_matches("data/images.npy", quote, "train"))
        self.assertTrue(_path_evidence_matches("validation/targets.npy", quote, "validation"))
        self.assertFalse(_path_evidence_matches("data/targets.npy", quote, "train"))
        self.assertFalse(_path_evidence_matches("validation/images.npy", quote, "validation"))
        self.assertFalse(_path_evidence_matches("elsewhere/data/images.npy", quote, "train"))

    def test_image_dtype_and_pixels_remain_unchanged(self):
        path = self.area / "images.npy"
        original = np.arange(96, dtype=np.uint8).reshape(2, 4, 4, 3)
        np.save(path, original)
        output = load_image_tensor(path, [4, 4, 3], "HWC", [0, 255])
        self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_array_equal(output, original)

    def test_table_document_column_labels_prepare_c_and_lazy_test(self):
        source = self.area / "source"
        for folder in ("data", "validation", "sealed"):
            (source / folder).mkdir(parents=True)
        train = np.arange(12 * 48, dtype=np.int32).astype(np.uint8).reshape(12, 4, 4, 3)
        valid = train[:4].copy()
        np.save(source / "data/images.npy", train)
        np.save(source / "data/labels.npy", np.asarray([0, 1] * 6).reshape(-1, 1))
        np.save(source / "validation/images.npy", valid)
        np.save(source / "validation/labels.npy", np.asarray([0, 1] * 2).reshape(-1, 1))
        (source / "sealed/images.npy").write_bytes(b"UNREAD")
        (source / "sealed/labels.npy").write_bytes(b"UNREAD")
        quote = ("| Directory | Files |\n|---|---|\n"
                 "| `data/` | `images.npy`, `labels.npy` |\n"
                 "| `validation/` | `images.npy`, `labels.npy` |\n"
                 "| `sealed/` | `images.npy`, `labels.npy` |")
        doc = "Image classification. Primary metric accuracy.\n" + quote
        (source / "instructions.md").write_text(doc, encoding="utf-8")
        layout = {"data_type": "image", "metric": "accuracy", "partitions": {
            role: {"features": [folder + "/images.npy"], "labels": folder + "/labels.npy"}
            for role, folder in (("train", "data"), ("validation", "validation"), ("test", "sealed"))},
            "evidence": [{"field": role, "document": "instructions.md", "quote": quote}
                         for role in ("train", "validation", "test")]
                        + [{"field": "metric", "document": "instructions.md", "quote": "Primary metric accuracy."}]}
        original_open = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotIn("sealed", path.parts)
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            resolution = resolve_layout(source, Reply(layout), self.area / "analysis",
                                        document_paths=["instructions.md"])
            discovered = discover_task_directory(source, self.area / "staged", resolution=resolution, max_trials=1)
            package = prepare_data(discovered["task_spec"], self.area / "prepared")
        self.assertEqual(package["status"], "success", package.get("error"))
        self.assertTrue(prepare_data(discovered["task_spec"], self.area / "prepared")["reused"])
        spec = discovered["task_spec"]
        plan = validate_plan({"schema_version": "2.0", "task_id": spec["task_id"], "data_version": spec["data_version"],
                              "experiment_id": "shape_check", "model": "random_forest", "params": {},
                              "random_seed": 42, "timeout_seconds": 60, "representation": "image_tensor",
                              "pipeline": default_pipeline("image_tensor")})
        def guard_evaluation(path, *args, **kwargs):
            self.assertNotIn(path.name, {"validation_labels.csv", "evaluation_manifest.json"})
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guard_evaluation):
            loaded = load_prepared_data(package["data_manifest_path"], plan)
        self.assertEqual(loaded.X_train.dtype, np.uint8)
        np.testing.assert_array_equal(loaded.X_train, train)
        np.testing.assert_array_equal(loaded.y_train, [0, 1] * 6)
        np.save(source / "sealed/images.npy", valid)
        np.save(source / "sealed/labels.npy", np.asarray([0, 1] * 2).reshape(-1, 1))
        test_config = materialize_final_test(discovered["test_descriptor_path"], self.area / "test_input")
        config = json.loads(test_config.read_text(encoding="utf-8"))
        self.assertEqual(config["n_samples"], 4)
        self.assertEqual(read_npy_array(test_config.parent / "y_test.npy").shape, (4,))


if __name__ == "__main__":
    unittest.main()
