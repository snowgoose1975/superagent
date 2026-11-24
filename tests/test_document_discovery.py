"""B instruction selection uses names first, and only opens validated documents."""
from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from data_b.llm_analysis import (
    AnalysisError,
    _inventory,
    resolve_layout,
    select_document_paths,
)
from data_b.task_directory import discover_task_directory
from evaluation_d.school_gateway import GatewayError

ROOT = Path(__file__).resolve().parents[1]


class DocumentDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.area = ROOT / ".cache/test-temp" / ("documents-" + uuid.uuid4().hex)
        self.source = self.area / "source"
        self.source.mkdir(parents=True)
        self.audit = self.area / "audit"
        self.audit.mkdir()
        self.doc = "Use records.tsv for training. Target column is label. Primary metric accuracy."
        (self.source / "使用指南.txt").write_text(self.doc, encoding="utf-8")
        (self.source / "dataset_details.json").write_text('{"purpose":"classification"}', encoding="utf-8")
        (self.source / "test.json").write_text("SEALED TEST DATA", encoding="utf-8")
        (self.source / "api_key.txt").write_text("SYNTHETIC_SECRET_MUST_NOT_READ", encoding="utf-8")
        (self.source / "records.tsv").write_text("text\tlabel\n" + "\n".join(
            f"{'good product' if i % 2 else 'bad product'}\t{i % 2}" for i in range(12)), encoding="utf-8")

    def gateway(self, replies):
        owner = self
        self.requests = []
        class Gateway:
            def call_llm(self, messages):
                owner.requests.append(json.loads(messages[1]["content"]))
                reply = replies[len(owner.requests) - 1]
                if isinstance(reply, Exception):
                    raise reply
                return {"text": json.dumps(reply, ensure_ascii=False)}
        return Gateway()

    def layout(self):
        return {"format": "tsv", "feature_prefixes": [], "target_column": "label", "id_columns": [],
                "partitions": {"train": {"features": ["records.tsv"], "labels": None}, "validation": None, "test": None},
                "metric": "accuracy", "reader_options": {}, "feature_columns": ["text"], "feature_types": {"text": "text"},
                "evidence": [{"field": field, "document": "使用指南.txt", "quote": self.doc}
                             for field in ("train", "target_column", "metric")], "unresolved_questions": []}

    def test_name_selection_precedes_document_and_training_reads(self):
        gateway = self.gateway([
            {"document_paths": ["使用指南.txt", "dataset_details.json"], "unresolved_questions": []},
            self.layout(),
        ])
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.json")
            self.assertNotEqual(path.name, "api_key.txt")
            if path.name in {"使用指南.txt", "dataset_details.json"}:
                self.assertGreaterEqual(len(self.requests), 1)
            if path.name == "records.tsv":
                self.assertGreaterEqual(len(self.requests), 2)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            result = resolve_layout(self.source, gateway, self.area / "analysis")
            discovered = discover_task_directory(self.source, self.area / "staged", resolution=result)
        self.assertEqual(discovered["task_spec"]["modality"]["kind"], "text")
        first, second = self.requests[:2]
        self.assertNotIn("documents", first)
        self.assertNotIn(self.doc, json.dumps(first, ensure_ascii=False))
        self.assertNotIn("test.json", first["document_candidates"])
        self.assertNotIn("api_key.txt", first["document_candidates"])
        self.assertEqual(second["documents"]["使用指南.txt"], self.doc)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue((self.area / "analysis/documents.json").is_file())

    def test_explicit_documents_skip_selection(self):
        result = resolve_layout(self.source, self.gateway([self.layout()]),
                                self.area / "explicit", document_paths=["使用指南.txt"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(list(result["documents"]), ["使用指南.txt"])

    def test_test_data_and_binary_candidates_are_rejected_without_reading(self):
        for path in ("test.json", "records.tsv", "../outside.txt"):
            with self.subTest(path=path):
                audit = self.area / uuid.uuid4().hex
                audit.mkdir()
                reply = {"document_paths": [path], "unresolved_questions": []}
                with self.assertRaises(AnalysisError):
                    select_document_paths(self.source, _inventory(self.source), self.gateway([reply, reply]), audit)

    def test_unknown_api_usage_stops_selection_without_retry(self):
        gateway = self.gateway([GatewayError("TIMEOUT", "Synthetic timeout")])
        with self.assertRaises(GatewayError):
            select_document_paths(self.source, _inventory(self.source), gateway, self.audit)
        self.assertEqual(len(self.requests), 1)

    def test_oversized_selected_document_is_not_opened(self):
        (self.source / "使用指南.txt").write_text("x" * 8001, encoding="utf-8")
        gateway = self.gateway([{"document_paths": ["使用指南.txt"], "unresolved_questions": []}])
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotEqual(path.name, "使用指南.txt")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard), self.assertRaisesRegex(AnalysisError, "8,000 bytes"):
            resolve_layout(self.source, gateway, self.area / "oversized")


if __name__ == "__main__":
    unittest.main()
