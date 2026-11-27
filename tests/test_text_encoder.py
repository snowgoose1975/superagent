"""Exercise the local Encoder flow with synthetic weights and vectors."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import main
from agent_a.upstream.llm_planner import LLMPlanner
from agent_a.upstream.planner import Planner
from evaluation_d.school_gateway import SchoolGateway
from executor_c import text_encoder
from executor_c.model_registry import (
    capabilities, default_pipeline, encoder_pipeline, normalize_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]


class TextEncoderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="superagent-local-encoder-"))

    def _task(self, *, classes=2, n_fields=2):
        source = self.root / "task"
        source.mkdir()
        headers = ["body_a", "body_b"][:n_fields]
        for split, count in (("train", 42), ("val", 12), ("test", 9)):
            with (source / f"{split}.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([*headers, "target"])
                for index in range(count):
                    label = index % classes
                    marker = "xyz"[label]
                    fields = [
                        f"{marker} some longer text containing the exact same words about events",
                        "another long story about an upcoming event in our city center",
                    ]
                    writer.writerow([*fields[:n_fields], label])
        (source / "README.md").write_text(
            "Classify text using " + " and ".join(f"`{name}`" for name in headers)
            + ". The class column is `target`. Report test accuracy.",
            encoding="utf-8",
        )
        return source

    def _fake_worker(self):
        (self.root / "sitecustomize.py").write_text(
            '''
from executor_c import text_encoder as e
import numpy as np

def prepare_mock(*args, **kwargs):
    return {"weight_cache_hit": True, "asset_bytes": 128, "prepare_seconds": 0.0}

def encode_mock(self, texts):
    vectors = np.zeros((len(texts), e.DIMENSION), dtype=np.float32)
    for index, text in enumerate(texts):
        words = text.split()
        marker = next((word for word in words if word in ("x", "y", "z")), None)
        if marker is not None:
            vectors[index, "xyz".index(marker)] = 1
    return vectors

e.prepare_weights = prepare_mock
e.LocalTextEncoder._encode = encode_mock
''',
            encoding="utf-8",
        )
        return os.pathsep.join([str(ROOT), str(self.root)])

    def test_offline_auto_task_freezes_encoder_and_predicts_in_new_process(self):
        source = self._task(classes=3)
        output = self.root / "output"
        weights = {"weight_cache_hit": False, "asset_bytes": 128, "prepare_seconds": 0.0}
        original_freeze = main.freeze_best
        frozen = False
        original_open = Path.open

        def open_checked(path, *args, **kwargs):
            if path.resolve() == source / "test.csv":
                self.assertTrue(frozen, "test was opened before freeze")
            return original_open(path, *args, **kwargs)

        def freeze(*args, **kwargs):
            nonlocal frozen
            path = original_freeze(*args, **kwargs)
            frozen = path.is_file()
            return path

        with (
            mock.patch.dict(os.environ, {"PYTHONPATH": self._fake_worker()}),
            mock.patch.object(main, "encoder_status", return_value="downloadable"),
            mock.patch.object(main, "prepare_weights", return_value=weights) as download,
            mock.patch.object(main, "freeze_best", side_effect=freeze),
            mock.patch.object(main, "evaluate_frozen_bundle",
                              wraps=main.evaluate_frozen_bundle) as evaluate,
            mock.patch.object(Path, "open", open_checked),
        ):
            baseline = self.root / "baseline-only"
            self.assertEqual(main.main([
                "run", "--task", str(source), "--output-root", str(baseline), "--max-trials", "1",
            ]), 0)
            download.assert_not_called()
            self.assertEqual(main.main([
                "run", "--task", str(source), "--output-root", str(output), "--max-trials", "2", "--performance",
            ]), 0)
            summary = main.read_json(next((output / "sessions").glob("session_*/summary.json")))
            frozen = False
            warm = main.read_json(main.run_project(
                task_directory=source, output_root=output, max_trials=2,
                pretrained="download", performance=True,
            ))
        self.assertEqual(warm["status"], "success", warm)
        warm_history = [json.loads(line) for line in (
            output / warm["history_path"]
        ).read_text(encoding="utf-8").splitlines()]
        self.assertNotEqual(warm["session_id"], summary["session_id"])
        self.assertTrue(all(record["cache_hit"] for record in warm_history[1]["encoder"]["partitions"]))
        self.assertEqual(summary["status"], "success", summary)
        self.assertEqual(summary["pretrained"]["mode"], "download")
        self.assertEqual(summary["real_llm_calls"], 0)
        self.assertEqual(download.call_count, 2)
        self.assertTrue(summary["pretrained"]["no_time_limit"])
        self.assertIsNone(evaluate.call_args.kwargs["encoder_timeout"])
        self.assertEqual(summary["test_evaluation"]["value"], 1.0)
        bundle = main.read_json(output / summary["bundle_path"])
        self.assertIn("encoder.json", bundle["hashes"])
        history = [json.loads(line) for line in (
            output / summary["history_path"]
        ).read_text(encoding="utf-8").splitlines()]
        self.assertIn("text_vectorizer", history[0]["plan"]["pipeline"])
        self.assertEqual(
            history[1]["plan"]["pipeline"]["text_encoder"]["params"]["lexical"], "char_tfidf",
        )
        experiment = (output / "runs" / history[1]["result_relpath"]).parent
        runtime = main.read_json(experiment / "encoder_runtime.json")
        self.assertTrue(runtime["no_time_limit"])
        stats = history[1]["encoder"]
        self.assertEqual([record["partition"] for record in stats["partitions"]],
                         ["train", "validation"])
        self.assertFalse(any(record["cache_hit"] for record in stats["partitions"]))

    def test_download_failure_keeps_classical_solution(self):
        source = self._task(n_fields=1)
        output = self.root / "fallback"
        with (
            mock.patch.object(main, "encoder_status", return_value="downloadable"),
            mock.patch.object(main, "prepare_weights", side_effect=text_encoder.EncoderError(
                "ENCODER_DOWNLOAD_TIMEOUT", "synthetic download timeout",
            )) as download,
        ):
            summary = main.read_json(main.run_project(
                task_directory=source, output_root=output, pretrained="download", max_trials=3,
            ))
        self.assertEqual(summary["status"], "success", summary)
        download.assert_called_once()
        self.assertEqual(summary["pretrained"]["status"], "trial_failed")
        history = [json.loads(line) for line in (
            output / summary["history_path"]
        ).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(history[1]["error"]["code"], "ENCODER_DOWNLOAD_TIMEOUT")
        self.assertIn("text_vectorizer", history[2]["plan"]["pipeline"])

    def test_required_completion_reports_encoder_failure_even_when_test_succeeds(self):
        source = self._task(n_fields=1)
        output = self.root / "strict"
        with (
            mock.patch.object(main, "encoder_status", return_value="downloadable"),
            mock.patch.object(main, "prepare_weights", side_effect=text_encoder.EncoderError(
                "ENCODER_DOWNLOAD_FAILED", "synthetic download failure",
            )),
        ):
            code = main.main([
                "run", "--task", str(source), "--output-root", str(output),
                "--max-trials", "2", "--require-complete",
            ])
        summary = main.read_json(next((output / "sessions").glob("session_*/summary.json")))
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "degraded")
        self.assertEqual(summary["test_evaluation"]["status"], "success")
        self.assertEqual(summary["trial_failures"][0]["error"]["code"], "ENCODER_DOWNLOAD_FAILED")

    def test_empty_school_response_switches_to_local_encoder_and_keeps_usage(self):
        source = self._task()
        output = self.root / "fallback"
        weights = {"weight_cache_hit": True, "asset_bytes": 128, "prepare_seconds": 0.0}

        def response(content, completion_tokens):
            return 200, {}, json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": completion_tokens},
            }).encode("utf-8")

        transport = mock.Mock(side_effect=[
            response(json.dumps({"model": "logistic_regression", "params": {}}), 80),
            response(None, 1024),
        ])
        gateway = SchoolGateway(
            ROOT / "configs" / "school-api.json", "synthetic-key",
            output / "sessions", max_calls=5, budget_rmb="20", transport=transport,
        )
        with (
            mock.patch.dict(os.environ, {"PYTHONPATH": self._fake_worker()}),
            mock.patch.object(main, "encoder_status", return_value="cached"),
            mock.patch.object(main, "prepare_weights", return_value=weights),
        ):
            summary = main.read_json(main.run_project(
                task_directory=source, output_root=output, pretrained="cache-only", max_trials=3,
                planner_mode="school", gateway=gateway, encoder_no_time_limit=True,
            ))
        self.assertEqual(summary["status"], "success", summary)
        self.assertEqual(summary["planner_fallback"]["error_code"], "INVALID_RESPONSE")
        self.assertEqual(summary["llm_gateway"]["usage_status"], "known")
        self.assertEqual(summary["llm_gateway"]["output_tokens"], 1104)
        self.assertEqual(transport.call_count, 2)
        self.assertFalse(gateway.can_call)
        history = [json.loads(line) for line in (
            output / summary["history_path"]
        ).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(history), 3)
        self.assertTrue(all(
            item["decision_source"] == "rules_after_school_response" for item in history[1:]
        ))
        self.assertIn("text_encoder", history[1]["plan"]["pipeline"])
        self.assertEqual(summary["test_evaluation"]["status"], "success")

    def test_empty_school_response_falls_back_once_and_resumes_school(self):
        source = self._task()
        output = self.root / "fallback"
        weights = {"weight_cache_hit": True, "asset_bytes": 128, "prepare_seconds": 0.0}

        def response(content, completion_tokens):
            return 200, {}, json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": completion_tokens},
            }).encode("utf-8")

        transport = mock.Mock(side_effect=[
            response(json.dumps({"model": "logistic_regression", "params": {}}), 80),
            response(None, 1024),
            response(json.dumps({
                "model": "logistic_regression", "params": {},
                "pipeline": encoder_pipeline(2),
            }), 100),
        ])
        gateway = SchoolGateway(
            ROOT / "configs" / "school-api.json", "synthetic-key",
            output / "sessions", max_calls=5, budget_rmb="20", transport=transport,
        )
        with (
            mock.patch.dict(os.environ, {"PYTHONPATH": self._fake_worker()}),
            mock.patch.object(main, "encoder_status", return_value="cached"),
            mock.patch.object(main, "prepare_weights", return_value=weights),
        ):
            summary = main.read_json(main.run_project(
                task_directory=source, output_root=output, pretrained="cache-only", max_trials=3,
                planner_mode="school", gateway=gateway, encoder_no_time_limit=True, performance=True,
            ))
        self.assertEqual(summary["status"], "success", summary)
        self.assertEqual(summary["planner_round_fallbacks"][0]["error_code"], "INVALID_RESPONSE")
        self.assertEqual(summary["llm_gateway"]["usage_status"], "known")
        self.assertEqual(summary["llm_gateway"]["output_tokens"], 1204)
        self.assertEqual(transport.call_count, 3)
        self.assertTrue(gateway.can_call)
        history = [json.loads(line) for line in (
            output / summary["history_path"]
        ).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(history), 3)
        self.assertEqual(history[1]["decision_source"], "rules_after_school_response")
        self.assertEqual(history[2]["decision_source"], "llm")
        self.assertIn("text_encoder", history[1]["plan"]["pipeline"])
        self.assertEqual(summary["test_evaluation"]["status"], "success")

    def test_known_layout_survives_two_settled_empty_b_replies(self):
        from evaluation_d.task_budget import TaskBudgetLedger

        source = self._task()
        output = self.root / "layout-fallback"
        config = main.read_json(ROOT / "configs" / "school-api.json")
        ledger = TaskBudgetLedger(
            self.root / "ledgers", "mock-layout", budget_rmb="20", profile=config["pricing_profile"],
        )
        contents = [None, None, json.dumps({"model": "logistic_regression", "params": {}})]
        transport = mock.Mock(side_effect=[
            (200, {}, json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100},
            }).encode()) for content in contents
        ])
        gateway = SchoolGateway(config, "synthetic-key", output / "sessions",
                                max_calls=3, transport=transport, task_budget_ledger=ledger)
        summary = main.read_json(main.run_project(
            task_directory=source, output_root=output, planner_mode="school",
            gateway=gateway, b_analysis="llm", max_trials=1,
        ))
        self.assertEqual(summary["status"], "success", summary)
        self.assertEqual(summary["b_fallbacks"][0]["phase"], "layout")
        self.assertEqual(summary["decision_sources"], ["llm"])
        self.assertEqual(summary["llm_gateway"]["call_attempts"], 3)
        self.assertEqual(summary["test_evaluation"]["status"], "success")

    def test_school_timeout_still_stops_before_local_fallback(self):
        source = self._task()
        output = self.root / "unknown-usage"
        transport = mock.Mock(side_effect=TimeoutError)
        gateway = SchoolGateway(
            ROOT / "configs" / "school-api.json", "synthetic-key",
            output / "sessions", max_calls=5, budget_rmb="20", transport=transport,
        )
        summary = main.read_json(main.run_project(
            task_directory=source, output_root=output, max_trials=3,
            planner_mode="school", gateway=gateway,
        ))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["error"]["code"], "TIMEOUT")
        self.assertEqual(summary["llm_gateway"]["usage_status"], "unknown")
        self.assertEqual(summary["attempted_trials"], 0)
        self.assertNotIn("planner_fallback", summary)
        self.assertEqual(transport.call_count, 1)

    def test_feature_cache_reuses_only_matching_partition_and_field_order(self):
        matrix = np.array([["left", "right"]], dtype=object)
        encoder = text_encoder.LocalTextEncoder()
        with mock.patch.object(encoder, "_encode", side_effect=lambda rows: np.full(
            (len(rows), text_encoder.DIMENSION), 1, dtype=np.float32,
        )) as encode:
            kwargs = {"cache_dir": self.root, "source_hash": "synthetic"}
            vector, stats = encoder.transform(
                matrix, ["a", "b"], "separate", partition="train", **kwargs,
            )
            self.assertFalse(stats["cache_hit"])
            repeated, stats = encoder.transform(
                matrix, ["a", "b"], "separate", partition="train", **kwargs,
            )
            self.assertTrue(stats["cache_hit"])
            np.testing.assert_array_equal(vector, repeated)
            self.assertEqual(encode.call_count, 2)
            for columns, partition in ((["b", "a"], "train"), (["a", "b"], "validation")):
                _, stats = encoder.transform(matrix, columns, "separate",
                                             partition=partition, **kwargs)
                self.assertFalse(stats["cache_hit"])
            self.assertEqual(encode.call_count, 6)
            for _ in range(2):
                _, stats = encoder.transform(matrix, ["a", "b"], "separate", partition="prediction", **kwargs)
                self.assertFalse(stats["cache_hit"])
            self.assertEqual(encode.call_count, 10)

    def test_snapshot_download_uses_fixed_anonymous_file_list_and_checksum(self):
        source = self.root / "remote"
        source.mkdir()
        for filename in text_encoder.FILES:
            (source / filename).write_text(filename, encoding="utf-8")
        weights = source / "model.safetensors"
        with mock.patch.object(text_encoder, "WEIGHT_SHA256", text_encoder._sha(weights)):
            fetch = mock.Mock(return_value=str(source))
            target = self.root / text_encoder.REVISION
            text_encoder._download_snapshot(self.root, target, fetch)
            text_encoder._verify(target)
            call = fetch.call_args.kwargs
            self.assertIs(call["token"], False)
            self.assertEqual(call["revision"], text_encoder.REVISION)
            self.assertEqual(call["allow_patterns"], list(text_encoder.FILES))
            weights.write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(text_encoder.EncoderError, "checksum"):
                text_encoder._verify(target)

    def test_download_retries_a_transport_error_without_replacing_the_cache(self):
        import sys

        downloader = mock.Mock(side_effect=[OSError("interrupted"), None])
        hub = mock.Mock(snapshot_download=mock.sentinel.fetch)
        lock = mock.Mock(FileLock=mock.Mock(return_value=mock.MagicMock()))
        request_errors = mock.Mock(RequestException=type("RequestException", (Exception,), {}))
        cache = self.root / "resume-cache"
        with (
            mock.patch.dict(sys.modules, {
                "huggingface_hub": hub, "filelock": lock, "requests.exceptions": request_errors,
            }),
            mock.patch.object(text_encoder, "_download_snapshot", downloader),
            mock.patch.object(text_encoder.time, "sleep") as pause,
        ):
            text_encoder._download(cache)
        args = (cache, cache / text_encoder.REVISION, mock.sentinel.fetch)
        self.assertEqual(downloader.call_args_list, [mock.call(*args), mock.call(*args)])
        pause.assert_called_once_with(2)

    def test_encoder_cannot_mix_with_vectorizer_or_accept_remote_paths(self):
        pipeline = encoder_pipeline(2)
        self.assertEqual(pipeline["text_encoder"]["params"]["field_mode"], "separate")
        with self.assertRaises(ValueError):
            normalize_pipeline("raw_text", {**default_pipeline("raw_text"), **pipeline})
        pipeline["text_encoder"]["params"]["repo"] = "another/model"
        with self.assertRaises(ValueError):
            normalize_pipeline("raw_text", pipeline)

    def test_school_proposal_uses_encoder_only_when_advertised(self):
        catalogue = capabilities()
        proposal = {
            "model": "logistic_regression", "params": {}, "pipeline": encoder_pipeline(2),
        }
        request = mock.Mock(return_value=json.dumps(proposal))
        planner = LLMPlanner(request, Planner())
        context = {
            "task_id": "synthetic", "data_version": "v1", "history": [],
            "search_profile": "expanded", "capabilities": catalogue,
            "data_overview": {"n_features": 2, "representations": ["raw_text"]},
        }
        unavailable = planner.next_plan(context)
        self.assertIn("text_vectorizer", unavailable["plan"]["pipeline"])
        offered = catalogue["pipeline_components"]["text_encoder"][text_encoder.ENCODER_ID]
        offered["availability"] = "cached"
        available = planner.next_plan(context)
        self.assertIn("text_encoder", available["plan"]["pipeline"])


if __name__ == "__main__":
    unittest.main()
