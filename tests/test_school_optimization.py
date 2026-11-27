"""Mock-only tests for the school planner context and budgeted gateway."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_a.controller import SchoolAController
from agent_a.upstream.llm_planner import LLMPlanner, _system_message
from agent_a.upstream.planner import Planner
from data_b.llm_analysis import AnalysisError, _request
from evaluation_d.school_gateway import GatewayError, SchoolGateway
from evaluation_d.task_budget import TaskBudgetLedger
from executor_c.model_registry import (
    PAIRWISE_MODEL_ID,
    capabilities,
    runtime_capabilities,
)

_MESSAGES = [{"role": "user", "content": "A synthetic planning request."}]
_CONFIG = {
    "url": "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
    "model": "test-model",
    "input_cny_per_million": "1.50",
    "output_cny_per_million": "2.00",
    "max_input_tokens": 4096,
    "max_tokens": 1024,
}


class SchoolOptimizationTests(unittest.TestCase):
    def test_gateway_records_bounded_completion_metadata_and_reopens_after_settlement(self):
        sensitive = "Synthetic string that must not be persisted."
        replies = iter([
            {
                "choices": [{"finish_reason": "length", "message": {
                    "content": None, "reasoning_content": sensitive,
                }}],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 1024,
                    "completion_tokens_details": {"reasoning_tokens": 1024},
                },
            },
            {
                "choices": [{"finish_reason": sensitive, "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 6},
            },
        ])

        def transport(_request, _timeout):
            return 200, {}, json.dumps(next(replies)).encode()

        ledger = Path(tempfile.mkdtemp(prefix="school-completion-metadata-"))
        gateway = SchoolGateway(
            _CONFIG, "mock-key", ledger, max_calls=2, budget_rmb="0.10",
            transport=transport,
        )
        with self.assertRaises(GatewayError) as invalid:
            gateway.call_llm(_MESSAGES)
        self.assertEqual(invalid.exception.code, "INVALID_RESPONSE")
        self.assertEqual(gateway.metadata()["last_completion"], {
            "attempt": 1, "has_usable_text": False, "output_limit_reached": True,
            "finish_reason": "length", "reasoning_tokens": 1024,
        })
        self.assertTrue(gateway.recover_empty_reply())
        self.assertEqual(gateway.call_llm(_MESSAGES)["text"], "{}")
        self.assertIsNone(gateway.metadata()["last_completion"]["finish_reason"])
        self.assertEqual(gateway.metadata()["call_attempts"], 2)
        self.assertFalse(gateway.can_call)
        calls = next(ledger.glob("*/calls.jsonl")).read_text(encoding="utf-8")
        self.assertNotIn(sensitive, calls)
        self.assertNotIn("mock-key", calls)
        self.assertEqual(
            [json.loads(row)["status"] for row in calls.splitlines()],
            ["reserved", "failed_known", "reserved", "complete"],
        )

    def test_unknown_usage_cannot_be_reopened(self):
        def timeout(_request, _window):
            raise TimeoutError("synthetic timeout")

        ledger = Path(tempfile.mkdtemp(prefix="school-unknown-usage-"))
        gateway = SchoolGateway(_CONFIG, "mock-key", ledger, max_calls=2, transport=timeout)
        with self.assertRaises(GatewayError) as failure:
            gateway.call_llm(_MESSAGES)
        self.assertEqual(failure.exception.code, "TIMEOUT")
        self.assertFalse(gateway.recover_empty_reply())
        self.assertEqual(gateway.metadata()["usage_status"], "unknown")
        self.assertFalse(gateway.can_call)

    def test_data_analysis_corrects_a_settled_empty_reply_on_shared_ledger(self):
        root = Path(tempfile.mkdtemp(prefix="b-analysis-empty-reply-"))
        ledger = TaskBudgetLedger(
            root / "budgets", "mock-task", budget_rmb="20",
            profile="deepseek-v4.1-flash-offpeak-2026-10-21",
        )
        messages = []

        def transport(request, _window):
            messages.append(json.loads(request.data)["messages"])
            content = None if len(messages) == 1 else '{"layout": "verified"}'
            payload = {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            return 200, {}, json.dumps(payload).encode()

        gateway = SchoolGateway(
            _CONFIG, "mock-key", root / "sessions", max_calls=3,
            budget_rmb="0.10", transport=transport, task_budget_ledger=ledger,
        )
        result = _request(
            gateway, "layout", {"files": ["train.csv"]}, "Check the layout.",
            lambda value: value, root,
        )
        self.assertEqual(result, {"layout": "verified"})
        self.assertIn("no usable text", messages[-1][-1]["content"])
        self.assertEqual(ledger.snapshot()["settled_count"], 2)
        self.assertEqual(gateway.metadata()["call_attempts"], 2)
        self.assertTrue(gateway.can_call)
        self.assertEqual(
            json.loads((root / "layout_invalid_0.json").read_text())["error_code"],
            "INVALID_RESPONSE",
        )

    def test_data_analysis_leaves_unknown_usage_and_repeated_empty_replies_blocked(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                root = Path(tempfile.mkdtemp(prefix="b-analysis-blocked-reply-"))

                def transport(_request, _window, *, conn_lost=unknown):
                    if conn_lost:
                        raise TimeoutError("Synthetic transport timeout")
                    payload = {
                        "choices": [{"message": {"content": None}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    }
                    return 200, {}, json.dumps(payload).encode()

                gateway = SchoolGateway(
                    _CONFIG, "mock-key", root / "sessions", max_calls=3,
                    budget_rmb="0.10", transport=transport,
                )
                with self.assertRaises(GatewayError if unknown else AnalysisError):
                    _request(
                        gateway, "quality", {}, "Check the profile.", lambda value: value, root,
                    )
                self.assertEqual(gateway.metadata()["call_attempts"], 1 if unknown else 2)
                self.assertFalse(gateway.can_call)
                self.assertFalse((root / "quality.json").exists())

    def test_long_history_keeps_best_lower_score_latest_run_and_b_statistics(self):
        history = [
            {
                "model": "random_forest",
                "params": {"n_estimators": i + 1, "min_samples_leaf": 1},
                "status": "success", "metric_value": 0.8 if i != 0 else 0.15,
            }
            for i in range(120)
        ]
        context = {
            "history": history,
            "greater_is_better": False,
            "capabilities": capabilities(),
            "data_overview": {
                "representations": ["dense_numeric"],
                "training_profile": {
                    "quality_summary": {"n_rows": 400, "ignored": "NoRawData"},
                    "column_statistics": [{
                        "name": "count", "kind": "numeric", "missing_fraction": 0.2,
                        "raw_values": ["NoRawData"], "length": {"p95": 20},
                    }],
                },
            },
        }
        planner = LLMPlanner(lambda _: None, Planner())
        system = _system_message(context)
        message = planner._user_message(context, system_message=system)
        byte_size = len(json.dumps(
            [{"role": "system", "content": system}, {"role": "user", "content": message}],
            ensure_ascii=False, separators=(",", ":"),
        ).encode())
        self.assertLessEqual(byte_size + 768, 16_384)
        payload = json.loads(message)
        shown = {record["trial"] for record in payload["history"]}
        self.assertTrue({1, 120}.issubset(shown))
        self.assertEqual(payload["omitted_history_trials"], 120 - len(shown))
        profile = payload["data_overview"]["training_profile"]
        self.assertEqual(profile["quality_summary"], {"n_rows": 400})
        self.assertEqual(profile["column_statistics_sample"][0]["length"], {"p95": 20})
        self.assertNotIn("NoRawData", message)

    def test_school_can_resume_after_one_early_stop(self):
        root = Path(tempfile.mkdtemp(prefix="school-early-stop-"))
        proposed = iter([
            {"action": "stop", "reason": "the run should end"},
            {"model": "logistic_regression", "params": {}},
        ])
        requests = []

        def transport(request, _window):
            requests.append(json.loads(request.data))
            return 200, {}, json.dumps({
                "choices": [{"message": {"content": json.dumps(next(proposed))}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 40},
            }).encode()

        gateway = SchoolGateway(
            {**_CONFIG, "max_input_tokens": 16384}, "mock-key", root,
            max_calls=2, budget_rmb="0.10", transport=transport,
        )
        controller = SchoolAController(
            "synthetic", "original", 2, gateway=gateway, local_after_paid_limit=True,
            max_trials=3,
        )
        context = {
            "training_profile": {"n_train": 10},
            "capabilities": capabilities(),
            "remaining_trials": 3,
            "history": [],
        }
        first = controller.decide(context)
        self.assertEqual(first["action"], "run")
        self.assertEqual(controller.round_fallback_reason, "SCHOOL_PLANNER_STOP")
        self.assertIsNone(controller.local_only_reason)
        next_decision = controller.decide(context)
        self.assertEqual(next_decision["model"], "logistic_regression")
        self.assertEqual(controller.last_decision_source, "llm")
        self.assertIn(
            "Exploration has not met its minimum yet",
            json.loads(requests[-1]["messages"][-1]["content"])["planner_feedback"],
        )

    def test_school_can_choose_the_documented_pairwise_candidate(self):
        input_context = {
            "task_id": "paired-text",
            "data_version": "original",
            "history": [],
            "capabilities": runtime_capabilities(
                "raw_text", min_class_count=10, native_table=False, performance=True,
                pairwise_text=True,
            ),
            "data_overview": {"representations": ["raw_text"], "n_features": 2},
        }
        reply = json.dumps({"model": PAIRWISE_MODEL_ID, "params": {}})
        planner = LLMPlanner(lambda _: {"text": reply}, Planner())
        result = planner.next_plan(input_context)["plan"]
        self.assertEqual(result["model"], PAIRWISE_MODEL_ID)
        vectorizer = result["pipeline"]["text_vectorizer"]
        self.assertEqual(vectorizer["params"]["field_mode"], "separate")
        self.assertEqual(vectorizer["params"]["analyzer"], "word_char")

    def test_follow_up_corrects_duplicate_and_accepts_action_envelope(self):
        collected = []
        replies = iter([
            {"model": "random_forest", "params": {}},
            {
                "action": "run", "model": "random_forest", "params": {"n_estimators": 200},
                "reason": "Untrusted rationale should not leave the response.",
            },
        ])

        def request(messages):
            collected.append(json.loads(messages[1]["content"]))
            return {"text": json.dumps(next(replies))}

        inspector = LLMPlanner(request, Planner())
        context = {
            "task_id": "synthetic",
            "data_version": "original",
            "history": [{
                "model": "random_forest", "params": {}, "status": "success", "metric_value": 0.5,
            }],
            "capabilities": capabilities(),
            "data_overview": {"representations": ["dense_numeric"]},
        }
        inspector.next_plan(context)
        next_plan = inspector.next_plan(context)
        self.assertIn("already tried", collected[-1]["planner_feedback"])
        self.assertEqual(next_plan["plan"]["params"]["n_estimators"], 200)
        self.assertNotIn("Untrusted rationale", next_plan["reason"])


if __name__ == "__main__":
    unittest.main()
