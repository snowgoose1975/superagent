"""A controllers and the selective adapter for the upstream A-agent package."""

from __future__ import annotations

import copy
import math
from typing import Any

from evaluation_d.school_gateway import GatewayError
from executor_c.contracts import validate_plan
from executor_c.diagnostics import safe_history_view
from executor_c.model_registry import (
    PAIRWISE_MODEL_ID,
    models_for_profile,
    normalize_pipeline,
    text_pipeline_candidates,
)

from .legacy_controller import OfflineController as OfflineController
from .upstream.decide import decide as upstream_decide
from .upstream.llm_planner import LLMPlanner
from .upstream.planner import Planner


class _DecisionEvents:
    """Keep only event names; upstream events may contain raw model replies."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def event(self, name: str, **_details: Any) -> None:
        self.names.append(name)


class PackageAController:
    """Adapt the classmate's deterministic planner to the v2 A interface."""

    mode = "offline_rules"
    strategy_source = "Classmate A: a-agent-v1/rule_based_hill_climb_v1"

    def __init__(
        self,
        task_id: str,
        data_version: str,
        n_features: int,
        *,
        max_trials: int = 3,
        patience: int = 4,
        seed: int = 42,
        timeout: int = 120,
        search_profile: str = "legacy",
        search_config: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.data_version = data_version
        self.n_features = n_features
        self.max_trials = max_trials
        self.patience = patience
        self.seed = seed
        self.timeout = timeout
        if search_profile not in {"legacy", "expanded"}:
            raise ValueError("search_profile must be legacy or expanded")
        self.search_profile = search_profile
        self.search_config = copy.deepcopy(search_config or {})
        self.planner = Planner(seed=seed)
        self.last_decision_source = "none"
        self.decision_sources: list[str] = []

    def _canonical_params(self, model: str, params: dict[str, Any]) -> dict[str, Any]:
        plan = validate_plan(
            {
                "schema_version": "1.0",
                "task_id": self.task_id,
                "data_version": self.data_version,
                "experiment_id": "preflight",
                "model": model,
                "params": copy.deepcopy(params),
                "random_seed": self.seed,
                "timeout_seconds": self.timeout,
            }
        )
        return plan["params"]

    def _map_history(self, history: Any) -> list[dict[str, Any]]:
        if not isinstance(history, list):
            raise ValueError("history must be a list")
        mapped: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict) or not isinstance(item.get("plan"), dict):
                raise ValueError("history entry is missing its plan")
            plan = item["plan"]
            if plan.get("task_id") != self.task_id or plan.get("data_version") != self.data_version:
                raise ValueError("history belongs to a different task or data version")
            status = item.get("status")
            evaluation = item.get("evaluation")
            succeeded = (
                status == "success"
                and isinstance(evaluation, dict)
                and evaluation.get("status") == "success"
            )
            value = evaluation.get("value") if succeeded else None
            if succeeded and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError("history evaluation is not a finite metric")
            mapped.append(
                {
                    "experiment_id": item.get("experiment_id"),
                    "model": plan["model"],
                    "params": self._canonical_params(plan["model"], plan["params"]),
                    "status": "success" if succeeded else status,
                    "metric_value": value,
                    "duration_seconds": item.get("duration_seconds"),
                    "baseline_delta": item.get("baseline_delta"),
                    "result_relpath": item.get("result_relpath"),
                    "error": item.get("error"),
                    "diagnostics": safe_history_view(item.get("diagnostics")),
                    **(
                        {
                            "representation": plan["representation"],
                            "pipeline": copy.deepcopy(plan["pipeline"]),
                        }
                        if plan.get("schema_version") == "2.0"
                        else {}
                    ),
                }
            )
        return mapped

    def _upstream_context(
        self, context: dict[str, Any], *, remaining_budget: float | None
    ) -> dict[str, Any]:
        remaining_trials = context.get("remaining_trials")
        if type(remaining_trials) is not int or remaining_trials < 0:
            raise ValueError("remaining_trials must be a nonnegative integer")
        profile = context.get("training_profile")
        if not isinstance(profile, dict):
            raise ValueError("training_profile must be an object")
        data_overview = {
            "n_train": profile.get("n_train"),
            "n_features": self.n_features,
            "training_profile": copy.deepcopy(profile),
            "representations": context.get("data_overview", {}).get(
                "representations", ["dense_numeric"]
            ),
        }
        search = copy.deepcopy(context.get("search", {}))
        search.update(self.search_config)
        search["phase"] = "legacy"
        search["pending_model_ids"] = []
        if self.search_profile == "expanded":
            models = context.get("capabilities", {}).get("models")
            available = (
                models_for_profile("expanded", data_overview["representations"])
                if models is None else [
                    name for name, spec in models.items() if "expanded" in spec.get("profiles", [])
                ]
            )
            tried_models = {
                item["model"] for item in self._map_history(context.get("history", []))
            }
            pending = [model for model in available if model not in tried_models]
            remaining = context.get("remaining_trials", 0)
            pending_text_recipes: list[dict[str, Any]] = []
            if data_overview["representations"] == ["raw_text"]:
                tried_recipes = set()
                for item in self._map_history(context.get("history", [])):
                    pipeline = normalize_pipeline("raw_text", item.get("pipeline", {}))
                    if pipeline.get("text_vectorizer", {}).get("name") != "tfidf":
                        continue
                    params = pipeline["text_vectorizer"]["params"]
                    strategy = tuple(
                        params[key]
                        for key in ("field_mode", "analyzer", "ngram_min", "ngram_max")
                    )
                    tried_recipes.add((item["model"], *strategy))
                pending_text_recipes = [
                    {
                        "model": model,
                        **{
                            key: pipeline["text_vectorizer"]["params"][key]
                            for key in ("field_mode", "analyzer", "ngram_min", "ngram_max")
                        },
                    }
                    for model in available
                    for pipeline in text_pipeline_candidates(self.n_features)
                    if (
                        model != PAIRWISE_MODEL_ID
                        or pipeline["text_vectorizer"]["params"]["field_mode"] == "separate"
                    )
                    and (
                        model,
                        *(
                            pipeline["text_vectorizer"]["params"][key]
                            for key in ("field_mode", "analyzer", "ngram_min", "ngram_max")
                        ),
                    ) not in tried_recipes
                ]
            coverage_size = len(pending_text_recipes) or len(pending)
            coverage_possible = (
                bool(search.get("require_family_coverage", True))
                and type(remaining) is int
                and remaining >= coverage_size
            )
            search["phase"] = "coverage" if coverage_possible and pending else (
                "refinement" if len(context.get("history", [])) >= search.get(
                    "min_exploration_trials", 8
                ) else "exploration"
            )
            if coverage_possible and pending_text_recipes:
                search["phase"] = "coverage"
            search["pending_model_ids"] = pending if coverage_possible else []
            search["pending_text_recipes"] = (
                pending_text_recipes if coverage_possible else []
            )
            search["available_model_ids"] = available
        return {
            "task_id": self.task_id,
            "data_version": self.data_version,
            "history": self._map_history(context.get("history", [])),
            "greater_is_better": context.get("primary_metric", {}).get(
                "greater_is_better", True
            ),
            "primary_metric": copy.deepcopy(context.get("primary_metric", {})),
            "validation_baseline": copy.deepcopy(context.get("validation_baseline")),
            "remaining_budget_rmb": remaining_budget,
            "planning_mode": self.mode,
            "max_experiments": self.max_trials,
            "patience": self.patience,
            "timeout_seconds": self.timeout,
            "planner": self.planner,
            "data_overview": data_overview,
            "capabilities": copy.deepcopy(context.get("capabilities", {})),
            "remaining_trials": remaining_trials,
            "search_profile": self.search_profile,
            "search": search,
            **(
                {"task_context": copy.deepcopy(context["task_context"])}
                if isinstance(context.get("task_context"), dict)
                else {}
            ),
        }

    def _finish(self, decision: dict[str, Any], source: str) -> dict[str, Any]:
        self.last_decision_source = source
        self.decision_sources.append(source)
        if decision.get("action") == "stop":
            return {
                "action": "stop",
                "reason": str(decision.get("reason", "planner stopped"))[:2000],
            }
        if decision.get("action") != "run" or not isinstance(decision.get("plan"), dict):
            raise ValueError("upstream A returned an invalid decision")
        proposal = decision["plan"]
        return {
            "action": "run",
            "model": proposal["model"],
            "params": self._canonical_params(proposal["model"], proposal["params"]),
            "reason": str(decision.get("reason", "A planner decision"))[:2000],
            **(
                {
                    "pipeline": normalize_pipeline(
                        proposal["representation"], proposal["pipeline"]
                    )
                }
                if proposal.get("schema_version") == "2.0"
                else {}
            ),
        }

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        remaining_trials = context.get("remaining_trials")
        if type(remaining_trials) is not int or remaining_trials <= 0:
            return self._finish(
                {"action": "stop", "reason": "main_trial_limit_reached"}, "trial_limit"
            )
        upstream_context = self._upstream_context(context, remaining_budget=None)
        decision = upstream_decide(upstream_context)
        return self._finish(decision, "rules")


class SchoolAController(PackageAController):
    """Use the school gateway for one LLM proposal, with safe rule fallback."""

    mode = "school_llm"
    strategy_source = "Classmate A: a-agent-v1/LLMPlanner + rule_based_hill_climb_v1 fallback"

    def __init__(
        self, task_id: str, data_version: str, n_features: int, *, gateway: Any,
        local_after_paid_limit: bool = False, **kwargs: Any
    ):
        super().__init__(task_id, data_version, n_features, **kwargs)
        self.gateway = gateway
        self.local_after_paid_limit = local_after_paid_limit
        self.gateway_metadata = gateway.metadata()
        self.local_only_reason: str | None = None
        self.consecutive_empty_replies = 0
        self.consecutive_early_stops = 0
        self.round_fallback_reason: str | None = None
        self._events = _DecisionEvents()
        self.llm_planner = LLMPlanner(
            self._call_gateway,
            self.planner,
            log=self._events,
            max_input_bytes=self.gateway_metadata.get("max_input_tokens", 16_384),
        )

    def _call_gateway(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        reply = self.gateway.call_llm(messages)
        if not isinstance(reply, dict):
            raise RuntimeError("school gateway returned an invalid response envelope")
        required = ("text", "input_tokens", "output_tokens", "cost_rmb")
        if any(key not in reply for key in required):
            raise RuntimeError("school gateway response is missing usage")
        if (
            type(reply["input_tokens"]) is not int
            or reply["input_tokens"] < 0
            or type(reply["output_tokens"]) is not int
            or reply["output_tokens"] < 0
            or type(reply["cost_rmb"]) not in (int, float)
            or not math.isfinite(reply["cost_rmb"])
            or reply["cost_rmb"] < 0
            or not isinstance(reply["text"], str)
        ):
            raise RuntimeError("school gateway response has invalid usage")
        return reply

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        self.last_decision_source = "none"
        self._events.names.clear()
        self.round_fallback_reason = None
        remaining_trials = context.get("remaining_trials")
        if type(remaining_trials) is not int or remaining_trials <= 0:
            return self._finish(
                {"action": "stop", "reason": "main_trial_limit_reached"}, "trial_limit"
            )
        if self.local_only_reason is not None:
            return self._decide_locally(context)
        if not self.gateway.can_call:
            if self.local_after_paid_limit:
                reason = getattr(self.gateway, "paid_limit_reason", None)
                if reason is not None:
                    self.local_only_reason = reason
                    return self._decide_locally(context)
            return self._finish(
                {"action": "stop", "reason": "school_gateway_call_unavailable"}, "gateway_stop"
            )
        search = context.get("search") or {}
        remaining = search.get("time_remaining_seconds")
        reserve = search.get("finalize_reserve_seconds")
        if type(remaining) in (int, float) and type(reserve) is int:
            request_window = float(remaining) - reserve
            try:
                self.gateway.set_request_timeout_cap(request_window)
            except Exception:
                return self._finish(
                    {"action": "stop", "reason": "session_time_finalize_reserve"},
                    "time_limit",
                )
        else:
            self.gateway.set_request_timeout_cap(None)
        remaining_budget = float(self.gateway.remaining_budget_rmb)
        if self.local_after_paid_limit and remaining_budget <= 0:
            self.local_only_reason = "BUDGET_EXCEEDED"
            return self._decide_locally(context)
        upstream_context = self._upstream_context(context, remaining_budget=remaining_budget)
        try:
            decision = upstream_decide({**upstream_context, "planner": self.llm_planner})
        except GatewayError as exc:
            if self.local_after_paid_limit and exc.code in {"TASK_BALANCE_EXHAUSTED", "BUDGET_EXCEEDED", "CALL_LIMIT"}:
                self.local_only_reason = exc.code
                return self._decide_locally(context)
            if exc.code != "INVALID_RESPONSE":
                raise
            metadata = self.gateway.metadata()
            if metadata.get("usage_status") != "known" or metadata.get("blocked_reason") != exc.code:
                raise
            self.consecutive_empty_replies += 1
            self.consecutive_early_stops = 0
            self.llm_planner.note_empty_reply()
            if (
                self.local_after_paid_limit and self.consecutive_empty_replies == 1
                and self.gateway.recover_empty_reply()
            ):
                self.round_fallback_reason = exc.code
                return self._decide_locally(context)
            self.local_only_reason = exc.code
            return self._decide_locally(context)
        self.consecutive_empty_replies = 0
        school_stop = "llm_planner_stop" in self._events.names
        upstream_search = upstream_context["search"]
        minimum = upstream_search.get("min_exploration_trials", 8)
        if (
            self.local_after_paid_limit and school_stop and (
                upstream_search["phase"] == "coverage" or len(context.get("history", [])) < minimum
            )
        ):
            self.consecutive_early_stops += 1
            if self.consecutive_early_stops == 1:
                self.round_fallback_reason = "SCHOOL_PLANNER_STOP"
                self.llm_planner.note_early_stop()
            else:
                self.local_only_reason = "SCHOOL_PLANNER_STOP"
            return self._decide_locally(context)
        self.consecutive_early_stops = 0
        if "llm_planner_decision" in self._events.names:
            source = "llm"
        elif "llm_planner_fallback" in self._events.names:
            source = "rules_fallback"
        elif decision.get("action") == "stop":
            source = "planner_stop"
        else:
            source = "rules"
        return self._finish(decision, source)

    def _decide_locally(self, context: dict[str, Any]) -> dict[str, Any]:
        rule_context = self._upstream_context(context, remaining_budget=None)
        rule_context["planning_mode"] = "offline_rules"
        return self._finish(upstream_decide(rule_context), "rules_after_school_response")
