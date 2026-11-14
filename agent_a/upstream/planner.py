"""Deterministic fallback planner over the C classifier capability registry."""
from __future__ import annotations

import json
from typing import Any

from executor_c.model_registry import (
    MODELS,
    candidates_for_profile,
    models_for_profile,
    normalize_params,
)

WIDE_FEATURE_THRESHOLD = 500


def params_key(model: str, params: dict[str, Any]) -> tuple[str, str]:
    try:
        normalized = normalize_params(model, params)
    except ValueError:
        normalized = params
    return model, json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)


def make_plan(
    task_id: str,
    data_version: str,
    experiment_id: str,
    model: str,
    params: dict[str, Any],
    seed: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": task_id,
        "data_version": data_version,
        "experiment_id": experiment_id,
        "model": model,
        "params": normalize_params(model, params),
        "random_seed": seed,
        "timeout_seconds": timeout_seconds,
    }


class Planner:
    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self._counter = 0

    def new_experiment_id(self) -> str:
        self._counter += 1
        return f"exp_{self._counter:03d}"

    def next_plan(self, context: dict[str, Any]) -> dict[str, Any] | None:
        task_id = context["task_id"]
        data_version = context["data_version"]
        history = context["history"]
        greater = bool(context.get("greater_is_better", True))
        timeout = int(context.get("timeout_seconds", 300))
        overview = context.get("data_overview") or {}
        profile = context.get("search_profile", "legacy")
        capabilities = context.get("capabilities", {})
        models = capabilities.get("models", {})
        representations = overview.get("representations", ["dense_numeric"])
        available = [
            model_id for model_id in models
            if model_id in MODELS
            and profile in MODELS[model_id].profiles
            and set(representations).intersection(MODELS[model_id].representations)
        ]
        if not available:
            available = models_for_profile(profile, representations)
        if not available:
            return None

        spaces = {
            model_id: candidates_for_profile(model_id, profile)
            for model_id in available
        }
        tried = {params_key(item["model"], item["params"]) for item in history}
        successful = [
            item for item in history
            if item.get("status") == "success" and item.get("metric_value") is not None
        ]
        best = (
            max(successful, key=lambda item: item["metric_value"] if greater else -item["metric_value"])
            if successful else None
        )

        def build(model_id: str, params: dict[str, Any], reason: str) -> dict[str, Any]:
            return {
                "plan": make_plan(
                    task_id, data_version, self.new_experiment_id(), model_id, params,
                    self.seed + len(history), timeout,
                ),
                "reason": reason,
            }

        wide = (overview.get("n_features") or 0) >= WIDE_FEATURE_THRESHOLD
        ordered = list(available)
        if wide and "logistic_regression" in ordered:
            ordered.remove("logistic_regression")
            ordered.insert(0, "logistic_regression")

        # Give each supported family one chance before tuning any family.
        for model_id in ordered:
            for params in spaces[model_id][:1]:
                if params_key(model_id, params) not in tried:
                    return build(
                        model_id, params,
                        f"run {model_id.replace('_', '-')} baseline using {profile} profile",
                    )

        if best is not None and best["model"] in spaces:
            model_id = best["model"]
            for params in spaces[model_id]:
                if params_key(model_id, params) not in tried:
                    return build(model_id, params, f"refine best {model_id} configuration")

        for model_id in ordered:
            for params in spaces[model_id]:
                if params_key(model_id, params) not in tried:
                    return build(model_id, params, f"explore {model_id} candidate")
        return None
