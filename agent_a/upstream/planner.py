"""Deterministic fallback planner over the C classifier capability registry."""
from __future__ import annotations

import json
from typing import Any

from executor_c.model_registry import (
    MODELS,
    candidates_for_profile,
    default_pipeline,
    models_for_profile,
    normalize_params,
    normalize_pipeline,
    text_pipeline_candidates,
)

WIDE_FEATURE_THRESHOLD = 500


def params_key(model: str, params: dict[str, Any]) -> tuple[str, str]:
    try:
        normalized = normalize_params(model, params)
    except ValueError:
        normalized = params
    return model, json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_key(
    model: str,
    params: dict[str, Any],
    pipeline: dict[str, Any] | None,
    representation: str,
) -> tuple[str, str]:
    """Identify a complete registered experiment, including its fitted representation."""
    normalized = normalize_params(model, params)
    recipe: dict[str, Any] = {"params": normalized}
    if representation == "raw_text":
        recipe["pipeline"] = normalize_pipeline(representation, pipeline or {})
    return model, json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
        representation = representations[0] if representations else "dense_numeric"
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
        tried = {
            recipe_key(
                item["model"],
                item["params"],
                item.get("pipeline"),
                item.get("representation", representation),
            )
            for item in history
        }
        successful = [
            item for item in history
            if item.get("status") == "success" and item.get("metric_value") is not None
        ]
        best = (
            max(successful, key=lambda item: item["metric_value"] if greater else -item["metric_value"])
            if successful else None
        )

        if representation == "raw_text":
            pipeline_options: list[dict[str, Any] | None] = text_pipeline_candidates(
                int(overview.get("n_features") or 1)
            )
        elif representation == "image_tensor":
            pipeline_options = [default_pipeline(representation)]
        else:
            pipeline_options = [None]

        def build(
            model_id: str,
            params: dict[str, Any],
            pipeline: dict[str, Any] | None,
            reason: str,
        ) -> dict[str, Any]:
            plan = make_plan(
                task_id, data_version, self.new_experiment_id(), model_id, params,
                self.seed + len(history), timeout,
            )
            if representation != "dense_numeric":
                plan.update(
                    schema_version="2.0",
                    representation=representation,
                    pipeline=pipeline or default_pipeline(representation),
                )
            return {
                "plan": plan,
                "reason": reason,
            }

        wide = (overview.get("n_features") or 0) >= WIDE_FEATURE_THRESHOLD
        ordered = list(available)
        if wide and "logistic_regression" in ordered:
            ordered.remove("logistic_regression")
            ordered.insert(0, "logistic_regression")

        # Give each supported family one chance before tuning any family.
        for model_id in ordered:
            for pipeline in pipeline_options:
                for params in spaces[model_id][:1]:
                    if recipe_key(model_id, params, pipeline, representation) not in tried:
                        mode = (
                            " with text strategy "
                            + json.dumps(pipeline["text_vectorizer"]["params"], sort_keys=True)
                            if representation == "raw_text" and pipeline is not None
                            else ""
                        )
                        return build(
                            model_id,
                            params,
                            pipeline,
                            f"run {model_id.replace('_', '-')} baseline{mode} using {profile} profile",
                        )

        if best is not None and best["model"] in spaces:
            model_id = best["model"]
            for pipeline in pipeline_options:
                for params in spaces[model_id]:
                    if recipe_key(model_id, params, pipeline, representation) not in tried:
                        return build(model_id, params, pipeline, f"refine best {model_id} configuration")

        for model_id in ordered:
            for pipeline in pipeline_options:
                for params in spaces[model_id]:
                    if recipe_key(model_id, params, pipeline, representation) not in tried:
                        return build(model_id, params, pipeline, f"explore {model_id} candidate")
        return None
