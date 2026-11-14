"""Single source of truth for the deliberately limited classifier catalogue.

Safe to import in the planner: no I/O, secrets, or estimator imports.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

CAPABILITY_SCHEMA_VERSION = "2.0"
REGISTRY_VERSION = "1.1"
MAX_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class Parameter:
    kind: str
    default: Any
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[Any, ...] = ()
    nullable: bool = False
    exclusive_minimum: bool = False

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.kind, "default": self.default}
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.choices:
            result["choices"] = list(self.choices)
        if self.nullable:
            result["nullable"] = True
        if self.exclusive_minimum:
            result["exclusive_minimum"] = True
        return result


@dataclass(frozen=True)
class Model:
    model_id: str
    builder_id: str
    parameters: dict[str, Parameter]
    fixed_settings: dict[str, Any]
    representations: tuple[str, ...] = ("dense_numeric",)
    profiles: tuple[str, ...] = ("expanded",)
    legacy_candidates: tuple[dict[str, Any], ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()


def _forest_parameters() -> dict[str, Parameter]:
    return {
        "n_estimators": Parameter("integer", 100, 1, 500),
        "max_depth": Parameter("integer", None, 1, 50, nullable=True),
        "min_samples_leaf": Parameter("integer", 1, 1, 100),
        "min_samples_split": Parameter("integer", 2, 2, 100),
        "max_features": Parameter(
            "ratio_or_feature_token", "sqrt", 0.0, 1.0,
            ("sqrt", "log2", None), exclusive_minimum=True,
        ),
        "criterion": Parameter("enum", "gini", choices=("gini", "entropy")),
        "class_weight": Parameter("optional_enum", None, choices=(None, "balanced")),
    }


MODELS: dict[str, Model] = {
    "random_forest": Model(
        "random_forest", "random_forest", _forest_parameters(),
        {"bootstrap": True, "n_jobs": 1},
        representations=("dense_numeric", "image_tensor"),
        profiles=("legacy", "expanded"),
        legacy_candidates=(
            {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1},
            {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1},
            {"n_estimators": 100, "max_depth": 30, "min_samples_leaf": 1},
            {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5},
            {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5},
            {"n_estimators": 100, "max_depth": 30, "min_samples_leaf": 5},
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 5},
            {"n_estimators": 300, "max_depth": 30, "min_samples_leaf": 1},
        ),
        candidates=(
            {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 2},
        ),
    ),
    "logistic_regression": Model(
        "logistic_regression", "logistic_regression",
        {
            "C": Parameter("number", 1.0, 0.0, 100.0, exclusive_minimum=True),
            "max_iter": Parameter("integer", 1000, 1, 2000),
            "class_weight": Parameter("optional_enum", None, choices=(None, "balanced")),
        },
        {"solver": "lbfgs", "scaler": "StandardScaler", "n_jobs": 1},
        representations=("dense_numeric", "raw_text", "image_tensor"),
        profiles=("legacy", "expanded"),
        legacy_candidates=(
            {"C": 1.0, "max_iter": 1000}, {"C": 0.1, "max_iter": 1000},
            {"C": 10.0, "max_iter": 1000}, {"C": 0.3, "max_iter": 1500},
            {"C": 3.0, "max_iter": 1500},
        ),
        candidates=(
            {"C": 1.0, "max_iter": 1000}, {"C": 0.1, "max_iter": 1000},
            {"C": 10.0, "max_iter": 1000}, {"C": 0.3, "max_iter": 1500},
            {"C": 3.0, "max_iter": 1500},
        ),
    ),
    "multinomial_nb": Model(
        "multinomial_nb",
        "multinomial_nb",
        {"alpha": Parameter("number", 1.0, 0.0, 100.0, exclusive_minimum=True)},
        {"fit_prior": True},
        representations=("raw_text",),
        candidates=(
            {"alpha": 0.1},
            {"alpha": 1.0},
            {"alpha": 10.0},
        ),
    ),
    "complement_nb": Model(
        "complement_nb",
        "complement_nb",
        {"alpha": Parameter("number", 1.0, 0.0, 100.0, exclusive_minimum=True)},
        {"fit_prior": True},
        representations=("raw_text",),
        candidates=(
            {"alpha": 0.1},
            {"alpha": 1.0},
            {"alpha": 10.0},
        ),
    ),
    "extra_trees": Model(
        "extra_trees", "extra_trees", _forest_parameters(),
        {"bootstrap": False, "n_jobs": 1},
        representations=("dense_numeric", "image_tensor"),
        candidates=(
            {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2},
            {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 2},
        ),
    ),
    "hist_gradient_boosting": Model(
        "hist_gradient_boosting", "hist_gradient_boosting",
        {
            "learning_rate": Parameter("number", 0.1, 0.001, 1.0),
            "max_iter": Parameter("integer", 100, 1, 500),
            "max_leaf_nodes": Parameter("integer", 31, 2, 127),
            "max_depth": Parameter("integer", None, 1, 50, nullable=True),
            "min_samples_leaf": Parameter("integer", 20, 1, 100),
            "l2_regularization": Parameter("number", 0.0, 0.0, 100.0),
        },
        {"categorical_features": None, "early_stopping": False},
        representations=("dense_numeric", "image_tensor"),
        candidates=(
            {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31},
            {"learning_rate": 0.03, "max_iter": 200, "max_leaf_nodes": 31},
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 15},
            {"learning_rate": 0.2, "max_iter": 400, "max_leaf_nodes": 31},
        ),
    ),
}


def normalize_params(model_id: str, params: Any) -> dict[str, Any]:
    """Validate an untrusted proposal and return defaults in stable types."""
    model = MODELS.get(model_id)
    if model is None:
        raise ValueError("model is not in the supported classifier registry")
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    if set(params) - set(model.parameters):
        raise ValueError("params contains unsupported fields")
    normalized: dict[str, Any] = {}
    for name, spec in model.parameters.items():
        value = params.get(name, spec.default)
        if value is None:
            if spec.nullable or spec.kind == "optional_enum" or None in spec.choices:
                normalized[name] = None
                continue
            raise ValueError(f"parameter {name} cannot be null")
        if spec.kind == "integer":
            valid = type(value) is int
        elif spec.kind == "number":
            valid = type(value) in (int, float) and math.isfinite(value)
            if valid:
                value = float(value)
        elif spec.kind in ("enum", "optional_enum"):
            valid = type(value) is str
        elif spec.kind == "ratio_or_feature_token":
            valid = (
                type(value) is str
                or type(value) in (int, float) and math.isfinite(value)
            )
            if valid and type(value) in (int, float):
                value = float(value)
        else:
            raise RuntimeError(f"unknown internal parameter type: {spec.kind}")
        if not valid:
            raise ValueError(f"parameter {name} has an invalid type")
        if spec.choices and value not in spec.choices:
            raise ValueError(f"parameter {name} is outside the allowed choices")
        if spec.kind in ("integer", "number"):
            lower_ok = value > spec.minimum if spec.exclusive_minimum else value >= spec.minimum
            if not lower_ok or value > spec.maximum:
                raise ValueError(f"parameter {name} is outside the allowed range")
        elif spec.kind == "ratio_or_feature_token" and type(value) is float:
            if not 0.0 < value <= 1.0:
                raise ValueError(f"parameter {name} must be a fraction in (0, 1]")
        normalized[name] = value
    return normalized


def model_capability(model_id: str) -> dict[str, Any]:
    model = MODELS[model_id]
    return {
        "builder_id": model.builder_id,
        "parameters": {name: spec.as_dict() for name, spec in model.parameters.items()},
        "representations": list(model.representations),
        "supports_binary": True,
        "supports_multiclass": True,
        "predict_proba": True,
        "fixed_settings": copy.deepcopy(model.fixed_settings),
        "profiles": list(model.profiles),
        "legacy_candidates": [copy.deepcopy(item) for item in model.legacy_candidates],
        "candidates": [copy.deepcopy(item) for item in model.candidates],
    }


def capabilities() -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "problem_type": "classification",
        "models": {model_id: model_capability(model_id) for model_id in MODELS},
        "pipeline_components": {
            "feature_selector": {
                "none": {},
                "variance_threshold": {
                    "parameters": {
                        "threshold": {
                            "type": "number",
                            "default": 0.0,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        }
                    }
                },
            },
            "text_vectorizer": {
                "tfidf": {
                    "candidate_strategies": [
                        {"field_mode": "shared", "analyzer": "word", "ngram_min": 1, "ngram_max": 1},
                        {"field_mode": "separate", "analyzer": "word", "ngram_min": 1, "ngram_max": 1},
                        {"field_mode": "shared", "analyzer": "char", "ngram_min": 2, "ngram_max": 5},
                        {"field_mode": "separate", "analyzer": "char", "ngram_min": 2, "ngram_max": 5},
                    ],
                    "parameters": {
                        "field_mode": {"type": "enum", "choices": ["shared", "separate"]},
                        "analyzer": {"type": "enum", "choices": ["word", "char"]},
                        "ngram_min": {"type": "integer", "minimum": 1, "maximum": 5},
                        "ngram_max": {"type": "integer", "minimum": 1, "maximum": 5},
                        "max_features": {"type": "integer", "minimum": 100, "maximum": 100000},
                        "min_df": {"type": "integer", "minimum": 1, "maximum": 20},
                    }
                }
            },
            "image_transform": {"flatten_scale": {}},
        },
        "max_timeout_seconds": MAX_TIMEOUT_SECONDS,
        "representations": ["dense_numeric", "raw_text", "image_tensor"],
    }


def default_pipeline(representation: str) -> dict[str, Any]:
    if not isinstance(representation, str):
        raise ValueError("representation is not supported by the pipeline registry")
    pipeline: dict[str, Any] = {
        "feature_selector": {"name": "none", "params": {}},
    }
    if representation == "raw_text":
        pipeline["text_vectorizer"] = {
            "name": "tfidf",
            "params": {
                "field_mode": "shared",
                "analyzer": "word",
                "ngram_min": 1,
                "ngram_max": 1,
                "max_features": 20_000,
                "min_df": 1,
            },
        }
    elif representation == "image_tensor":
        pipeline["image_transform"] = {"name": "flatten_scale", "params": {}}
    elif representation != "dense_numeric":
        raise ValueError("representation is not supported by the pipeline registry")
    return pipeline


def text_pipeline_candidates(n_features: int) -> list[dict[str, Any]]:
    """Return a small, representation-diverse starting portfolio for raw text."""
    if type(n_features) is not int or n_features < 1:
        raise ValueError("n_features must be a positive integer")
    modes = ("shared", "separate") if n_features > 1 else ("shared",)
    candidates = []
    for analyzer, ngram_min, ngram_max in (("word", 1, 1), ("char", 2, 5)):
        for field_mode in modes:
            pipeline = default_pipeline("raw_text")
            params = pipeline["text_vectorizer"]["params"]
            params.update(
                field_mode=field_mode,
                analyzer=analyzer,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
            )
            candidates.append(pipeline)
    return candidates


def normalize_pipeline(representation: str, pipeline: Any) -> dict[str, Any]:
    """Validate the finite Plan 2.0 component vocabulary and fill defaults."""
    if not isinstance(representation, str) or not isinstance(pipeline, dict):
        raise ValueError("pipeline must be an object")
    allowed = {"feature_selector"}
    if representation == "raw_text":
        allowed.add("text_vectorizer")
    elif representation == "image_tensor":
        allowed.add("image_transform")
    elif representation != "dense_numeric":
        raise ValueError("pipeline representation is not supported")
    if set(pipeline) - allowed:
        raise ValueError("pipeline contains an unsupported component")
    normalized = default_pipeline(representation)
    feature = pipeline.get("feature_selector", normalized["feature_selector"])
    if not isinstance(feature, dict) or set(feature) != {"name", "params"}:
        raise ValueError("feature_selector must have name and params")
    if feature["name"] == "none":
        if feature["params"] != {}:
            raise ValueError("none feature_selector takes no parameters")
        normalized["feature_selector"] = {"name": "none", "params": {}}
    elif feature["name"] == "variance_threshold":
        params = feature["params"]
        if not isinstance(params, dict) or set(params) - {"threshold"}:
            raise ValueError("variance_threshold parameters are invalid")
        threshold = params.get("threshold", 0.0)
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("variance threshold must be finite and in [0, 1]")
        normalized["feature_selector"] = {
            "name": "variance_threshold",
            "params": {"threshold": float(threshold)},
        }
    else:
        raise ValueError("feature_selector name is not supported")

    if representation == "raw_text":
        text = pipeline.get("text_vectorizer", normalized["text_vectorizer"])
        if (
            not isinstance(text, dict)
            or set(text) != {"name", "params"}
            or text["name"] != "tfidf"
        ):
            raise ValueError("raw_text requires the registered tfidf vectorizer")
        params = text["params"]
        keys = {
            "field_mode", "analyzer", "ngram_min", "ngram_max", "max_features", "min_df"
        }
        if not isinstance(params, dict) or set(params) - keys:
            raise ValueError("tfidf parameters contain unsupported fields")
        defaults = normalized["text_vectorizer"]["params"]
        values = {name: params.get(name, default) for name, default in defaults.items()}
        if not isinstance(values["field_mode"], str) or values["field_mode"] not in {
            "shared", "separate"
        }:
            raise ValueError("tfidf field_mode must be shared or separate")
        if not isinstance(values["analyzer"], str) or values["analyzer"] not in {"word", "char"}:
            raise ValueError("tfidf analyzer must be word or char")
        for field, low, high in (
            ("ngram_min", 1, 5),
            ("ngram_max", 1, 5),
            ("max_features", 100, 100_000),
            ("min_df", 1, 20),
        ):
            if type(values[field]) is not int or not low <= values[field] <= high:
                raise ValueError(f"tfidf {field} is outside the supported integer range")
        if values["ngram_min"] > values["ngram_max"]:
            raise ValueError("tfidf ngram_min cannot exceed ngram_max")
        normalized["text_vectorizer"] = {"name": "tfidf", "params": values}
    elif representation == "image_tensor":
        image = pipeline.get("image_transform", normalized["image_transform"])
        if (
            not isinstance(image, dict)
            or set(image) != {"name", "params"}
            or image["name"] != "flatten_scale"
            or image["params"] != {}
        ):
            raise ValueError("image_tensor requires the fixed flatten_scale transform")
    return normalized


def models_for_profile(profile: str, representations: list[str] | None = None) -> list[str]:
    allowed = set(representations or ["dense_numeric"])
    return [
        model_id for model_id, model in MODELS.items()
        if profile in model.profiles and allowed.intersection(model.representations)
    ]


def candidates_for_profile(model_id: str, profile: str) -> list[dict[str, Any]]:
    model = MODELS[model_id]
    if profile not in model.profiles:
        return []
    candidates = model.legacy_candidates if profile == "legacy" else model.candidates
    return [normalize_params(model_id, item) for item in candidates]
