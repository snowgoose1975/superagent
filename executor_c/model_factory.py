"""Construct only the fixed classifier implementations declared by the registry."""

from __future__ import annotations

from typing import Any

from .model_registry import MODELS, normalize_params, normalize_pipeline


def _classifier(model_id: str, params: dict[str, Any], seed: int) -> Any:
    fixed = MODELS[model_id].fixed_settings
    if model_id == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            **params,
            bootstrap=fixed["bootstrap"],
            n_jobs=fixed["n_jobs"],
            random_state=seed,
        )
    if model_id == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            **params,
            bootstrap=fixed["bootstrap"],
            n_jobs=fixed["n_jobs"],
            random_state=seed,
        )
    if model_id == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        params_copy = dict(params)
        params_copy.pop("class_weight", None)
        return LogisticRegression(
            **params_copy,
            class_weight=params["class_weight"],
            solver=fixed["solver"],
            random_state=seed,
        )
    if model_id == "multinomial_nb":
        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB(alpha=params["alpha"], fit_prior=fixed["fit_prior"])
    if model_id == "complement_nb":
        from sklearn.naive_bayes import ComplementNB

        return ComplementNB(alpha=params["alpha"], fit_prior=fixed["fit_prior"])
    if model_id == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            **params,
            categorical_features=fixed["categorical_features"],
            early_stopping=fixed["early_stopping"],
            random_state=seed,
        )
    raise ValueError("model is not in the supported classifier registry")


def build_estimator(
    model_id: str,
    params: dict[str, Any],
    seed: int,
    *,
    representation: str = "dense_numeric",
    pipeline: dict[str, Any] | None = None,
    modality: dict[str, Any] | None = None,
) -> Any:
    """Build a finite, module-defined sklearn pipeline for a validated plan."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("random seed is outside the supported range")
    normalized = normalize_params(model_id, params)
    if representation not in MODELS[model_id].representations:
        raise ValueError("model does not support the requested representation")
    if pipeline is None and representation == "dense_numeric":
        if model_id == "logistic_regression":
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), _classifier(model_id, normalized, seed))
        return _classifier(model_id, normalized, seed)

    pipeline_spec = normalize_pipeline(representation, pipeline or {})
    from sklearn.pipeline import Pipeline

    steps: list[tuple[str, Any]] = []
    if representation == "raw_text":
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = pipeline_spec["text_vectorizer"]["params"]
        vectorizer_kwargs = {
            "analyzer": vectorizer["analyzer"],
            "ngram_range": (vectorizer["ngram_min"], vectorizer["ngram_max"]),
            "max_features": vectorizer["max_features"],
            "min_df": vectorizer["min_df"],
        }
        if vectorizer["field_mode"] == "separate":
            if not isinstance(modality, dict) or not isinstance(modality.get("text_columns"), list):
                raise ValueError("separate text vectorization requires text column declarations")
            transformers = [
                (f"text_{index}", TfidfVectorizer(**vectorizer_kwargs), index)
                for index, _ in enumerate(modality["text_columns"])
            ]
            steps.append(
                ("text_vectorizer", ColumnTransformer(transformers, remainder="drop"))
            )
        else:
            steps.append(("text_vectorizer", TfidfVectorizer(**vectorizer_kwargs)))
    elif representation == "image_tensor":
        if not isinstance(modality, dict) or set(modality) != {
            "kind", "image_shape", "channel_layout", "value_range"
        }:
            raise ValueError("image model requires a validated modality manifest")
        from executor_c.modalities import ImageArrayFlattener

        low, high = modality["value_range"]
        steps.append(("image_flatten_scale", ImageArrayFlattener(low, high)))

    selector = pipeline_spec["feature_selector"]
    if selector["name"] == "variance_threshold":
        from sklearn.feature_selection import VarianceThreshold

        steps.append(("feature_selector", VarianceThreshold(selector["params"]["threshold"])))
    steps.append(("classifier", _classifier(model_id, normalized, seed)))
    return Pipeline(steps)
