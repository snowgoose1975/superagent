"""Construct only the fixed classifier implementations declared by the registry."""

from __future__ import annotations

from typing import Any

from .model_registry import (
    MAX_RBF_TRAIN_ELEMENTS,
    MAX_RBF_TRAIN_ROWS,
    MODELS,
    PAIRWISE_MODEL_ID,
    normalize_params,
    normalize_pipeline,
    paired_text_pipeline,
    validate_model_pipeline,
)


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
    if model_id == PAIRWISE_MODEL_ID:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**params, **fixed, random_state=seed)
    if model_id == "rbf_svc":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        return CalibratedClassifierCV(
            SVC(**params, cache_size=fixed["cache_size_mb"], random_state=seed),
            method=fixed["calibration"], cv=fixed["cv"],
        )
    if model_id == "multinomial_nb":
        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB(alpha=params["alpha"], fit_prior=fixed["fit_prior"])
    if model_id == "complement_nb":
        from sklearn.naive_bayes import ComplementNB

        return ComplementNB(alpha=params["alpha"], fit_prior=fixed["fit_prior"])
    if model_id == "sgd_logistic":
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier(**params, **fixed, random_state=seed)
    if model_id == "linear_svc":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        return CalibratedClassifierCV(
            LinearSVC(**params, random_state=seed),
            method=fixed["calibration"], cv=fixed["cv"],
        )
    if model_id == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params, **fixed, random_state=seed)
    if model_id == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params, **fixed, random_seed=seed, verbose=False)
    if model_id == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            **params,
            categorical_features=fixed["categorical_features"],
            early_stopping=fixed["early_stopping"],
            random_state=seed,
        )
    raise ValueError("model is not in the supported classifier registry")


def pack_encoder_features(text, vectors):
    """Keep text and float vectors together for the fitted hybrid pipeline."""
    import numpy as np

    features = np.empty((len(text), text.shape[1] + vectors.shape[1]), dtype=object)
    features[:, :text.shape[1]] = text
    features[:, text.shape[1]:] = vectors
    return features


def _numeric_encoder_features(values):
    import numpy as np

    return np.asarray(values, dtype=np.float32)


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
    if pipeline is None and representation == "dense_numeric" and model_id != "rbf_svc":
        if model_id in {"logistic_regression", "linear_svc"}:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), _classifier(model_id, normalized, seed))
        return _classifier(model_id, normalized, seed)

    if pipeline is None and model_id == PAIRWISE_MODEL_ID:
        pipeline = paired_text_pipeline()
    pipeline_spec = normalize_pipeline(representation, pipeline or {})
    validate_model_pipeline(model_id, pipeline_spec)
    from sklearn.pipeline import Pipeline

    steps: list[tuple[str, Any]] = []
    if representation == "raw_text" and "text_vectorizer" in pipeline_spec:
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

        component = pipeline_spec["text_vectorizer"]
        vectorizer = component["params"]

        def vectorize():
            if component["name"] == "hashing":
                analyzer = vectorizer["analyzer"]
                return HashingVectorizer(
                    analyzer=analyzer, n_features=vectorizer["n_features"],
                    alternate_sign=False, ngram_range=(3, 5) if analyzer == "char" else (1, 2),
                )
            if vectorizer["analyzer"] != "word_char":
                return TfidfVectorizer(
                    analyzer=vectorizer["analyzer"],
                    ngram_range=(vectorizer["ngram_min"], vectorizer["ngram_max"]),
                    max_features=vectorizer["max_features"], min_df=vectorizer["min_df"],
                )
            from sklearn.pipeline import FeatureUnion

            settings = {key: vectorizer[key] for key in ("min_df", "max_features")}
            return FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **settings)),
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), **settings)),
            ])

        if model_id == PAIRWISE_MODEL_ID:
            if not isinstance(modality, dict) or len(modality.get("text_columns", [])) != 2:
                raise ValueError("Pairwise preference classification requires exactly two text fields")
            from executor_c.modalities import PairwiseTextFeatures

            steps.append(("paired_features", PairwiseTextFeatures(vectorize())))
        elif vectorizer["field_mode"] == "separate":
            if not isinstance(modality, dict) or not isinstance(modality.get("text_columns"), list):
                raise ValueError("separate text vectorization requires text column declarations")
            transformers = [
                (f"text_{index}", vectorize(), index)
                for index, _ in enumerate(modality["text_columns"])
            ]
            steps.append(
                ("text_vectorizer", ColumnTransformer(transformers, remainder="drop"))
            )
        else:
            steps.append(("text_vectorizer", vectorize()))
    elif pipeline_spec.get("text_encoder", {}).get("params", {}).get("lexical") == "char_tfidf":
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import FunctionTransformer

        n_fields = len(modality["text_columns"])
        if model_id == PAIRWISE_MODEL_ID:
            from executor_c.modalities import PairwiseTextFeatures

            if n_fields != 2:
                raise ValueError("Pairwise preference classification requires exactly two text fields")
            from sklearn.pipeline import FeatureUnion

            lexical_settings = {"min_df": 2, "max_features": 40_000, "sublinear_tf": True}
            vectorizer = FeatureUnion([
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), **lexical_settings)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **lexical_settings)),
            ])
            steps.append(("paired_features", PairwiseTextFeatures(vectorizer, semantic_weight=0.5)))
        else:
            transformers = [
                (f"char_{index}", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                                max_features=20_000), index)
                for index in range(n_fields)
            ]
            transformers.append((
                "semantic", FunctionTransformer(_numeric_encoder_features), slice(n_fields, None),
            ))
            steps.append(("hybrid_features", ColumnTransformer(transformers, sparse_threshold=1.0)))
    elif representation == "image_tensor":
        if not isinstance(modality, dict) or set(modality) != {
            "kind", "image_shape", "channel_layout", "value_range"
        }:
            raise ValueError("image model requires a validated modality manifest")
        from executor_c.modalities import ImageArrayFlattener, ImageGradientFeatures, ImageSpatialPyramid

        low, high = modality["value_range"]
        transform = {
            "gradient_cells": ImageGradientFeatures,
            "spatial_pyramid": ImageSpatialPyramid,
            "flatten_scale": ImageArrayFlattener,
        }[pipeline_spec["image_transform"]["name"]]
        steps.append(("image_transform", transform(low, high)))

    if model_id == "rbf_svc":
        from sklearn.preprocessing import StandardScaler
        from executor_c.modalities import FitMatrixLimit

        steps.extend([
            ("rbf_fit_limit", FitMatrixLimit(MAX_RBF_TRAIN_ROWS, MAX_RBF_TRAIN_ELEMENTS)),
            ("scale", StandardScaler()),
        ])

    selector = pipeline_spec["feature_selector"]
    if selector["name"] == "variance_threshold":
        from sklearn.feature_selection import VarianceThreshold

        steps.append(("feature_selector", VarianceThreshold(selector["params"]["threshold"])))
    steps.append(("classifier", _classifier(model_id, normalized, seed)))
    return Pipeline(steps)
