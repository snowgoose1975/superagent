"""Regression coverage for task-gated kernels and pairwise prediction symmetry."""

import json

import numpy as np

from executor_c.model_factory import build_estimator, pack_encoder_features
from executor_c.model_registry import (
    PAIRWISE_MODEL_ID,
    TEXT_ENCODER_ID,
    encoder_pipeline,
    paired_text_pipeline,
    runtime_capabilities,
)
from main import run_project


def test_explicit_dense_pipeline_matches_legacy_classifier_scaling():
    from executor_c.model_registry import default_pipeline

    X = np.asarray([[0, 11], [1, 200], [3, 700], [5, 0], [10, 1], [15, 100]], dtype=float)
    y = np.asarray([0, 0, 0, 1, 1, 1])
    legacy = build_estimator("logistic_regression", {}, 3).fit(X, y)
    explicit = build_estimator(
        "logistic_regression", {}, 3, pipeline=default_pipeline("dense_numeric"),
    ).fit(X, y)
    np.testing.assert_allclose(explicit.predict_proba(X), legacy.predict_proba(X))


def test_pairwise_hybrid_proba_mirrors_and_survives_reload(tmp_path):
    from joblib import dump, load

    texts = np.asarray([
        ["learn about bright cats", "read about gray dogs"],
        ["read about gray dogs", "learn about bright cats"],
        ["learn about green cats", "read about brown dogs"],
        ["read about brown dogs", "learn about green cats"],
    ], dtype=object)
    vectors = np.asarray([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0.9, 0.1, 0.1, 0.9],
        [0.1, 0.9, 0.9, 0.1],
    ], dtype=np.float32)
    features = pack_encoder_features(texts, vectors)
    mirrored = pack_encoder_features(texts[:, ::-1], vectors[:, [2, 3, 0, 1]])
    hybrid = encoder_pipeline(2)
    hybrid["text_encoder"]["params"]["lexical"] = "char_tfidf"
    model = build_estimator(
        PAIRWISE_MODEL_ID,
        {"C": 0.3},
        42,
        representation="raw_text",
        pipeline=hybrid,
        modality={"text_columns": ["left", "right"]},
    ).fit(features, [1, 0, 1, 0])
    path = tmp_path / "model.joblib"
    dump(model, path)
    np.testing.assert_allclose(
        model.predict_proba(features), load(path).predict_proba(mirrored)[:, ::-1], atol=1e-12,
    )


def test_advanced_capabilities_are_filtered_by_task_and_size():
    options = dict(min_class_count=3, native_table=False, performance=True)
    text = runtime_capabilities("raw_text", **options)
    assert PAIRWISE_MODEL_ID not in text["models"]
    assert text["pipeline_components"]["text_encoder"][TEXT_ENCODER_ID]["supported_models"] == [
        "logistic_regression",
    ]
    paired = runtime_capabilities("raw_text", pairwise_text=True, **options)
    assert PAIRWISE_MODEL_ID in paired["models"]
    assert not paired["models"][PAIRWISE_MODEL_ID]["supports_multiclass"]
    assert "rbf_svc" in runtime_capabilities(
        "dense_numeric", n_train=1200, n_features=12, **options,
    )["models"]
    assert "rbf_svc" not in runtime_capabilities(
        "dense_numeric", n_train=5000, n_features=1000, **options,
    )["models"]
    assert "rbf_svc" in runtime_capabilities(
        "image_tensor", n_train=300, image_shape=[32, 32, 3], **options,
    )["models"]
    assert "rbf_svc" not in runtime_capabilities(
        "image_tensor", n_train=5000, image_shape=[32, 32, 3], **options,
    )["models"]


def test_spatial_pyramid_kernel_has_valid_image_probabilities():
    data = np.zeros((18, 8, 8, 3), dtype=np.uint8)
    labels = np.arange(len(data)) % 2
    data[labels == 0, 1:3, :] = 210
    data[labels == 1, :, 5:8] = 210
    estimator = build_estimator(
        "rbf_svc", {}, 42, representation="image_tensor",
        pipeline={"image_transform": {"name": "spatial_pyramid", "params": {}}},
        modality={
            "kind": "image",
            "image_shape": [8, 8, 3],
            "channel_layout": "HWC",
            "value_range": [0, 255],
        },
    ).fit(data, labels)
    np.testing.assert_allclose(estimator.predict_proba(data).sum(axis=1), 1)
    np.testing.assert_array_equal(estimator.predict(data), labels)


def test_paired_lexical_fit_and_frozen_inference(tmp_path):
    source = tmp_path / "pair_task"
    source.mkdir()
    (source / "README.md").write_text(
        "Use the left and right text columns. The binary label_pairwise column is 1 if the "
        "left option won and 0 if the right option won. Report test accuracy.\n",
        encoding="utf-8",
    )
    (source / "info.json").write_text(json.dumps({
        "target_column": "label_pairwise",
        "feature_types": {"left": "text", "right": "text"},
    }), encoding="utf-8")
    body = (
        "left,right,label_pairwise\n"
        "bright green cats,quiet brown dogs,1\n"
        "quiet brown dogs,bright green cats,0\n"
        "vivid green cats,calm brown dogs,1\n"
        "calm brown dogs,vivid green cats,0\n"
    )
    for split in ("train", "val", "test"):
        (source / f"{split}.csv").write_text(body, encoding="utf-8")

    class Controller:
        def decide(self, _):
            return {
                "action": "run", "model": PAIRWISE_MODEL_ID, "params": {},
                "pipeline": paired_text_pipeline(), "reason": "Check frozen pairwise inference",
            }

    summary = json.loads(run_project(
        task_directory=source,
        output_root=tmp_path / "work",
        max_trials=1,
        performance=True,
        controller=Controller(),
    ).read_text())
    assert summary["status"] == "success", summary
    assert summary["test_evaluation"]["value"] == 1.0


def test_paired_lexical_preserves_character_vocabulary_when_words_are_pruned():
    texts = np.asarray([
        ["\u7532\u732b\u5f97", "\u4e19\u72d7\u5931"],
        ["\u4e01\u72d7\u5931", "\u4e59\u732b\u5f97"],
    ], dtype=object)
    model = build_estimator(
        PAIRWISE_MODEL_ID, {}, 42, representation="raw_text", pipeline=paired_text_pipeline(),
        modality={"text_columns": ["left", "right"]},
    ).fit(texts, [1, 0])
    vocabulary = model.named_steps["paired_features"].vectorizer_
    assert [name for name, _ in vocabulary.transformer_list] == ["char"]
    np.testing.assert_array_equal(model.predict(texts), [1, 0])

    single_characters = np.asarray([["\u732b", "\u72d7"], ["\u72d7", "\u732b"]], dtype=object)
    model.fit(single_characters, [1, 0])
    vocabulary = model.named_steps["paired_features"].vectorizer_
    assert vocabulary.analyzer == "char"
    assert vocabulary.ngram_range == (1, 5)
    np.testing.assert_array_equal(model.predict(single_characters), [1, 0])


def test_paired_hybrid_still_predicts_with_an_empty_lexical_vocabulary():
    texts = np.full((4, 2), "", dtype=object)
    vectors = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    features = pack_encoder_features(texts, vectors)
    hybrid = encoder_pipeline(2)
    hybrid["text_encoder"]["params"]["lexical"] = "char_tfidf"
    model = build_estimator(
        PAIRWISE_MODEL_ID, {}, 42, representation="raw_text", pipeline=hybrid,
        modality={"text_columns": ["left", "right"]},
    ).fit(features, [1, 0, 1, 0])
    assert model.named_steps["paired_features"].vectorizer_ is None
    np.testing.assert_array_equal(model.predict(features), [1, 0, 1, 0])
