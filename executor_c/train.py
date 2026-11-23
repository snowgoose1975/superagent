"""Train a classifier from a validated plan and generate validation predictions.

This module builds models from the capability registry and uses validation
data for inference. The executor normally calls this module in a subprocess.

"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from executor_c.contracts import (
    ContractError,
    load_prepared_data,
    read_json,
    validate_plan,
    validate_prediction_file,
    write_json,
)
from executor_c.model_factory import build_estimator, pack_encoder_features
from executor_c.diagnostics import experiment_signature, make_diagnostics, write_worker_snapshot
from data_b.modality_readers import combine_text_columns


def _construct_model(plan: dict[str, Any], manifest: dict[str, Any]) -> Any:
    """Construct a registered model from a validated plan."""
    try:
        return build_estimator(
            plan["model"],
            plan["params"],
            plan["random_seed"],
            representation=manifest.get("representation", "dense_numeric"),
            pipeline=plan.get("pipeline"),
            modality=manifest.get("modality"),
        )
    except ValueError as exc:
        raise ContractError("UNSUPPORTED_MODEL", "The model construction request is unsupported") from exc


def _ordered_predictions(
    model: Any, X_valid: np.ndarray, class_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate prediction shapes and values and order probabilities by class ID."""
    raw_pred = np.asarray(model.predict(X_valid))
    if raw_pred.shape == (len(X_valid), 1):
        raw_pred = raw_pred[:, 0]
    raw_prob = np.asarray(model.predict_proba(X_valid), dtype=np.float64)
    expected_shape = (len(X_valid), len(class_ids))
    if raw_pred.shape != (len(X_valid),) or raw_prob.shape != expected_shape:
        raise RuntimeError("Prediction shape differs from the validation sample or class count")
    model_classes = np.asarray(model.classes_)
    if model_classes.ndim != 1 or len(model_classes) != len(class_ids):
        raise RuntimeError("The model class list has an invalid shape")
    try:
        column_for_id = {int(class_id): i for i, class_id in enumerate(model_classes.tolist())}
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("The model returned an unrecognized class ID") from exc
    if len(column_for_id) != len(class_ids) or set(column_for_id) != set(class_ids):
        raise RuntimeError("The fitted model classes differ from the manifest")
    probs = raw_prob[:, [column_for_id[class_id] for class_id in class_ids]]
    if not np.isfinite(probs).all() or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise RuntimeError("Model probabilities must be finite and in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Model probabilities in each row must sum to 1")
    if not np.isin(raw_pred, class_ids).all():
        raise RuntimeError("Model predictions contain a class ID absent from the manifest")
    return raw_pred.astype(np.int64), probs


def _diagnostic_array_bytes(values: np.ndarray) -> int:
    if values.dtype.kind == "O":
        return int(values.nbytes + sum(len(item.encode("utf-8")) for item in values.flat))
    return int(values.nbytes)


def _write_predictions(
    path: Path,
    sample_ids: list[str],
    class_ids: list[int],
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Write a new UTF-8 predictions CSV with the specified column order."""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "y_pred", *(f"p_{class_id}" for class_id in class_ids)])
        for sample_id, prediction, row in zip(sample_ids, predictions, probabilities, strict=True):
            writer.writerow([sample_id, int(prediction), *(format(float(p), ".17g") for p in row)])


def run(plan_path: Path, manifest_path: Path, run_dir: Path) -> None:
    """Load inputs, train, save, and verify artifacts in an existing run_dir."""
    started = time.perf_counter()
    fitting_started = False
    phase = "input"
    experiment_id = run_dir.name
    random_seed: int | None = None
    data = None
    model = None
    fit_complete = False
    timings: dict[str, float | None] = {
        "input": None, "fit": None, "predict": None, "serialize": None,
        "reload_check": None, "total": None,
    }
    captured_warnings: list[dict[str, str]] = []
    warnings_truncated = False
    signature: str | None = None

    def snapshot(status: str, failure: dict[str, str] | None = None) -> None:
        timings["total"] = max(0.0, time.perf_counter() - started)
        try:
            write_worker_snapshot(
                run_dir / "worker_diagnostics.json",
                make_diagnostics(
                    experiment_id=experiment_id,
                    status=status,
                    phase="complete" if status == "success" else phase,
                    timings_seconds=timings,
                    model_bytes=(
                        (run_dir / "model.joblib").stat().st_size
                        if (run_dir / "model.joblib").is_file()
                        else None
                    ),
                    input_shape={
                        "train": list(data.X_train.shape) if data is not None else None,
                        "validation": list(data.X_valid.shape) if data is not None else None,
                    },
                    input_matrix_bytes=(
                        _diagnostic_array_bytes(data.X_train)
                        + _diagnostic_array_bytes(data.X_valid)
                        if data is not None
                        else None
                    ),
                    random_seed=random_seed,
                    effective_iterations=(
                        _effective_iterations(model) if fit_complete and model is not None else None
                    ),
                    warnings=captured_warnings,
                    warnings_truncated=warnings_truncated,
                    failure=failure,
                    signature=signature,
                ),
            )
        except Exception:
            # Diagnostic failure must not replace the original worker outcome.
            pass

    try:
        input_started = time.perf_counter()
        raw_plan = read_json(plan_path, error_code="PLAN_INVALID")
        plan = validate_plan(raw_plan)
        experiment_id = plan["experiment_id"]
        random_seed = plan["random_seed"]
        data = load_prepared_data(manifest_path, plan)
        manifest = data.manifest
        signature = experiment_signature(plan, data.input_hashes)
        timings["input"] = time.perf_counter() - input_started
        snapshot("running")
        model = _construct_model(plan, manifest)
        encoder_config = plan.get("pipeline", {}).get("text_encoder")
        if encoder_config is not None:
            from executor_c.text_encoder import EncoderError, LocalTextEncoder, encoder_manifest

            phase = "text_encoding"
            snapshot("running")
            encoder_runtime = read_json(run_dir / "encoder_runtime.json")
            if encoder_runtime.get("enabled") is not True:
                raise ContractError("ENCODER_DISABLED", "The encoder is disabled for this experiment")
            encoder = LocalTextEncoder()
            columns = manifest["modality"]["text_columns"]
            field_mode = encoder_config["params"]["field_mode"]
            cache_dir = Path(encoder_runtime["feature_cache"])
            stats = []
            matrices = []
            for partition, text in (("train", data.X_train), ("validation", data.X_valid)):
                try:
                    matrix, record = encoder.transform(
                        text, columns, field_mode, cache_dir=cache_dir,
                        source_hash=data.input_hashes[f"{partition}_features"], partition=partition,
                    )
                except EncoderError as exc:
                    raise ContractError(exc.code, str(exc)) from exc
                if encoder_config["params"].get("lexical") == "char_tfidf":
                    matrix = pack_encoder_features(text, matrix)
                matrices.append(matrix)
                stats.append(record)
            X_train, X_valid = matrices
            write_json(run_dir / "encoder.json", {
                "resource": encoder_manifest(), "packages": encoder.versions,
                "device": encoder.device, "partitions": stats,
                "weights": encoder_runtime["weights"],
            })
        elif plan["model"] == "catboost":
            from executor_c.native_table import load_native_tables

            X_train, X_valid, cat_columns = load_native_tables(
                manifest_path.parent, len(data.X_train), len(data.X_valid)
            )
            model.set_params(cat_features=cat_columns)
        elif manifest.get("representation") == "raw_text":
            modality = manifest["modality"]
            field_mode = plan["pipeline"]["text_vectorizer"]["params"]["field_mode"]
            if field_mode == "separate":
                X_train, X_valid = data.X_train, data.X_valid
            else:
                X_train = combine_text_columns(data.X_train, modality["text_columns"])
                X_valid = combine_text_columns(data.X_valid, modality["text_columns"])
        else:
            X_train, X_valid = data.X_train, data.X_valid
        print(f"Model: {plan['model']}", flush=True)

        # Classify errors after fit starts as execution failures.
        fitting_started = True
        phase = "fit"
        snapshot("running")
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, data.y_train)
        fit_complete = True
        timings["fit"] = time.perf_counter() - fit_started
        warnings_truncated = len(caught) > 20
        captured_warnings = [
            {
                "category": warning.category.__name__,
                "phase": "fit",
                "message": str(warning.message),
            }
            for warning in caught[:20]
        ]
        snapshot("running")
        print("Training complete.", flush=True)
        phase = "prediction"
        snapshot("running")
        predict_started = time.perf_counter()
        predictions, probabilities = _ordered_predictions(model, X_valid, data.class_ids)
        timings["predict"] = time.perf_counter() - predict_started

        phase = "serialize"
        snapshot("running")
        serialize_started = time.perf_counter()
        model_path = run_dir / "model.joblib"
        # Write the joblib model to a new file to preserve historical artifacts.
        with model_path.open("xb") as stream:
            joblib.dump(model, stream)
        timings["serialize"] = time.perf_counter() - serialize_started

        # Reload the model here for a small probe. Integration tests check loading in a new process.
        phase = "reload_check"
        snapshot("running")
        reload_started = time.perf_counter()
        restored = joblib.load(model_path)
        probe_size = min(len(data.X_valid), 5)
        # Compare identical batch shapes: float32 kernels may round differently
        # for a short probe than for the full validation matrix.
        probe_pred, probe_prob = _ordered_predictions(model, X_valid[:probe_size], data.class_ids)
        restored_pred, restored_prob = _ordered_predictions(
            restored, X_valid[:probe_size], data.class_ids
        )
        if not np.array_equal(restored_pred, probe_pred) or not np.allclose(
            restored_prob, probe_prob, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("The saved model predictions changed after reloading")
        timings["reload_check"] = time.perf_counter() - reload_started

        prediction_path = run_dir / "predictions.csv"
        _write_predictions(
            prediction_path, data.sample_ids, data.class_ids, predictions, probabilities
        )
        count = validate_prediction_file(
            prediction_path, sample_ids=data.sample_ids, class_ids=data.class_ids
        )
        write_json(
            run_dir / "worker_metadata.json",
            {
                "class_ids": data.class_ids,
                "n_predictions": count,
                "input_hashes": data.input_hashes,
            },
        )
        snapshot("success")
        print("Predictions and model artifacts were saved and verified.", flush=True)
    except ContractError as exc:
        status = (
            "rejected"
            if exc.code
            in {"PLAN_INVALID", "DATA_INVALID", "UNSUPPORTED_MODEL", "UNSUPPORTED_PARAMETER"}
            and not fitting_started
            else "failed"
        )
        snapshot(status, {"code": exc.code, "message": exc.message})
        _record_error(run_dir, status, exc.code, exc.message)
        raise
    except Exception as exc:
        if phase in {"serialize", "reload_check"}:
            code = "ARTIFACT_INVALID"
        else:
            code = "TRAINING_ERROR" if fitting_started else "DATA_INVALID"
        snapshot("failed", {"code": code, "message": f"{type(exc).__name__}: {exc}"})
        _record_error(run_dir, "failed", code, str(exc))
        raise


def _effective_iterations(model: Any) -> int | None:
    if model is None:
        return None
    if hasattr(model, "steps") and model.steps:
        model = model.steps[-1][1]
    if hasattr(model, "n_estimators"):
        value = getattr(model, "n_estimators")
    elif hasattr(model, "n_iter_"):
        value = np.asarray(model.n_iter_).max()
    elif hasattr(model, "named_steps"):
        estimator = model.named_steps.get("logisticregression")
        value = (
            np.asarray(estimator.n_iter_).max()
            if estimator is not None and hasattr(estimator, "n_iter_")
            else None
        )
    else:
        value = None
    return int(value) if value is not None and np.isfinite(value) else None


def _record_error(run_dir: Path, status: str, code: str, message: str) -> None:
    """Record a structured error when possible and keep the traceback in stderr."""
    print(traceback.format_exc(), file=sys.stderr, end="")
    try:
        write_json(
            run_dir / "worker_error.json",
            {"status": status, "error": {"code": code, "message": message}},
        )
    except Exception:
        print("Cannot write worker_error.json. See the traceback above.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a registered classifier from a validated plan")
    parser.add_argument("--plan", required=True, type=Path, help="Absolute path to the experiment plan JSON")
    parser.add_argument(
        "--data-manifest", required=True, type=Path, help="Absolute path to the data manifest JSON"
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Existing directory for this run")
    args = parser.parse_args(argv)
    try:
        run(args.plan, args.data_manifest, args.run_dir)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
