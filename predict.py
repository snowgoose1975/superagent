"""Standalone inference for a frozen bundle."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import joblib
import numpy as np
from data_b.modality_readers import (
    combine_text_columns,
    load_image_tensor,
    load_text_matrix,
)
from data_b.raw_reader import load_feature_frame
from executor_c.contracts import read_json, resolve_data_path, sha256_file
from executor_c.model_registry import normalize_pipeline


def predict_bundle(
    bundle_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    id_prefix: str = "prediction",
    *,
    pretrained: str = "cache-only",
) -> Path:
    """Apply the frozen transformer and classifier to new data and emit class IDs and probabilities."""
    path = Path(bundle_path).resolve()
    bundle = read_json(path)
    if (
        bundle.get("schema_version") not in {"1.0", "2.0"}
        or bundle.get("training_policy", {}).get("refit_on_validation") is not False
    ):
        raise ValueError("Unsupported frozen bundle schema or training policy")
    for name, expected in bundle["hashes"].items():
        artifact = resolve_data_path(path.parent, name)
        if sha256_file(artifact) != expected:
            raise ValueError("A frozen artifact checksum does not match")
    if not isinstance(id_prefix, str) or not id_prefix or any(c in id_prefix for c in "\r\n\0"):
        raise ValueError("id_prefix must be a nonempty single-line string")
    if bundle.get("ensemble"):
        class_ids = [item["id"] for item in bundle["classes"]]
        with tempfile.TemporaryDirectory(prefix="superagent-ensemble-") as directory:
            probability = None
            for index, member in enumerate(bundle["ensemble"]):
                if member not in bundle["hashes"] or "/" not in member:
                    raise ValueError("The ensemble member manifest lacks a frozen checksum")
                member_path = resolve_data_path(path.parent, member)
                member_bundle = read_json(member_path)
                if member_bundle.get("ensemble") or member_bundle["classes"] != bundle["classes"]:
                    raise ValueError("The ensemble member classes do not match the frozen manifest")
                member_output = Path(directory) / f"{index}.csv"
                predict_bundle(member_path, features_path, member_output, id_prefix,
                               pretrained=pretrained)
                with member_output.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.reader(stream))[1:]
                matrix = np.asarray([row[2:] for row in rows], dtype=np.float64)
                if probability is None:
                    probability = matrix
                else:
                    if matrix.shape != probability.shape:
                        raise ValueError("Ensemble member prediction dimensions differ")
                    probability += matrix
            probability /= len(bundle["ensemble"])
        predictions = np.asarray(class_ids)[np.argmax(probability, axis=1)]
        return _write_predictions(output_path, id_prefix, class_ids, predictions, probability)

    model = joblib.load(resolve_data_path(path.parent, bundle["artifacts"]["model_path"]))
    if bundle["schema_version"] == "1.0":
        frame = load_feature_frame(
            Path(features_path).resolve(), bundle["feature_columns"], bundle["feature_types"]
        )
        if bundle.get("input_representation") == "native_table":
            from executor_c.native_table import native_features

            X = native_features(
                frame, bundle["feature_types"],
                missing_values=bundle.get("preprocessing", {}).get("missing_values", {}),
            )
        else:
            preprocessor = joblib.load(
                resolve_data_path(path.parent, bundle["artifacts"]["preprocessor_path"])
            )
            X = preprocessor.transform(frame)
        n_rows = len(frame)
    else:
        representation = bundle.get("representation")
        modality = bundle.get("modality")
        feature_path = Path(features_path).resolve()
        if representation == "raw_text" and isinstance(modality, dict):
            X = load_text_matrix(
                feature_path,
                modality["text_columns"],
                max_characters=modality["max_characters"],
            )
            frozen_plan = read_json(
                resolve_data_path(path.parent, "plan.json"), error_code="FINAL_BUNDLE_INVALID"
            )
            pipeline = normalize_pipeline(representation, frozen_plan.get("pipeline", {}))
            if "text_encoder" in pipeline:
                from executor_c.text_encoder import LocalTextEncoder, encoder_manifest, prepare_weights

                if "encoder.json" not in bundle["hashes"]:
                    raise ValueError("The frozen plan lacks an encoder resource manifest")
                descriptor = read_json(resolve_data_path(path.parent, "encoder.json"))
                if descriptor.get("resource") != encoder_manifest():
                    raise ValueError("The frozen encoder manifest does not match the installed implementation")
                prepare_weights(pretrained)
                encoder = LocalTextEncoder()
                vectors, stats = encoder.transform(
                    X, modality["text_columns"], pipeline["text_encoder"]["params"]["field_mode"]
                )
                if pipeline["text_encoder"]["params"].get("lexical") == "char_tfidf":
                    from executor_c.model_factory import pack_encoder_features

                    X = pack_encoder_features(X, vectors)
                else:
                    X = vectors
                print(
                    f"[C] Encoded {stats['rows']} inference rows on {encoder.device} in "
                    f"{stats['seconds']:.3f} seconds",
                    flush=True,
                )
            elif pipeline["text_vectorizer"]["params"]["field_mode"] != "separate":
                X = combine_text_columns(X, modality["text_columns"])
        elif representation == "image_tensor" and isinstance(modality, dict):
            X = load_image_tensor(
                feature_path,
                modality["image_shape"],
                modality["channel_layout"],
                modality["value_range"],
            )
        else:
            raise ValueError("Unsupported representation or modality in the 2.0 bundle")
        n_rows = len(X)
    predictions = np.asarray(model.predict(X)).reshape(-1)
    raw_probability = np.asarray(model.predict_proba(X), dtype=np.float64)
    class_ids = [item["id"] for item in bundle["classes"]]
    class_positions = {int(label): index for index, label in enumerate(model.classes_)}
    if set(class_positions) != set(class_ids) or not np.isin(predictions, class_ids).all():
        raise ValueError("Model classes differ from the frozen manifest")
    probability = raw_probability[:, [class_positions[label] for label in class_ids]]
    if (
        probability.shape != (n_rows, len(class_ids))
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
        or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("The frozen model emitted invalid probabilities")
    return _write_predictions(output_path, id_prefix, class_ids, predictions, probability)


def _write_predictions(output_path, id_prefix, class_ids, predictions, probability) -> Path:
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "y_pred", *[f"p_{item}" for item in class_ids]])
        for index, (label, probabilities) in enumerate(zip(predictions, probability, strict=True)):
            writer.writerow(
                [
                    f"{id_prefix}_{index:09d}",
                    int(label),
                    *[format(float(p), ".17g") for p in probabilities],
                ]
            )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id-prefix", default="prediction")
    parser.add_argument("--pretrained", choices=("cache-only", "download"), default="cache-only")
    args = parser.parse_args()
    print(predict_bundle(
        args.bundle, args.features, args.output, args.id_prefix, pretrained=args.pretrained,
    ))


if __name__ == "__main__":
    main()
