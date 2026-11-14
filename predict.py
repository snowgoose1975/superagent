"""冻结后的独立预测入口；不拟合，不评分，不调用 Agent 再选方案。"""

from __future__ import annotations

import argparse
import csv
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
) -> Path:
    """新输入使用冻结的 B 转换器和 C 模型，输出编码类别和全部概率。"""
    path = Path(bundle_path).resolve()
    bundle = read_json(path)
    if (
        bundle.get("schema_version") not in {"1.0", "2.0"}
        or bundle.get("training_policy", {}).get("refit_on_validation") is not False
    ):
        raise ValueError("冻结 bundle schema 或训练策略不受支持")
    for name, expected in bundle["hashes"].items():
        artifact = resolve_data_path(path.parent, name)
        if sha256_file(artifact) != expected:
            raise ValueError("冻结工件摘要不一致，拒绝推理")
    if not isinstance(id_prefix, str) or not id_prefix or any(c in id_prefix for c in "\r\n\0"):
        raise ValueError("id_prefix 必须是非空单行字符串")
    model = joblib.load(resolve_data_path(path.parent, bundle["artifacts"]["model_path"]))
    if bundle["schema_version"] == "1.0":
        preprocessor = joblib.load(
            resolve_data_path(path.parent, bundle["artifacts"]["preprocessor_path"])
        )
        frame = load_feature_frame(
            Path(features_path).resolve(), bundle["feature_columns"], bundle["feature_types"]
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
            if pipeline["text_vectorizer"]["params"]["field_mode"] != "separate":
                X = combine_text_columns(X, modality["text_columns"])
        elif representation == "image_tensor" and isinstance(modality, dict):
            X = load_image_tensor(
                feature_path,
                modality["image_shape"],
                modality["channel_layout"],
                modality["value_range"],
            )
        else:
            raise ValueError("bundle 2.0 的 representation/modality 不受支持")
        n_rows = len(X)
    predictions = np.asarray(model.predict(X))
    raw_probability = np.asarray(model.predict_proba(X), dtype=np.float64)
    class_ids = [item["id"] for item in bundle["classes"]]
    class_positions = {int(label): index for index, label in enumerate(model.classes_)}
    if set(class_positions) != set(class_ids) or not np.isin(predictions, class_ids).all():
        raise ValueError("模型类别与冻结清单不一致")
    probability = raw_probability[:, [class_positions[label] for label in class_ids]]
    if (
        probability.shape != (n_rows, len(class_ids))
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
        or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("冻结模型输出了非法概率")
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
    args = parser.parse_args()
    print(predict_bundle(args.bundle, args.features, args.output, args.id_prefix))


if __name__ == "__main__":
    main()
