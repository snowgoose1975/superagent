"""Score a frozen final bundle once on a separately configured test partition."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from data_b.npy_reader import read_npy_array
from data_b.raw_reader import typed_category_token
from evaluation_d.metric_contracts import score_predictions, validate_metric_spec
from executor_c.contracts import read_json, resolve_data_path, sha256_file, write_json
from predict import predict_bundle

_TEST_CONFIG_FIELDS = {
    "schema_version",
    "task_id",
    "source_root",
    "features_path",
    "labels_path",
    "n_samples",
}


class FinalTestError(ValueError):
    """A test evaluation failed after the final bundle was already frozen."""

    code = "FINAL_TEST_FAILED"


def _load_test_config(config_path: str | Path, expected_task_id: str) -> tuple[dict[str, Any], Path, Path]:
    path = Path(config_path).expanduser().resolve(strict=True)
    config = read_json(path, error_code="FINAL_TEST_CONFIG_INVALID")
    if set(config) != _TEST_CONFIG_FIELDS or config.get("schema_version") != "1.0":
        raise ValueError("test config 必须符合 schema 1.0 且字段完整")
    if config.get("task_id") != expected_task_id:
        raise ValueError("test config 的 task_id 与冻结 bundle 不一致")
    if (
        not isinstance(config.get("source_root"), str)
        or not config["source_root"]
        or type(config.get("n_samples")) is not int
        or config["n_samples"] < 1
    ):
        raise ValueError("test config 的 source_root/n_samples 不合法")
    root = (path.parent / config["source_root"]).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("test config 的 source_root 必须是现存目录")
    features = resolve_data_path(root, config["features_path"])
    labels = resolve_data_path(root, config["labels_path"])
    if features == labels:
        raise ValueError("test 特征和标签必须是不同文件")
    return config, features, labels


def _load_encoded_labels(labels_path: Path, mapping: dict[str, Any], class_ids: list[int]) -> list[int]:
    label_values = mapping.get("label_values")
    classes = mapping.get("classes")
    if (
        not isinstance(label_values, list)
        or len(label_values) != len(class_ids)
        or not isinstance(classes, list)
        or [item.get("id") if isinstance(item, dict) else None for item in classes]
        != class_ids
    ):
        raise ValueError("冻结 label_mapping 与 bundle 的类别定义不一致")
    tokens = [typed_category_token(value) for value in label_values]
    if any(not isinstance(token, str) for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("冻结 label_mapping 的原始类别值无效或重复")
    label_to_id = {token: class_id for token, class_id in zip(tokens, class_ids, strict=True)}
    raw = read_npy_array(labels_path)
    if raw.ndim != 1:
        raise ValueError("test 标签必须是一维数组")
    encoded: list[int] = []
    for value in raw:
        token = typed_category_token(value)
        if not isinstance(token, str) or token not in label_to_id:
            raise ValueError("test 标签包含冻结 label_mapping 未声明的类别")
        encoded.append(label_to_id[token])
    return encoded


def _read_predictions(path: Path, class_ids: list[int]) -> tuple[list[int], list[list[float]]]:
    expected_header = ["sample_id", "y_pred", *[f"p_{class_id}" for class_id in class_ids]]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != expected_header:
                raise ValueError("test 预测 CSV 表头与冻结类别不匹配")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("无法读取 test 预测 CSV") from exc
    if not rows:
        raise ValueError("test 预测不能为空")
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    for index, row in enumerate(rows):
        if len(row) != len(expected_header) or row[0] != f"test_{index:09d}":
            raise ValueError("test 预测行结构或顺序标识无效")
        try:
            label = int(row[1])
            probability = [float(value) for value in row[2:]]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("test 预测类别或概率不是合法数字") from exc
        if str(label) != row[1] or label not in class_ids:
            raise ValueError("test 预测包含非法类别")
        y_pred.append(label)
        probabilities.append(probability)
    return y_pred, probabilities


def evaluate_frozen_bundle(
    bundle_path: str | Path,
    test_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Predict and score test rows without fitting or changing the frozen model."""
    destination = Path(output_dir).expanduser().resolve()
    if not destination.is_dir():
        raise ValueError("test 报告目录必须已存在")
    report_path = destination / "test_evaluation.json"
    predictions_path = destination / "test_predictions.csv"
    bundle_file = Path(bundle_path).expanduser().resolve()
    bundle: dict[str, Any] = {}
    try:
        bundle_file = bundle_file.resolve(strict=True)
        bundle = read_json(bundle_file, error_code="FINAL_BUNDLE_INVALID")
        if (
            bundle.get("schema_version") not in {"1.0", "2.0"}
            or not isinstance(bundle.get("final_id"), str)
            or not bundle["final_id"].startswith("final_")
            or bundle.get("selection", {}).get("partition") != "validation"
            or bundle.get("training_policy", {}).get("refit_on_validation") is not False
        ):
            raise ValueError("只允许评估已冻结的 validation 最优 final bundle")
        classes = bundle.get("classes")
        class_ids = [item.get("id") if isinstance(item, dict) else None for item in classes]
        if class_ids != list(range(len(class_ids))) or len(class_ids) < 2:
            raise ValueError("冻结 bundle 的类别 ID 必须是完整的 0..K-1 序列")
        metric = validate_metric_spec(
            {
                "name": bundle["selection"].get("metric"),
                "greater_is_better": bundle["selection"].get("greater_is_better"),
            }
        )
        source_inputs = None
        requested_test_config = read_json(test_config_path, error_code="FINAL_TEST_CONFIG_INVALID")
        if requested_test_config.get("schema_version") == "2.0":
            from data_b.task_directory import materialize_final_test

            if (
                requested_test_config.get("kind") != "task_directory"
                or requested_test_config.get("task_id") != bundle["task_id"]
            ):
                raise ValueError("自动 test 描述与冻结 bundle 不一致")
            # CSV/多数组 test 的转换也必须发生在 final bundle 冻结之后。
            test_input_dir = destination / "test_input"
            test_config_path = materialize_final_test(test_config_path, test_input_dir)
            source_inputs = read_json(test_input_dir / "test_source.json")
        config, features_path, labels_path = _load_test_config(
            test_config_path, bundle["task_id"]
        )
        config_hash = sha256_file(Path(test_config_path).expanduser().resolve(strict=True))
        feature_hash = sha256_file(features_path)
        labels_hash = sha256_file(labels_path)
        mapping_path = resolve_data_path(bundle_file.parent, bundle["artifacts"]["label_mapping_path"])
        mapping = read_json(mapping_path, error_code="FINAL_BUNDLE_INVALID")

        predict_bundle(bundle_file, features_path, predictions_path, id_prefix="test")
        y_true = _load_encoded_labels(labels_path, mapping, class_ids)
        y_pred, probabilities = _read_predictions(predictions_path, class_ids)
        if (
            len(y_true) != config["n_samples"]
            or len(y_pred) != config["n_samples"]
            or len(y_true) != len(y_pred)
        ):
            raise ValueError("test 特征/标签行数与 n_samples 声明不一致")
        metric_value = score_predictions(metric["name"], y_true, y_pred, probabilities, class_ids)
        accuracy = score_predictions("accuracy", y_true, y_pred, probabilities, class_ids)
        if (
            sha256_file(features_path) != feature_hash
            or sha256_file(labels_path) != labels_hash
            or sha256_file(Path(test_config_path).expanduser().resolve(strict=True)) != config_hash
        ):
            raise ValueError("test 输入或配置在评估期间发生变化")
        report = {
            "schema_version": "1.0",
            "task_id": bundle["task_id"],
            "data_version": bundle["data_version"],
            "session_id": bundle["session_id"],
            "final_id": bundle["final_id"],
            "experiment_id": bundle["experiment_id"],
            "status": "success",
            "selection_partition": "validation",
            "metric": metric["name"],
            "value": metric_value,
            "greater_is_better": metric["greater_is_better"],
            "accuracy": accuracy,
            "n_samples": len(y_true),
            "artifacts": {
                "predictions_path": predictions_path.name,
                "bundle_sha256": sha256_file(bundle_file),
                "test_config_sha256": config_hash,
                "features_sha256": feature_hash,
                "labels_sha256": labels_hash,
                "predictions_sha256": sha256_file(predictions_path),
            },
        }
        if source_inputs is not None:
            report["artifacts"]["source_inputs"] = source_inputs
        write_json(report_path, report)
        return report
    except Exception as exc:
        report = {
            "schema_version": "1.0",
            "task_id": bundle.get("task_id"),
            "data_version": bundle.get("data_version"),
            "session_id": bundle.get("session_id"),
            "final_id": bundle.get("final_id"),
            "experiment_id": bundle.get("experiment_id"),
            "status": "failed",
            "artifacts": {
                "predictions_path": predictions_path.name if predictions_path.exists() else None,
            },
            "error": {"code": getattr(exc, "code", "FINAL_TEST_FAILED"), "message": str(exc)},
        }
        try:
            write_json(report_path, report)
        except FileExistsError:
            pass
        except OSError as report_exc:
            raise FinalTestError(
                f"{exc}；test 失败报告也无法写入：{report_exc}"
            ) from exc
        raise FinalTestError(str(exc)) from exc


def format_test_result(report: dict[str, Any]) -> list[str]:
    """Terminal lines for the configured metric and the familiar accuracy score."""
    if report.get("status") != "success":
        return [f"[Test] 失败：{report.get('error', {}).get('message', '未知错误')}"]
    n_samples = report["n_samples"]
    metric = report["metric"]
    lines = []
    if metric == "accuracy":
        lines.append(f"[Test] accuracy = {report['value']:.6f} ({report['value']:.2%})，n = {n_samples}")
    else:
        lines.append(f"[Test] {metric} = {report['value']:.6f}，n = {n_samples}")
        lines.append(f"[Test] accuracy = {report['accuracy']:.6f} ({report['accuracy']:.2%})，n = {n_samples}")
    return lines

