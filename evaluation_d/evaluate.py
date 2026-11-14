"""D 独立评分：按 sample_id 对齐真实验证标签并计算 accuracy。"""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from executor_c.consumer import load_successful_predictions
from executor_c.contracts import read_json, resolve_data_path, write_json
from evaluation_d.metric_contracts import score_predictions, validate_metric_spec


def _read_csv(path: Path, expected_header: list[str]) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != expected_header:
                raise ValueError(f"{path.name} 表头必须为 {expected_header!r}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"无法读取 {path.name}") from exc
    if any(len(row) != len(expected_header) for row in rows):
        raise ValueError(f"{path.name} 含列数错误的行")
    return rows


def _resolve_manifest_file(manifest_path: Path, relative: Any) -> Path:
    """按清单所在目录解析相对路径，并拒绝越出目录的路径。"""
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in PurePosixPath(relative).parts
    ):
        raise ValueError("evaluation_manifest 中的数据路径必须是目录内相对 POSIX 路径")
    return resolve_data_path(manifest_path.parent, relative)


def evaluate(
    result_path: Path,
    evaluation_manifest_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """只对 C 的 success 产物评分，不修改 C 的 result.json。"""
    result_path = Path(result_path).expanduser().resolve(strict=True)
    evaluation_manifest_path = Path(evaluation_manifest_path).expanduser().resolve(strict=True)
    if result_path.name != "result.json" or not result_path.is_file():
        raise ValueError("result_path 必须指向 C 的 result.json")

    # consumer 负责 success、预测路径、概率及行结构检查。
    runs_root = result_path.parents[3]
    relpath = result_path.relative_to(runs_root).as_posix()
    result, predictions = load_successful_predictions(relpath, runs_root)

    evaluation_manifest = read_json(evaluation_manifest_path, "DATA_INVALID")
    expected_fields = {
        "schema_version",
        "task_id",
        "data_version",
        "data_manifest_path",
        "labels_path",
        "primary_metric",
        "positive_class_id",
    }
    if set(evaluation_manifest) != expected_fields:
        raise ValueError("evaluation_manifest 字段缺失或包含未知字段")
    if evaluation_manifest["schema_version"] != "1.0":
        raise ValueError("evaluation_manifest schema_version 不受支持")
    identity = (result["task_id"], result["data_version"])
    if (evaluation_manifest["task_id"], evaluation_manifest["data_version"]) != identity:
        raise ValueError("evaluation_manifest 与 C result 的 task_id/data_version 不一致")
    metric = validate_metric_spec(evaluation_manifest["primary_metric"])

    data_manifest_path = _resolve_manifest_file(
        evaluation_manifest_path, evaluation_manifest["data_manifest_path"]
    )
    data_manifest = read_json(data_manifest_path, "DATA_INVALID")
    if (data_manifest.get("task_id"), data_manifest.get("data_version")) != identity:
        raise ValueError("B data_manifest 与 C result 的 task_id/data_version 不一致")
    classes = data_manifest.get("classes")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or [item.get("id") if isinstance(item, dict) else None for item in classes]
        != list(range(len(classes)))
        or result.get("class_ids") != list(range(len(classes)))
    ):
        raise ValueError("B classes 与 C class_ids 必须为相同的 0..K-1 序列")

    labels_path = _resolve_manifest_file(
        evaluation_manifest_path, evaluation_manifest["labels_path"]
    )
    label_rows = _read_csv(labels_path, ["sample_id", "y_true"])
    labels: dict[str, int] = {}
    class_ids = result["class_ids"]
    for line_number, row in enumerate(label_rows, start=2):
        sample_id, label_text = row
        if not sample_id.strip() or sample_id in labels:
            raise ValueError(f"标签文件第 {line_number} 行 sample_id 为空或重复")
        try:
            label = int(label_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"标签文件第 {line_number} 行 y_true 必须是整数") from exc
        if str(label) != label_text or label not in class_ids:
            raise ValueError(f"标签文件第 {line_number} 行 y_true 不是有效类别整数")
        labels[sample_id] = label

    prediction_ids = [row["sample_id"] for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("C 预测中 sample_id 重复")
    if set(prediction_ids) != set(labels):
        missing = len(set(prediction_ids) - set(labels))
        extra = len(set(labels) - set(prediction_ids))
        raise ValueError(
            f"C 预测与 D 标签 sample_id 集合不一致（缺少标签 {missing}，多余标签 {extra}）"
        )
    n_samples = len(predictions)
    if n_samples == 0:
        raise ValueError("验证预测和标签不能为空")
    y_true = [labels[row["sample_id"]] for row in predictions]
    y_pred = [int(row["y_pred"]) for row in predictions]
    probabilities = [
        [float(row[f"p_{class_id}"]) for class_id in class_ids]
        for row in predictions
    ]
    value = score_predictions(metric["name"], y_true, y_pred, probabilities, class_ids)

    evaluation = {
        "schema_version": "1.0",
        "task_id": result["task_id"],
        "data_version": result["data_version"],
        "experiment_id": result["experiment_id"],
        "status": "success",
        "metric": metric["name"],
        "value": value,
        "greater_is_better": metric["greater_is_better"],
        "n_samples": n_samples,
    }
    if output_path is not None:
        write_json(Path(output_path), evaluation, exclusive=True)
    return evaluation
