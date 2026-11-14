"""Read and structurally validate a successful C prediction artifact.

This module deliberately does not read validation labels or compute metrics.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from executor_c.contracts import read_json, validate_identity


def load_successful_predictions(
    result_relpath: str | Path, runs_root: str | Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Load a successful result and its predictions without scoring them."""
    root = Path(runs_root).expanduser().resolve(strict=True)
    rel_text = str(result_relpath)
    rel = PurePosixPath(rel_text)
    if (
        "\\" in rel_text
        or rel.is_absolute()
        or PureWindowsPath(rel_text).drive
        or ".." in rel.parts
    ):
        raise ValueError("result_relpath 必须是 runs_root 内的相对路径")
    if rel.name != "result.json" or len(rel.parts) != 4:
        raise ValueError(
            "result_relpath 必须为 <task_id>/<data_version>/<experiment_id>/result.json"
        )
    result_path = (root / Path(*rel.parts)).resolve(strict=True)
    if not result_path.is_relative_to(root) or not result_path.is_file():
        raise ValueError("result.json 路径越出 runs_root 或不是文件")
    result = read_json(result_path, error_code="DATA_INVALID")
    if not isinstance(result, dict) or result.get("schema_version") != "1.0":
        raise ValueError("result.json 结构或 schema_version 不受支持")
    if result.get("status") != "success":
        raise ValueError(f"只可读取 success 结果，当前状态为 {result.get('status')!r}")
    identity = {key: result.get(key) for key in ("task_id", "data_version", "experiment_id")}
    if validate_identity(identity) != tuple(rel.parts[:3]):
        raise ValueError("result_relpath 与 result.json 中的运行身份不一致")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("predictions_path"), str):
        raise ValueError("成功结果缺少 artifacts.predictions_path")
    prediction_text = artifacts["predictions_path"]
    prediction_rel = PurePosixPath(prediction_text)
    if (
        "\\" in prediction_text
        or prediction_rel.is_absolute()
        or PureWindowsPath(prediction_text).drive
        or ".." in prediction_rel.parts
    ):
        raise ValueError("predictions_path 必须相对 result.json 目录且不得越界")
    prediction_path = (result_path.parent / Path(*prediction_rel.parts)).resolve(strict=True)
    if (
        not prediction_path.is_relative_to(result_path.parent.resolve())
        or not prediction_path.is_file()
    ):
        raise ValueError("预测文件路径越出本次实验目录或不是文件")

    class_ids = result.get("class_ids")
    if (
        not isinstance(class_ids, list)
        or len(class_ids) < 2
        or any(type(value) is not int for value in class_ids)
        or class_ids != list(range(len(class_ids)))
    ):
        raise ValueError("result.json 的 class_ids 必须为从 0 开始的整数序列")
    expected_header = ["sample_id", "y_pred", *(f"p_{class_id}" for class_id in class_ids)]
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_header:
            raise ValueError(f"预测 CSV 列必须为 {expected_header!r}")
        rows = list(reader)
    if not rows:
        raise ValueError("预测 CSV 不能没有数据行")
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "")
        if not sample_id or not sample_id.strip() or sample_id in seen:
            raise ValueError(f"第 {line_number} 行 sample_id 为空或重复")
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"第 {line_number} 行列数与 CSV 表头不一致")
        seen.add(sample_id)
        try:
            prediction_text = row["y_pred"]
            prediction = int(prediction_text)
            if str(prediction) != prediction_text:
                raise ValueError("y_pred 不是规范整数文本")
            probabilities = [float(row[f"p_{class_id}"]) for class_id in class_ids]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"第 {line_number} 行预测值不是有效数字") from exc
        if prediction not in class_ids or not all(
            math.isfinite(p) and 0 <= p <= 1 for p in probabilities
        ):
            raise ValueError(f"第 {line_number} 行类别或概率超出允许范围")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError(f"第 {line_number} 行概率和不为 1")
    if type(result.get("n_predictions")) is not int or result["n_predictions"] <= 0:
        raise ValueError("result.json 的 n_predictions 必须是正整数")
    if result["n_predictions"] != len(rows):
        raise ValueError("预测行数与 result.json 的 n_predictions 不一致")
    return result, rows
