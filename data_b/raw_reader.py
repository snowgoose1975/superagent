"""按任务显式声明读取原始特征；不读取标签、test 或推断列语义。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .npy_reader import read_npy_array


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return False


def typed_category_token(value: Any) -> str | float:
    """把类别值编码成带类型的稳定字符串，避免数字 1 与字符串“1”碰撞。"""
    if _is_missing(value):
        return np.nan
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is bool:
        kind, text = "bool", "true" if value else "false"
    elif type(value) is int:
        kind, text = "int", str(value)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("分类特征不允许无穷值")
        kind, text = "float", value.hex()
    elif type(value) is str:
        kind, text = "str", value
    elif isinstance(value, bytes):
        kind, text = "bytes", value.hex()
    else:
        raise ValueError(f"不支持的类别标量类型：{type(value).__name__}")
    return json.dumps([kind, text], ensure_ascii=False, separators=(",", ":"))


def _numeric_column(values: np.ndarray, name: str) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        if _is_missing(value):
            output[index] = np.nan
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if type(value) is bool or not isinstance(value, (int, float, str)):
            raise ValueError(f"数值特征 {name!r} 含不支持的值类型")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"数值特征 {name!r} 含空字符串；缺失值必须使用 None/NaN")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"数值特征 {name!r} 含非数值内容") from exc
        if not math.isfinite(number):
            raise ValueError(f"数值特征 {name!r} 含无穷值")
        output[index] = number
    return output


def _read_matrix(path: Path, feature_columns: list[str]) -> np.ndarray:
    values = read_npy_array(path)
    if values.ndim != 2 or values.shape[1] != len(feature_columns):
        raise ValueError(f"特征数组 {path.name} 的列数/维度与 feature_columns 不符")
    return values


def load_feature_frame(
    features_path: str | Path,
    feature_columns: list[str],
    feature_types: dict[str, str],
) -> pd.DataFrame:
    """读取单个特征 NPY 并按声明转换；不会访问其配对标签或其它分区。"""
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns 必须是非空且不重复的列名列表")
    if set(feature_types) != set(feature_columns):
        raise ValueError("feature_types 必须且只能覆盖 feature_columns")
    if any(value not in {"categorical", "numeric"} for value in feature_types.values()):
        raise ValueError("feature_types 的值必须为 categorical 或 numeric")
    matrix = _read_matrix(Path(features_path), feature_columns)
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(feature_columns):
        values = matrix[:, index]
        if feature_types[name] == "numeric":
            columns[name] = _numeric_column(values, name)
        else:
            columns[name] = np.asarray(
                [typed_category_token(value) for value in values], dtype=object
            )
    return pd.DataFrame(columns, columns=feature_columns)
