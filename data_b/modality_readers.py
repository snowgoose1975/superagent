"""Strict NPY readers for declared text and image classification inputs."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from .npy_reader import read_npy_array

MAX_TEXT_COLUMNS = 8
MAX_TEXT_CHARACTERS = 100_000
MAX_IMAGE_PIXELS = 1_000_000
MAX_TEXT_CORPUS_CHARACTERS = 8_000_000
MAX_TEXT_JSONL_BYTES = 128_000_000


def load_text_matrix(
    path: str | Path, text_columns: list[str], *, max_characters: int = MAX_TEXT_CHARACTERS
) -> np.ndarray:
    if (
        not isinstance(text_columns, list)
        or not 1 <= len(text_columns) <= MAX_TEXT_COLUMNS
        or any(not isinstance(name, str) or not name.strip() for name in text_columns)
        or len(set(text_columns)) != len(text_columns)
    ):
        raise ValueError("text_columns 必须是 1–8 个唯一非空列名")
    values = read_npy_array(path)
    if values.ndim == 1 and len(text_columns) == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != len(text_columns):
        raise ValueError("文本数组必须是与 text_columns 匹配的二维矩阵")
    if type(max_characters) is not int or not 1 <= max_characters <= MAX_TEXT_CHARACTERS:
        raise ValueError("max_characters 超出允许范围")
    output = np.empty(values.shape, dtype=object)
    total_characters = 0
    for index, raw in np.ndenumerate(values):
        if isinstance(raw, np.generic):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="strict")
        if type(raw) is not str or "\0" in raw or len(raw) > max_characters:
            raise ValueError("文本元素必须为合法 UTF-8 字符串且不超过最大长度")
        total_characters += len(raw)
        if total_characters > MAX_TEXT_CORPUS_CHARACTERS:
            raise ValueError("文本分区总字符数超过 8000000 的安全上限")
        output[index] = raw
    return output


def write_text_matrix_jsonl(path: str | Path, values: np.ndarray) -> None:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.dtype.kind not in "OUS":
        raise ValueError("文本 JSONL 输入必须是二维字符串矩阵")
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in matrix:
            stream.write(json.dumps(list(row), ensure_ascii=False, allow_nan=False) + "\n")


def read_text_matrix_jsonl(
    path: str | Path,
    *,
    n_samples: int,
    text_columns: list[str],
    max_characters: int,
) -> np.ndarray:
    if (
        type(n_samples) is not int
        or n_samples < 1
        or not isinstance(text_columns, list)
        or not 1 <= len(text_columns) <= MAX_TEXT_COLUMNS
        or type(max_characters) is not int
        or not 1 <= max_characters <= MAX_TEXT_CHARACTERS
    ):
        raise ValueError("文本 JSONL 读取约束非法")
    if Path(path).stat().st_size > MAX_TEXT_JSONL_BYTES:
        raise ValueError("文本 JSONL 超过 128 MB 的安全上限")
    rows: list[list[str]] = []
    total_characters = 0
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for line in stream:
            if len(rows) >= n_samples or len(line) > len(text_columns) * max_characters * 6 + 4096:
                raise ValueError("文本 JSONL 行数或单行字节数超出清单上限")
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError("文本 JSONL 存在非法行") from exc
            if (
                not isinstance(row, list)
                or len(row) != len(text_columns)
                or any(type(item) is not str or "\0" in item or len(item) > max_characters for item in row)
            ):
                raise ValueError("文本 JSONL 行与 modality schema 不一致")
            total_characters += sum(map(len, row))
            if total_characters > MAX_TEXT_CORPUS_CHARACTERS:
                raise ValueError("文本分区总字符数超过 8000000 的安全上限")
            rows.append(row)
    if len(rows) != n_samples:
        raise ValueError("文本 JSONL 行数与 Manifest 不一致")
    return np.asarray(rows, dtype=object)


def combine_text_columns(values: np.ndarray, text_columns: list[str]) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != len(text_columns) or matrix.dtype.kind not in "OUS":
        raise ValueError("raw_text 输入 shape/dtype 与 text_columns 不一致")
    rows = [
        "\n".join(f"{name}: {value}" for name, value in zip(text_columns, row, strict=True))
        for row in matrix
    ]
    return np.asarray(rows, dtype=object)


def validate_image_spec(
    image_shape: Any, channel_layout: Any, value_range: Any
) -> tuple[list[int], str, tuple[float, float]]:
    if (
        not isinstance(image_shape, list)
        or len(image_shape) not in (2, 3)
        or any(type(size) is not int or not 1 <= size <= 4096 for size in image_shape)
        or math.prod(image_shape) > MAX_IMAGE_PIXELS
    ):
        raise ValueError("image_shape 必须是合法 HW/HWC 且单张不超过一百万像素")
    expected_layout = "HW" if len(image_shape) == 2 else "HWC"
    if channel_layout != expected_layout:
        raise ValueError(f"channel_layout 必须为 {expected_layout}")
    if (
        not isinstance(value_range, list)
        or len(value_range) != 2
        or any(type(item) not in (int, float) or not math.isfinite(item) for item in value_range)
        or value_range[0] >= value_range[1]
    ):
        raise ValueError("value_range 必须含两个递增的有限数")
    return list(image_shape), expected_layout, (float(value_range[0]), float(value_range[1]))


def load_image_tensor(
    path: str | Path,
    image_shape: list[int],
    channel_layout: str,
    value_range: list[float] | tuple[float, float],
) -> np.ndarray:
    shape, _, limits = validate_image_spec(image_shape, channel_layout, list(value_range))
    values = read_npy_array(path)
    if values.ndim != len(shape) + 1 or tuple(values.shape[1:]) != tuple(shape):
        raise ValueError("image NPY shape 必须是 (N, *image_shape)")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("image NPY 必须是有限整数或浮点像素")
    if (values < limits[0]).any() or (values > limits[1]).any():
        raise ValueError("图像像素超出声明的 value_range")
    return np.asarray(values, dtype=np.float32)
