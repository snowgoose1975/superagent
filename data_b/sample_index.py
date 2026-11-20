"""Optional, versioned sample provenance without changing the C manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path, PureWindowsPath
from typing import Any


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sample_index(root: Path, part: dict[str, Any], split: str) -> tuple[list[dict], dict]:
    relative = part.get("sample_ids_path")
    if relative is None:
        return [
            {"sample_id": f"{split}_{i:09d}", "source_file": part["features_path"],
             "source_row": str(i), "original_id": ""}
            for i in range(part["n_samples"])
        ], {}
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or PureWindowsPath(relative).drive or ".." in Path(relative).parts):
        raise ValueError("sample_ids_path 必须为 source_root 内的相对路径")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("sample_ids_path 越出 source_root 或不是文件")
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("样本索引超过 128 MiB 上限")
    before = _digest(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"sample_id", "source_file", "source_row", "original_id"}
        if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not {"sample_id"} <= set(reader.fieldnames) <= expected):
            raise ValueError("样本索引必须包含 sample_id，且仅含支持的来源字段")
        rows = []
        for i, row in enumerate(reader):
            if i >= part["n_samples"] or None in row or any(value is None for value in row.values()):
                raise ValueError("样本索引行数或行宽错误")
            sample_id = row["sample_id"]
            if not sample_id.strip() or any(c in sample_id for c in "\r\n\0"):
                raise ValueError("样本 ID 必须为非空单行字符串")
            source_row = row.get("source_row", str(i))
            if not source_row.isascii() or not source_row.isdecimal():
                raise ValueError("source_row 必须为从零开始的非负行号")
            rows.append({"sample_id": sample_id,
                         "source_file": row.get("source_file", part["features_path"]),
                         "source_row": source_row,
                         "original_id": row.get("original_id", "")})
    if len(rows) != part["n_samples"] or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("样本索引行数不一致或 ID 重复")
    after = _digest(path)
    if before != after:
        raise ValueError("样本索引在读取期间发生变化")
    return rows, {"sample_index": {"relative_path": relative, "sha256": after}}


def save_sample_index(path: Path, rows: list[dict], split: str | None = None) -> None:
    columns = ["sample_id", "source_file", "source_row", "original_id"]
    if split is not None:
        columns = ["split", *columns]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(({"split": split, **row} if split is not None else row))


def save_combined_index(path: Path, train: list[dict], validation: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "sample_id", "source_file", "source_row", "original_id"])
        writer.writeheader()
        for split, rows in (("train", train), ("validation", validation)):
            for row in rows:
                writer.writerow({"split": split, **row})
