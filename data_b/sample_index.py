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
        raise ValueError("sample_ids_path must be relative to source_root")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("sample_ids_path must identify a file within source_root")
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("The sample index exceeds the 128 MiB limit")
    before = _digest(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"sample_id", "source_file", "source_row", "original_id"}
        if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not {"sample_id"} <= set(reader.fieldnames) <= expected):
            raise ValueError("The sample index requires sample_id and allows only supported provenance fields")
        rows = []
        for i, row in enumerate(reader):
            if i >= part["n_samples"] or None in row or any(value is None for value in row.values()):
                raise ValueError("The sample index row count or width is invalid")
            sample_id = row["sample_id"]
            if not sample_id.strip() or any(c in sample_id for c in "\r\n\0"):
                raise ValueError("Sample IDs must be nonempty single-line strings")
            source_row = row.get("source_row", str(i))
            if not source_row.isascii() or not source_row.isdecimal():
                raise ValueError("source_row must be a nonnegative zero-based row number")
            rows.append({"sample_id": sample_id,
                         "source_file": row.get("source_file", part["features_path"]),
                         "source_row": source_row,
                         "original_id": row.get("original_id", "")})
    if len(rows) != part["n_samples"] or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("The sample index has an incorrect row count or duplicate IDs")
    after = _digest(path)
    if before != after:
        raise ValueError("The sample index changed during reading")
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
