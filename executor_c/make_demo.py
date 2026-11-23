"""Generate a frozen, synthetic B-style data package for local demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import uuid
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def _safe_id(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) or ".." in value:
        raise ValueError(f"{field} must contain 1–64 ASCII letters, digits, underscores, or hyphens and exclude '..'")
    return value


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def make_demo(
    data_root: str | Path,
    *,
    task_id: str | None = None,
    data_version: str = "v1",
    n_classes: int = 2,
    random_seed: int = 42,
) -> Path:
    """Create an exclusive demo package and return its absolute manifest path.

    The generated validation labels are for a separate mock D workflow. C's
    manifest intentionally does not reference them.
    """
    task_id = _safe_id(task_id or f"demo_{uuid.uuid4().hex[:12]}", "task_id")
    data_version = _safe_id(data_version, "data_version")
    if type(n_classes) is not int or n_classes not in (2, 3):
        raise ValueError("n_classes must be 2 or 3")
    if type(random_seed) is not int or not 0 <= random_seed <= 2**32 - 1:
        raise ValueError("random_seed must be between 0 and 2**32 - 1")

    package = Path(data_root).expanduser().resolve() / task_id / data_version
    package.mkdir(parents=True, exist_ok=False)
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_seed,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X.astype(np.float64),
        y.astype(np.int64),
        test_size=0.25,
        random_state=random_seed,
        stratify=y,
    )
    for name, array in (
        ("X_train.npy", X_train),
        ("y_train.npy", y_train),
        ("X_valid.npy", X_valid),
    ):
        with (package / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)

    ids = [f"valid_{index:06d}" for index in range(1, len(y_valid) + 1)]
    with (package / "valid_ids.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id"])
        writer.writerows([[sample_id] for sample_id in ids])
    with (package / "validation_labels.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "y_true"])
        writer.writerows(
            [[sample_id, int(label)] for sample_id, label in zip(ids, y_valid, strict=True)]
        )

    classes = [{"id": i, "name": f"class_{i}"} for i in range(n_classes)]
    _write_json(
        package / "data_manifest.json",
        {
            "schema_version": "1.0",
            "task_id": task_id,
            "data_version": data_version,
            "problem_type": "classification",
            "n_features": 8,
            "feature_names": [f"feature_{i}" for i in range(8)],
            "classes": classes,
            "train": {
                "features_path": "X_train.npy",
                "labels_path": "y_train.npy",
                "n_samples": len(y_train),
            },
            "validation": {
                "features_path": "X_valid.npy",
                "sample_ids_path": "valid_ids.csv",
                "n_samples": len(y_valid),
            },
            "preprocessor_path": None,
        },
    )
    _write_json(
        package / "evaluation_manifest.json",
        {
            "schema_version": "1.0",
            "task_id": task_id,
            "data_version": data_version,
            "data_manifest_path": "data_manifest.json",
            "labels_path": "validation_labels.csv",
            "primary_metric": {"name": "accuracy", "greater_is_better": True},
            "positive_class_id": 1 if n_classes == 2 else None,
        },
    )
    return (package / "data_manifest.json").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic data package for the executor demo")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--n-classes", type=int, choices=(2, 3), default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    print(
        make_demo(
            args.data_root,
            task_id=args.task_id,
            n_classes=args.n_classes,
            random_seed=args.random_seed,
        )
    )


if __name__ == "__main__":
    main()
