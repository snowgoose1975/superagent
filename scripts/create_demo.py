"""Generate a standalone synthetic coupon-classification task from the project template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "configs" / "coupon.json"


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_npy_exclusive(path: Path, values: np.ndarray, *, allow_pickle: bool) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=allow_pickle)


def _labels(count: int, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    # Keep class counts within one sample of each other and shuffle with a fixed seed.
    result = np.arange(count, dtype=np.int64) % n_classes
    rng.shuffle(result)
    return result


def _features(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_rows = len(labels)
    values = np.empty((n_rows, 21), dtype=object)
    levels = [
        [55, 65, 80, 95],
        ["7AM", "10AM", "2PM", "6PM"],
        ["coffee_house", "restaurant", "carry_out"],
        ["2h", "1d"],
        ["Female", "Male"],
        ["21", "26", "31", "41", "50plus"],
        ["Single", "Married partner", "Unmarried partner"],
        ["0", "1"],
        ["Some college", "Bachelor degree", "Graduate degree"],
        ["Student", "Office", "Service", "Other"],
        ["$25000 - $37499", "$37500 - $49999", "$50000 - $62499"],
        ["never", "less1", "1~3", "gt8"],
        ["never", "less1", "1~3", "gt8"],
        ["never", "less1", "1~3", "gt8"],
        ["never", "less1", "1~3", "gt8"],
        ["never", "less1", "1~3", "gt8"],
        ["0", "1"],
        ["0", "1"],
        ["0", "1"],
        ["0", "1"],
        ["0", "1"],
    ]
    for column, choices in enumerate(levels):
        for row in range(n_rows):
            values[row, column] = choices[int(rng.integers(0, len(choices)))]

    # Include numeric categories, a learnable signal, and imperfectly correlated features.
    values[:, 0] = (55 + 10 * labels + rng.integers(0, 4, size=n_rows)).tolist()
    for row, label in enumerate(labels):
        values[row, 2] = "coffee_house" if int(label) % 2 else "restaurant"
    # Use NumPy NaN for missing values so the static reader recognizes them.
    for row in range(n_rows):
        if row % 17 == 0:
            values[row, 10] = float("nan")
        if row % 23 == 0:
            values[row, 5] = float("nan")
    return values


def create_demo(
    output_root: Path,
    *,
    n_train: int = 120,
    n_valid: int = 40,
    n_classes: int = 2,
) -> Path:
    """Create new data and a task config and return the absolute path to demo.json."""
    output_root = Path(output_root).expanduser().resolve()
    if type(n_classes) is not int or n_classes not in (2, 3):
        raise ValueError("n_classes must be 2 or 3")
    if type(n_train) is not int or n_train < n_classes:
        raise ValueError("n_train must be a positive integer with at least one sample per class")
    if type(n_valid) is not int or n_valid < n_classes:
        raise ValueError("n_valid must be a positive integer with at least one sample per class")

    raw_dir = output_root / "data" / "raw" / "synthetic_coupon"
    config_dir = output_root / "configs"
    config_path = config_dir / "demo.json"
    if raw_dir.exists() or config_path.exists():
        raise FileExistsError("The synthetic task directory or configs/demo.json already exists")

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    if len(template.get("feature_columns", [])) != 21:
        raise ValueError("The coupon.json template must declare 21 columns to generate matching samples")

    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=False)
    seed = 20261110 + n_classes * 1009 + n_train * 7 + n_valid
    rng = np.random.default_rng(seed)
    y_train = _labels(n_train, n_classes, rng)
    y_valid = _labels(n_valid, n_classes, rng)
    X_train = _features(y_train, rng)
    X_valid = _features(y_valid, rng)
    _write_npy_exclusive(raw_dir / "C_train.npy", X_train, allow_pickle=True)
    _write_npy_exclusive(raw_dir / "C_val.npy", X_valid, allow_pickle=True)
    _write_npy_exclusive(raw_dir / "y_train.npy", y_train, allow_pickle=False)
    _write_npy_exclusive(raw_dir / "y_val.npy", y_valid, allow_pickle=False)

    template.update(
        task_id="synthetic_coupon",
        data_version="demo_v1",
        source_root="../data/raw/synthetic_coupon",
        label_values=list(range(n_classes)),
        label_names=[f"class{index}" for index in range(n_classes)],
        partitions={
            "train": {
                "features_path": "C_train.npy",
                "labels_path": "y_train.npy",
                "n_samples": n_train,
            },
            "validation": {
                "features_path": "C_val.npy",
                "labels_path": "y_val.npy",
                "n_samples": n_valid,
            },
        },
        train_policy={"max_trials": 3, "random_seed": 42, "timeout_seconds": 120},
    )
    _write_json_exclusive(config_path, template)
    return config_path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-classes", type=int, choices=(2, 3), default=2)
    args = parser.parse_args(argv)
    print(create_demo(args.output_root, n_classes=args.n_classes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
