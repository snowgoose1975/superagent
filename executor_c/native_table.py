"""Raw tabular features shared by native categorical training and inference."""
from pathlib import Path
from typing import Any

import pandas as pd

from executor_c.contracts import read_json, sha256_file


def native_features(
    frame: pd.DataFrame, kinds: dict[str, str], *,
    missing_values: dict[str, list[int | float]] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    for column, kind in kinds.items():
        if kind == "categorical":
            result[column] = result[column].map(
                lambda value: "missing" if pd.isna(value) else f"value:{value}"
            )
        else:
            numbers = pd.to_numeric(result[column], errors="raise").astype(float)
            markers = (missing_values or {}).get(column, [])
            result[column] = numbers.mask(numbers.isin(markers))
    return result


def load_native_tables(
    directory: Path, n_train: int, n_validation: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    hashes = read_json(directory / "package_result.json")["artifact_sha256"]
    paths = [
        directory / "metadata" / "profile.json",
        directory / "native_train.json",
        directory / "native_validation.json",
    ]
    for path in paths:
        if sha256_file(path) != hashes.get(path.relative_to(directory).as_posix()):
            raise ValueError("native table does not match the prepared data package")
    profile = read_json(paths[0])
    columns = profile["feature_columns"]
    kinds = profile["feature_types"]

    def load(path: Path, expected_rows: int) -> pd.DataFrame:
        payload: dict[str, Any] = read_json(path)
        if payload["columns"] != columns or len(payload["data"]) != expected_rows:
            raise ValueError("native table does not match the partition schema")
        return native_features(
            pd.DataFrame(payload["data"], columns=columns), kinds,
            missing_values=profile.get("preprocessing", {}).get("missing_values", {}),
        )

    categories = [column for column in columns if kinds[column] == "categorical"]
    return load(paths[1], n_train), load(paths[2], n_validation), categories
