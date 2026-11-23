"""Read raw features according to the task's declared column types."""

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
    """Encode categories as stable typed strings so numeric 1 and textual 1 are distinct."""
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
            raise ValueError("Categorical features cannot contain infinity")
        kind, text = "float", value.hex()
    elif type(value) is str:
        kind, text = "str", value
    elif isinstance(value, bytes):
        kind, text = "bytes", value.hex()
    else:
        raise ValueError(f"Unsupported categorical scalar type: {type(value).__name__}")
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
            raise ValueError(f"Numeric feature {name!r} contains an unsupported value type")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Numeric feature {name!r} contains an empty string. Use None or NaN for missing values.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Numeric feature {name!r} contains a nonnumeric value") from exc
        if not math.isfinite(number):
            raise ValueError(f"Numeric feature {name!r} contains infinity")
        output[index] = number
    return output


def _read_matrix(path: Path, feature_columns: list[str]) -> np.ndarray:
    values = read_npy_array(path)
    if values.ndim != 2 or values.shape[1] != len(feature_columns):
        raise ValueError(f"Feature array {path.name} dimensions do not match feature_columns")
    return values


def load_feature_frame(
    features_path: str | Path,
    feature_columns: list[str],
    feature_types: dict[str, str],
) -> pd.DataFrame:
    """Read one NPY feature file and apply its declared column conversions."""
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must contain unique nonempty names")
    if set(feature_types) != set(feature_columns):
        raise ValueError("feature_types must cover exactly feature_columns")
    if any(value not in {"categorical", "numeric", "text"} for value in feature_types.values()):
        raise ValueError("feature_types values must be categorical, numeric, or text")
    matrix = _read_matrix(Path(features_path), feature_columns)
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(feature_columns):
        values = matrix[:, index]
        if feature_types[name] == "numeric":
            columns[name] = _numeric_column(values, name)
        elif feature_types[name] == "text":
            text = []
            for value in values:
                if _is_missing(value):
                    value = ""
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="strict")
                if not isinstance(value, str) or "\0" in value or len(value) > 100_000:
                    raise ValueError(f"Text feature {name!r} must be a string within the allowed bounds")
                text.append(value)
            columns[name] = np.asarray(text, dtype=object)
        else:
            columns[name] = np.asarray(
                [typed_category_token(value) for value in values], dtype=object
            )
    return pd.DataFrame(columns, columns=feature_columns)
