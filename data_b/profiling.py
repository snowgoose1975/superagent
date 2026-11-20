"""Aggregate training-only statistics; never sample raw rows or validation labels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def table_profile(frame: pd.DataFrame, feature_types: dict[str, str]) -> dict:
    fields = []
    for name in frame:
        values = frame[name]
        missing = values.isna()
        unique = int(values.nunique(dropna=True))
        field = {"name": name, "kind": feature_types[name], "missing_count": int(missing.sum()),
                 "missing_fraction": float(missing.mean()), "unique_count": unique,
                 "constant": unique <= 1}
        counts = values.value_counts(dropna=True)
        field["rare_values_le_2"] = int((counts <= 2).sum())
        if feature_types[name] == "numeric":
            observed = values.dropna().to_numpy(dtype=float)
            field["numeric"] = ({"min": float(observed.min()), "max": float(observed.max()),
                                  "median": float(np.median(observed))} if len(observed) else None)
        if feature_types[name] == "text":
            field["length"] = length_statistics(values.fillna("").tolist())
        fields.append(field)
    # Detailed fields stay on disk. A receives a bounded summary and selected fields.
    total = sum(field["missing_count"] for field in fields)
    return {"column_statistics": fields,
            "quality_summary": {"missing_fraction": total / max(1, frame.size),
                                "columns_with_missing": sum(f["missing_count"] > 0 for f in fields),
                                "constant_columns": sum(f["constant"] for f in fields),
                                "high_cardinality_columns": sum(f["unique_count"] > 100 for f in fields),
                                "duplicate_train_rows": int(frame.duplicated().sum())}}


def length_statistics(values) -> dict:
    lengths = np.asarray([len(value) for value in values], dtype=np.int64)
    return {"mean": float(lengths.mean()), "p50": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)), "max": int(lengths.max()),
            "empty_count": int((lengths == 0).sum())}


def modality_profile(values: np.ndarray, modality: dict) -> dict:
    if modality["kind"] == "text":
        fields = [{"name": name, "kind": "text", "length": length_statistics(values[:, i])}
                  for i, name in enumerate(modality["text_columns"])]
        return {"column_statistics": fields,
                "quality_summary": {"empty_text_cells": sum(f["length"]["empty_count"] for f in fields)}}
    return {"modality": modality,
            "quality_summary": {"pixel_min": float(np.min(values)),
                                "pixel_max": float(np.max(values)),
                                "pixel_mean": float(np.mean(values, dtype=np.float64))}}
