"""Table preprocessing fitted exclusively on training rows."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def validate_preprocessing(options, feature_types):
    if options is None:
        return {}
    allowed = {"numeric_imputation", "add_missing_indicator", "categorical_min_frequency", "missing_values"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Unknown preprocessing component or parameter")
    if options.get("numeric_imputation", "median") not in {"median", "mean", "most_frequent"}:
        raise ValueError("Unsupported numeric imputation strategy")
    if type(options.get("add_missing_indicator", False)) is not bool:
        raise ValueError("add_missing_indicator must be a boolean")
    frequency = options.get("categorical_min_frequency")
    if frequency is not None and (type(frequency) is not int or not 2 <= frequency <= 1000):
        raise ValueError("categorical_min_frequency must be an integer between 2 and 1000")
    markers = options.get("missing_values", {})
    if not isinstance(markers, dict) or len(markers) > 32:
        raise ValueError("missing_values must be a bounded mapping from columns to markers")
    for column, values in markers.items():
        if feature_types.get(column) != "numeric" or not isinstance(values, list) or not 1 <= len(values) <= 8:
            raise ValueError("Missing markers are supported for numeric columns with at most 8 markers per column")
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in values):
            raise ValueError("Numeric missing markers must be finite numbers")
    return dict(options)


def _text_documents(values):
    documents = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        text = value if isinstance(value, str) else ""
        documents.append(text if re.search(r"\w", text) else "__b_empty_text__")
    return documents


class TablePreprocessor:
    """Transform declared columns in order and keep unique output feature names."""

    def __init__(
        self, feature_types: dict[str, str], max_features: int = 4096,
        options: dict | None = None, *, allow_frequency_fallback: bool = False,
    ):
        self.feature_types = dict(feature_types)
        self.max_features = max_features
        self.allow_frequency_fallback = allow_frequency_fallback
        self.options = validate_preprocessing(options, feature_types)

    def fit(self, frame: pd.DataFrame) -> "TablePreprocessor":
        if frame.empty or not frame.columns.is_unique:
            raise ValueError("Training features require rows and unique column names")
        self.original_feature_names_ = list(frame.columns)
        if set(self.feature_types) != set(self.original_feature_names_):
            raise ValueError("feature_types must match every training feature column")
        if any(v not in {"categorical", "numeric", "text"} for v in self.feature_types.values()):
            raise ValueError("Feature types must be categorical, numeric, or text")
        self.numeric_ = [
            c for c in self.original_feature_names_ if self.feature_types[c] == "numeric"
        ]
        self.categorical_ = [
            c for c in self.original_feature_names_ if self.feature_types[c] == "categorical"
        ]
        self.text_ = [c for c in self.original_feature_names_ if self.feature_types[c] == "text"]
        self.frequency_maps_ = {}
        frame = self._ordered(frame)
        options = getattr(self, "options", {})
        frequency = options.get("categorical_min_frequency")
        base_features = len(self.numeric_)
        if options.get("add_missing_indicator", False):
            base_features += sum(frame[c].isna().any() for c in self.numeric_)
        encoded_categories = 0
        for column in self.categorical_:
            counts = frame[column].fillna("__B_MISSING__").value_counts()
            encoded_categories += int((counts >= frequency).sum()) + int((counts < frequency).any()) if frequency else len(counts)
        base_features += encoded_categories
        minimum = base_features + len(self.text_)
        if (minimum > self.max_features or len(frame) * minimum * 8 > 256 * 1024 * 1024):
            if self.allow_frequency_fallback:
                base_features -= encoded_categories
                for column in self.categorical_:
                    self.frequency_maps_[column] = frame[column].value_counts(normalize=True).to_dict()
                base_features += len(self.frequency_maps_)
        width_limit = min(self.max_features, (256 * 1024 * 1024) // (len(frame) * 8))
        self.text_max_features_ = min(512, (width_limit - base_features) // len(self.text_)) if self.text_ else 0
        if self.text_ and self.text_max_features_ < 1:
            raise ValueError("The dense feature budget cannot retain all text fields")
        estimate = base_features + self.text_max_features_ * len(self.text_)
        if estimate > self.max_features:
            raise ValueError(f"Dense feature limit exceeded: {estimate} > {self.max_features}")
        if len(frame) * estimate * 8 > 256 * 1024 * 1024:
            raise ValueError("Dense training would exceed 256 MiB. Use a lower-dimensional representation")
        transforms = []
        if self.numeric_:
            transforms.append(
                ("num", SimpleImputer(strategy=options.get("numeric_imputation", "median"),
                                      add_indicator=options.get("add_missing_indicator", False),
                                      keep_empty_features=True), self.numeric_)
            )
        if self.frequency_maps_:
            transforms.append(("frequency", "passthrough", self.categorical_))
        elif self.categorical_:
            transforms.append(
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(
                            strategy="constant",
                            fill_value="__B_MISSING__",
                            keep_empty_features=True,
                        ),
                        OneHotEncoder(
                            handle_unknown="infrequent_if_exist" if frequency else "ignore",
                            min_frequency=frequency, sparse_output=False, dtype=np.float64
                        ),
                    ),
                    self.categorical_,
                )
            )
        for index, column in enumerate(self.text_):
            transforms.append((f"text_{index}", make_pipeline(
                FunctionTransformer(_text_documents, validate=False, feature_names_out="one-to-one"),
                TfidfVectorizer(max_features=self.text_max_features_, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True),
            ), [column]))
        self.transformer_ = ColumnTransformer(transforms, remainder="drop", sparse_threshold=0)
        self.transformer_.fit(self._ordered(frame))
        generated = self.transformer_.get_feature_names_out().tolist()
        self.feature_names_ = [f"{index:04d}::{name}" for index, name in enumerate(generated)]
        if (
            len(set(self.feature_names_)) != len(self.feature_names_)
            or len(self.feature_names_) > self.max_features
        ):
            raise ValueError("Generated feature names are duplicated or exceed the dense feature limit")
        return self

    def _ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.original_feature_names_) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")
        ordered = frame.loc[:, self.original_feature_names_].copy()
        for column, markers in getattr(self, "options", {}).get("missing_values", {}).items():
            ordered[column] = ordered[column].mask(ordered[column].isin(markers), np.nan)
        if maps := getattr(self, "frequency_maps_", {}):
            for column, mapping in maps.items():
                ordered[column] = ordered[column].map(mapping).fillna(0.0).astype(float)
        return ordered

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if len(frame) * len(self.feature_names_) * 8 > 256 * 1024 * 1024:
            raise ValueError("Dense transformation would exceed 256 MiB. Use smaller batches")
        output = np.asarray(self.transformer_.transform(self._ordered(frame)), dtype=np.float64)
        if output.ndim != 2 or not np.isfinite(output).all():
            raise ValueError("Preprocessing must return a finite two-dimensional array")
        return output

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
