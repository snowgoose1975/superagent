"""B 的表格预处理器：语义由任务声明，所有学习转换只拟合训练行。"""

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
        raise ValueError("未知预处理组件或参数")
    if options.get("numeric_imputation", "median") not in {"median", "mean", "most_frequent"}:
        raise ValueError("不支持的数值填补策略")
    if type(options.get("add_missing_indicator", False)) is not bool:
        raise ValueError("add_missing_indicator 必须是布尔值")
    frequency = options.get("categorical_min_frequency")
    if frequency is not None and (type(frequency) is not int or not 2 <= frequency <= 1000):
        raise ValueError("低频类别阈值必须是 2–1000 的整数")
    markers = options.get("missing_values", {})
    if not isinstance(markers, dict) or len(markers) > 32:
        raise ValueError("missing_values 必须为有界字段映射")
    for column, values in markers.items():
        if feature_types.get(column) != "numeric" or not isinstance(values, list) or not 1 <= len(values) <= 8:
            raise ValueError("首版特殊缺失标记仅支持数值特征，每列最多 8 个值")
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in values):
            raise ValueError("数值缺失标记必须是有限数字")
    return dict(options)


def _text_documents(values):
    documents = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        text = value if isinstance(value, str) else ""
        documents.append(text if re.search(r"\w", text) else "__b_empty_text__")
    return documents


class TablePreprocessor:
    """按固定列顺序处理数值/类别特征并保留稳定、全局唯一输出名称。"""

    def __init__(self, feature_types: dict[str, str], max_features: int = 4096, options: dict | None = None):
        self.feature_types = dict(feature_types)
        self.max_features = max_features
        self.options = validate_preprocessing(options, feature_types)

    def fit(self, frame: pd.DataFrame) -> "TablePreprocessor":
        if frame.empty or not frame.columns.is_unique:
            raise ValueError("训练特征必须有样本且列名唯一")
        self.original_feature_names_ = list(frame.columns)
        if set(self.feature_types) != set(self.original_feature_names_):
            raise ValueError("feature_types 必须与训练特征列完全一致")
        if any(v not in {"categorical", "numeric", "text"} for v in self.feature_types.values()):
            raise ValueError("只支持 categorical、numeric 和 text 特征类型")
        self.numeric_ = [
            c for c in self.original_feature_names_ if self.feature_types[c] == "numeric"
        ]
        self.categorical_ = [
            c for c in self.original_feature_names_ if self.feature_types[c] == "categorical"
        ]
        self.text_ = [c for c in self.original_feature_names_ if self.feature_types[c] == "text"]
        frame = self._ordered(frame)
        options = getattr(self, "options", {})
        frequency = options.get("categorical_min_frequency")
        base_features = len(self.numeric_)
        if options.get("add_missing_indicator", False):
            base_features += sum(frame[c].isna().any() for c in self.numeric_)
        for column in self.categorical_:
            counts = frame[column].fillna("__B_MISSING__").value_counts()
            base_features += int((counts >= frequency).sum()) + int((counts < frequency).any()) if frequency else len(counts)
        self.text_max_features_ = min(512, (self.max_features - base_features) // len(self.text_)) if self.text_ else 0
        if self.text_ and self.text_max_features_ < 1:
            raise ValueError("稠密特征预算不足以保留每个文本字段")
        estimate = base_features + self.text_max_features_ * len(self.text_)
        if estimate > self.max_features:
            raise ValueError(f"稠密特征上限超出：{estimate} > {self.max_features}")
        if len(frame) * estimate * 8 > 256 * 1024 * 1024:
            raise ValueError("稠密预处理估算超过 256 MiB；需要稀疏或低维表示")
        transforms = []
        if self.numeric_:
            transforms.append(
                ("num", SimpleImputer(strategy=options.get("numeric_imputation", "median"),
                                      add_indicator=options.get("add_missing_indicator", False),
                                      keep_empty_features=True), self.numeric_)
            )
        if self.categorical_:
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
            raise ValueError("生成的特征名重复或稠密特征上限超出")
        return self

    def _ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.original_feature_names_) - set(frame.columns)
        if missing:
            raise ValueError(f"缺少输入特征：{sorted(missing)}")
        ordered = frame.loc[:, self.original_feature_names_].copy()
        for column, markers in getattr(self, "options", {}).get("missing_values", {}).items():
            ordered[column] = ordered[column].mask(ordered[column].isin(markers), np.nan)
        return ordered

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if len(frame) * len(self.feature_names_) * 8 > 256 * 1024 * 1024:
            raise ValueError("稠密转换估算超过 256 MiB；需要分批或低维表示")
        output = np.asarray(self.transformer_.transform(self._ordered(frame)), dtype=np.float64)
        if output.ndim != 2 or not np.isfinite(output).all():
            raise ValueError("预处理结果必须是有限二维数组")
        return output

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
