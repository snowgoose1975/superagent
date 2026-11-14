"""B 的表格预处理器：语义由任务声明，所有学习转换只拟合训练行。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


class TablePreprocessor:
    """按固定列顺序处理数值/类别特征并保留稳定、全局唯一输出名称。"""

    def __init__(self, feature_types: dict[str, str], max_features: int = 4096):
        self.feature_types = dict(feature_types)
        self.max_features = max_features

    def fit(self, frame: pd.DataFrame) -> "TablePreprocessor":
        if frame.empty or not frame.columns.is_unique:
            raise ValueError("训练特征必须有样本且列名唯一")
        self.original_feature_names_ = list(frame.columns)
        if set(self.feature_types) != set(self.original_feature_names_):
            raise ValueError("feature_types 必须与训练特征列完全一致")
        if any(v not in {"categorical", "numeric"} for v in self.feature_types.values()):
            raise ValueError("只支持 categorical 和 numeric 特征类型")
        self.numeric_ = [
            c for c in self.original_feature_names_ if self.feature_types[c] == "numeric"
        ]
        self.categorical_ = [
            c for c in self.original_feature_names_ if self.feature_types[c] == "categorical"
        ]
        estimate = len(self.numeric_) + sum(
            frame[c].nunique(dropna=True) + int(frame[c].isna().any()) for c in self.categorical_
        )
        if estimate > self.max_features:
            raise ValueError(f"稠密特征上限超出：{estimate} > {self.max_features}")
        transforms = []
        if self.numeric_:
            transforms.append(
                ("num", SimpleImputer(strategy="median", keep_empty_features=True), self.numeric_)
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
                            handle_unknown="ignore", sparse_output=False, dtype=np.float64
                        ),
                    ),
                    self.categorical_,
                )
            )
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
        return frame.loc[:, self.original_feature_names_]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        output = np.asarray(self.transformer_.transform(self._ordered(frame)), dtype=np.float64)
        if output.ndim != 2 or not np.isfinite(output).all():
            raise ValueError("预处理结果必须是有限二维数组")
        return output

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
