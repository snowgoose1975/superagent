"""Fixed, picklable representation transforms used by Plan 2.0 pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ImageArrayFlattener(TransformerMixin, BaseEstimator):
    """Flatten declared HW/HWC tensors and scale the declared pixel interval to [0, 1]."""

    def __init__(self, value_min: float, value_max: float):
        self.value_min = value_min
        self.value_max = value_max

    def fit(self, X: np.ndarray, y: Any = None) -> "ImageArrayFlattener":
        self._validate(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = self._validate(X)
        flattened = values.reshape(len(values), -1).astype(np.float32, copy=False)
        return (flattened - self.value_min) / (self.value_max - self.value_min)

    def _validate(self, X: Any) -> np.ndarray:
        values = np.asarray(X)
        if (
            values.ndim not in (3, 4)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or not np.isfinite(self.value_min)
            or not np.isfinite(self.value_max)
            or self.value_min >= self.value_max
            or (values < self.value_min).any()
            or (values > self.value_max).any()
        ):
            raise ValueError("image tensor violates the declared shape or pixel range")
        return values
