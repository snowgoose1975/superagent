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


class ImageGradientFeatures(ImageArrayFlattener):
    """Pool local gradient orientations and channel statistics over a fixed grid."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = self._validate(X)
        height, width = values.shape[1:3]
        parts = []
        for start in range(0, len(values), 64):
            batch = values[start:start + 64].astype(np.float32)
            batch = (batch - self.value_min) / (self.value_max - self.value_min)
            if batch.ndim == 3:
                batch = batch[..., None]
            gray = batch[..., :3].mean(axis=-1)
            dx = np.diff(gray, axis=2, append=gray[:, :, -1:])
            dy = np.diff(gray, axis=1, append=gray[:, -1:, :])
            orientation = (np.arctan2(dy, dx) % np.pi) * (9 / np.pi)
            bins = np.minimum(orientation.astype(np.int8), 8)
            magnitude = np.hypot(dx, dy)
            cells = []
            y_edges = np.linspace(0, height, min(height, 4) + 1, dtype=int)
            x_edges = np.linspace(0, width, min(width, 4) + 1, dtype=int)
            for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
                for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
                    patch = batch[:, y0:y1, x0:x1]
                    cell_bins = bins[:, y0:y1, x0:x1]
                    weights = magnitude[:, y0:y1, x0:x1]
                    hist = np.column_stack([
                        np.where(cell_bins == index, weights, 0).sum(axis=(1, 2))
                        for index in range(9)
                    ])
                    hist /= np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-6)
                    cells.extend([hist, patch.mean(axis=(1, 2)), patch.std(axis=(1, 2))])
            parts.append(np.column_stack(cells))
        return np.concatenate(parts, axis=0)


class PairwiseTextFeatures(TransformerMixin, BaseEstimator):
    """Map two comparable texts to signed differences in a single vocabulary."""

    def __init__(self, vectorizer, semantic_weight: float = 0.0):
        self.vectorizer = vectorizer
        self.semantic_weight = semantic_weight

    def _validate(self, X):
        values = np.asarray(X, dtype=object)
        expected = 2 if self.semantic_weight == 0.0 else None
        if (
            values.ndim != 2 or values.shape[1] < 2
            or (expected is not None and values.shape[1] != expected)
            or (self.semantic_weight > 0 and (values.shape[1] <= 2 or values.shape[1] % 2))
        ):
            raise ValueError("Pairwise text features require two texts with matching embedding widths")
        return values

    @staticmethod
    def _fit_vocabulary(vectorizer, documents):
        try:
            return vectorizer.fit(documents)
        except ValueError as exc:
            if not str(exc).lower().startswith((
                "empty vocabulary", "after pruning, no terms remain",
                "max_df corresponds to < documents than min_df",
            )):
                raise
            return None

    def fit(self, X, y=None):
        from sklearn.base import clone
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        values = self._validate(X)
        if y is not None and not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Pairwise preference classification requires binary class IDs 0 and 1")
        documents = np.concatenate([values[:, 0], values[:, 1]])
        vectorizer = clone(self.vectorizer)
        components = vectorizer.transformer_list if isinstance(vectorizer, FeatureUnion) else ()
        self.vectorizer_ = self._fit_vocabulary(vectorizer, documents)
        if self.vectorizer_ is None and components:
            # A missing word vocabulary should not discard trained character features.
            fitted = [
                (name, model) for name, component in components
                if (model := self._fit_vocabulary(clone(component), documents)) is not None
            ]
            if fitted:
                self.vectorizer_ = vectorizer.set_params(transformer_list=fitted)
        if self.vectorizer_ is None:
            vocabulary_limits = [
                item.max_features for item in ([item for _, item in components] or [vectorizer])
                if item.max_features is not None
            ]
            fallback = TfidfVectorizer(
                analyzer="char", ngram_range=(1, 5), max_features=min(vocabulary_limits, default=40_000),
            )
            self.vectorizer_ = self._fit_vocabulary(fallback, documents)
        return self

    def transform(self, X):
        from scipy.sparse import csr_matrix, hstack
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "vectorizer_")
        values = self._validate(X)
        if self.vectorizer_ is None:
            lexical = csr_matrix((len(values), 1), dtype=np.float32)
        else:
            lexical = self.vectorizer_.transform(values[:, 0]) - self.vectorizer_.transform(values[:, 1])
        if not self.semantic_weight:
            return lexical
        vectors = np.asarray(values[:, 2:], dtype=np.float32)
        midpoint = vectors.shape[1] // 2
        semantic = self.semantic_weight * (vectors[:, :midpoint] - vectors[:, midpoint:])
        return hstack((lexical, csr_matrix(semantic)), format="csr")


class ImageSpatialPyramid(ImageArrayFlattener):
    """Pool bounded intensity and edge statistics at several spatial scales."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = self._validate(X)
        height, width = values.shape[1:3]
        grids = dict.fromkeys((min(height, scale), min(width, scale)) for scale in (1, 2, 4, 8))
        parts = []
        # Keep each temporary batch at roughly 1 MiB of source pixels.
        batch_size = max(1, min(64, 262_144 // max(1, int(np.prod(values.shape[1:])))))
        for start in range(0, len(values), batch_size):
            pixels = values[start:start + batch_size].astype(np.float32)
            pixels = (pixels - self.value_min) / (self.value_max - self.value_min)
            if pixels.ndim == 3:
                pixels = pixels[..., None]
            gray = pixels[..., :3].mean(axis=-1)
            dx = np.abs(np.diff(gray, axis=2, append=gray[:, :, -1:]))
            dy = np.abs(np.diff(gray, axis=1, append=gray[:, -1:, :]))
            cells = []
            for gy, gx in grids:
                ys = np.linspace(0, height, gy + 1, dtype=int)
                xs = np.linspace(0, width, gx + 1, dtype=int)
                for y0, y1 in zip(ys[:-1], ys[1:]):
                    for x0, x1 in zip(xs[:-1], xs[1:]):
                        patch = pixels[:, y0:y1, x0:x1]
                        cells.extend([
                            patch.mean(axis=(1, 2)),
                            patch.std(axis=(1, 2)),
                            dx[:, y0:y1, x0:x1].mean(axis=(1, 2)),
                            dy[:, y0:y1, x0:x1].mean(axis=(1, 2)),
                        ])
            parts.append(np.column_stack(cells))
        return np.concatenate(parts, axis=0)


class FitMatrixLimit(TransformerMixin, BaseEstimator):
    """Reject oversized fits before an estimator with quadratic scaling runs."""

    def __init__(self, max_rows: int, max_elements: int):
        self.max_rows = max_rows
        self.max_elements = max_elements

    def fit(self, X, y=None):
        if len(X) > self.max_rows or np.prod(X.shape) > self.max_elements:
            raise ValueError(
                f"RBF SVC supports up to {self.max_rows} training rows "
                f"and {self.max_elements} training feature values"
            )
        return self

    def transform(self, X):
        return X
