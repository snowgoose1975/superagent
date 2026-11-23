"""Read bounded NPY inputs, fit preprocessing, and export immutable data packages."""

from .prepare import prepare_data
from .raw_reader import load_feature_frame

__all__ = ["load_feature_frame", "prepare_data"]
