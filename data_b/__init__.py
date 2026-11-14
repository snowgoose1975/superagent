"""B 模块：受限 NPY 读取、声明式预处理和不可变数据包导出。"""

from .prepare import prepare_data
from .raw_reader import load_feature_frame

__all__ = ["load_feature_frame", "prepare_data"]
