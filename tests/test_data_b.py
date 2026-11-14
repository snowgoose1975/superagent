"""B 的合成 NPY 契约测试；所有输入与输出目录均留存在 .cache 中。"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
from data_b.npy_reader import read_npy_array
from data_b.prepare import prepare_data
from data_b.preprocessing import TablePreprocessor
from data_b.raw_reader import load_feature_frame

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _area() -> Path:
    path = REPOSITORY_ROOT / ".cache" / "test-temp" / f"team-data-b-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_npy(path: Path, values: object) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=True)


def _task(area: Path, *, task_id: str = "b_test", version: str = "v1") -> dict:
    source = area / f"source-{task_id}"
    source.mkdir()
    # category_numeric 故意是数字类别；valid 中出现 train 未见过的 30。
    train_x = np.asarray(
        [[10, "b_c", "z", 1.0], [20, "q", "c", 2.0], [10, "b_c", "z", 3.0], [20, "q", "c", 4.0]],
        dtype=object,
    )
    valid_x = np.asarray([[30, "b_c", "z", 5.0], [10, "new", "new", 6.0]], dtype=object)
    _write_npy(source / "train_features.npy", train_x)
    _write_npy(source / "train_labels.npy", np.asarray(["no", "yes", "no", "yes"], dtype="U3"))
    _write_npy(source / "valid_features.npy", valid_x)
    _write_npy(source / "valid_labels.npy", np.asarray(["no", "yes"], dtype="U3"))
    # 故意不是有效 NPY；无配置引用时 B 不得尝试打开它。
    (source / "test_features.npy").write_bytes(b"test must remain unread")
    return {
        "schema_version": "1.0",
        "task_id": task_id,
        "data_version": version,
        "source_format": "npy_array_package",
        "source_root": str(source.resolve()),
        "feature_columns": ["category_numeric", "x", "x_y", "number"],
        "feature_types": {
            "category_numeric": "categorical",
            "x": "categorical",
            "x_y": "categorical",
            "number": "numeric",
        },
        "label_values": ["no", "yes"],
        "label_names": ["No", "Yes"],
        "primary_metric": {"name": "accuracy", "greater_is_better": True},
        "partitions": {
            "train": {
                "features_path": "train_features.npy",
                "labels_path": "train_labels.npy",
                "n_samples": 4,
            },
            "validation": {
                "features_path": "valid_features.npy",
                "labels_path": "valid_labels.npy",
                "n_samples": 2,
            },
        },
        "train_policy": {"max_trials": 3, "random_seed": 42, "timeout_seconds": 120},
    }


def _prepare(task: dict, area: Path) -> dict:
    return prepare_data(task, area / "prepared")


def test_declared_categorical_unknown_validation_and_train_only_profile() -> None:
    area = _area()
    task = _task(area)
    result = _prepare(task, area)
    assert result["status"] == "success", result

    package = Path(result["data_manifest_path"]).parent
    manifest = json.loads(Path(result["data_manifest_path"]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(result["label_mapping_path"]).read_text(encoding="utf-8"))
    assert mapping["label_values"] == ["no", "yes"]
    assert mapping["label_names"] == ["No", "Yes"]
    x_train = np.load(package / "X_train.npy", allow_pickle=False)
    x_valid = np.load(package / "X_valid.npy", allow_pickle=False)
    assert x_train.dtype == np.float64
    assert x_valid.dtype == np.float64

    processor = __import__("joblib").load(result["preprocessor_path"])
    categories = (
        processor.transformer_.named_transformers_["cat"]
        .named_steps["onehotencoder"]
        .categories_[0]
    )
    assert len(categories) == 2
    assert not any('"int","30"' in item for item in categories)
    # category_numeric 的两列 one-hot 结果只有 0/1；未知值 30 的该列组全为 0。
    cat_indices = [
        i for i, name in enumerate(manifest["feature_names"]) if "category_numeric" in name
    ]
    assert len(cat_indices) == 2
    assert set(np.unique(x_train[:, cat_indices]).tolist()) <= {0.0, 1.0}
    assert np.array_equal(x_valid[0, cat_indices], np.zeros(len(cat_indices)))

    profile = json.loads(Path(result["profile_path"]).read_text(encoding="utf-8"))
    rendered = json.dumps(profile, ensure_ascii=False)
    assert "valid_features.npy" not in rendered
    assert "valid_labels.npy" not in rendered
    assert "validation_label" not in rendered
    assert profile["n_train"] == 4
    assert result["profile_path"] == str(Path(result["profile_path"]).resolve())
    assert not (package / "test_features.npy").exists()


def test_feature_names_are_unique_when_encoder_names_collide() -> None:
    frame = __import__("pandas").DataFrame({"x": ["y_z", "a"], "x_y": ["z", "b"]})
    processor = TablePreprocessor({"x": "categorical", "x_y": "categorical"})
    processor.fit(frame)
    assert len(processor.feature_names_) == len(set(processor.feature_names_))
    assert processor.original_feature_names_ == ["x", "x_y"]
    assert processor.feature_names_[0].startswith("0000::")
    assert processor.feature_names_[1].startswith("0001::")


def test_same_content_reuses_cache_without_rewriting_and_ignores_policy() -> None:
    area = _area()
    task = _task(area)
    first = _prepare(task, area)
    assert first["status"] == "success", first
    package_result = Path(first["data_manifest_path"]).parent / "package_result.json"
    profile = Path(first["profile_path"])
    manifest = Path(first["data_manifest_path"])
    before = {path: path.read_bytes() for path in (package_result, profile, manifest)}
    task["train_policy"] = {"max_trials": 9, "random_seed": 9, "timeout_seconds": 240}
    second = _prepare(task, area)
    assert second["status"] == "success", second
    assert second["reused"] is True
    assert {path: path.read_bytes() for path in before} == before


def test_reuse_after_moving_prepared_directory_returns_current_absolute_paths() -> None:
    area = _area()
    task = _task(area)
    first = _prepare(task, area)
    assert first["status"] == "success", first
    old_package = Path(first["data_manifest_path"]).parent
    frozen_record = (old_package / "package_result.json").read_bytes()
    new_root = area / "moved-prepared"
    new_package = new_root / task["task_id"] / task["data_version"]
    new_package.parent.mkdir(parents=True)
    shutil.copytree(old_package, new_package)

    reused = prepare_data(task, new_root)
    assert reused["status"] == "success", reused
    assert reused["reused"] is True
    for field, filename in (
        ("data_manifest_path", "data_manifest.json"),
        ("evaluation_manifest_path", "evaluation_manifest.json"),
        ("profile_path", "metadata/profile.json"),
        ("preprocessor_path", "preprocessor.joblib"),
        ("label_mapping_path", "label_mapping.json"),
    ):
        assert Path(reused[field]) == (new_package / filename).resolve()
    assert (new_package / "package_result.json").read_bytes() == frozen_record


def test_source_change_and_prepared_change_are_rejected_without_overwrite() -> None:
    area = _area()
    task = _task(area)
    first = _prepare(task, area)
    assert first["status"] == "success", first
    package = Path(first["data_manifest_path"]).parent
    original = (package / "data_manifest.json").read_bytes()

    source_file = Path(task["source_root"]) / "train_features.npy"
    changed = np.load(source_file, allow_pickle=True)
    changed[0, 3] = 11.0
    source_file.write_bytes(b"")
    with source_file.open("wb") as stream:
        np.save(stream, changed, allow_pickle=True)
    changed_result = _prepare(task, area)
    assert changed_result["status"] == "failed"
    assert "源配置/数据已变化" in changed_result["error"]["message"]
    assert (package / "data_manifest.json").read_bytes() == original

    # 使用另一份新版本先准备，再改 prepared 数组，以单独覆盖工件摘要拒绝分支。
    task2 = _task(area, task_id="b_test_artifact", version="v1")
    second = _prepare(task2, area)
    assert second["status"] == "success", second
    package2 = Path(second["data_manifest_path"]).parent
    x_train = np.load(package2 / "X_train.npy", allow_pickle=False)
    x_train[0, 0] = 12345.0
    with (package2 / "X_train.npy").open("wb") as stream:
        np.save(stream, x_train, allow_pickle=False)
    tampered = _prepare(task2, area)
    assert tampered["status"] == "failed"
    assert "产物缺失或已变化" in tampered["error"]["message"]


def test_incomplete_version_and_explicit_test_partition_are_rejected() -> None:
    area = _area()
    task = _task(area)
    incomplete = area / "prepared" / task["task_id"] / task["data_version"]
    incomplete.mkdir(parents=True)
    (incomplete / "keep.txt").write_text("preserve", encoding="utf-8")
    result = _prepare(task, area)
    assert result["status"] == "failed"
    assert "没有完整成功记录" in result["error"]["message"]
    assert (incomplete / "keep.txt").read_text(encoding="utf-8") == "preserve"

    task_with_test = _task(area, task_id="b_test_partition", version="v1")
    task_with_test["partitions"]["test"] = {
        "features_path": "test_features.npy",
        "labels_path": "test_labels.npy",
        "n_samples": 2,
    }
    rejected = _prepare(task_with_test, area)
    assert rejected["status"] == "failed"
    assert "只能包含 train 与 validation" in rejected["error"]["message"]
    assert not (area / "prepared" / "b_test_partition" / "v1").exists()


def test_metric_requires_supported_name_and_boolean_direction() -> None:
    area = _area()
    task = _task(area)
    task["primary_metric"]["greater_is_better"] = 1
    result = _prepare(task, area)
    assert result["status"] == "failed"
    assert "primary_metric 必须是" in result["error"]["message"]
    assert not (area / "prepared" / task["task_id"] / task["data_version"]).exists()


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _UnknownGlobal:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def test_object_npy_unknown_global_is_rejected_without_execution() -> None:
    area = _area()
    marker = area / "should_not_exist.txt"
    array_path = area / "hostile.npy"
    _write_npy(array_path, np.asarray([_UnknownGlobal(marker)], dtype=object))
    with pytest.raises(ValueError, match="Unsupported pickle GLOBAL"):
        read_npy_array(array_path)
    assert not marker.exists()


def test_raw_reader_keeps_integer_categories_discrete() -> None:
    area = _area()
    path = area / "features.npy"
    _write_npy(path, np.asarray([[1], [2], [1]], dtype=np.int64))
    frame = load_feature_frame(path, ["coupon_code"], {"coupon_code": "categorical"})
    assert frame["coupon_code"].nunique() == 2
    assert all(isinstance(value, str) for value in frame["coupon_code"])
