"""Immutable B data packages for declared raw-text and image-array tasks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation_d.metric_contracts import validate_metric_spec

from .modality_readers import (
    load_image_tensor,
    load_text_matrix,
    validate_image_spec,
    write_text_matrix_jsonl,
)
from .npy_reader import read_npy_array
from .raw_reader import typed_category_token

_ARTIFACTS = (
    "X_train.data",
    "y_train.npy",
    "X_valid.data",
    "valid_ids.csv",
    "validation_labels.csv",
    "label_mapping.json",
    "evaluation_manifest.json",
    "metadata/profile.json",
    "data_manifest.json",
)
_TASK_FIELDS = {
    "schema_version",
    "task_id",
    "data_version",
    "source_format",
    "source_root",
    "modality",
    "label_values",
    "label_names",
    "primary_metric",
    "partitions",
    "train_policy",
}
_SPLIT_FIELDS = {"features_path", "labels_path", "n_samples"}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: Any, field: str) -> str:
    import re

    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) or ".." in value:
        raise ValueError(f"{field} 必须是 1–64 位安全 ASCII 编号")
    return value


def _validate_spec(spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if not isinstance(spec, dict) or set(spec) != _TASK_FIELDS:
        raise ValueError("2.0 任务配置字段缺失或包含未支持字段")
    if spec["schema_version"] != "2.0":
        raise ValueError("模态任务必须使用 TaskSpec 2.0")
    task_id = _safe_id(spec["task_id"], "task_id")
    data_version = _safe_id(spec["data_version"], "data_version")
    source_root = Path(spec["source_root"]).expanduser()
    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError("source_root 必须是现存目录的绝对路径")
    modality = spec["modality"]
    if not isinstance(modality, dict) or not isinstance(modality.get("kind"), str):
        raise ValueError("modality 必须声明 kind")
    if spec["source_format"] == "text_array_package" and modality.get("kind") == "text":
        expected = {"kind", "text_columns", "max_characters"}
        if set(modality) != expected:
            raise ValueError("文本 modality 字段不受支持")
        columns = modality["text_columns"]
        if (
            not isinstance(columns, list)
            or not 1 <= len(columns) <= 8
            or any(not isinstance(name, str) or not name.strip() for name in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError("text_columns 必须是 1–8 个唯一非空列名")
        if (
            type(modality["max_characters"]) is not int
            or not 1 <= modality["max_characters"] <= 100_000
        ):
            raise ValueError("max_characters 必须在 1–100000 之间")
        modality = {
            "kind": "text",
            "text_columns": list(columns),
            "max_characters": modality["max_characters"],
        }
    elif spec["source_format"] == "image_array_package" and modality.get("kind") == "image":
        expected = {"kind", "image_shape", "channel_layout", "value_range"}
        if set(modality) != expected:
            raise ValueError("图像 modality 字段不受支持")
        shape, layout, value_range = validate_image_spec(
            modality["image_shape"], modality["channel_layout"], modality["value_range"]
        )
        modality = {
            "kind": "image",
            "image_shape": shape,
            "channel_layout": layout,
            "value_range": list(value_range),
        }
    else:
        raise ValueError("source_format 与 modality.kind 必须声明为文本或图像数组")
    validate_metric_spec(spec["primary_metric"])
    partitions = spec["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation"}:
        raise ValueError("partitions 必须且只能包含 train 和 validation")
    for name, split in partitions.items():
        if (
            not isinstance(split, dict)
            or set(split) != _SPLIT_FIELDS
            or type(split["n_samples"]) is not int
            or split["n_samples"] < 1
        ):
            raise ValueError(f"{name} 必须声明合法 features_path/labels_path/n_samples")
        for key in ("features_path", "labels_path"):
            relative = split[key]
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
            ):
                raise ValueError(f"{name}.{key} 必须是 source_root 内相对路径")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name}.{key} 必须是 source_root 内相对路径")
            if not (source_root / path).resolve().is_relative_to(source_root.resolve()):
                raise ValueError(f"{name}.{key} 越出 source_root")
            if not (source_root / path).is_file():
                raise ValueError(f"{name}.{key} 文件不存在")
    values, names = spec["label_values"], spec["label_names"]
    if (
        not isinstance(values, list)
        or len(values) < 2
        or not isinstance(names, list)
        or len(names) != len(values)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("label_values/label_names 必须等长、唯一且至少包含两个类别")
    label_tokens = [typed_category_token(item) for item in values]
    if any(not isinstance(token, str) for token in label_tokens) or len(
        set(label_tokens)
    ) != len(label_tokens):
        raise ValueError("label_values 必须是唯一、非缺失的标量")
    policy = spec["train_policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "max_trials",
        "random_seed",
        "timeout_seconds",
    }:
        raise ValueError("train_policy 必须含 max_trials/random_seed/timeout_seconds")
    if type(policy["max_trials"]) is not int or not 1 <= policy["max_trials"] <= 20:
        raise ValueError("max_trials 必须是 1–20 的整数")
    if type(policy["random_seed"]) is not int or not 0 <= policy["random_seed"] <= 2**32 - 1:
        raise ValueError("random_seed 超出范围")
    if type(policy["timeout_seconds"]) is not int or not 1 <= policy["timeout_seconds"] <= 600:
        raise ValueError("timeout_seconds 必须是 1–600 的整数")
    normalized = dict(spec)
    normalized.update(task_id=task_id, data_version=data_version, modality=modality)
    return source_root.resolve(), normalized


def _load_split(
    source_root: Path, spec: dict[str, Any], split_name: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split = spec["partitions"][split_name]
    feature_path = (source_root / split["features_path"]).resolve()
    label_path = (source_root / split["labels_path"]).resolve()
    before = {"features": _sha256(feature_path), "labels": _sha256(label_path)}
    if spec["modality"]["kind"] == "text":
        X = load_text_matrix(
            feature_path,
            spec["modality"]["text_columns"],
            max_characters=spec["modality"]["max_characters"],
        )
    else:
        X = load_image_tensor(
            feature_path,
            spec["modality"]["image_shape"],
            spec["modality"]["channel_layout"],
            spec["modality"]["value_range"],
        )
    y = read_npy_array(label_path)
    if y.ndim != 1 or len(y) != split["n_samples"] or len(X) != split["n_samples"]:
        raise ValueError(f"{split_name} 特征/标签行数与 n_samples 不一致")
    after = {"features": _sha256(feature_path), "labels": _sha256(label_path)}
    if before != after:
        raise ValueError(f"{split_name} 源文件在读取期间发生变化")
    return X, y, {
        "features": {
            "relative_path": split["features_path"],
            "sha256": after["features"],
        },
        "labels": {"relative_path": split["labels_path"], "sha256": after["labels"]},
    }


def _signature(spec: dict[str, Any], source_hashes: dict[str, Any]) -> str:
    code_hashes = {
        name: _sha256(Path(__file__).with_name(name))
        for name in (
            "prepare_modalities.py",
            "modality_readers.py",
            "npy_reader.py",
            "raw_reader.py",
        )
    }
    material = {
        "task": {
            key: spec[key]
            for key in (
                "schema_version",
                "task_id",
                "data_version",
                "source_format",
                "modality",
                "label_values",
                "label_names",
                "primary_metric",
                "partitions",
                "train_policy",
            )
        },
        "source_hashes": source_hashes,
        "preparation_code_sha256": code_hashes,
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def _cache(directory: Path, signature: str) -> dict[str, Any]:
    result_path = directory / "package_result.json"
    if not result_path.is_file():
        raise ValueError("该 data_version 已存在但未完整发布；保留目录并使用新版本")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "success" or result.get("source_signature") != signature:
        raise ValueError("该 data_version 的配置/源文件/预处理代码已变化；保留原目录并使用新版本")
    expected = result.get("artifact_sha256")
    if not isinstance(expected, dict) or set(expected) != set(_ARTIFACTS):
        raise ValueError("模态缓存的产物清单不完整；保留原目录并使用新版本")
    for relative, digest in expected.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or _sha256(path) != digest:
            raise ValueError("模态缓存产物缺失/变化/路径越界；保留原目录并使用新版本")
    result.update(
        data_manifest_path=str((directory / "data_manifest.json").resolve()),
        evaluation_manifest_path=str((directory / "evaluation_manifest.json").resolve()),
        profile_path=str((directory / "metadata" / "profile.json").resolve()),
        preprocessor_path=None,
        label_mapping_path=str((directory / "label_mapping.json").resolve()),
        reused=True,
    )
    return result


def _save_npy(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def prepare_modalities(task_spec: dict[str, Any], data_root: str | Path) -> dict[str, Any]:
    """Prepare a versioned v2.0 raw-text or image-tensor B package."""
    fresh: Path | None = None
    try:
        source_root, spec = _validate_spec(task_spec)
        directory = Path(data_root).expanduser().resolve() / spec["task_id"] / spec["data_version"]
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=False)
            fresh = directory
        X_train_raw, y_train_raw, train_hashes = _load_split(source_root, spec, "train")
        X_valid_raw, y_valid_raw, valid_hashes = _load_split(source_root, spec, "validation")
        label_tokens = [typed_category_token(item) for item in spec["label_values"]]
        label_to_id = {token: index for index, token in enumerate(label_tokens)}

        def encode(raw: np.ndarray, name: str, require_all: bool) -> np.ndarray:
            output = []
            for value in raw:
                token = typed_category_token(value)
                if not isinstance(token, str) or token not in label_to_id:
                    raise ValueError(f"{name} 标签不在 label_values 声明中")
                output.append(label_to_id[token])
            labels = np.asarray(output, dtype=np.int64)
            if require_all and set(np.unique(labels).tolist()) != set(range(len(label_tokens))):
                raise ValueError(f"{name} 标签未覆盖全部已声明类别")
            return labels

        y_train = encode(y_train_raw, "train", True)
        y_valid = encode(y_valid_raw, "validation", False)
        source_hashes = {"train": train_hashes, "validation": valid_hashes}
        signature = _signature(spec, source_hashes)
        if fresh is None:
            return _cache(directory, signature)

        if spec["modality"]["kind"] == "text":
            feature_names = list(spec["modality"]["text_columns"])
            representation = "raw_text"
        else:
            feature_names = None
            representation = "image_tensor"
        class_names = spec["label_names"]
        classes = [{"id": index, "name": name} for index, name in enumerate(class_names)]
        metadata_dir = directory / "metadata"
        metadata_dir.mkdir()
        if representation == "raw_text":
            write_text_matrix_jsonl(directory / "X_train.data", X_train_raw)
            write_text_matrix_jsonl(directory / "X_valid.data", X_valid_raw)
        else:
            _save_npy(directory / "X_train.data", X_train_raw)
            _save_npy(directory / "X_valid.data", X_valid_raw)
        _save_npy(directory / "y_train.npy", y_train)
        valid_ids = [f"validation_{index:09d}" for index in range(len(X_valid_raw))]
        with (directory / "valid_ids.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["sample_id"])
            writer.writerows([[item] for item in valid_ids])
        with (directory / "validation_labels.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["sample_id", "y_true"])
            writer.writerows(
                [[item, int(label)] for item, label in zip(valid_ids, y_valid, strict=True)]
            )
        train_counts = {
            name: int(np.sum(y_train == index)) for index, name in enumerate(class_names)
        }
        _write_json(
            metadata_dir / "profile.json",
            {
                "schema_version": "1.0",
                "task_id": spec["task_id"],
                "data_version": spec["data_version"],
                "source_format": spec["source_format"],
                "representation": representation,
                "n_train": len(X_train_raw),
                "n_features_raw": (
                    len(feature_names)
                    if feature_names is not None
                    else int(np.prod(spec["modality"]["image_shape"]))
                ),
                "feature_names": feature_names,
                "image_shape": spec["modality"].get("image_shape"),
                "train_label_counts": train_counts,
            },
        )
        mapping = {
            "classes": classes,
            "label_values": spec["label_values"],
            "label_names": class_names,
            "representation": representation,
            "modality": spec["modality"],
            "output_label_mode": "encoded",
        }
        _write_json(directory / "label_mapping.json", mapping)
        _write_json(
            directory / "evaluation_manifest.json",
            {
                "schema_version": "1.0",
                "task_id": spec["task_id"],
                "data_version": spec["data_version"],
                "data_manifest_path": "data_manifest.json",
                "labels_path": "validation_labels.csv",
                "primary_metric": spec["primary_metric"],
                "positive_class_id": None,
            },
        )
        manifest = {
            "schema_version": "2.0",
            "task_id": spec["task_id"],
            "data_version": spec["data_version"],
            "problem_type": "classification",
            "representation": representation,
            "n_features": (
                len(feature_names)
                if feature_names is not None
                else int(np.prod(spec["modality"]["image_shape"]))
            ),
            "modality": spec["modality"],
            "classes": classes,
            "train": {
                "features_path": "X_train.data",
                "labels_path": "y_train.npy",
                "n_samples": len(X_train_raw),
            },
            "validation": {
                "features_path": "X_valid.data",
                "sample_ids_path": "valid_ids.csv",
                "n_samples": len(X_valid_raw),
            },
            "preprocessor_path": None,
        }
        manifest_bytes = _json_bytes(manifest)
        file_hashes = {
            name: _sha256(directory / name) for name in _ARTIFACTS if name != "data_manifest.json"
        }
        file_hashes["data_manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
        result = {
            "status": "success",
            "task_id": spec["task_id"],
            "data_version": spec["data_version"],
            "data_manifest_path": str((directory / "data_manifest.json").resolve()),
            "evaluation_manifest_path": str((directory / "evaluation_manifest.json").resolve()),
            "profile_path": str((metadata_dir / "profile.json").resolve()),
            "preprocessor_path": None,
            "label_mapping_path": str((directory / "label_mapping.json").resolve()),
            "source_signature": signature,
            "source_file_sha256": source_hashes,
            "artifact_sha256": file_hashes,
            "reused": False,
            "error": None,
        }
        _write_json(directory / "package_result.json", result)
        with (directory / "data_manifest.json").open("xb") as stream:
            stream.write(manifest_bytes)
        return result
    except Exception as exc:
        if fresh is not None:
            try:
                _write_json(
                    fresh / "failure.json",
                    {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
                )
            except OSError:
                pass
        return {
            "status": "failed",
            "task_id": task_spec.get("task_id") if isinstance(task_spec, dict) else None,
            "data_version": task_spec.get("data_version") if isinstance(task_spec, dict) else None,
            "data_manifest_path": None,
            "evaluation_manifest_path": None,
            "profile_path": None,
            "preprocessor_path": None,
            "label_mapping_path": None,
            "reused": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
