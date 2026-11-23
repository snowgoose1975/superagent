"""Freeze declared NPY train and validation partitions for C."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .npy_reader import read_npy_array
from .preprocessing import TablePreprocessor, validate_preprocessing
from .profiling import table_profile
from .raw_reader import load_feature_frame, typed_category_token
from .sample_index import load_sample_index, save_combined_index
from evaluation_d.metric_contracts import validate_metric_spec

_TASK_FIELDS = {
    "schema_version",
    "task_id",
    "data_version",
    "source_format",
    "source_root",
    "feature_columns",
    "feature_types",
    "label_values",
    "label_names",
    "primary_metric",
    "partitions",
    "train_policy",
}
_PARTITIONS = {"train", "validation"}
_SPLIT_FIELDS = {"features_path", "labels_path", "n_samples"}
_ARTIFACTS = (
    "X_train.npy",
    "y_train.npy",
    "X_valid.npy",
    "valid_ids.csv",
    "validation_labels.csv",
    "preprocessor.joblib",
    "label_mapping.json",
    "evaluation_manifest.json",
    "metadata/profile.json",
    "metadata/sample_index.csv",
    "data_manifest.json",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


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
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)
        or ".." in value
    ):
        raise ValueError(f"{field} must contain 1 to 64 safe ASCII characters")
    return value


def _source_file(root: Path, value: Any) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Partition files require a relative path within source_root")
    if ".." in Path(value).parts:
        raise ValueError("Partition file paths cannot contain ..")
    path = (root / value).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Partition file path escapes source_root") from exc
    if not path.is_file():
        raise ValueError(f"Partition file does not exist: {relative.as_posix()}")
    return path, relative.as_posix()


def _validate_spec(task_spec: dict[str, Any]) -> tuple[Path, list[str], dict[str, str]]:
    if not isinstance(task_spec, dict) or not _TASK_FIELDS <= set(task_spec) <= _TASK_FIELDS | {"preprocessing"}:
        extra = sorted(set(task_spec) - _TASK_FIELDS) if isinstance(task_spec, dict) else []
        missing = (
            sorted(_TASK_FIELDS - set(task_spec))
            if isinstance(task_spec, dict)
            else sorted(_TASK_FIELDS)
        )
        raise ValueError(f"Task config fields differ. Missing: {missing}. Unexpected: {extra}")
    if task_spec["schema_version"] != "1.0" or task_spec["source_format"] != "npy_array_package":
        raise ValueError("Expected schema 1.0 and npy_array_package")
    root = Path(task_spec["source_root"]).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("source_root must be an existing absolute directory")
    root = root.resolve()
    columns = task_spec["feature_columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(c, str) or not c.strip() for c in columns)
    ):
        raise ValueError("feature_columns must be a non-empty list of column names")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns cannot contain duplicates")
    kinds = task_spec["feature_types"]
    if not isinstance(kinds, dict) or set(kinds) != set(columns):
        raise ValueError("feature_types must cover exactly feature_columns")
    if any(kind not in {"categorical", "numeric", "text"} for kind in kinds.values()):
        raise ValueError("feature_types values must be categorical, numeric, or text")
    validate_preprocessing(task_spec.get("preprocessing"), kinds)
    validate_metric_spec(task_spec["primary_metric"])
    policy = task_spec["train_policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "max_trials",
        "random_seed",
        "timeout_seconds",
    }:
        raise ValueError("train_policy requires max_trials, random_seed, and timeout_seconds")
    if type(policy["max_trials"]) is not int or policy["max_trials"] < 1:
        raise ValueError("max_trials must be positive")
    if type(policy["random_seed"]) is not int or not 0 <= policy["random_seed"] <= 2**32 - 1:
        raise ValueError("random_seed is outside its supported range")
    if type(policy["timeout_seconds"]) is not int or policy["timeout_seconds"] < 1:
        raise ValueError("timeout_seconds must be positive")
    partitions = task_spec["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != _PARTITIONS:
        raise ValueError("partitions must contain only train and validation")
    for name, part in partitions.items():
        if not isinstance(part, dict) or not _SPLIT_FIELDS <= set(part) <= _SPLIT_FIELDS | {"sample_ids_path"}:
            raise ValueError(f"{name} requires features_path, labels_path, and n_samples. sample_ids_path is optional")
        if type(part["n_samples"]) is not int or part["n_samples"] < 1:
            raise ValueError(f"{name}.n_samples must be positive")
    values, names = task_spec["label_values"], task_spec["label_names"]
    if (
        not isinstance(values, list)
        or len(values) < 2
        or not isinstance(names, list)
        or len(names) != len(values)
    ):
        raise ValueError("label_values and label_names must have equal lengths and at least two classes")
    tokens = [typed_category_token(value) for value in values]
    if any(not isinstance(token, str) for token in tokens) or len(tokens) != len(set(tokens)):
        raise ValueError("label_values must be unique, non-missing scalars")
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(names) != len(
        set(names)
    ):
        raise ValueError("label_names must be unique non-empty strings")
    _safe_id(task_spec["task_id"], "task_id")
    _safe_id(task_spec["data_version"], "data_version")
    return root, list(columns), dict(kinds)


def _load_partition(
    root: Path, name: str, spec: dict[str, Any], columns: list[str], kinds: dict[str, str]
):
    part = spec["partitions"][name]
    feature_path, feature_relative = _source_file(root, part["features_path"])
    label_path, label_relative = _source_file(root, part["labels_path"])
    feature_hash_before, label_hash_before = _sha256(feature_path), _sha256(label_path)
    frame = load_feature_frame(feature_path, columns, kinds)
    labels = read_npy_array(label_path)
    if labels.ndim != 1 or len(labels) != part["n_samples"] or len(frame) != part["n_samples"]:
        raise ValueError(f"{name} features or labels do not match n_samples")
    feature_hash_after, label_hash_after = _sha256(feature_path), _sha256(label_path)
    if (feature_hash_before, label_hash_before) != (feature_hash_after, label_hash_after):
        raise ValueError(f"{name} source files changed while being read")
    hashes = {
        "features": {"relative_path": feature_relative, "sha256": feature_hash_after},
        "labels": {"relative_path": label_relative, "sha256": label_hash_after},
    }
    sample_rows, sample_hash = load_sample_index(root, part, name)
    hashes.update(sample_hash)
    frame.index = [row["sample_id"] for row in sample_rows]
    return frame, labels, hashes, sample_rows


def _label_token(value: Any) -> str:
    token = typed_category_token(value)
    if not isinstance(token, str):
        raise ValueError("Labels cannot contain None or NaN")
    return token


def _source_signature(
    spec: dict[str, Any], columns: list[str], kinds: dict[str, str], hashes: dict[str, Any]
) -> str:
    schema = {
        "schema_version": spec["schema_version"],
        "task_id": spec["task_id"],
        "data_version": spec["data_version"],
        "source_format": spec["source_format"],
        "feature_columns": columns,
        "feature_types": {c: kinds[c] for c in columns},
        "label_values": [_label_token(v) for v in spec["label_values"]],
        "label_names": spec["label_names"],
        "primary_metric": spec["primary_metric"],
        "preprocessing": spec.get("preprocessing", {}),
        "n_samples": {name: spec["partitions"][name]["n_samples"] for name in sorted(_PARTITIONS)},
        "partitions": hashes,
        "preparation_code_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("prepare.py", "raw_reader.py", "preprocessing.py", "npy_reader.py",
                         "profiling.py", "sample_index.py")
        },
    }
    semantic = hashlib.sha256(_json_bytes(schema)).hexdigest()
    return semantic


_NATIVE_SPLITS = ("native_train.json", "native_validation.json")


def _read_cache(directory: Path, signature: str) -> dict[str, Any] | None:
    result_path = directory / "package_result.json"
    if not result_path.is_file():
        raise ValueError("Existing version has no complete successful package. Use a new version")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Cannot read package_result.json. Use a new version"
        ) from exc
    if result.get("status") != "success" or result.get("source_signature") != signature:
        raise ValueError("This version is incomplete or its inputs changed. Use a new version")
    expected = result.get("artifact_sha256")
    if not isinstance(expected, dict) or set(expected) not in (set(_ARTIFACTS), set(_ARTIFACTS + _NATIVE_SPLITS)):
        raise ValueError("The cached artifact manifest is incomplete. Use a new version")
    for relative, digest in expected.items():
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError as exc:
            raise ValueError("Cached artifact path escapes the package") from exc
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Cached artifact is missing or changed: {relative}. Use a new version")
    # Resolve returned paths after moving the package while keeping its manifest immutable.
    result.update(
        {
            "data_manifest_path": str((directory / "data_manifest.json").resolve()),
            "evaluation_manifest_path": str((directory / "evaluation_manifest.json").resolve()),
            "profile_path": str((directory / "metadata" / "profile.json").resolve()),
            "preprocessor_path": str((directory / "preprocessor.joblib").resolve()),
            "label_mapping_path": str((directory / "label_mapping.json").resolve()),
            "sample_index_path": str((directory / "metadata" / "sample_index.csv").resolve()),
        }
    )
    result["reused"] = True
    return result


def _save_array(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def _save_ids(path: Path, ids: list[str], values: list[int] | None = None) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id"] if values is None else ["sample_id", "y_true"])
        for index, sample_id in enumerate(ids):
            writer.writerow([sample_id] if values is None else [sample_id, values[index]])


def prepare_data(task_spec: dict[str, Any], data_root: str | Path) -> dict[str, Any]:
    """Prepare or reuse an immutable package and return absolute paths for C and D."""
    if isinstance(task_spec, dict) and task_spec.get("schema_version") == "2.0":
        from .prepare_modalities import prepare_modalities

        return prepare_modalities(task_spec, data_root)
    fresh_directory: Path | None = None
    try:
        source_root, columns, kinds = _validate_spec(task_spec)
        base = Path(data_root).expanduser().resolve()
        directory = base / task_spec["task_id"] / task_spec["data_version"]
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=False)
            fresh_directory = directory
        train_frame, train_raw_labels, train_hashes, train_index = _load_partition(
            source_root, "train", task_spec, columns, kinds
        )
        valid_frame, valid_raw_labels, valid_hashes, valid_index = _load_partition(
            source_root, "validation", task_spec, columns, kinds
        )
        if train_frame.shape[1] != valid_frame.shape[1]:
            raise ValueError("Train and validation feature columns differ")
        label_tokens = [_label_token(value) for value in task_spec["label_values"]]
        label_to_id = {token: index for index, token in enumerate(label_tokens)}

        def encode(raw: np.ndarray, split: str, require_all_classes: bool) -> np.ndarray:
            encoded = []
            for value in raw:
                token = _label_token(value)
                if token not in label_to_id:
                    raise ValueError(f"{split} labels contain a value absent from label_values")
                encoded.append(label_to_id[token])
            result = np.asarray(encoded, dtype=np.int64)
            if require_all_classes and set(np.unique(result).tolist()) != set(
                range(len(label_tokens))
            ):
                raise ValueError(f"{split} labels do not cover all declared classes")
            return result

        y_train = encode(train_raw_labels, "train", require_all_classes=True)
        y_valid = encode(valid_raw_labels, "validation", require_all_classes=False)
        hashes = {"train": train_hashes, "validation": valid_hashes}
        signature = _source_signature(task_spec, columns, kinds, hashes)
        if fresh_directory is None:
            result = _read_cache(directory, signature)
            return result
        metadata = directory / "metadata"
        metadata.mkdir()
        try:
            preprocessor = TablePreprocessor(
                kinds, options=task_spec.get("preprocessing"), allow_frequency_fallback=True,
            )
            X_train = preprocessor.fit_transform(train_frame)
            X_valid = preprocessor.transform(valid_frame)
            if X_train.shape[1] < 1:
                raise ValueError("Preprocessing yielded no usable features")
            train_counts = {
                name: int(np.sum(y_train == index))
                for index, name in enumerate(task_spec["label_names"])
            }
            profile = {
                "schema_version": "1.0",
                "task_id": task_spec["task_id"],
                "data_version": task_spec["data_version"],
                "source_format": task_spec["source_format"],
                "n_train": len(X_train),
                "n_features_raw": len(columns),
                "feature_columns": columns,
                "feature_types": {c: kinds[c] for c in columns},
                "train_label_counts": train_counts,
                "preprocessing": task_spec.get("preprocessing", {}),
                **table_profile(train_frame, kinds),
                "table_encoding": "frequency" if preprocessor.frequency_maps_ else "one_hot",
            }
            _write_json(metadata / "profile.json", profile)
            for name, frame in zip(_NATIVE_SPLITS, (train_frame, valid_frame), strict=True):
                rows = frame.astype(object).where(frame.notna(), None).values.tolist()
                payload = json.dumps({"columns": columns, "data": rows}, allow_nan=False)
                with (directory / name).open("x", encoding="utf-8") as stream:
                    stream.write(payload)
            save_combined_index(metadata / "sample_index.csv", train_index, valid_index)
            _save_array(directory / "X_train.npy", X_train)
            _save_array(directory / "y_train.npy", y_train)
            _save_array(directory / "X_valid.npy", X_valid)
            valid_ids = [row["sample_id"] for row in valid_index]
            _save_ids(directory / "valid_ids.csv", valid_ids)
            _save_ids(directory / "validation_labels.csv", valid_ids, y_valid.tolist())
            with (directory / "preprocessor.joblib").open("xb") as stream:
                joblib.dump(preprocessor, stream)
            classes = [
                {"id": index, "name": name} for index, name in enumerate(task_spec["label_names"])
            ]
            mapping = {
                "classes": classes,
                "label_values": task_spec["label_values"],
                "label_names": task_spec["label_names"],
                "feature_columns": columns,
                "id_column": None,
            }
            _write_json(directory / "label_mapping.json", mapping)
            _write_json(
                directory / "evaluation_manifest.json",
                {
                    "schema_version": "1.0",
                    "task_id": task_spec["task_id"],
                    "data_version": task_spec["data_version"],
                    "data_manifest_path": "data_manifest.json",
                    "labels_path": "validation_labels.csv",
                    "primary_metric": task_spec["primary_metric"],
                    "positive_class_id": None,
                },
            )
            manifest = {
                "schema_version": "1.0",
                "task_id": task_spec["task_id"],
                "data_version": task_spec["data_version"],
                "problem_type": "classification",
                "n_features": int(X_train.shape[1]),
                "feature_names": preprocessor.feature_names_,
                "classes": classes,
                "train": {
                    "features_path": "X_train.npy",
                    "labels_path": "y_train.npy",
                    "n_samples": len(X_train),
                },
                "validation": {
                    "features_path": "X_valid.npy",
                    "sample_ids_path": "valid_ids.csv",
                    "n_samples": len(X_valid),
                },
                "preprocessor_path": "preprocessor.joblib",
            }
            manifest_bytes = _json_bytes(manifest)
            file_hashes = {
                name: _sha256(directory / name)
                for name in _ARTIFACTS + _NATIVE_SPLITS
                if name != "data_manifest.json"
            }
            file_hashes["data_manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
            paths = {
                "status": "success",
                "task_id": task_spec["task_id"],
                "data_version": task_spec["data_version"],
                "data_manifest_path": str((directory / "data_manifest.json").resolve()),
                "evaluation_manifest_path": str((directory / "evaluation_manifest.json").resolve()),
                "profile_path": str((metadata / "profile.json").resolve()),
                "preprocessor_path": str((directory / "preprocessor.joblib").resolve()),
                "label_mapping_path": str((directory / "label_mapping.json").resolve()),
                "sample_index_path": str((metadata / "sample_index.csv").resolve()),
                "source_signature": signature,
                "source_file_sha256": hashes,
                "artifact_sha256": file_hashes,
                "reused": False,
                "error": None,
            }
            _write_json(directory / "package_result.json", paths)
            with (directory / "data_manifest.json").open("xb") as stream:
                stream.write(manifest_bytes)
            return paths
        except Exception as exc:
            try:
                _write_json(
                    directory / "failure.json",
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except OSError:
                pass
            raise
    except Exception as exc:
        if fresh_directory is not None:
            try:
                _write_json(
                    fresh_directory / "failure.json",
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
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
