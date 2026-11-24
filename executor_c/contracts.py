"""Shared input and artifact contracts for data preparation, training, and evaluation."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from executor_c.model_registry import (
    MAX_TIMEOUT_SECONDS as REGISTRY_MAX_TIMEOUT_SECONDS,
    MODELS,
    capabilities as registry_capabilities,
    normalize_params,
    normalize_pipeline,
    validate_model_pipeline,
)

SCHEMA_VERSION = "1.0"
MAX_TIMEOUT_SECONDS = REGISTRY_MAX_TIMEOUT_SECONDS
_MIB = 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * _MIB
_MAX_DENSE_PAYLOAD_BYTES = 256 * _MIB
_MAX_NPY_HEADER_BYTES = 64 * 1024 + 12
_MAX_LABEL_ROWS = 2_000_000
_MAX_IDS_BYTES = 128 * _MIB
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
PLAN_FIELDS = {
    "schema_version",
    "task_id",
    "data_version",
    "experiment_id",
    "model",
    "params",
    "random_seed",
    "timeout_seconds",
}
PLAN_V2_FIELDS = PLAN_FIELDS | {"representation", "pipeline"}
MODEL_PARAMETERS = {
    model_id: {name: parameter.as_dict() for name, parameter in model.parameters.items()}
    for model_id, model in MODELS.items()
}


class ContractError(ValueError):
    """Input or artifact contract violation suitable for the result record."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class InvocationError(ValueError):
    """Invocation error raised before the experiment directory can be created."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def capabilities() -> dict[str, Any]:
    """Return the planner's model inventory. The executor validates every plan."""
    return registry_capabilities()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("JSON cannot contain NaN or Infinity")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON floating-point numbers must be finite")
    return number


def read_json(
    path: str | Path, error_code: str = "DATA_INVALID", *, max_bytes: int | None = None,
) -> dict[str, Any]:
    """Read a standard JSON object while keeping raw input out of error messages."""
    try:
        if max_bytes is None:
            content = Path(path).read_text(encoding="utf-8-sig")
        else:
            with Path(path).open("rb") as stream:
                raw = stream.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise ContractError(error_code, "The JSON file exceeds its size limit")
            content = raw.decode("utf-8-sig")
        value = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ContractError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        raise ContractError(error_code, "Could not read a valid UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ContractError(error_code, "The JSON root must be an object")
    return value


def write_json(path: str | Path, value: Any, exclusive: bool = True) -> None:
    """Write a new JSON file by default to preserve plans, snapshots, and results."""
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def publish_json(path: str | Path, value: Any) -> None:
    """Publish complete JSON from a temporary file. Each experiment directory has one writer."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("The final result already exists")
    pending = destination.with_name(destination.stem + ".pending.json")
    write_json(pending, value)
    pending.rename(destination)


def sha256_file(path: str | Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_MIB), b""):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError("File grew beyond its size limit during hashing")
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scikit-learn", "joblib", "xgboost-cpu", "xgboost", "catboost"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _fields(value: Any, expected: set[str], code: str, context: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError(code, f"{context} has missing or unsupported fields")


def _integer(value: Any, minimum: int, maximum: int | None = None) -> bool:
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def validate_identity(plan: Any) -> tuple[str, str, str]:
    if not isinstance(plan, dict):
        raise InvocationError("PLAN_INVALID", "The plan must be an object")
    identifiers = []
    for key in ("task_id", "data_version", "experiment_id"):
        value = plan.get(key)
        if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
            raise InvocationError("PLAN_INVALID", f"{key} must be a safe ASCII identifier from 1 to 64 characters")
        identifiers.append(value)
    return tuple(identifiers)


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ContractError("PLAN_INVALID", "The experiment plan must be an object")
    version = plan.get("schema_version")
    expected = PLAN_FIELDS if version == "1.0" else PLAN_V2_FIELDS if version == "2.0" else set()
    _fields(plan, expected, "PLAN_INVALID", "experiment plan")
    try:
        validate_identity(plan)
    except InvocationError as exc:
        raise ContractError(exc.code, exc.message) from exc
    if version not in {"1.0", "2.0"}:
        raise ContractError("PLAN_INVALID", "Unsupported schema_version")
    model = plan["model"]
    if not isinstance(model, str) or model not in MODELS:
        raise ContractError("UNSUPPORTED_MODEL", "model must appear in the capabilities.json registry")
    if not isinstance(plan["params"], dict):
        raise ContractError("PLAN_INVALID", "params must be an object")
    try:
        resolved = normalize_params(model, plan["params"])
    except ValueError as exc:
        code = "UNSUPPORTED_PARAMETER" if "unsupported fields" in str(exc) else "PLAN_INVALID"
        raise ContractError(code, str(exc)) from exc
    if not _integer(plan["random_seed"], 0, 2**32 - 1):
        raise ContractError("PLAN_INVALID", "random_seed must be an integer from 0 to 2**32 - 1")
    if not _integer(plan["timeout_seconds"], 1, MAX_TIMEOUT_SECONDS):
        raise ContractError("PLAN_INVALID", "timeout_seconds must be an integer from 1 to 600")
    normalized = {**plan, "params": resolved}
    if version == "2.0":
        representation = plan["representation"]
        if not isinstance(representation, str) or representation not in {
            "dense_numeric", "raw_text", "image_tensor"
        }:
            raise ContractError("PLAN_INVALID", "Unsupported representation")
        if representation not in MODELS[model].representations:
            raise ContractError("UNSUPPORTED_MODEL", "The model does not support this plan representation")
        try:
            normalized["pipeline"] = normalize_pipeline(representation, plan["pipeline"])
            validate_model_pipeline(model, normalized["pipeline"])
        except (TypeError, ValueError) as exc:
            raise ContractError("PLAN_INVALID", str(exc)) from exc
    return normalized


def resolve_data_path(base_dir: str | Path, relative: Any) -> Path:
    """Resolve paths inside the manifest directory independently of the working directory."""
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in PurePosixPath(relative).parts
    ):
        raise ContractError("DATA_INVALID", "The data file must use a relative POSIX path within its package")
    base = Path(base_dir).resolve()
    try:
        candidate = (base / relative).resolve()
        candidate.relative_to(base)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ContractError("DATA_INVALID", "The file path escapes the data package") from exc
    if not candidate.is_file():
        raise ContractError("DATA_INVALID", "The manifest entry must point to an existing regular file")
    return candidate


def validate_manifest(manifest_path: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata first. The worker validates array contents within its time limit."""
    manifest = read_json(manifest_path, max_bytes=_MAX_MANIFEST_BYTES)
    required_v1 = {
        "schema_version",
        "task_id",
        "data_version",
        "problem_type",
        "n_features",
        "feature_names",
        "classes",
        "train",
        "validation",
        "preprocessor_path",
    }
    required_v2 = (required_v1 - {"feature_names"}) | {"representation", "modality"}
    version = manifest.get("schema_version")
    required = required_v1 if version == "1.0" else required_v2 if version == "2.0" else set()
    _fields(manifest, required, "DATA_INVALID", "data_manifest")
    if version not in {"1.0", "2.0"}:
        raise ContractError("DATA_INVALID", "Unsupported data manifest schema_version")
    if plan["schema_version"] != version:
        raise ContractError("DATA_INVALID", "The plan and data manifest schema versions must match")
    for field in ("task_id", "data_version"):
        if manifest[field] != plan[field]:
            raise ContractError("DATA_INVALID", f"Manifest {field} does not match the plan")
    if manifest["problem_type"] != "classification":
        raise ContractError("DATA_INVALID", "Only classification is supported")
    if not _integer(manifest["n_features"], 1):
        raise ContractError("DATA_INVALID", "n_features must be a positive integer")
    names = manifest.get("feature_names")
    if version == "1.0" and (
        not isinstance(names, list)
        or not all(isinstance(name, str) and name.strip() for name in names)
        or len(names) != manifest["n_features"]
        or len(set(names)) != len(names)
    ):
        raise ContractError("DATA_INVALID", "feature_names must be nonempty and unique and match the column count")
    classes = manifest["classes"]
    if not isinstance(classes, list) or len(classes) < 2:
        raise ContractError("DATA_INVALID", "classes must contain at least two classes")
    for index, item in enumerate(classes):
        _fields(item, {"id", "name"}, "DATA_INVALID", "classes entry")
        if type(item["id"]) is not int or item["id"] != index:
            raise ContractError("DATA_INVALID", "classes must use consecutive IDs ordered from 0 to K-1")
        if not isinstance(item["name"], str) or not item["name"].strip():
            raise ContractError("DATA_INVALID", "A class name must be a nonempty string")
    if len({item["name"] for item in classes}) != len(classes):
        raise ContractError("DATA_INVALID", "Class names must be unique")
    if version == "2.0":
        representation = manifest["representation"]
        if not isinstance(representation, str) or representation not in {"raw_text", "image_tensor"}:
            raise ContractError("DATA_INVALID", "Manifest 2.0 supports raw_text and image_tensor")
        if plan.get("representation") != representation:
            raise ContractError("DATA_INVALID", "The plan and data manifest declare different representations")
        modality = manifest["modality"]
        if (
            not isinstance(modality, dict)
            or not isinstance(modality.get("kind"), str)
            or modality.get("kind") not in {"text", "image"}
        ):
            raise ContractError("DATA_INVALID", "The Manifest 2.0 modality declaration is invalid")
        if representation == "raw_text":
            if (
                set(modality) != {"kind", "text_columns", "max_characters"}
                or modality.get("kind") != "text"
                or not isinstance(modality.get("text_columns"), list)
                or not 1 <= len(modality["text_columns"]) <= 8
                or any(not isinstance(name, str) or not name.strip() for name in modality["text_columns"])
                or len(set(modality["text_columns"])) != len(modality["text_columns"])
                or manifest["n_features"] != len(modality["text_columns"])
                or not _integer(modality.get("max_characters"), 1, 100_000)
            ):
                raise ContractError("DATA_INVALID", "The raw_text modality does not match n_features")
        else:
            if set(modality) != {"kind", "image_shape", "channel_layout", "value_range"}:
                raise ContractError("DATA_INVALID", "The image modality has unsupported fields")
            try:
                from data_b.modality_readers import validate_image_spec

                shape, layout, value_range = validate_image_spec(
                    modality["image_shape"], modality["channel_layout"], modality["value_range"]
                )
            except (KeyError, ValueError) as exc:
                raise ContractError("DATA_INVALID", "The image modality parameters are invalid") from exc
            if (
                modality.get("kind") != "image"
                or math.prod(shape) != manifest["n_features"]
                or layout != modality["channel_layout"]
                or list(value_range) != [float(item) for item in modality["value_range"]]
            ):
                raise ContractError("DATA_INVALID", "The image modality does not match n_features")
    _fields(
        manifest["train"],
        {"features_path", "labels_path", "n_samples"},
        "DATA_INVALID",
        "train",
    )
    validation_fields = {"features_path", "sample_ids_path", "n_samples"}
    _fields(manifest["validation"], validation_fields, "DATA_INVALID", "validation")
    base = Path(manifest_path).resolve().parent
    for split in ("train", "validation"):
        if not _integer(manifest[split]["n_samples"], 1, _MAX_LABEL_ROWS):
            raise ContractError("DATA_INVALID", f"{split}.n_samples is outside the supported row count")
        if version == "1.0" and manifest[split]["n_samples"] * manifest["n_features"] * 4 > _MAX_DENSE_PAYLOAD_BYTES:
            raise ContractError("DATA_INVALID", f"{split} dense feature matrix exceeds the 256 MiB limit")
        for key in ("features_path", "labels_path", "sample_ids_path"):
            if key in manifest[split]:
                resolve_data_path(base, manifest[split][key])
    if manifest["preprocessor_path"] is not None:
        resolve_data_path(base, manifest["preprocessor_path"])
    if version == "2.0" and manifest["preprocessor_path"] is not None:
        raise ContractError("DATA_INVALID", "Manifest 2.0 must leave fitted transformers to the plan pipeline")
    return manifest


def read_sample_ids(path: str | Path, expected_count: int) -> list[str]:
    if Path(path).stat().st_size > _MAX_IDS_BYTES:
        raise ContractError("DATA_INVALID", "Validation IDs exceed the input size limit")
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != ["sample_id"]:
                raise ContractError("DATA_INVALID", "valid_ids.csv must have exactly one header, sample_id")
            ids = []
            for row in reader:
                if len(ids) >= expected_count:
                    raise ContractError("DATA_INVALID", "Validation IDs exceed the manifest row count")
                if (
                    len(row) != 1
                    or not row[0].strip()
                    or any(character in row[0] for character in "\r\n\0")
                ):
                    raise ContractError("DATA_INVALID", "Validation ID rows must contain one nonempty string")
                ids.append(row[0])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("DATA_INVALID", "Could not read the validation ID CSV") from exc
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ContractError("DATA_INVALID", "Validation IDs contain duplicates or differ from the manifest row count")
    return ids


@dataclass
class PreparedData:
    manifest: dict[str, Any]
    X_train: Any
    y_train: Any
    X_valid: Any
    sample_ids: list[str]
    class_ids: list[int]
    input_hashes: dict[str, str]


def _input_paths(manifest_path: str | Path, manifest: dict[str, Any]) -> dict[str, Path]:
    base = Path(manifest_path).resolve().parent
    files = {
        "manifest": Path(manifest_path).resolve(),
        "train_features": resolve_data_path(base, manifest["train"]["features_path"]),
        "train_labels": resolve_data_path(base, manifest["train"]["labels_path"]),
        "validation_features": resolve_data_path(base, manifest["validation"]["features_path"]),
        "validation_ids": resolve_data_path(base, manifest["validation"]["sample_ids_path"]),
    }
    if manifest["preprocessor_path"] is not None:
        files["preprocessor"] = resolve_data_path(base, manifest["preprocessor_path"])
    return files


def _hash_prepared_inputs(files: dict[str, Path], representation: str) -> dict[str, str]:
    from data_b.modality_readers import MAX_TEXT_JSONL_BYTES
    from data_b.npy_reader import ReaderLimits

    feature_limit = {
        "dense_numeric": _MAX_DENSE_PAYLOAD_BYTES + _MAX_NPY_HEADER_BYTES,
        "raw_text": MAX_TEXT_JSONL_BYTES,
        "image_tensor": ReaderLimits().max_payload_bytes + _MAX_NPY_HEADER_BYTES,
    }[representation]
    limits = {
        "manifest": _MAX_MANIFEST_BYTES,
        "train_features": feature_limit,
        "train_labels": _MAX_LABEL_ROWS * 8 + _MAX_NPY_HEADER_BYTES,
        "validation_features": feature_limit,
        "validation_ids": _MAX_IDS_BYTES,
        "preprocessor": 1024 * _MIB,
    }
    # Check every input before streaming any potentially large file.
    for name, path in files.items():
        if path.stat().st_size > limits[name]:
            raise ValueError(f"The prepared {name} file exceeds its size limit")
    return {name: sha256_file(path, max_bytes=limits[name]) for name, path in files.items()}


def load_prepared_data(manifest_path: str | Path, plan: dict[str, Any]) -> PreparedData:
    """Load the executor inputs and leave validation labels to the evaluator."""
    import numpy as np
    from data_b.npy_reader import ReaderLimits, read_npy_array

    manifest = validate_manifest(manifest_path, plan)
    files = _input_paths(manifest_path, manifest)
    representation = manifest.get("representation", "dense_numeric")
    try:
        hashes_before = _hash_prepared_inputs(files, representation)
        y_train = read_npy_array(
            files["train_labels"], expected_shape=(manifest["train"]["n_samples"],),
            allowed_dtype_kinds="i", allowed_item_sizes=(4, 8),
        )
        if representation == "raw_text":
            from data_b.modality_readers import read_text_matrix_jsonl

            modality = manifest["modality"]
            X_train = read_text_matrix_jsonl(
                files["train_features"],
                n_samples=manifest["train"]["n_samples"],
                text_columns=modality["text_columns"],
                max_characters=modality["max_characters"],
            )
            X_valid = read_text_matrix_jsonl(
                files["validation_features"],
                n_samples=manifest["validation"]["n_samples"],
                text_columns=modality["text_columns"],
                max_characters=modality["max_characters"],
            )
        else:
            if representation == "image_tensor":
                shape = tuple(manifest["modality"]["image_shape"])
                limits = None
                kinds, item_sizes = "fiu", None
            else:
                shape = (manifest["n_features"],)
                # Image-index tables can have more columns than the legacy tabular bound.
                limits = ReaderLimits(
                    max_columns=manifest["n_features"], max_payload_bytes=_MAX_DENSE_PAYLOAD_BYTES,
                )
                kinds, item_sizes = "f", (4, 8)
            X_train, X_valid = [
                read_npy_array(
                    files[f"{split}_features"], limits,
                    expected_shape=(manifest[split]["n_samples"], *shape),
                    allowed_dtype_kinds=kinds, allowed_item_sizes=item_sizes,
                )
                for split in ("train", "validation")
            ]
    except (OSError, ValueError, EOFError, TypeError) as exc:
        raise ContractError("DATA_INVALID", "Could not read train or validation features matching the manifest") from exc
    for name, value, count in (
        ("X_train", X_train, manifest["train"]["n_samples"]),
        ("X_valid", X_valid, manifest["validation"]["n_samples"]),
    ):
        if not isinstance(value, np.ndarray):
            raise ContractError("DATA_INVALID", f"{name} must be a .npy array")
        if representation == "dense_numeric":
            if value.ndim != 2 or value.shape != (count, manifest["n_features"]):
                raise ContractError("DATA_INVALID", f"{name} shape does not match the manifest")
            if value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8):
                raise ContractError("DATA_INVALID", f"{name} must be float32 or float64")
            if not np.isfinite(value).all():
                raise ContractError("DATA_INVALID", f"{name} contains NaN or infinity")
        elif representation == "raw_text":
            modality = manifest["modality"]
            if (
                value.ndim != 2
                or value.shape != (count, len(modality["text_columns"]))
                or value.dtype.kind != "O"
                or any(type(item) is not str for item in value.flat)
            ):
                raise ContractError("DATA_INVALID", f"{name} text shape or dtype differs from the manifest")
            if any(len(item) > modality["max_characters"] or "\0" in item for item in value.flat):
                raise ContractError("DATA_INVALID", f"{name} exceeds the text length or character limits")
        else:
            modality = manifest["modality"]
            expected = (count, *modality["image_shape"])
            if value.shape != expected or value.dtype.kind not in "fiu":
                raise ContractError("DATA_INVALID", f"{name} image shape or dtype differs from the manifest")
            lower, upper = modality["value_range"]
            if not np.isfinite(value).all() or (value < lower).any() or (value > upper).any():
                raise ContractError("DATA_INVALID", f"{name} contains invalid image pixels or pixels outside the declared range")
    if (
        not isinstance(y_train, np.ndarray)
        or y_train.ndim != 1
        or y_train.shape[0] != X_train.shape[0]
        or y_train.dtype.kind != "i"
        or y_train.dtype.itemsize not in (4, 8)
    ):
        raise ContractError("DATA_INVALID", "y_train must be a one-dimensional int32 or int64 array matching the training row count")
    class_ids = [item["id"] for item in manifest["classes"]]
    if not np.array_equal(np.unique(y_train), np.asarray(class_ids)):
        raise ContractError("DATA_INVALID", "Training labels must cover exactly classes 0..K-1")
    sample_ids = read_sample_ids(files["validation_ids"], X_valid.shape[0])
    try:
        hashes = _hash_prepared_inputs(files, representation)
    except (OSError, ValueError) as exc:
        raise ContractError("DATA_INVALID", "An input became invalid while reading") from exc
    if hashes != hashes_before:
        raise ContractError("DATA_INVALID", "Inputs changed while reading. Freeze a data version and retry.")
    return PreparedData(manifest, X_train, y_train, X_valid, sample_ids, class_ids, hashes)


def validate_prediction_file(
    path: str | Path,
    *,
    sample_ids: list[str],
    class_ids: list[int],
) -> int:
    """Validate probability columns, row alignment, and probability values."""
    expected_header = ["sample_id", "y_pred"] + [f"p_{item}" for item in class_ids]
    try:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != expected_header:
                raise ContractError("ARTIFACT_INVALID", "The prediction header or probability column order is invalid")
            count = 0
            for row in reader:
                if len(row) != len(expected_header) or count >= len(sample_ids):
                    raise ContractError("ARTIFACT_INVALID", "The prediction column or row count is invalid")
                if row[0] != sample_ids[count]:
                    raise ContractError("ARTIFACT_INVALID", "Prediction sample IDs differ from the prepared data order")
                try:
                    label = int(row[1])
                    probabilities = [float(item) for item in row[2:]]
                except (ValueError, OverflowError) as exc:
                    raise ContractError("ARTIFACT_INVALID", "Predictions contain a nonnumeric class ID or probability") from exc
                if str(label) != row[1] or label not in class_ids:
                    raise ContractError("ARTIFACT_INVALID", "Predictions contain a class ID outside the accepted integer IDs")
                if (
                    not all(math.isfinite(item) and 0 <= item <= 1 for item in probabilities)
                    or abs(math.fsum(probabilities) - 1.0) > 1e-6
                ):
                    raise ContractError("ARTIFACT_INVALID", "Prediction probabilities contain invalid values or row sums")
                count += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("ARTIFACT_INVALID", "Could not read the full predictions CSV") from exc
    if count != len(sample_ids):
        raise ContractError("ARTIFACT_INVALID", "Prediction row count differs from the validation sample count")
    return count


def inspect_artifacts(
    run_dir: str | Path,
    manifest_path: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify the worker artifacts before publishing a successful result."""
    run_dir = Path(run_dir).resolve()
    for name, limit in (
        ("model.joblib", 1024 * _MIB),
        ("predictions.csv", 512 * _MIB),
        ("worker_metadata.json", _MIB),
    ):
        file_path = run_dir / name
        if not file_path.is_file() or file_path.resolve().parent != run_dir:
            raise ContractError("ARTIFACT_INVALID", "Training did not generate all experiment artifacts")
        if not 0 < file_path.stat().st_size <= limit:
            raise ContractError("ARTIFACT_INVALID", f"{name} is empty or exceeds the artifact size limit")
    metadata = read_json(run_dir / "worker_metadata.json", error_code="ARTIFACT_INVALID", max_bytes=_MIB)
    class_ids = [item["id"] for item in manifest["classes"]]
    if (
        metadata.get("class_ids") != class_ids
        or not isinstance(metadata.get("class_ids"), list)
        or any(type(item) is not int for item in metadata["class_ids"])
    ):
        raise ContractError("ARTIFACT_INVALID", "The worker class mapping differs from the data manifest")
    if (
        type(metadata.get("n_predictions")) is not int
        or metadata["n_predictions"] != manifest["validation"]["n_samples"]
    ):
        raise ContractError("ARTIFACT_INVALID", "The worker prediction row count differs from the manifest")
    input_hashes = metadata.get("input_hashes")
    expected_hashes = {
        "manifest",
        "train_features",
        "train_labels",
        "validation_features",
        "validation_ids",
    }
    if manifest["preprocessor_path"] is not None:
        expected_hashes.add("preprocessor")
    if (
        not isinstance(input_hashes, dict)
        or set(input_hashes) != expected_hashes
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
            for value in input_hashes.values()
        )
    ):
        raise ContractError("ARTIFACT_INVALID", "The worker lacks the full input file checksums")
    try:
        if read_json(manifest_path, max_bytes=_MAX_MANIFEST_BYTES) != manifest:
            raise ContractError("ARTIFACT_INVALID", "The data manifest description changed during the run")
        files = _input_paths(manifest_path, manifest)
        current_hashes = _hash_prepared_inputs(files, manifest.get("representation", "dense_numeric"))
    except (OSError, ValueError) as exc:
        raise ContractError("ARTIFACT_INVALID", "An input became unavailable during the run. Check the frozen data version.") from exc
    if input_hashes != current_hashes:
        raise ContractError("ARTIFACT_INVALID", "An input changed during the run. Use a new data version.")
    ids_path = files["validation_ids"]
    ids = read_sample_ids(ids_path, manifest["validation"]["n_samples"])
    validate_prediction_file(run_dir / "predictions.csv", sample_ids=ids, class_ids=class_ids)
    return {
        "n_predictions": metadata["n_predictions"],
        "class_ids": class_ids,
        "input_hashes": input_hashes,
    }
