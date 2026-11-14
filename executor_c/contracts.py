"""A/B/C/D 共用的输入与文件约定；这里不训练模型、不计算评分。"""

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
)

SCHEMA_VERSION = "1.0"
MAX_TIMEOUT_SECONDS = REGISTRY_MAX_TIMEOUT_SECONDS
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
    """可记录在结果中的输入/产物协议错误。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class InvocationError(ValueError):
    """尚不能建立独立实验目录的调用错误，不会覆盖旧结果。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def capabilities() -> dict[str, Any]:
    """A 可以把这份能力清单放进规划提示；C 仍会重新检查输出。"""
    return registry_capabilities()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 中存在重复字段")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("JSON 不允许 NaN 或 Infinity")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON 浮点数必须是有限值")
    return number


def read_json(path: str | Path, error_code: str = "DATA_INVALID") -> dict[str, Any]:
    """读取标准 JSON 对象；不回显原始文件内容，以免错误输入进入日志。"""
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (OSError, ValueError, UnicodeError) as exc:
        raise ContractError(error_code, "无法读取合法的 UTF-8 JSON 对象") from exc
    if not isinstance(value, dict):
        raise ContractError(error_code, "JSON 最外层必须是对象")
    return value


def write_json(path: str | Path, value: Any, exclusive: bool = True) -> None:
    """默认使用 x 模式，避免意外覆盖计划、快照和运行证据。"""
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def publish_json(path: str | Path, value: Any) -> None:
    """写完临时文件后再发布最终结果；每个实验目录只有一个写入者。"""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("最终结果已经存在，禁止覆盖")
    pending = destination.with_name(destination.stem + ".pending.json")
    write_json(pending, value)
    pending.rename(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scikit-learn", "joblib"):
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
        raise ContractError(code, f"{context} 的字段缺失或包含未支持字段，请对照接口说明")


def _integer(value: Any, minimum: int, maximum: int | None = None) -> bool:
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def validate_identity(plan: Any) -> tuple[str, str, str]:
    if not isinstance(plan, dict):
        raise InvocationError("PLAN_INVALID", "计划必须是对象")
    identifiers = []
    for key in ("task_id", "data_version", "experiment_id"):
        value = plan.get(key)
        if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
            raise InvocationError("PLAN_INVALID", f"{key} 必须是 1–64 位安全英文编号")
        identifiers.append(value)
    return tuple(identifiers)


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ContractError("PLAN_INVALID", "实验计划必须是对象")
    version = plan.get("schema_version")
    expected = PLAN_FIELDS if version == "1.0" else PLAN_V2_FIELDS if version == "2.0" else set()
    _fields(plan, expected, "PLAN_INVALID", "实验计划")
    try:
        validate_identity(plan)
    except InvocationError as exc:
        raise ContractError(exc.code, exc.message) from exc
    if version not in {"1.0", "2.0"}:
        raise ContractError("PLAN_INVALID", "不支持此 schema_version")
    model = plan["model"]
    if not isinstance(model, str) or model not in MODELS:
        raise ContractError("UNSUPPORTED_MODEL", "model 不在 capabilities.json 的模型清单中")
    if not isinstance(plan["params"], dict):
        raise ContractError("PLAN_INVALID", "params 必须是对象")
    try:
        resolved = normalize_params(model, plan["params"])
    except ValueError as exc:
        code = "UNSUPPORTED_PARAMETER" if "unsupported fields" in str(exc) else "PLAN_INVALID"
        raise ContractError(code, str(exc)) from exc
    if not _integer(plan["random_seed"], 0, 2**32 - 1):
        raise ContractError("PLAN_INVALID", "random_seed 必须是 0 到 2**32-1 的整数")
    if not _integer(plan["timeout_seconds"], 1, MAX_TIMEOUT_SECONDS):
        raise ContractError("PLAN_INVALID", "timeout_seconds 必须是 1 到 600 的整数")
    normalized = {**plan, "params": resolved}
    if version == "2.0":
        representation = plan["representation"]
        if not isinstance(representation, str) or representation not in {
            "dense_numeric", "raw_text", "image_tensor"
        }:
            raise ContractError("PLAN_INVALID", "representation 不受支持")
        if representation not in MODELS[model].representations:
            raise ContractError("UNSUPPORTED_MODEL", "模型不支持计划声明的 representation")
        try:
            normalized["pipeline"] = normalize_pipeline(representation, plan["pipeline"])
        except (TypeError, ValueError) as exc:
            raise ContractError("PLAN_INVALID", str(exc)) from exc
    return normalized


def resolve_data_path(base_dir: str | Path, relative: Any) -> Path:
    """文件位置只能位于清单目录内；不依赖当前工作目录。"""
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in PurePosixPath(relative).parts
    ):
        raise ContractError("DATA_INVALID", "数据路径必须是包内的相对 POSIX 路径")
    base = Path(base_dir).resolve()
    try:
        candidate = (base / relative).resolve()
        candidate.relative_to(base)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ContractError("DATA_INVALID", "数据文件路径越出数据包") from exc
    if not candidate.is_file():
        raise ContractError("DATA_INVALID", "清单指向的数据文件不存在或不是普通文件")
    return candidate


def validate_manifest(manifest_path: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    """先检查元数据；数组内容在训练子进程内检查，计入时间限制。"""
    manifest = read_json(manifest_path)
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
        raise ContractError("DATA_INVALID", "数据清单 schema_version 不受支持")
    if plan["schema_version"] != version:
        raise ContractError("DATA_INVALID", "Plan 与 data_manifest schema_version 必须匹配")
    for field in ("task_id", "data_version"):
        if manifest[field] != plan[field]:
            raise ContractError("DATA_INVALID", f"数据清单的 {field} 与计划不一致")
    if manifest["problem_type"] != "classification":
        raise ContractError("DATA_INVALID", "首版只支持 classification")
    if not _integer(manifest["n_features"], 1):
        raise ContractError("DATA_INVALID", "n_features 必须是正整数")
    names = manifest.get("feature_names")
    if version == "1.0" and (
        not isinstance(names, list)
        or not all(isinstance(name, str) and name.strip() for name in names)
        or len(names) != manifest["n_features"]
        or len(set(names)) != len(names)
    ):
        raise ContractError("DATA_INVALID", "feature_names 必须是与列数一致的非空唯一名称列表")
    classes = manifest["classes"]
    if not isinstance(classes, list) or len(classes) < 2:
        raise ContractError("DATA_INVALID", "classes 必须至少包含两个类别")
    for index, item in enumerate(classes):
        _fields(item, {"id", "name"}, "DATA_INVALID", "classes 条目")
        if type(item["id"]) is not int or item["id"] != index:
            raise ContractError("DATA_INVALID", "classes 必须按 0..K-1 的连续整数 id 排列")
        if not isinstance(item["name"], str) or not item["name"].strip():
            raise ContractError("DATA_INVALID", "类别 name 必须是非空字符串")
    if len({item["name"] for item in classes}) != len(classes):
        raise ContractError("DATA_INVALID", "类别 name 不能重复")
    if version == "2.0":
        representation = manifest["representation"]
        if not isinstance(representation, str) or representation not in {"raw_text", "image_tensor"}:
            raise ContractError("DATA_INVALID", "Manifest 2.0 仅支持 raw_text/image_tensor")
        if plan.get("representation") != representation:
            raise ContractError("DATA_INVALID", "计划与数据清单 representation 不一致")
        modality = manifest["modality"]
        if (
            not isinstance(modality, dict)
            or not isinstance(modality.get("kind"), str)
            or modality.get("kind") not in {"text", "image"}
        ):
            raise ContractError("DATA_INVALID", "Manifest 2.0 modality 声明非法")
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
                raise ContractError("DATA_INVALID", "raw_text modality 与 n_features 不一致")
        else:
            if set(modality) != {"kind", "image_shape", "channel_layout", "value_range"}:
                raise ContractError("DATA_INVALID", "image modality 字段不受支持")
            try:
                from data_b.modality_readers import validate_image_spec

                shape, layout, value_range = validate_image_spec(
                    modality["image_shape"], modality["channel_layout"], modality["value_range"]
                )
            except (KeyError, ValueError) as exc:
                raise ContractError("DATA_INVALID", "image modality 参数非法") from exc
            if (
                modality.get("kind") != "image"
                or math.prod(shape) != manifest["n_features"]
                or layout != modality["channel_layout"]
                or list(value_range) != [float(item) for item in modality["value_range"]]
            ):
                raise ContractError("DATA_INVALID", "image modality 与 n_features 不一致")
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
        if not _integer(manifest[split]["n_samples"], 1):
            raise ContractError("DATA_INVALID", f"{split}.n_samples 必须是正整数")
        for key in ("features_path", "labels_path", "sample_ids_path"):
            if key in manifest[split]:
                resolve_data_path(base, manifest[split][key])
    if manifest["preprocessor_path"] is not None:
        resolve_data_path(base, manifest["preprocessor_path"])
    if version == "2.0" and manifest["preprocessor_path"] is not None:
        raise ContractError("DATA_INVALID", "Manifest 2.0 的拟合转换器必须由 C 的 Plan pipeline 管理")
    return manifest


def read_sample_ids(path: str | Path, expected_count: int) -> list[str]:
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != ["sample_id"]:
                raise ContractError("DATA_INVALID", "valid_ids.csv 表头必须只有 sample_id")
            ids = []
            for row in reader:
                if (
                    len(row) != 1
                    or not row[0].strip()
                    or any(character in row[0] for character in "\r\n\0")
                ):
                    raise ContractError("DATA_INVALID", "验证编号行必须是单列非空字符串")
                ids.append(row[0])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("DATA_INVALID", "无法读取验证样本编号 CSV") from exc
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ContractError("DATA_INVALID", "验证编号重复，或行数与清单不符")
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


def load_prepared_data(manifest_path: str | Path, plan: dict[str, Any]) -> PreparedData:
    """读取 B 的白名单输入，不打开 evaluation_manifest 或验证真实标签。"""
    import numpy as np
    from data_b.npy_reader import read_npy_array

    manifest = validate_manifest(manifest_path, plan)
    files = _input_paths(manifest_path, manifest)
    hashes_before = {name: sha256_file(path) for name, path in files.items()}
    representation = manifest.get("representation", "dense_numeric")
    try:
        y_train = np.load(files["train_labels"], allow_pickle=False)
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
                X_train = read_npy_array(files["train_features"])
                X_valid = read_npy_array(files["validation_features"])
            else:
                X_train = np.load(files["train_features"], allow_pickle=False)
                X_valid = np.load(files["validation_features"], allow_pickle=False)
    except (OSError, ValueError, EOFError, TypeError) as exc:
        raise ContractError("DATA_INVALID", "无法读取符合清单限制的训练/验证特征") from exc
    for name, value, count in (
        ("X_train", X_train, manifest["train"]["n_samples"]),
        ("X_valid", X_valid, manifest["validation"]["n_samples"]),
    ):
        if not isinstance(value, np.ndarray):
            raise ContractError("DATA_INVALID", f"{name} 必须是 .npy 数组")
        if representation == "dense_numeric":
            if value.ndim != 2 or value.shape != (count, manifest["n_features"]):
                raise ContractError("DATA_INVALID", f"{name} 的 shape 与清单不一致")
            if value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8):
                raise ContractError("DATA_INVALID", f"{name} 必须是 float32 或 float64")
            if not np.isfinite(value).all():
                raise ContractError("DATA_INVALID", f"{name} 包含 NaN 或无穷值")
        elif representation == "raw_text":
            modality = manifest["modality"]
            if (
                value.ndim != 2
                or value.shape != (count, len(modality["text_columns"]))
                or value.dtype.kind != "O"
                or any(type(item) is not str for item in value.flat)
            ):
                raise ContractError("DATA_INVALID", f"{name} 文本 shape/dtype 与清单不一致")
            if any(len(item) > modality["max_characters"] or "\0" in item for item in value.flat):
                raise ContractError("DATA_INVALID", f"{name} 文本内容越过长度或字符限制")
        else:
            modality = manifest["modality"]
            expected = (count, *modality["image_shape"])
            if value.shape != expected or value.dtype.kind not in "fiu":
                raise ContractError("DATA_INVALID", f"{name} 图像 shape/dtype 与清单不一致")
            lower, upper = modality["value_range"]
            if not np.isfinite(value).all() or (value < lower).any() or (value > upper).any():
                raise ContractError("DATA_INVALID", f"{name} 图像像素无效或超出声明范围")
    if (
        not isinstance(y_train, np.ndarray)
        or y_train.ndim != 1
        or y_train.shape[0] != X_train.shape[0]
        or y_train.dtype.kind != "i"
        or y_train.dtype.itemsize not in (4, 8)
    ):
        raise ContractError("DATA_INVALID", "y_train 必须是与训练行数一致的 int32/int64 一维数组")
    class_ids = [item["id"] for item in manifest["classes"]]
    if not np.array_equal(np.unique(y_train), np.asarray(class_ids)):
        raise ContractError("DATA_INVALID", "训练标签必须恰好覆盖全部 0..K-1 类别")
    sample_ids = read_sample_ids(files["validation_ids"], X_valid.shape[0])
    hashes = {name: sha256_file(path) for name, path in files.items()}
    if hashes != hashes_before:
        raise ContractError("DATA_INVALID", "读取期间输入文件发生变化，请冻结数据版本后重试")
    return PreparedData(manifest, X_train, y_train, X_valid, sample_ids, class_ids, hashes)


def validate_prediction_file(
    path: str | Path,
    *,
    sample_ids: list[str],
    class_ids: list[int],
) -> int:
    """检查列含义、逐行样本对齐和概率，不计算任何评分。"""
    expected_header = ["sample_id", "y_pred"] + [f"p_{item}" for item in class_ids]
    try:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != expected_header:
                raise ContractError("ARTIFACT_INVALID", "预测文件表头或类别概率列顺序错误")
            count = 0
            for row in reader:
                if len(row) != len(expected_header) or count >= len(sample_ids):
                    raise ContractError("ARTIFACT_INVALID", "预测文件的列数或行数错误")
                if row[0] != sample_ids[count]:
                    raise ContractError("ARTIFACT_INVALID", "预测样本编号与 B 的顺序不一致")
                try:
                    label = int(row[1])
                    probabilities = [float(item) for item in row[2:]]
                except (ValueError, OverflowError) as exc:
                    raise ContractError("ARTIFACT_INVALID", "预测类别或概率不是合法数字") from exc
                if str(label) != row[1] or label not in class_ids:
                    raise ContractError("ARTIFACT_INVALID", "预测类别不在有效整数类别集合内")
                if (
                    not all(math.isfinite(item) and 0 <= item <= 1 for item in probabilities)
                    or abs(math.fsum(probabilities) - 1.0) > 1e-6
                ):
                    raise ContractError("ARTIFACT_INVALID", "预测概率范围、有限性或概率和不合法")
                count += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("ARTIFACT_INVALID", "无法读取完整的预测 CSV") from exc
    if count != len(sample_ids):
        raise ContractError("ARTIFACT_INVALID", "预测行数不等于验证样本数")
    return count


def inspect_artifacts(
    run_dir: str | Path,
    manifest_path: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """父执行器独立检查产物后，才允许发布 success。"""
    run_dir = Path(run_dir).resolve()
    for name in ("model.joblib", "predictions.csv", "worker_metadata.json"):
        file_path = run_dir / name
        if not file_path.is_file() or file_path.resolve().parent != run_dir:
            raise ContractError("ARTIFACT_INVALID", "训练未生成完整的本实验产物")
        if file_path.stat().st_size == 0:
            raise ContractError("ARTIFACT_INVALID", "训练产物为空")
    metadata = read_json(run_dir / "worker_metadata.json", error_code="ARTIFACT_INVALID")
    class_ids = [item["id"] for item in manifest["classes"]]
    if (
        metadata.get("class_ids") != class_ids
        or not isinstance(metadata.get("class_ids"), list)
        or any(type(item) is not int for item in metadata["class_ids"])
    ):
        raise ContractError("ARTIFACT_INVALID", "worker 类别映射与数据清单不一致")
    if (
        type(metadata.get("n_predictions")) is not int
        or metadata["n_predictions"] != manifest["validation"]["n_samples"]
    ):
        raise ContractError("ARTIFACT_INVALID", "worker 预测行数与清单不一致")
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
        raise ContractError("ARTIFACT_INVALID", "worker 缺少完整输入文件摘要")
    try:
        if read_json(manifest_path) != manifest:
            raise ContractError("ARTIFACT_INVALID", "运行中数据清单描述发生变化")
        files = _input_paths(manifest_path, manifest)
        current_hashes = {name: sha256_file(path) for name, path in files.items()}
    except (ContractError, OSError) as exc:
        raise ContractError("ARTIFACT_INVALID", "运行中输入文件失效，请检查冻结的数据版本") from exc
    if input_hashes != current_hashes:
        raise ContractError("ARTIFACT_INVALID", "运行中输入文件发生变化，请使用新数据版本")
    ids_path = files["validation_ids"]
    ids = read_sample_ids(ids_path, manifest["validation"]["n_samples"])
    validate_prediction_file(run_dir / "predictions.csv", sample_ids=ids, class_ids=class_ids)
    return {
        "n_predictions": metadata["n_predictions"],
        "class_ids": class_ids,
        "input_hashes": input_hashes,
    }
