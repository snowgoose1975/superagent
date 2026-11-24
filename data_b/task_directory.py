"""Discover bounded classification inputs without opening the final-test files.

Only train/validation are inspected during discovery.  A descriptor records the
selected reader; D materializes its test inputs after the model is frozen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .npy_reader import normalize_classification_labels, read_npy_array
from .raw_reader import typed_category_token
from .sample_index import save_sample_index

_MAX_CSV_BYTES = 64 * 1024 * 1024
_MAX_ROWS = 2_000_000
_MAX_ELEMENTS = 8_000_000
_MAX_FIELD_CHARACTERS = 100_000
_MAX_TEXT_CHARACTERS = 8_000_000
_SPLITS = {"train": "train", "val": "validation", "valid": "validation",
           "validation": "validation", "test": "test"}
_MISSING = {"", "nan", "na", "n/a", "null", "none"}


class TaskDirectoryError(ValueError):
    """The task directory does not have an unambiguous supported input contract."""

    code = "TASK_DIRECTORY_INVALID"


def _missing_content(value: Any) -> bool:
    if value is None or isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return True
    return isinstance(value, (str, bytes, np.str_, np.bytes_)) and not value.strip()


def _keep_present_samples(reader: dict, split: str, sample_rows: list[dict], reasons: list[list[str]]) -> list[int]:
    """Record the user-selected missing-content policy without renumbering rows."""
    kept = [index for index, items in enumerate(reasons) if not items]
    dropped = []
    n_dropped = 0
    counts: dict[str, int] = {}
    for row, items in zip(sample_rows, reasons, strict=True):
        if items:
            n_dropped += 1
            if len(dropped) < 1000:
                dropped.append({**row, "reasons": items})
        for reason in items:
            counts[reason] = counts.get(reason, 0) + 1
    reader.setdefault("sample_filtering", {})[split] = {
        "input_count": len(sample_rows), "kept_count": len(kept), "dropped_count": n_dropped,
        "reason_counts": counts, "dropped_samples": dropped,
        "details_truncated": n_dropped > 1000,
    }
    if split == "test" and n_dropped:
        details = ", ".join(f"{reason}: {count}" for reason, count in sorted(counts.items())[:8])
        raise TaskDirectoryError(
            f"Final test requires all {len(sample_rows)} rows. "
            f"{n_dropped} rows have missing required content ({details})"
        )
    if not kept:
        raise TaskDirectoryError(f"No usable {split} samples remain after removing missing text/images")
    return kept


def _filter_array_samples(values, labels, sample_rows, reader, split):
    if len(values) != len(sample_rows) or (labels is not None and len(values) != len(labels)):
        raise TaskDirectoryError("Features, labels and sample IDs have different row counts")
    if reader["output_kind"] == "image":
        if values.dtype.kind in "biu":
            reasons = [[] for _ in values]
        elif values.dtype.kind == "f":
            missing = np.isnan(values).reshape(len(values), -1).all(axis=1)
            reasons = [["missing_image"] if value else [] for value in missing]
        else:
            reasons = [["missing_image"] if all(_missing_content(value) for value in image.flat) else [] for image in values]
    else:
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        columns = reader["feature_columns"]
        if values.ndim != 2 or values.shape[1] != len(columns):
            raise TaskDirectoryError("Feature width does not match the training schema")
        kinds = reader["feature_types"]
        text_indices = [(index, name) for index, name in enumerate(columns) if kinds[name] == "text"]
        reasons = [[f"missing_text:{name}" for index, name in text_indices if _missing_content(row[index])]
                   for row in values]
    kept = _keep_present_samples(reader, split, sample_rows, reasons)
    if len(kept) == len(values):
        return values, labels, sample_rows
    return values[kept], None if labels is None else labels[kept], [sample_rows[index] for index in kept]


def _filter_csv_samples(root, header, rows, reader, source, split):
    from .input_readers import filter_record_samples

    records = [dict(zip(header, row, strict=True)) for row in rows]
    _, positions = filter_record_samples(root, records, reader, source, split, reader["feature_types"])
    return [rows[index] for index in positions], positions


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file(root: Path, relative: Any) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            or any(part.startswith(".") for part in Path(relative).parts)):
        raise TaskDirectoryError("Task inputs must use visible relative paths within the task directory")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise TaskDirectoryError(f"Task file is missing or escapes the task directory: {relative}")
    return path


def _scalar(value: Any) -> Any:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    token = typed_category_token(value)
    if not isinstance(token, str):
        raise TaskDirectoryError("Classification labels cannot be missing")
    if type(value) not in (str, int, float, bool):
        raise TaskDirectoryError("Classification labels must be strings, numbers, or booleans")
    return value


def _labels(values: np.ndarray) -> np.ndarray:
    values = normalize_classification_labels(values)
    if values.ndim != 1 or not 1 <= len(values) <= _MAX_ROWS:
        raise TaskDirectoryError("Classification labels must be a non-empty one-dimensional array")
    return np.asarray([_scalar(value) for value in values], dtype=object)


def _label_schema(values: np.ndarray) -> tuple[list[Any], list[str]]:
    unique = {typed_category_token(value): value for value in values}
    labels = [unique[token] for token in sorted(unique)]
    if not 2 <= len(labels) <= 1024:
        raise TaskDirectoryError("Classification requires 2 to 1024 training classes")
    names = [str(value) for value in labels]
    if len(set(names)) != len(names):
        names = [f"{type(value).__name__}:{value}" for value in labels]
    return labels, names


def _save_array(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=True)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    if path.stat().st_size > _MAX_CSV_BYTES:
        raise TaskDirectoryError("CSV exceeds the 64 MiB input limit")
    rows: list[list[str]] = []
    total_characters = 0
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if (not header or len(header) > 4096 or len(set(header)) != len(header)
                    or any(not name.strip() or "\0" in name for name in header)):
                raise TaskDirectoryError("CSV requires unique non-empty column names")
            for row in reader:
                if (len(row) != len(header) or len(rows) >= _MAX_ROWS
                        or (len(rows) + 1) * len(header) > _MAX_ELEMENTS):
                    raise TaskDirectoryError("CSV row width is inconsistent or input exceeds row or element limits")
                if any("\0" in value or len(value) > _MAX_FIELD_CHARACTERS for value in row):
                    raise TaskDirectoryError("CSV contains NUL or a cell exceeds 100000 characters")
                total_characters += sum(map(len, row))
                if total_characters > _MAX_TEXT_CHARACTERS:
                    raise TaskDirectoryError("CSV exceeds the 8000000-character limit")
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TaskDirectoryError(f"Unable to read CSV safely: {path.name}") from exc
    if not rows:
        raise TaskDirectoryError("CSV cannot be empty")
    return header, rows


def _read_npy_matrix(root: Path, paths: list[str]) -> tuple[np.ndarray, list[int]]:
    arrays = [read_npy_array(_source_file(root, relative)) for relative in paths]
    if len(arrays) == 1:
        return arrays[0], [arrays[0].shape[1] if arrays[0].ndim == 2 else 1]
    if (any(array.ndim != 2 for array in arrays)
            or len({len(array) for array in arrays}) != 1):
        raise TaskDirectoryError("N/C features must be two-dimensional and have equal row counts")
    widths = [array.shape[1] for array in arrays]
    if sum(widths) > 4096 or len(arrays[0]) * sum(widths) > _MAX_ELEMENTS:
        raise TaskDirectoryError("Combined N/C features exceed the dimension limit")
    output = np.empty((len(arrays[0]), sum(widths)), dtype=object)
    offset = 0
    for array, width in zip(arrays, widths, strict=True):
        output[:, offset:offset + width] = array
        offset += width
    return output, widths


def _metric(
    info: dict[str, Any], readme: str, *, declarations_only: bool = False,
) -> tuple[dict[str, Any], str]:
    accuracy_zh = "\u51c6\u786e\u7387"
    log_loss_zh = "\u5bf9\u6570\u635f\u5931"
    cross_entropy_zh = "\u4ea4\u53c9\u71b5"
    macro_f1_zh = "\u5b8f"
    aliases = {accuracy_zh: "accuracy", log_loss_zh: "log_loss", cross_entropy_zh: "log_loss",
               f"{macro_f1_zh}f1": "f1_macro", f"{macro_f1_zh} f1": "f1_macro",
               "accuracy": "accuracy", "acc": "accuracy", "f1_macro": "f1_macro",
               "macro_f1": "f1_macro", "macro-f1": "f1_macro", "macro f1": "f1_macro",
               "f1 macro": "f1_macro", "log_loss": "log_loss", "logloss": "log_loss",
               "log loss": "log_loss", "log-loss": "log_loss", "cross_entropy": "log_loss"}
    explicit: Any = None
    for field in ("primary_metric", "metric", "evaluation_metric", "eval_metric"):
        if field in info:
            explicit = info[field]
            if isinstance(explicit, dict):
                explicit = explicit.get("name")
            break
    if explicit is not None:
        if not isinstance(explicit, str) or explicit.strip().lower() not in aliases:
            raise TaskDirectoryError(f"info.json declares an unsupported metric: {explicit!r}")
        name, source = aliases[explicit.strip().lower()], "info.json"
    else:
        text = re.sub(r"[*`]", "", readme.lower())
        # A primary declaration outranks later diagnostic metric mentions.  In
        # particular, an unsupported primary AUC cannot silently become accuracy.
        metric_patterns = {
            "accuracy": rf"(?<![a-z0-9_])(?:accuracy|acc)(?![a-z0-9_])|{accuracy_zh}",
            "f1_macro": rf"(?<![a-z0-9_])(?:macro[ _-]f1|f1[ _-]macro)(?![a-z0-9_])|{macro_f1_zh}[ _-]*f1",
            "log_loss": rf"(?<![a-z0-9_])(?:log[ _-]?loss|cross[ _-]entropy)(?![a-z0-9_])|{log_loss_zh}|{cross_entropy_zh}",
        }

        def declared_name(fragment: str) -> str:
            fragment = re.sub(r"^(?:(?:the|test|validation|set)\s+)+", "", fragment.strip())
            fragment = fragment.lstrip('"\'“‘')
            for name, pattern in metric_patterns.items():
                if re.match(pattern, fragment):
                    return name
            raise TaskDirectoryError(f"README declares an unsupported metric: {fragment[:60]!r}")

        primary = re.findall(
            r"\b(?:primary|main)[ _-]+(?:evaluation[ _-]+)?metric\s*(?:is\s+|[:=]\s*)?([^\n.;]+)",
            text)
        primary.extend(re.findall(
            r"\u4e3b(?:\u8981)?(?:\u8bc4\u4f30|\u8bc4\u4ef7)?\u6307\u6807"
            r"\s*(?:\u4e3a|\u662f|[:=\uff1a])?\s*([^\n.;\u3002\uff1b\uff0c]+)", text))
        final = re.findall(
            r"\breport\s+(?:the\s+)?(?:final\s+)?test(?:\s+set)?\s+"
            r"(?:(?:primary\s+)?metrics?\s*[:=]\s*)?([^\n.;]+)", text)
        declarations = primary or final
        if declarations:
            declared = {declared_name(fragment) for fragment in declarations}
            if len(declared) != 1:
                raise TaskDirectoryError("README contains conflicting primary metrics")
            name = declared.pop()
            return {"name": name, "greater_is_better": name != "log_loss"}, "README_primary" if primary else "README_test_report"
        if declarations_only:
            return {"name": "accuracy", "greater_is_better": True}, "default_unspecified"
        found = set()
        for name, pattern in metric_patterns.items():
            if re.search(pattern, text):
                found.add(name)
        if len(found) > 1:
            raise TaskDirectoryError("README does not identify one primary metric")
        if found:
            name, source = found.pop(), "README"
        elif re.search(r"\b(?:auc|roc_auc|rmse|mae|mse|precision|recall|f1|r2)\b", text):
            raise TaskDirectoryError("Unsupported README metric. Use accuracy, f1_macro, or log_loss")
        else:
            name, source = "accuracy", "default_unspecified"
    return {"name": name, "greater_is_better": name != "log_loss"}, source


def _declared_columns(info: dict[str, Any], readme: str, width: int) -> list[str] | None:
    for field in ("feature_columns", "feature_names", "columns"):
        columns = info.get(field)
        if isinstance(columns, list):
            if (len(columns) != width or len(set(columns)) != width
                    or any(not isinstance(name, str) or not name.strip() for name in columns)):
                raise TaskDirectoryError(f"info.json {field} does not match the training feature count")
            return list(columns)
    match = re.search(r"columns[^\n]*?from left to right[^\n]*?\b(?:are|:)\s*(.+)",
                      readme, flags=re.IGNORECASE)
    if match:
        columns = re.findall(r"`([^`]+)`", match.group(1))
        if columns:
            if len(columns) != width or len(set(columns)) != width:
                raise TaskDirectoryError("README column order does not match the training feature width")
            return columns
    return None


def _target_column(info: dict[str, Any], readme: str, columns: list[str]) -> str:
    declarations = []
    for field in ("target_column", "label_column", "target", "target_name"):
        if field in info:
            value = info[field]
            value = value.get("name") if isinstance(value, dict) else value
            if not isinstance(value, str):
                raise TaskDirectoryError(f"info.json {field} must identify the target column")
            declarations.append(value)
    declarations.extend(re.findall(
        r"\b(?:target|label|response)(?:\s+column)?\s*(?:is|:|=)?\s*[`\"']([^`\"']+)[`\"']",
        readme, flags=re.IGNORECASE))
    declarations = list(dict.fromkeys(declarations))
    if declarations:
        if len(declarations) != 1 or declarations[0] not in columns:
            raise TaskDirectoryError("The README or info.json target is ambiguous or absent from the CSV")
        return declarations[0]
    candidates = [name for name in columns if name.casefold() in {"label", "labels", "target", "y"}]
    if len(candidates) != 1:
        raise TaskDirectoryError("CSV requires one target column declared in README or info.json")
    return candidates[0]


def _id_columns(info: dict[str, Any], readme: str, columns: list[str], target: str) -> list[str]:
    declared = info.get("id_columns", info.get("id_column", []))
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list) or any(not isinstance(value, str) for value in declared):
        raise TaskDirectoryError("info id_column or id_columns must name one or more columns")
    declared = [*declared, *re.findall(
        r"\b(?:id|identifier)(?:\s+column)?\s*(?:is|:|=)?\s*`([^`]+)`",
        readme, flags=re.IGNORECASE)]
    if any(name not in columns or name == target for name in declared):
        raise TaskDirectoryError("A declared ID column is missing or overlaps the target")
    return [name for name in columns if name != target and (
        name in declared or name.casefold() in {"id", "sample_id", "row_id", "index"})]


def _csv_input_columns(
    info: dict[str, Any], readme: str, header: list[str], target: str, ids: list[str]
) -> tuple[list[str], list[str]]:
    """Respect explicit input declarations before considering remaining columns."""
    metadata: list[list[str]] = []
    for field in ("feature_columns", "feature_names"):
        if field in info:
            names = info[field]
            if (not isinstance(names, list) or not names
                    or any(not isinstance(name, str) or not name.strip() for name in names)
                    or len(names) != len(set(names))):
                raise TaskDirectoryError(f"info.json {field} must list unique non-empty input columns")
            metadata.append(list(names))
    if len(metadata) > 1 and metadata[0] != metadata[1]:
        raise TaskDirectoryError("info.json contains conflicting input column declarations")
    statements = re.findall(
        r"\buse\s+((?:`[^`]+`|[^`\n.;])+?)\s+as\s+(?:the\s+)?(?:inputs?|(?:input\s+)?features)\b",
        readme, flags=re.IGNORECASE)
    declarations: list[list[str]] = []
    for statement in statements:
        names = re.findall(r"`([^`]+)`", statement)
        if not names:
            continue
        # File-use instructions do not declare feature columns.
        if all(re.search(r"\.(?:csv|npy|json|txt|parquet)$", name, flags=re.IGNORECASE) for name in names):
            continue
        if len(names) != len(set(names)):
            raise TaskDirectoryError("README lists duplicate input columns")
        declarations.append(names)
    declared = [*metadata, *declarations]
    for names in declared:
        if target in names:
            raise TaskDirectoryError("The target label cannot be used as an input feature")
        if any(name not in header or name in ids for name in names):
            raise TaskDirectoryError("A declared input feature is absent from the CSV or overlaps an ID")
    if declared and any(set(names) != set(declared[0]) for names in declared[1:]):
        raise TaskDirectoryError("README and info.json disagree on input columns")
    columns = declared[0] if declared else [name for name in header if name != target and name not in ids]
    if not columns:
        raise TaskDirectoryError("No features remain after excluding target and ID columns")
    ignored = [name for name in header if name not in columns and name != target and name not in ids]
    return columns, ignored


def _numeric_values(values: list[Any]) -> list[float | None] | None:
    output: list[float | None] = []
    for raw in values:
        if raw is None or (isinstance(raw, str) and raw.strip().casefold() in _MISSING):
            output.append(None)
            continue
        if isinstance(raw, np.generic):
            raw = raw.item()
        if isinstance(raw, float) and math.isnan(raw):
            output.append(None)
            continue
        if type(raw) is bool:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        output.append(value)
    return output if any(value is not None for value in output) else None


def _is_text_column(name: str, values: list[Any]) -> bool:
    strings = [value.decode("utf-8") if isinstance(value, bytes) else value
               for value in values if isinstance(value, (str, bytes)) and value]
    if not strings:
        return False
    lengths = np.asarray([len(value) for value in strings])
    space_fraction = sum(bool(re.search(r"\s", value)) for value in strings) / len(strings)
    semantic = bool(re.search(r"(?:text|headline|title|sentence|review|article|question|content|body)",
                              name, flags=re.IGNORECASE))
    return bool(np.max(lengths) > 200 or np.mean(lengths) >= 50
                or (np.median(lengths) >= 24 and space_fraction >= 0.4)
                or (semantic and np.mean(lengths) >= 8 and space_fraction >= 0.4))


def _text_matrix(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or not 1 <= values.shape[1] <= 8:
        raise TaskDirectoryError("Text tasks support 1 to 8 input columns")
    output = np.empty(values.shape, dtype=object)
    total = 0
    for index, value in np.ndenumerate(values):
        value = value.item() if isinstance(value, np.generic) else value
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="strict")
        if not isinstance(value, str) or "\0" in value or len(value) > _MAX_FIELD_CHARACTERS:
            raise TaskDirectoryError("Text cells must be valid strings with at most 100000 characters")
        total += len(value)
        if total > _MAX_TEXT_CHARACTERS:
            raise TaskDirectoryError("Text partition exceeds the 8000000-character limit")
        output[index] = value
    return output


def _column_kinds(columns: list[str], values: dict[str, list], info: dict) -> dict[str, str]:
    declared = info.get("feature_types")
    if declared is not None:
        if (not isinstance(declared, dict) or set(declared) != set(columns)
                or any(kind not in {"numeric", "categorical", "text"} for kind in declared.values())):
            raise TaskDirectoryError("info feature_types must cover all inputs with numeric, categorical, or text")
        return dict(declared)
    kinds = {}
    for name in columns:
        column = values[name]
        has_padded_code = any(
            isinstance(value, str) and re.fullmatch(r"0[0-9]+", value.strip()) for value in column
        )
        if not has_padded_code and _numeric_values(column) is not None:
            kinds[name] = "numeric"
        elif _is_text_column(name, column):
            kinds[name] = "text"
        else:
            kinds[name] = "categorical"
    return kinds


def _configure_npy_columns(values, widths, reader, columns, info):
    """Infer a two-dimensional NPY schema from the complete training partition."""
    from .input_readers import _set_text

    if values.ndim != 2 or not 1 <= values.shape[1] <= 4096:
        raise TaskDirectoryError("Table and text inputs must be non-empty and two-dimensional")
    prefixes = reader["feature_prefixes"]
    families = list(zip(prefixes, widths, strict=True))
    if columns is None:
        columns = [f"{prefix}_{index}" for prefix, width in families for index in range(width)]
    if len(columns) != values.shape[1]:
        raise TaskDirectoryError("NPY columns do not match the feature width")

    kinds = _column_kinds(columns, {name: list(values[:, i]) for i, name in enumerate(columns)}, info)
    if "feature_types" not in info:
        forced = {"n": "numeric", "c": "categorical"}
        column_prefixes = [prefix for prefix, width in families for _ in range(width)]
        kinds.update({name: forced[prefix] for name, prefix in zip(columns, column_prefixes, strict=True)
                      if prefix in forced})
    reader.update(output_kind="table", feature_columns=columns, feature_types=kinds)

    implicit_text = False
    if len(columns) == 1 and prefixes[0] not in {"n", "c"}:
        text_types = (str, bytes, np.str_, np.bytes_)
        column = values[:, 0]
        implicit_text = (any(isinstance(value, text_types) for value in column)
                         and all(isinstance(value, text_types) or _missing_content(value) for value in column))
    if prefixes == ["text"] or implicit_text or all(kind == "text" for kind in kinds.values()):
        _set_text(reader, columns)


def _csv_sample_rows(header: list[str], rows: list[list[str]], reader: dict, source: str, split: str,
                     source_rows: list[int] | None = None) -> list[dict]:
    indices = [header.index(name) for name in reader["id_columns"]]
    output = []
    for position, row in enumerate(rows):
        i = source_rows[position] if source_rows is not None else position
        parts = [row[index] for index in indices]
        if any(not value.strip() or any(c in value for c in "\r\n\0") for value in parts):
            raise TaskDirectoryError("Source sample IDs must be non-empty and contain no newline or NUL")
        original = parts[0] if len(parts) == 1 else json.dumps(parts, ensure_ascii=False, separators=(",", ":")) if parts else ""
        output.append({"sample_id": original if parts else f"{split}_{i:09d}",
                       "original_id": original, "source_file": source, "source_row": str(i)})
    if len({row["sample_id"] for row in output}) != len(output):
        raise TaskDirectoryError("Duplicate source sample IDs prevent unambiguous tracing")
    return output


def _array_sample_rows(count: int, files: dict, split: str) -> list[dict]:
    paths = files["feature_paths"]
    source = paths[0] if len(paths) == 1 else json.dumps(paths, separators=(",", ":"))
    return [{"sample_id": f"{split}_{i:09d}", "original_id": "", "source_file": source, "source_row": str(i)}
            for i in range(count)]


def _csv_arrays(
    header: list[str], rows: list[list[str]], reader: dict[str, Any], *, allow_unlabeled: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    target = reader["target_column"]
    has_labels = target in header
    expected = [*reader["feature_columns"], *reader["id_columns"], *reader.get("ignored_columns", [])]
    if has_labels or not allow_unlabeled:
        expected.append(target)
    if set(header) != set(expected):
        raise TaskDirectoryError("CSV columns differ from the discovered features, labels, or IDs")
    indices = [header.index(name) for name in reader["feature_columns"]]
    values = np.asarray([[row[index] for index in indices] for row in rows], dtype=object)
    labels = None
    if has_labels:
        target_index = header.index(target)
        labels = _labels(np.asarray([row[target_index] for row in rows], dtype=object))
    if reader["output_kind"] == "text":
        return _text_matrix(values), labels
    for column_index, name in enumerate(reader["feature_columns"]):
        raw = list(values[:, column_index])
        if reader["feature_types"][name] == "numeric":
            converted = _numeric_values(raw)
            if converted is None and any(value.strip().casefold() not in _MISSING for value in raw):
                raise TaskDirectoryError(f"Numeric CSV column {name!r} contains non-numeric values")
            if reader.get("preserve_feature_lexemes", False):
                values[:, column_index] = [None if value.strip().casefold() in _MISSING else value for value in raw]
            else:
                values[:, column_index] = converted if converted is not None else [None] * len(raw)
        elif reader["feature_types"][name] == "text":
            values[:, column_index] = raw
        else:
            values[:, column_index] = [None if _missing_content(value) else value for value in raw]
    return values, labels


def _image_modality(values: np.ndarray) -> dict[str, Any]:
    if values.ndim not in (3, 4) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise TaskDirectoryError("Image NPY must be a finite numeric (N,H,W[,C]) array")
    shape = list(values.shape[1:])
    if any(not 1 <= size <= 4096 for size in shape) or math.prod(shape) > 1_000_000:
        raise TaskDirectoryError("Image dimensions exceed the supported limit")
    if len(shape) == 3 and shape[-1] not in (1, 3, 4):
        raise TaskDirectoryError("Image arrays must use HW or HWC channel layout")
    low, high = float(np.min(values)), float(np.max(values))
    if values.dtype == np.uint8:
        limits = [0.0, 255.0]
    elif low >= 0 and high <= 1:
        limits = [0.0, 1.0]
    elif low >= 0 and high <= 255:
        limits = [0.0, 255.0]
    else:
        limits = [low, high] if low < high else [low - 0.5, high + 0.5]
    return {"kind": "image", "image_shape": shape,
            "channel_layout": "HW" if len(shape) == 2 else "HWC", "value_range": limits}


def _npy_features(values: np.ndarray, reader: dict[str, Any]) -> np.ndarray:
    kind = reader["output_kind"]
    if kind == "text":
        return _text_matrix(values)
    if kind == "image":
        modality = reader["modality"]
        if (values.ndim != len(modality["image_shape"]) + 1
                or list(values.shape[1:]) != modality["image_shape"]
                or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or (values < modality["value_range"][0]).any()
                or (values > modality["value_range"][1]).any()):
            raise TaskDirectoryError("Image shape or range does not match the training schema")
        return values
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != len(reader["feature_columns"]):
        raise TaskDirectoryError("Feature width does not match the training schema")
    output = values.astype(object)
    for index, name in enumerate(reader["feature_columns"]):
        if reader["feature_types"][name] == "numeric":
            converted = _numeric_values(list(output[:, index]))
            # Completely missing numeric columns remain valid for the B imputer.
            if converted is None:
                present = [value for value in output[:, index]
                           if value is not None and not (isinstance(value, float) and math.isnan(value))]
                if present:
                    raise TaskDirectoryError(f"Numeric NPY column {name!r} contains non-numeric values")
                converted = [None] * len(output)
            output[:, index] = converted
        elif reader["feature_types"][name] == "text":
            for value in output[:, index]:
                if not isinstance(value, str) or "\0" in value or len(value) > _MAX_FIELD_CHARACTERS:
                    raise TaskDirectoryError(f"Text NPY column {name!r} must contain bounded strings")
        else:
            for value in output[:, index]:
                typed_category_token(value)
    return output


def _inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in root.iterdir():
        if path.name.startswith("."):
            continue
        # Only names are needed for test discovery; its contents/header are unread.
        key = path.name.casefold()
        if key in inventory:
            raise TaskDirectoryError("The task directory contains case-insensitive filename collisions")
        inventory[key] = path.name
    return inventory


def _pick_split_files(inventory: dict[str, str], extension: str, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    for alias, split in _SPLITS.items():
        name = f"{prefix}{alias}.{extension}".casefold()
        if name in inventory:
            if split in found:
                raise TaskDirectoryError(f"Multiple candidate files match the {split} split")
            found[split] = inventory[name]
    return found


def _holdout_train(
    values: np.ndarray, labels: np.ndarray, random_seed: int, *, return_indices: bool = False
) -> tuple:
    """Reserve validation rows from train only; keep every class in both parts."""
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(labels):
        groups.setdefault(typed_category_token(value), []).append(index)
    rng = np.random.default_rng(random_seed)
    train_indices, valid_indices = [], []
    for token in sorted(groups):
        indices = np.asarray(groups[token], dtype=int)
        if len(indices) < 2:
            raise TaskDirectoryError("A stratified holdout requires at least two training samples per class")
        rng.shuffle(indices)
        n_valid = max(1, min(len(indices) - 1, round(len(indices) * 0.2)))
        valid_indices.extend(indices[:n_valid])
        train_indices.extend(indices[n_valid:])
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    result = (values[train_indices], labels[train_indices], values[valid_indices], labels[valid_indices])
    return (*result, train_indices, valid_indices) if return_indices else result


def _check_declared_metadata(
    info: dict[str, Any], reader: dict[str, Any], *, n_source_train: int,
    n_validation: int, n_classes: int,
) -> None:
    expected = {"train_size": n_source_train, "n_train": n_source_train,
                "val_size": n_validation, "valid_size": n_validation,
                "validation_size": n_validation, "n_val": n_validation,
                "num_classes": n_classes, "n_classes": n_classes}
    if reader["output_kind"] == "table":
        counts = list(reader["feature_types"].values())
        expected.update(n_num_features=counts.count("numeric"),
                        n_cat_features=counts.count("categorical"))
    for field, actual in expected.items():
        if field in info and (type(info[field]) is not int or info[field] != actual):
            raise TaskDirectoryError(f"info.json {field}={info[field]!r} disagrees with the observed value {actual}")


def _describes_pairwise_preference(text: str) -> bool:
    """Require complementary label meanings rather than relying on the target name."""
    text = re.sub(r"[`*#]", "", text.lower())
    if not re.search(r"\b(higher|lower|better|worse|wins?|won|preferred|winner)\b", text):
        return False
    roles = r"(?:(?:the|option|headline|text)\s+){0,2}(a|left|first|b|right|second)\b"
    positive = re.search(r"\b(?:1|true)\s+(?:if|when|means)\s+" + roles, text)
    negative = re.search(r"\b(?:0|false)\s+(?:if|when|means)\s+" + roles, text)
    first = {"a", "left", "first"}
    return (
        positive is not None and negative is not None
        and (positive.group(1) in first) != (negative.group(1) in first)
    )


def discover_task_directory(
    task_dir: Path,
    workspace_dir: Path,
    *,
    max_trials: int = 20,
    random_seed: int = 42,
    timeout_seconds: int = 120,
    resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover train/validation, stage immutable inputs, and defer all test reads."""
    root = Path(task_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise TaskDirectoryError("The task path must be a directory")
    destination = Path(workspace_dir).expanduser().resolve()
    if destination.is_relative_to(root):
        raise TaskDirectoryError("Staging must be outside the read-only task directory")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise TaskDirectoryError("Task staging must be empty")
    if (type(max_trials) is not int or max_trials < 1
            or type(random_seed) is not int or not 0 <= random_seed <= 2**32 - 1
            or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600):
        raise TaskDirectoryError("Trial count, seed, or timeout is outside the supported range")
    inventory = _inventory(root) if resolution is None else resolution["inventory"]
    hashes: dict[str, str] = {}
    train_image_sources: set[str] = set()
    readme = ""
    info: dict[str, Any] = {}
    readmes = [value for key, value in inventory.items()
               if re.fullmatch(r"readme(?:\.(?:md|txt|rst))?", key)]
    if len(readmes) > 1:
        raise TaskDirectoryError("Multiple README files make the task instructions ambiguous")
    metadata_files = ([*readmes, *([inventory["info.json"]] if "info.json" in inventory else [])]
                      if resolution is None else list(resolution["documents"]))
    for relative in metadata_files:
        path = _source_file(root, relative)
        if path.stat().st_size > 128 * 1024:
            raise TaskDirectoryError("Task instructions or info.json exceed 128 KiB")
        hashes[relative] = _sha256(path)
        text = path.read_text(encoding="utf-8-sig")
        if resolution is None and relative.casefold() == "info.json":
            info = json.loads(text)
            if not isinstance(info, dict):
                raise TaskDirectoryError("info.json must be an object")
        else:
            readme += text + "\n\n"
    if resolution is not None:
        info = dict(resolution["metadata"])
    if str(info.get("task_type", "classification")).lower() in {"regression", "reg", "continuous"}:
        raise TaskDirectoryError("Automatic discovery supports classification. info.json declares regression")
    metric, metric_source = _metric(info, readme)
    csv_files = _pick_split_files(inventory, "csv")
    feature_families = {prefix: _pick_split_files(inventory, "npy", f"{prefix}_")
                        for prefix in ("n", "c", "x", "features", "text", "images")}
    feature_families = {prefix: files for prefix, files in feature_families.items() if files}
    reader: dict[str, Any]
    split_files: dict[str, dict[str, Any]] = {}
    from .input_readers import (
        IMAGE_EXTENSIONS,
        additional_contract,
        read_split,
        reader_hash,
    )

    external = additional_contract(root, inventory, info, resolution,
                                   allow_auto=not csv_files and not feature_families)
    if external is not None:
        reader, split_files = external
    elif csv_files:
        if feature_families or "train" not in csv_files:
            raise TaskDirectoryError("CSV tasks require a unique train.csv and cannot include NPY feature splits")
        reader = {"format": "csv", "output_kind": "table"}
        if resolution is not None:
            reader["preserve_feature_lexemes"] = True
        split_files = {split: {"csv_path": path} for split, path in csv_files.items()}
    else:
        if not feature_families:
            raise TaskDirectoryError("Unsupported task format. Use N/C/X_train.npy or train/val/test.csv")
        prefixes = [prefix for prefix in ("n", "c", "x", "features", "text", "images") if prefix in feature_families]
        if len(prefixes) > 1 and set(prefixes) != {"n", "c"}:
            raise TaskDirectoryError("NPY features are ambiguous. Only N+C arrays can be combined automatically")
        split_names = set(feature_families[prefixes[0]])
        if ("train" not in split_names
                or any(set(feature_families[prefix]) != split_names for prefix in prefixes)):
            raise TaskDirectoryError("NPY feature arrays require matching training and optional validation/test splits")
        label_families = [_pick_split_files(inventory, "npy", f"{prefix}_")
                          for prefix in ("y", "label", "labels", "target")]
        label_families = [files for files in label_families if files]
        if len(label_families) != 1 or not split_names - {"test"} <= set(label_families[0]) <= split_names:
            raise TaskDirectoryError("NPY features require a unique matching y/label split")
        reader = {"format": "npy", "feature_prefixes": prefixes, "output_kind": "table"}
        split_files = {split: {"feature_paths": [feature_families[prefix][split] for prefix in prefixes],
                               "labels_path": label_families[0].get(split)}
                       for split in ("train", "validation", "test") if split in split_names}
    # Hash/read train and validation only. No test file stat, header, hash, or values.
    for split in (() if external is not None else ("train", "validation")):
        if split not in split_files:
            continue
        files = split_files[split]
        relatives = [files["csv_path"]] if reader["format"] == "csv" else [*files["feature_paths"], files["labels_path"]]
        for relative in relatives:
            hashes[relative] = _sha256(_source_file(root, relative))
    if external is not None:
        active_reads = set()

        def touch(relative):
            active_reads.add(relative)
            digest = _sha256(_source_file(root, relative))
            if relative in hashes and hashes[relative] != digest:
                raise TaskDirectoryError("Source changed during reading")
            hashes[relative] = digest

        before_train = set(hashes)
        train_X, train_y, train_index = read_split(root, split_files["train"], reader, "train", touch, fit=True)
        train_relatives = sorted(set(hashes) - before_train)
        if "validation" in split_files:
            active_reads.clear()
            valid_X, valid_y, valid_index = read_split(root, split_files["validation"], reader, "validation", touch)
            if active_reads & set(train_relatives):
                raise TaskDirectoryError("Training and validation reuse the same source samples")
        if reader["output_kind"] == "table" and reader.get("mixed_inputs"):
            info["feature_types"] = reader["feature_types"]
    elif reader["format"] == "csv":
        train_header, train_rows = _read_csv(_source_file(root, split_files["train"]["csv_path"]))
        target = _target_column(info, readme, train_header)
        ids = _id_columns(info, readme, train_header, target)
        columns, ignored = _csv_input_columns(info, readme, train_header, target, ids)
        reader.update(target_column=target, id_columns=ids, feature_columns=columns, ignored_columns=ignored)
        train_columns = {name: [row[train_header.index(name)] for row in train_rows] for name in columns}
        # Budget preflight and a semantic image-index reader must identify the
        # same training resources, even before LLM types are available. Only
        # columns whose values all resolve to existing image files qualify.
        for values in train_columns.values():
            present = [value for value in values if not _missing_content(value)]
            if not present or not all(isinstance(value, str) and Path(value).suffix.lower() in IMAGE_EXTENSIONS for value in present):
                continue
            resources = {}
            for relative in set(present):
                try:
                    resources[relative] = _source_file(root, relative)
                except (TaskDirectoryError, OSError):
                    continue  # Missing references are removed by the selected image reader.
            for relative, image_path in resources.items():
                hashes[relative] = _sha256(image_path)
                train_image_sources.add(relative)
        kinds = _column_kinds(columns, train_columns, info)
        if all(kind == "text" for kind in kinds.values()):
            reader["output_kind"] = "text"
            reader["modality"] = {"kind": "text", "text_columns": columns, "max_characters": _MAX_FIELD_CHARACTERS}
        reader["feature_types"] = kinds
        train_rows, train_positions = _filter_csv_samples(root, train_header, train_rows, reader, split_files["train"]["csv_path"], "train")
        train_X, train_y = _csv_arrays(train_header, train_rows, reader)
        train_index = _csv_sample_rows(train_header, train_rows, reader, split_files["train"]["csv_path"], "train", train_positions)
        if "validation" in split_files:
            valid_header, valid_rows = _read_csv(_source_file(root, split_files["validation"]["csv_path"]))
            valid_rows, valid_positions = _filter_csv_samples(root, valid_header, valid_rows, reader,
                                                              split_files["validation"]["csv_path"], "validation")
            valid_X, valid_y = _csv_arrays(valid_header, valid_rows, reader)
            valid_index = _csv_sample_rows(valid_header, valid_rows, reader, split_files["validation"]["csv_path"], "validation", valid_positions)
    else:
        train_X, widths = _read_npy_matrix(root, split_files["train"]["feature_paths"])
        train_y = normalize_classification_labels(read_npy_array(_source_file(root, split_files["train"]["labels_path"])))
        train_index = _array_sample_rows(len(train_y), split_files["train"], "train")
        if not 1 <= len(train_X) <= _MAX_ROWS:
            raise TaskDirectoryError("Training features are empty or exceed the row limit")
        if train_X.ndim in (3, 4):
            if len(reader["feature_prefixes"]) != 1:
                raise TaskDirectoryError("Image tasks cannot combine feature sources")
            reader.update(output_kind="image", feature_columns=[])
            train_X, train_y, train_index = _filter_array_samples(train_X, train_y, train_index, reader, "train")
            reader["modality"] = _image_modality(train_X)
        else:
            if train_X.ndim == 1:
                train_X = train_X.reshape(-1, 1)
            columns = _declared_columns(info, readme, train_X.shape[1]) if train_X.ndim == 2 else None
            _configure_npy_columns(train_X, widths, reader, columns, info)
            train_X, train_y, train_index = _filter_array_samples(train_X, train_y, train_index, reader, "train")
        train_X = _npy_features(train_X, reader)
        train_y = _labels(train_y)
        if "validation" in split_files:
            valid_X, _ = _read_npy_matrix(root, split_files["validation"]["feature_paths"])
            valid_y = read_npy_array(_source_file(root, split_files["validation"]["labels_path"]))
            valid_index = _array_sample_rows(len(valid_y), split_files["validation"], "validation")
            valid_X, valid_y, valid_index = _filter_array_samples(valid_X, valid_y, valid_index, reader, "validation")
            valid_X = _npy_features(valid_X, reader)
            valid_y = _labels(valid_y)
    if len(train_X) != len(train_y):
        raise TaskDirectoryError("Training features and labels have different row counts")
    n_source_train = reader.get("sample_filtering", {}).get("train", {}).get("input_count", len(train_y))
    split_strategy: dict[str, Any] = {"kind": "provided_validation"}
    if "validation" not in split_files:
        train_X, train_y, valid_X, valid_y, train_rows, valid_rows = _holdout_train(
            train_X, train_y, random_seed, return_indices=True)
        source_index = train_index
        train_index = [source_index[i] for i in train_rows]
        valid_index = [source_index[i] for i in valid_rows]
        split_strategy = {"kind": "stratified_train_holdout", "fraction": 0.2, "random_seed": random_seed}
    if len(valid_X) != len(valid_y):
        raise TaskDirectoryError("Training or validation features and labels have different row counts")
    if external is not None and ({row["sample_id"] for row in train_index} & {row["sample_id"] for row in valid_index}):
        raise TaskDirectoryError("Training and validation sample IDs overlap")
    label_values, label_names = _label_schema(train_y)
    _check_declared_metadata(info, reader, n_source_train=n_source_train,
                             n_validation=reader.get("sample_filtering", {}).get("validation", {}).get("input_count", len(valid_y)),
                             n_classes=len(label_values))
    for relative, digest in hashes.items():
        if _sha256(_source_file(root, relative)) != digest:
            raise TaskDirectoryError("Development data or instructions changed during discovery")
    code_hash = reader_hash()
    train_policy = {"max_trials": max_trials, "random_seed": random_seed,
                    "timeout_seconds": timeout_seconds}
    identity = {"sources": hashes, "reader": reader, "metric": metric, "split_strategy": split_strategy,
                "label_values": label_values, "reader_code_sha256": code_hash,
                "preparation_code_sha256": {
                    name: _sha256(Path(__file__).with_name(name))
                    for name in ("prepare.py", "prepare_modalities.py", "preprocessing.py",
                                 "raw_reader.py", "npy_reader.py", "modality_readers.py",
                                 "sample_index.py", "profiling.py")
                },
                "train_policy": train_policy}
    version = "auto_" + hashlib.sha256(_json_bytes(identity)).hexdigest()[:24]
    task_id = re.sub(r"[^A-Za-z0-9_-]+", "_", root.name).strip("_")[:48]
    if not task_id:
        task_id = "task_" + hashlib.sha256(root.name.encode("utf-8")).hexdigest()[:16]
    spec: dict[str, Any] = {
        "schema_version": "1.0" if reader["output_kind"] == "table" else "2.0",
        "task_id": task_id, "data_version": version,
        "source_format": {"table": "npy_array_package", "text": "text_array_package", "image": "image_array_package"}[reader["output_kind"]],
        "source_root": str(destination), "label_values": label_values, "label_names": label_names,
        "primary_metric": metric, "partitions": {},
        "train_policy": train_policy,
    }
    if reader["output_kind"] == "table":
        spec.update(feature_columns=reader["feature_columns"], feature_types=reader["feature_types"])
    else:
        spec["modality"] = reader["modality"]
    for split, X, y in (("train", train_X, train_y), ("validation", valid_X, valid_y)):
        feature_name, label_name = f"X_{split}.npy", f"y_{split}.npy"
        _save_array(destination / feature_name, X)
        _save_array(destination / label_name, y)
        index_name = f"sample_index_{split}.csv"
        save_sample_index(destination / index_name, train_index if split == "train" else valid_index)
        spec["partitions"][split] = {"features_path": feature_name, "labels_path": label_name,
                                     "n_samples": len(y), "sample_ids_path": index_name}
    declared_test_size = info.get("test_size", info.get("n_test"))
    if declared_test_size is not None and (type(declared_test_size) is not int or declared_test_size < 1):
        raise TaskDirectoryError("info.json test_size must be a positive integer")
    descriptor_path = None
    if "test" in split_files:
        descriptor = {"schema_version": "2.0", "kind": "task_directory", "task_id": task_id,
                      "source_root": str(root), "reader": reader,
                      "test_files": split_files["test"], "n_samples": declared_test_size,
                      "primary_metric": metric, "reader_code_sha256": code_hash,
                      "development_files": sorted({
                          _source_file(root, name).relative_to(root).as_posix() for name in hashes
                      })}
        descriptor_path = destination / "test_descriptor.json"
        _write_json(descriptor_path, descriptor)
    paragraphs = [
        paragraph.strip() for paragraph in readme.split("\n\n")
        if paragraph.strip() and not all(
            line.lstrip().startswith("#") for line in paragraph.splitlines() if line.strip()
        )
    ]
    description = re.sub(r"\s+", " ", re.sub(r"[`*#]", "", paragraphs[0] if paragraphs else "")).strip()
    context = {"description": description[:1000], "objective": description.split(". ")[0][:400],
               "task_type": "classification", "metric": metric["name"],
               "input_columns": reader["feature_columns"][:32],
               "label_semantics": {"values": label_values[:32], "description": description[:400]}}
    if resolution is not None:
        # B needs file-role evidence; A only needs the resolved learning task.
        # Do not forward the full B document (which contains deferred test paths).
        objective = "Classify " + str(reader.get("target_column", "the declared labels")) + " from the declared development features."
        context.update(description=objective, objective=objective,
                       label_semantics={"values": label_values[:32], "description": "Training label schema"})
    if "target_column" in reader:
        context["target_column"] = reader["target_column"]
    if reader["output_kind"] == "text" and len(reader["feature_columns"]) == len(label_values) == 2:
        task_description = (
            readme if resolution is None else "\n".join(resolution["documents"].values())
        )
        context["pairwise_preference"] = (
            info.get("pairwise_preference") is True
            or _describes_pairwise_preference(task_description)
        )
    train_files = split_files["train"]
    if external is None:
        train_relatives = [train_files["csv_path"]] if reader["format"] == "csv" else [*train_files["feature_paths"], train_files["labels_path"]]
        train_relatives = [*train_relatives, *sorted(train_image_sources)]
    report = {"schema_version": "1.0", "task_id": task_id, "data_version": version,
              "source_root": str(root), "source_files_sha256": hashes,
              "reader_code_sha256": code_hash, "reader": reader, "metric_source": metric_source,
              "n_train": len(train_y), "n_validation": len(valid_y),
              "n_source_train": n_source_train, "validation_split": split_strategy,
              "sample_filtering": reader.get("sample_filtering", {}),
              "test_read_during_discovery": False, "test_files": split_files.get("test"),
              "validation_files": split_files.get("validation"),
              "overlapping_partition_ids": len({r["sample_id"] for r in train_index}
                                               & {r["sample_id"] for r in valid_index}),
              "dropped_id_columns": reader.get("id_columns", []),
              "budget_fingerprint": hashlib.sha256(_json_bytes(sorted(hashes[name] for name in train_relatives))).hexdigest()}
    _write_json(destination / "task.auto.json", spec)
    if external is not None:
        _write_json(destination / "reader_rules.json", {
            "schema_version": "1.0", "reader_sha256": reader_hash(), "reader": reader,
        })
    _write_json(destination / "discovery_report.json", report)
    _write_json(destination / "task_context.json", context)
    return {"task_spec": spec, "test_descriptor_path": descriptor_path,
            "task_context": context, "discovery_report": report}


def materialize_final_test(descriptor_path: Path, output_dir: Path) -> Path:
    """Read the deferred test inputs using the fixed train-derived reader only."""
    descriptor_file = Path(descriptor_path).expanduser().resolve(strict=True)
    descriptor_hash = _sha256(descriptor_file)
    descriptor = json.loads(descriptor_file.read_text(encoding="utf-8"))
    if (not isinstance(descriptor, dict) or descriptor.get("schema_version") != "2.0"
            or descriptor.get("kind") != "task_directory"):
        raise TaskDirectoryError("Unsupported automatic test descriptor")
    from .input_readers import FORMATS, read_split, reader_hash

    code_hash = reader_hash()
    if descriptor.get("reader_code_sha256") != code_hash:
        raise TaskDirectoryError("Task reader changed. Preserve the old descriptor and rediscover the task")
    root = Path(descriptor["source_root"]).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise TaskDirectoryError("The test descriptor source_root must be a directory")
    destination = Path(output_dir).expanduser().resolve()
    if destination.is_relative_to(root):
        raise TaskDirectoryError("Final test staging must be outside the task directory")
    reader, files = descriptor["reader"], descriptor["test_files"]
    before = {}
    development_files = set(descriptor["development_files"])

    def test_source_path(relative):
        path = _source_file(root, relative)
        if path.relative_to(root).as_posix() in development_files:
            raise TaskDirectoryError("Final test reuses a development input file")
        return path

    if reader.get("format") in FORMATS:
        def touch(relative):
            digest = _sha256(test_source_path(relative))
            if relative in before and before[relative] != digest:
                raise TaskDirectoryError("Test input changed during reading")
            before[relative] = digest

        X, y, sample_rows = read_split(root, files, reader, "test", touch, prediction=None)
        relatives = list(before)
    elif reader.get("format") == "csv":
        relatives = [files["csv_path"]]
    elif reader.get("format") == "npy":
        relatives = [*files["feature_paths"]]
        if files.get("labels_path") is not None:
            relatives.append(files["labels_path"])
    else:
        raise TaskDirectoryError("Unsupported test descriptor reader")
    if reader.get("format") not in FORMATS:
        before = {name: _sha256(test_source_path(name)) for name in relatives}
    if reader["format"] == "csv":
        header, rows = _read_csv(_source_file(root, files["csv_path"]))
        rows, source_rows = _filter_csv_samples(root, header, rows, reader, files["csv_path"], "test")
        X, y = _csv_arrays(header, rows, reader, allow_unlabeled=True)
        sample_rows = _csv_sample_rows(header, rows, reader, files["csv_path"], "test", source_rows)
    elif reader["format"] == "npy":
        X, _ = _read_npy_matrix(root, files["feature_paths"])
        y = (read_npy_array(_source_file(root, files["labels_path"]))
             if files.get("labels_path") is not None else None)
        sample_rows = _array_sample_rows(len(X), files, "test")
        X, y, sample_rows = _filter_array_samples(X, y, sample_rows, reader, "test")
        X = _npy_features(X, reader)
        if y is not None:
            y = _labels(y)
    if y is not None and len(X) != len(y):
        raise TaskDirectoryError("Test features and labels have different row counts")
    declared = descriptor.get("n_samples")
    original_count = reader.get("sample_filtering", {}).get("test", {}).get("input_count", len(X))
    if declared is not None and declared != original_count:
        raise TaskDirectoryError("Test row count differs from the declared test_size")
    after = {name: _sha256(test_source_path(name)) for name in relatives}
    if before != after or _sha256(descriptor_file) != descriptor_hash:
        raise TaskDirectoryError("Test input or descriptor changed during materialization")
    destination.mkdir(parents=True, exist_ok=True)
    _save_array(destination / "X_test.npy", X)
    label_name = "y_test.npy" if y is not None else None
    if label_name is not None:
        _save_array(destination / label_name, y)
    save_sample_index(destination / "sample_index_test.csv", sample_rows)
    config = {"schema_version": "1.0", "task_id": descriptor["task_id"],
              "source_root": str(destination), "features_path": "X_test.npy",
              "labels_path": label_name, "n_samples": len(X)}
    config_path = destination / "test_config.json"
    _write_json(config_path, config)
    _write_json(destination / "test_source.json", {
        "source_root": str(root), "source_files_sha256": after,
        "descriptor_sha256": descriptor_hash, "reader_code_sha256": descriptor["reader_code_sha256"],
        "reader": reader, "n_samples": len(X),
        "original_n_samples": original_count,
        "sample_filtering": reader.get("sample_filtering", {}).get("test", {}),
        "sample_index_path": "sample_index_test.csv",
        "sample_index_sha256": _sha256(destination / "sample_index_test.csv"),
    })
    return config_path
