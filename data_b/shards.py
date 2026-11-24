"""Explicit row-shard composition, preserving the existing B/C output contract."""
from __future__ import annotations

import copy
import json
import re

import numpy as np

MAX_SHARDS = 128
SHARD_FORMATS = {"csv", "tsv", "convert", "npy", "text_lines"}


def partition_shards(part):
    from .input_readers import InputReaderError

    if not isinstance(part, dict):
        raise InputReaderError("A partition must be an object")
    if "shards" not in part:
        return [part]
    if set(part) != {"shards"} or not isinstance(part["shards"], list) or not 1 <= len(part["shards"]) <= MAX_SHARDS:
        raise InputReaderError("A sharded partition contains only a nonempty shards list (at most 128)")
    if any(not isinstance(item, dict) or set(item) != {"features", "labels"} for item in part["shards"]):
        raise InputReaderError("Each shard requires features and labels; nested shards are unsupported")
    return part["shards"]


def automatic_contract(inventory, info):
    """Recognize numbered CSV/TSV splits only; arbitrary names use LLM/metadata."""
    from .input_readers import InputReaderError
    from .task_directory import _SPLITS, _pick_split_files

    grouped = {}
    pattern = r"(train|val|valid|validation|test)(?:[_-](?:part|shard))?[_-](\d+)\.(csv|tsv)"
    for name, relative in inventory.items():
        match = re.fullmatch(pattern, name)
        if match:
            alias, number, fmt = match.groups()
            role = _SPLITS[alias]
            key = (fmt, role)
            numbered = grouped.setdefault(key, {})
            if int(number) in numbered:
                raise InputReaderError("Duplicate shard numbers or ambiguous split aliases")
            numbered[int(number)] = relative
    if not grouped:
        return None
    formats = {fmt for fmt, _ in grouped}
    if len(formats) != 1:
        raise InputReaderError("Mixed CSV/TSV shard families require an explicit reader contract")
    fmt = next(iter(formats))
    single = _pick_split_files(inventory, fmt)
    parts = {role: {"features": [path], "labels": None} for role, path in single.items()}
    for (_, role), numbered in grouped.items():
        if role in parts:
            raise InputReaderError("A split contains both a monolithic file and numbered shards")
        parts[role] = {"shards": [{"features": [numbered[i]], "labels": None} for i in sorted(numbered)]}
    if "train" not in parts:
        raise InputReaderError("Numbered shards require a training partition")
    return {"format": fmt, "options": {}, "partitions": parts,
            "target_column": info.get("target_column"), "id_columns": info.get("id_columns", []),
            "feature_columns": info.get("feature_columns", []), "feature_types": info.get("feature_types", {})}


def build_contract(contract):
    """Validate path roles without opening any sources, including deferred test."""
    from pathlib import PurePosixPath, PureWindowsPath

    from .input_readers import InputReaderError, validate_options

    fmt, parts = contract["format"], contract["partitions"]
    if fmt not in SHARD_FORMATS:
        raise InputReaderError("Row shards support CSV/TSV, flat conversion records, NPY, and text_lines")
    options = contract.get("options", {})
    if fmt in {"csv", "npy"}:
        if options:
            raise InputReaderError("CSV/NPY shard reading options are fixed by the program")
    else:
        validate_options(fmt, options)
    if not isinstance(parts, dict) or "train" not in parts or parts["train"] is None or set(parts) - {"train", "validation", "test"}:
        raise InputReaderError("Shard partitions require train and optional validation/test")
    prefixes = contract.get("feature_prefixes") or ["x"]
    if fmt == "npy" and (not isinstance(prefixes, list) or prefixes not in [["n", "c"], ["x"], ["n"], ["c"], ["text"], ["images"], ["features"]]):
        raise InputReaderError("NPY shards require one feature family, or numeric and categorical families")
    files, used = {}, []
    expected = len(prefixes) if fmt == "npy" else 1
    for role, part in parts.items():
        if part is None:
            continue
        output = []
        for item in partition_shards(part):
            features, labels = item.get("features"), item.get("labels")
            if set(item) != {"features", "labels"} or not isinstance(features, list) or len(features) != expected:
                raise InputReaderError("Shard feature counts must match the training reader")
            if fmt in {"npy", "text_lines"}:
                if not isinstance(labels, str):
                    raise InputReaderError("Every NPY/text shard requires its own labels path")
            elif labels is not None:
                raise InputReaderError("Record shard labels must come from the target column")
            for name in [*features, *([labels] if labels is not None else [])]:
                if (not isinstance(name, str) or not name or "\\" in name or PureWindowsPath(name).drive
                        or PurePosixPath(name).is_absolute() or any(p.startswith(".") for p in PurePosixPath(name).parts)):
                    raise InputReaderError("Shard paths must be visible relative paths within the task")
                if any(name == other or name.startswith(other + "/") or other.startswith(name + "/") for other in used):
                    raise InputReaderError("Shard sources overlap within or across partitions")
                used.append(name)
            output.append({"feature_paths": features, "labels_path": labels})
        files[role] = {"shards": output}
    columns, ids, kinds = contract.get("feature_columns", []), contract.get("id_columns", []), contract.get("feature_types", {})
    for names in (columns, ids):
        if not isinstance(names, list) or len(names) > 4096 or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
            raise InputReaderError("Invalid shard feature/ID column names")
    if not isinstance(kinds, dict) or (kinds and set(kinds) != set(columns)) or any(k not in {"numeric", "categorical", "text"} for k in kinds.values()):
        raise InputReaderError("Shard feature_types must cover declared feature columns")
    if set(columns) & {contract.get("target_column"), *ids}:
        raise InputReaderError("Shard target and IDs cannot be features")
    reader = {"format": "shards", "options": {}, "base_format": fmt, "base_options": options,
              "feature_prefixes": prefixes, "target_column": contract.get("target_column"),
              "id_columns": ids, "feature_columns": columns, "declared_types": kinds}
    return reader, files


def _restore_rows(rows, origins):
    for row in rows:
        source, local = origins[int(row["source_row"])]
        row.update(source_file=source, source_row=str(local))


def _records(root, shards, reader, split, touch, *, fit, prediction):
    from . import input_readers as ir
    from . import task_directory as td

    fmt, records, origins, total_bytes, characters = reader["base_format"], [], [], 0, 0
    expected = None if fit else set(reader["source_columns"])
    for shard in shards:
        source = shard["feature_paths"][0]
        path = ir.source_path(root, source)
        total_bytes += path.stat().st_size
        if total_bytes > ir.MAX_BYTES:
            raise ir.InputReaderError("Combined record shards exceed the 64 MiB input budget")
        touch(source)
        if fmt == "csv":
            header, rows = td._read_csv(path)
            rows = [dict(zip(header, row, strict=True)) for row in rows]
        else:
            rows = ir._record_rows(path, fmt, reader["base_options"], skip_field=reader["target_column"] if prediction else None)
            header = list(rows[0])
        if fit and expected is None:
            expected = set(header)
            reader["source_columns"] = header
            if reader["target_column"] is None:
                reader["target_column"] = td._target_column({}, "", header)
            if not reader["id_columns"]:
                reader["id_columns"] = td._id_columns({}, "", header, reader["target_column"])
        actual = set(header)
        allowed = expected - {reader["target_column"]} if prediction else expected
        if not allowed <= actual <= expected or (not prediction and actual != expected):
            raise ir.InputReaderError("Shard columns differ from the training schema")
        if any(set(row) != actual for row in rows):
            raise ir.InputReaderError("Inconsistent fields within a record shard")
        characters += sum(len(str(value)) for row in rows for value in row.values())
        if characters > ir.MAX_CHARACTERS:
            raise ir.InputReaderError("Combined record shards exceed the character budget")
        if len(records) + len(rows) > ir.MAX_ROWS or (len(records) + len(rows)) * len(header) > ir.MAX_ELEMENTS:
            raise ir.InputReaderError("Combined record shards exceed the row/element budget")
        origins.extend((source, i) for i in range(len(rows)))
        records.extend(rows)
    X, y, index = ir._records_to_arrays(records, reader, "shards", split, fit=fit, prediction=prediction)
    _restore_rows(index, origins)
    _restore_rows(reader["sample_filtering"][split]["dropped_samples"], origins)
    return X, y, index


def _arrays(root, shards, reader, split, touch, *, fit, prediction):
    from . import input_readers as ir
    from . import task_directory as td
    from .npy_reader import (
        ReaderLimits,
        normalize_classification_labels,
        read_npy_array,
    )

    chunks, targets, origins = [], [], []
    limits = ReaderLimits()
    rows = elements = payload = characters = source_bytes = 0
    fmt = reader["base_format"]
    for shard in shards:
        paths = shard["feature_paths"]
        for name in [*paths, *([shard["labels_path"]] if shard.get("labels_path") else [])]:
            path = ir.source_path(root, name)
            source_bytes += path.stat().st_size
            limit = limits.max_object_payload_bytes if fmt == "text_lines" or chunks and chunks[0].dtype.hasobject else limits.max_payload_bytes
            if source_bytes > limit:
                raise ir.InputReaderError("Combined array shard sources exceed the byte budget")
            touch(name)
        if fmt == "text_lines":
            texts = ir._text(ir.source_path(root, paths[0])).splitlines()
            characters += sum(map(len, texts))
            if characters > ir.MAX_CHARACTERS:
                raise ir.InputReaderError("Combined text shards exceed the corpus character budget")
            X = np.asarray(texts, dtype=object).reshape(-1, 1)
            y = np.asarray(["__prediction__"] * len(X) if prediction else
                           ir._text(ir.source_path(root, shard["labels_path"])).splitlines(), dtype=object)
            if not prediction and any(not str(label).strip() for label in y):
                raise ir.InputReaderError("Text shard labels cannot be empty")
        else:
            X, current_widths = td._read_npy_matrix(root, paths)
            if fit and not chunks:
                reader["family_widths"] = current_widths
            if current_widths != reader.get("family_widths"):
                raise ir.InputReaderError("NPY shard feature-family widths differ from training")
            y = np.asarray(["__prediction__"] * len(X), dtype=object) if prediction else normalize_classification_labels(
                read_npy_array(ir.source_path(root, shard["labels_path"])))
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        if not len(X) or len(X) != len(y):
            raise ir.InputReaderError("Each shard's feature and label row counts must match and be nonempty")
        expected_shape = reader.get("source_shape")
        if fit and not chunks:
            reader["source_shape"], reader["source_dtype"] = list(X.shape[1:]), X.dtype.str
        elif list(X.shape[1:]) != expected_shape or X.dtype.str != reader["source_dtype"]:
            raise ir.InputReaderError("Shard shape/dtype differs from the training schema")
        rows += len(X)
        elements += X.size
        payload += X.nbytes + y.nbytes
        object_data = X.dtype.hasobject or fmt == "text_lines"
        if (rows > (ir.MAX_ROWS if fmt == "text_lines" else limits.max_rows)
                or elements > (ir.MAX_ELEMENTS if object_data else limits.max_elements)
                or max(payload, source_bytes) > (limits.max_object_payload_bytes if object_data else limits.max_payload_bytes)):
            raise ir.InputReaderError("Combined array shards exceed the row/element/byte budget")
        chunks.append(X)
        targets.append(y)
        source = paths[0] if len(paths) == 1 else json.dumps(paths, separators=(",", ":"))
        origins.extend((source, i) for i in range(len(X)))
    X = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
    y = np.concatenate([part.astype(object) for part in targets])
    index = [{"sample_id": f"{split}_{i:09d}", "source_file": name, "source_row": str(local), "original_id": ""}
             for i, (name, local) in enumerate(origins)]
    if fit:
        if fmt == "text_lines":
            ir._set_text(reader, ["text"])
        elif X.ndim in (3, 4):
            reader.update(output_kind="image", feature_columns=[], modality=td._image_modality(X))
        else:
            info = {"feature_types": reader["declared_types"]} if reader["declared_types"] else {}
            td._configure_npy_columns(X, reader["family_widths"], reader, reader["feature_columns"] or None, info)
    X, y, index = td._filter_array_samples(X, y, index, reader, split)
    return td._npy_features(X, reader), td._labels(y), index


def read_shards(root, files, reader, split, touch, *, fit=False, prediction=False):
    from .input_readers import InputReaderError

    shards = files.get("shards", [files])
    if not isinstance(shards, list) or not 1 <= len(shards) <= MAX_SHARDS:
        raise InputReaderError("Invalid shard count")
    working = copy.deepcopy(reader)
    if working["base_format"] in {"csv", "tsv", "convert"}:
        result = _records(root, shards, working, split, touch, fit=fit, prediction=prediction)
    else:
        result = _arrays(root, shards, working, split, touch, fit=fit, prediction=prediction)
    if len({row["sample_id"] for row in result[2]}) != len(result[2]):
        raise InputReaderError("Duplicate sample IDs across shards")
    reader.update(working)
    return result
