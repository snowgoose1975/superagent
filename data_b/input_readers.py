"""Additional raw readers and bounded declarative format conversion.

All results use the existing staged NPY/TaskSpec contract. No generated Python
is executed. Only selected development inputs are opened during discovery.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .task_directory import TaskDirectoryError

FORMATS = {"tsv", "text_lines", "text_directory", "image_directory", "image_csv", "convert"}
READER_OPTION_FIELDS = {
    "tsv": set(), "text_lines": set(), "text_directory": set(),
    "image_directory": set(),
    "image_csv": {"image_column", "image_base", "auxiliary_types"},
    "convert": {"source_format", "field_paths", "delimiter"},
}
MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 100_000
MAX_CHARACTERS = 8_000_000
MAX_ELEMENTS = 8_000_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class InputReaderError(TaskDirectoryError):
    code = "B_INPUT_INVALID"


def select_reader(root, layout):
    """Build reader settings from validated file roles/types, without LLM calls.

    Only the documented training source is inspected. The resulting rules are
    reused unchanged for validation, test and frozen raw-input predictions.
    """
    result = copy.deepcopy(layout)
    fmt = result["format"]
    source = result["partitions"]["train"]["features"][0]
    mapping = result.pop("column_mapping", {})
    if not isinstance(mapping, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                           for key, value in mapping.items()):
        raise InputReaderError("column_mapping must map output column names to source column names")
    if fmt == "auto":
        path = Path(root) / source
        if path.is_dir():
            entries = _directory_files(root, source, IMAGE_EXTENSIONS | {".txt"})
            extensions = {Path(name).suffix.lower() for name, _ in entries}
            if extensions <= IMAGE_EXTENSIONS:
                result["format"] = "image_directory"
            elif extensions == {".txt"}:
                result["format"] = "text_directory"
            else:
                raise InputReaderError("Mixed class directories require an explicit sample index table")
            return result
        result["format"] = fmt = "convert"
    if fmt not in {"csv", "tsv", "convert"}:
        return result
    path = source_path(root, source)
    if path.stat().st_size > MAX_BYTES:
        raise InputReaderError("Raw file exceeds the 64 MiB reader limit")
    options = result["reader_options"]
    json_source = path.suffix.lower() == ".json" or options.get("source_format") == "json"
    if json_source:
        rows = _json(_text(path))
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise InputReaderError("JSON input must be a root array of flat records")
        header = list(rows[0])
        delimiter = None
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(65536)
        delimiter = options.get("delimiter", "\t" if fmt == "tsv" else ",")
        if "delimiter" not in options:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
            except csv.Error:
                pass  # Normal readers still validate every row and required field.
        header = next(csv.reader(io.StringIO(sample), delimiter=delimiter), [])
    if not header or len(set(header)) != len(header) or not all(isinstance(key, str) and key for key in header):
        raise InputReaderError("Record input requires unique nonempty source columns")
    if not mapping and options.get("field_paths"):
        validate_options("convert", options)
        mapping = {key: values[0] for key, values in options["field_paths"].items()}
    fields = mapping or {key: key for key in header}
    required = [*result["feature_columns"], *result["id_columns"], result["target_column"]]
    if any(key not in fields or fields[key] not in header for key in required):
        raise InputReaderError("A declared feature, ID or target is absent from the training source columns")
    if not mapping and not json_source and ((fmt == "csv" and delimiter == ",") or (fmt == "tsv" and delimiter == "\t")):
        return result
    result["format"] = "convert"
    result["reader_options"] = {"source_format": "json" if json_source else "delimited",
                                "field_paths": {key: [value] for key, value in fields.items()}}
    if not json_source:
        result["reader_options"]["delimiter"] = delimiter
    return result


def source_path(root, relative, *, directory=False):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PureWindowsPath(relative).drive or PurePosixPath(relative).is_absolute()
            or any(part.startswith(".") for part in PurePosixPath(relative).parts)):
        raise InputReaderError("Input paths must be visible relative paths inside the task")
    candidate = Path(root) / relative
    # Reject symlinks rather than accepting aliases across split boundaries.
    if any(parent.is_symlink() for parent in [candidate, *candidate.parents] if parent != Path(root)):
        raise InputReaderError("Symlinked input paths are unsupported")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()) or not (path.is_dir() if directory else path.is_file()):
        raise InputReaderError("Input path is missing, has the wrong kind, or escapes the task")
    return path


def validate_options(format_name, options):
    if format_name not in FORMATS or not isinstance(options, dict):
        raise InputReaderError("Unknown reader or invalid reader_options")
    allowed = READER_OPTION_FIELDS[format_name]
    if set(options) - allowed:
        raise InputReaderError("Unknown reader option; executable code is not accepted")
    if format_name == "image_csv":
        if not isinstance(options.get("image_column"), str) or not options["image_column"].strip():
            raise InputReaderError("image_csv requires image_column")
        if "image_base" in options and (not isinstance(options["image_base"], str) or not options["image_base"]):
            raise InputReaderError("image_base must be a relative directory")
        if "auxiliary_types" in options and (
            not isinstance(options["auxiliary_types"], dict)
            or any(kind not in {"numeric", "categorical", "text"} for kind in options["auxiliary_types"].values())
        ):
            raise InputReaderError("Invalid auxiliary column types")
    if format_name == "convert":
        if options.get("source_format") not in {"json", "delimited"}:
            raise InputReaderError("Conversion supports flat JSON arrays or delimited records")
        if not isinstance(options.get("field_paths"), dict) or not 1 <= len(options["field_paths"]) <= 128:
            raise InputReaderError("Conversion needs a bounded field_paths mapping")
        for path in options["field_paths"].values():
            if not isinstance(path, list) or len(path) != 1 or not isinstance(path[0], str) or not path[0]:
                raise InputReaderError("Conversion only supports a single top-level field name")
        if any(not isinstance(name, str) or not name.strip() for name in options["field_paths"]):
            raise InputReaderError("Converted column names must be nonempty strings")
        delimiter = options.get("delimiter", ",")
        if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in "\r\n\0":
            raise InputReaderError("delimiter must be one printable separator")
    return dict(options)


def _text(path):
    if path.stat().st_size > MAX_BYTES:
        raise InputReaderError("Raw file exceeds the 64 MiB reader limit")
    value = path.read_text(encoding="utf-8-sig")
    if "\0" in value or len(value) > MAX_CHARACTERS:
        raise InputReaderError("Text exceeds the character limit or contains NUL")
    return value


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise InputReaderError("Duplicate JSON object key")
        result[name] = value
    return result


def _json(value):
    def invalid(_):
        raise InputReaderError("Nonfinite JSON values are unsupported")
    return json.loads(value, object_pairs_hook=_unique, parse_constant=invalid)


def _record_rows(path, format_name, options, *, skip_field=None):
    text = _text(path)
    source = options.get("source_format", format_name)
    if source == "json":
        rows = _json(text)
    elif source in {"tsv", "delimited"}:
        delimiter = "\t" if format_name == "tsv" else options.get("delimiter", ",")
        parsed = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        if not parsed or len(set(parsed[0])) != len(parsed[0]) or not all(parsed[0]):
            raise InputReaderError("Delimited input requires unique nonempty headers")
        if any(len(row) != len(parsed[0]) for row in parsed[1:]):
            raise InputReaderError("Delimited row width differs from its header")
        rows = [dict(zip(parsed[0], row, strict=True)) for row in parsed[1:]]
    else:
        raise InputReaderError("Unsupported external record format")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS or any(not isinstance(row, dict) for row in rows):
        raise InputReaderError("Expected a nonempty bounded list of record objects")
    if any(isinstance(value, (dict, list)) for row in rows for value in row.values()):
        raise InputReaderError("Record fields must be flat scalar values; nested JSON is unsupported")
    if format_name == "convert":
        fields = options["field_paths"]
        try:
            rows = [{name: row[keys[0]] for name, keys in fields.items() if name != skip_field} for row in rows]
        except KeyError as exc:
            raise InputReaderError("A configured top-level conversion field is missing") from exc
    return rows


def _directory_files(root, relative, extensions):
    directory = source_path(root, relative, directory=True)
    paths = []
    for category in sorted(directory.iterdir()):
        if category.name.startswith("."):
            continue
        if category.is_symlink() or not category.is_dir():
            raise InputReaderError("Class directory reader expects class subdirectories, without symlinks")
        for path in sorted(category.iterdir()):
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in extensions or not path.is_file() or path.is_symlink():
                raise InputReaderError("Unexpected entry in a class directory")
            paths.append((path.relative_to(root).as_posix(), category.name))
            if len(paths) > MAX_ROWS:
                raise InputReaderError("Directory exceeds the sample limit")
    if not paths:
        raise InputReaderError("Class directory contains no supported samples")
    return paths


def _image(root, relative, touch):
    from PIL import Image, ImageOps, UnidentifiedImageError

    path = source_path(root, relative)
    if path.suffix.lower() not in IMAGE_EXTENSIONS or path.stat().st_size > MAX_BYTES:
        raise InputReaderError("Expected a bounded JPG/PNG file")
    touch(relative)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG"} or image.width * image.height > 16_000_000:
                    raise InputReaderError("Unsupported image encoding or pixel count")
                image = ImageOps.exif_transpose(image).convert("RGB")
                return np.asarray(image, dtype=np.uint8)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputReaderError(f"Cannot decode image: {relative}") from exc


def _reader_image(root, relative, reader, touch, *, fit):
    image = _image(root, relative, touch)
    shape = list(image.shape)
    expected = reader.get("decoded_image_shape")
    if expected is None:
        if not fit:
            raise InputReaderError("Frozen image dimensions are missing; rediscover the training task")
        reader["decoded_image_shape"] = shape
    elif shape != expected:
        raise InputReaderError(f"Image size differs from training: {relative} has {shape}, expected {expected}")
    return image


def _image_batch(root, paths, reader, touch, *, fit):
    images = []
    elements = 0
    for relative in paths:
        image = _reader_image(root, relative, reader, touch, fit=fit)
        elements += image.size
        if elements > MAX_ELEMENTS:
            raise InputReaderError("Decoded image split exceeds the array element limit")
        images.append(image)
    if not images:
        raise InputReaderError("Image split must contain at least one image")
    return np.stack(images)


def filter_record_samples(root, records, reader, source, split, kinds, *, source_rows=None, image_column=None, touch=None):
    from . import task_directory as td

    positions = list(range(len(records))) if source_rows is None else source_rows
    text_columns = [name for name, kind in kinds.items() if kind == "text"]
    reasons, sample_rows = [], []
    for position, row in zip(positions, records, strict=True):
        if any(name not in row for name in text_columns) or image_column is not None and image_column not in row:
            raise InputReaderError("A declared text/image column is absent from the input")
        missing = [f"missing_text:{name}" for name in text_columns if td._missing_content(row[name])]
        if image_column is not None:
            value = row[image_column]
            if td._missing_content(value):
                missing.append(f"missing_image:{image_column}")
            elif not isinstance(value, str):
                raise InputReaderError("Image paths must be strings")
            else:
                relative = (PurePosixPath(reader["options"].get("image_base", "")) / value).as_posix()
                try:
                    source_path(root, relative)
                except FileNotFoundError:
                    missing.append(f"missing_image_file:{image_column}")
                else:
                    if touch is not None:
                        touch(relative)  # Identity includes existing references, even on a dropped row.
        parts = [str(row.get(key, "")) for key in reader.get("id_columns", [])]
        sample_rows.append({"source_file": source, "source_row": str(position), "original_id": "|".join(parts)})
        reasons.append(missing)
    kept = td._keep_present_samples(reader, split, sample_rows, reasons)
    return [records[index] for index in kept], [positions[index] for index in kept]


def _indices(ids, sources, source_rows=None):
    if len(ids) != len(set(ids)) or any(not value.strip() or any(c in value for c in "\r\n\0") for value in ids):
        raise InputReaderError("Sample IDs must be unique nonempty single-line strings")
    return [{"sample_id": value, "original_id": value, "source_file": source,
             "source_row": str(source_rows[i] if source_rows is not None else i)}
            for i, (value, source) in enumerate(zip(ids, sources, strict=True))]


def read_split(root, files, reader, split, touch, *, fit=False, prediction=False):
    """Read only an explicitly selected split; touch hashes every actual input."""
    from . import task_directory as td

    fmt, options = reader["format"], reader["options"]
    paths = files["feature_paths"]
    if len(paths) != 1:
        raise InputReaderError("Additional readers require one source per split")
    source = paths[0]
    if fmt in {"image_directory", "text_directory"}:
        extensions = IMAGE_EXTENSIONS if fmt == "image_directory" else {".txt"}
        directory = source_path(root, source, directory=True)
        visible = [p for p in sorted(directory.iterdir()) if not p.name.startswith(".")]
        if prediction and visible and all(p.is_file() and p.suffix.lower() in extensions for p in visible):
            entries = [(p.relative_to(root).as_posix(), "__prediction__") for p in visible]
        else:
            entries = _directory_files(root, source, extensions)
        ids = [name for name, _ in entries]
        y = td._labels(np.asarray([label for _, label in entries], dtype=object))
        if fmt == "image_directory":
            X = _image_batch(root, [name for name, _ in entries], reader, touch, fit=fit)
            if fit:
                reader.update(output_kind="image", modality=td._image_modality(X), feature_columns=[])
        else:
            values = []
            total = 0
            for name, _ in entries:
                touch(name)
                text = _text(source_path(root, name))
                total += len(text)
                if total > MAX_CHARACTERS:
                    raise InputReaderError("Text directory exceeds the corpus character limit")
                values.append([text])
            if fit:
                _set_text(reader, ["text"])
            raw = np.asarray(values, dtype=object)
            raw, y, sample_rows = td._filter_array_samples(raw, y, _indices(ids, [name for name, _ in entries]), reader, split)
            return td._text_matrix(raw), y, sample_rows
        sample_rows = _indices(ids, [name for name, _ in entries])
        td._keep_present_samples(reader, split, sample_rows, [[] for _ in entries])
        return X, y, sample_rows
    touch(source)
    path = source_path(root, source)
    if fmt == "text_lines":
        label_path = files.get("labels_path")
        if not label_path and not prediction:
            raise InputReaderError("text_lines requires a separate labels file")
        texts = _text(path).splitlines()
        if prediction:
            labels = ["__prediction__"] * len(texts)
        else:
            touch(label_path)
            labels = _text(source_path(root, label_path)).splitlines()
        if not 1 <= len(texts) <= MAX_ROWS or len(texts) != len(labels):
            raise InputReaderError("Text and label line counts must match; labels cannot be empty")
        if fit:
            _set_text(reader, ["text"])
        raw, labels, sample_rows = td._filter_array_samples(
            np.asarray(texts, dtype=object).reshape(-1, 1), np.asarray(labels, dtype=object),
            td._array_sample_rows(len(labels), files, split), reader, split)
        if any(not label.strip() for label in labels):
            raise InputReaderError("Labels cannot be empty on retained text samples")
        return td._text_matrix(raw), td._labels(labels), sample_rows
    if fmt == "image_csv":
        header, values = td._read_csv(path)
        records = [dict(zip(header, row, strict=True)) for row in values]
        if len(records) > MAX_ROWS:
            raise InputReaderError("Image index exceeds the sample limit")
        column = options["image_column"]
        if column not in header or column == reader["target_column"] or column in reader["id_columns"]:
            raise InputReaderError("image_column is absent or overlaps a label/ID")
        auxiliary = options.get("auxiliary_types", {})
        records, source_rows = filter_record_samples(root, records, reader, source, split, auxiliary,
                                                     image_column=column, touch=touch)
        base = options.get("image_base", "")
        image_paths = [(PurePosixPath(base) / row[column]).as_posix() for row in records]
        if len(set(image_paths)) != len(image_paths):
            raise InputReaderError("Repeated image paths prevent unambiguous sample tracing")
        if auxiliary:
            if set(auxiliary) - set(header) or set(auxiliary) & {column, reader["target_column"], *reader["id_columns"]}:
                raise InputReaderError("Auxiliary columns are absent or overlap labels/IDs/images")
            # Keep every source pixel, in HWC order. Only invertible value
            # scaling is applied; no resizing, pooling, or spatial compression.
            vectors = []
            for name in image_paths:
                image = _reader_image(root, name, reader, touch, fit=fit)
                if len(records) * (image.size + len(auxiliary)) > MAX_ELEMENTS:
                    raise InputReaderError("Full-resolution mixed input exceeds the array element limit")
                vectors.append(image.reshape(-1).astype(float) / 255)
            if not vectors:
                raise InputReaderError("Image index must contain at least one sample")
            names = [f"__image_pixel_{i}" for i in range(len(vectors[0]))]
            if set(names) & set(header):
                raise InputReaderError("Reserved image feature names collide with source columns")
            for row, vector in zip(records, vectors, strict=True):
                row.update(dict(zip(names, vector, strict=True)))
            if fit:
                reader["declared_types"] = {**auxiliary, **dict.fromkeys(names, "numeric")}
                reader["feature_columns"] = list(reader["declared_types"])
                reader["mixed_inputs"] = ["image", *sorted(set(auxiliary.values()))]
        else:
            images = _image_batch(root, image_paths, reader, touch, fit=fit)
            y = td._labels(np.asarray(["__prediction__" if prediction else row[reader["target_column"]]
                                      for row in records], dtype=object))
            ids = _record_ids(records, reader, split, source_rows)
            if fit:
                reader.update(output_kind="image", modality=td._image_modality(images), feature_columns=[])
            return images, y, _indices(ids, image_paths, source_rows)
    else:
        records = _record_rows(path, fmt, options, skip_field=reader["target_column"] if prediction else None)
    return _records_to_arrays(records, reader, source, split, fit=fit, prediction=prediction,
                              source_rows=source_rows if fmt == "image_csv" else None, filter_missing=fmt != "image_csv")


def _set_text(reader, columns):
    reader.update(output_kind="text", feature_columns=columns,
                  feature_types=dict.fromkeys(columns, "text"),
                  modality={"kind": "text", "text_columns": columns, "max_characters": 100_000})


def _record_ids(records, reader, split, source_rows=None):
    columns = reader["id_columns"]
    if not columns:
        return [f"{split}_{i:09d}" for i in (source_rows if source_rows is not None else range(len(records)))]
    try:
        if any(row[key] is None or isinstance(row[key], (dict, list)) for row in records for key in columns):
            raise InputReaderError("Sample IDs must be nonmissing scalar values")
        return [json.dumps([row[key] for key in columns], ensure_ascii=False, separators=(",", ":"))
                if len(columns) > 1 else str(row[columns[0]]) for row in records]
    except KeyError as exc:
        raise InputReaderError("A declared ID field is missing") from exc


def _records_to_arrays(records, reader, source, split, *, fit, prediction=False, source_rows=None, filter_missing=True):
    from . import task_directory as td

    target, ids = reader["target_column"], reader["id_columns"]
    if not isinstance(target, str) or not target:
        raise InputReaderError("Record input requires a declared target_column")
    if fit:
        columns = reader["feature_columns"] or [name for name in records[0] if name not in {target, *ids}]
        if not columns or set(columns) & {target, *ids}:
            raise InputReaderError("Features must be nonempty and exclude labels and IDs")
        try:
            kinds = td._column_kinds(columns, {c: [row[c] for row in records] for c in columns},
                                     {"feature_types": reader["declared_types"]} if reader["declared_types"] else {})
        except KeyError as exc:
            raise InputReaderError("A declared feature field is missing") from exc
        reader.update(feature_columns=columns, feature_types=kinds, output_kind="table")
        if all(kind == "text" for kind in kinds.values()):
            _set_text(reader, columns)
    columns = reader["feature_columns"]
    if filter_missing:
        records, source_rows = filter_record_samples(root=None, records=records, reader=reader, source=source,
                                                     split=split, kinds=reader["feature_types"], source_rows=source_rows)
    if len(records) * len(columns) > MAX_ELEMENTS:
        raise InputReaderError("Record input exceeds the array element limit")
    try:
        X = np.asarray([[row[c] for c in columns] for row in records], dtype=object)
        y = td._labels(np.asarray(["__prediction__" if prediction else row[target] for row in records], dtype=object))
    except KeyError as exc:
        raise InputReaderError("A feature or target field is missing") from exc
    if any(isinstance(value, (dict, list)) for value in X.flat):
        raise InputReaderError("Nested feature values are unsupported")
    raw = X.copy()
    X = td._npy_features(X, reader)
    # Keep numeric-looking lexical values until the final B type decision;
    # a documented category such as "001" must never turn into "1".
    if reader["output_kind"] == "table":
        for index, name in enumerate(columns):
            if reader["feature_types"][name] == "numeric":
                for i, value in enumerate(raw[:, index]):
                    if isinstance(value, str) and value.strip().casefold() not in td._MISSING:
                        X[i, index] = value
    return X, y, _indices(_record_ids(records, reader, split, source_rows), [source] * len(records), source_rows)


def reader_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def normalize_prediction_input(rules_path, root, relative, destination):
    """Apply frozen raw-input rules without needing or emitting any labels."""
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    if rules.get("schema_version") != "1.0" or rules.get("reader_sha256") != reader_hash():
        raise InputReaderError("Frozen input reader differs from the installed implementation")
    reader = rules["reader"]
    validate_options(reader["format"], reader["options"])
    hashes = {}
    from .task_directory import _sha256

    def touch(name):
        hashes[name] = _sha256(source_path(root, name))
    X, _, sample_rows = read_split(Path(root).resolve(), {"feature_paths": [relative], "labels_path": None},
                         reader, "prediction", touch, prediction=True)
    if any(_sha256(source_path(root, name)) != value for name, value in hashes.items()):
        raise InputReaderError("Prediction source changed during conversion")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        np.save(stream, X, allow_pickle=True)
    from .sample_index import save_sample_index
    save_sample_index(target.parent / "sample_index.csv", sample_rows)
    with (target.parent / "filtering_report.json").open("x", encoding="utf-8") as stream:
        json.dump(reader.get("sample_filtering", {}).get("prediction", {}), stream, ensure_ascii=False, indent=2)
    return target


def additional_contract(root, inventory, info, resolution=None, *, allow_auto=True):
    """Resolve added formats from a validated LLM plan, info.json, or conventions."""
    if resolution is not None and resolution.get("external_reader"):
        contract = resolution["external_reader"]
    elif info.get("reader_format"):
        contract = {"format": info["reader_format"], "options": info.get("reader_options", {}),
                    "partitions": info.get("partitions"), "target_column": info.get("target_column"),
                    "id_columns": info.get("id_columns", []), "feature_columns": info.get("feature_columns", []),
                    "feature_types": info.get("feature_types", {})}
    elif allow_auto:
        from .task_directory import _pick_split_files
        candidates = [(fmt, _pick_split_files(inventory, fmt)) for fmt in ("tsv",)]
        candidates = [(fmt, parts) for fmt, parts in candidates if parts]
        if len(candidates) > 1:
            raise InputReaderError("Multiple input formats require an explicit reader_format")
        if not candidates:
            return None
        fmt, parts = candidates[0]
        contract = {"format": fmt, "options": {}, "partitions": {
            role: {"features": [path], "labels": None} for role, path in parts.items()},
            "target_column": info.get("target_column"), "id_columns": info.get("id_columns", []),
            "feature_columns": info.get("feature_columns", []), "feature_types": info.get("feature_types", {})}
    else:
        return None
    fmt = contract["format"]
    options = validate_options(fmt, contract.get("options", {}))
    parts = contract.get("partitions")
    if parts is None:
        from .task_directory import _SPLITS, _pick_split_files
        extension = "csv" if fmt == "image_csv" else "txt" if fmt == "text_lines" else fmt
        names = ({role: inventory[alias] for alias, role in _SPLITS.items() if alias in inventory}
                 if fmt.endswith("_directory") else _pick_split_files(inventory, extension))
        parts = {role: {"features": [path], "labels": None} for role, path in names.items()}
        if fmt == "text_lines":
            labels = _pick_split_files(inventory, "txt", "y_")
            for role, part in parts.items():
                part["labels"] = labels.get(role)
    if not isinstance(parts, dict) or "train" not in parts or set(parts) - {"train", "validation", "test"}:
        raise InputReaderError("Reader partitions require train and optional validation/test")
    used, files = [], {}
    for role, part in parts.items():
        if part is None:
            continue
        if not isinstance(part, dict) or set(part) != {"features", "labels"} or not isinstance(part["features"], list) or len(part["features"]) != 1:
            raise InputReaderError("Each split needs one feature source and a labels field")
        names = [*part["features"], *([part["labels"]] if part["labels"] is not None else [])]
        if (fmt == "text_lines") != (part["labels"] is not None):
            raise InputReaderError("Only text_lines uses separate labels in additional readers")
        for name in names:
            if (not isinstance(name, str) or not name or "\\" in name or PureWindowsPath(name).drive
                    or PurePosixPath(name).is_absolute() or any(p.startswith(".") for p in PurePosixPath(name).parts)):
                raise InputReaderError("Split sources must use visible relative paths")
            if any(name == other or name.startswith(other + "/") or other.startswith(name + "/") for other in used):
                raise InputReaderError("Reader split sources overlap")
            used.append(name)
        files[role] = {"feature_paths": part["features"], "labels_path": part["labels"]}
    if "train" not in files:
        raise InputReaderError("The train split cannot be null")
    columns, ids = contract.get("feature_columns", []), contract.get("id_columns", [])
    for names in (columns, ids):
        if not isinstance(names, list) or len(names) > 128 or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names):
            raise InputReaderError("Invalid or duplicate feature/ID column names")
    if set(columns) & {contract.get("target_column"), *ids}:
        raise InputReaderError("Label and ID fields cannot be features")
    kinds = contract.get("feature_types", {})
    if not isinstance(kinds, dict) or (kinds and set(kinds) != set(columns)) or any(x not in {"numeric", "categorical", "text"} for x in kinds.values()):
        raise InputReaderError("feature_types must cover the declared feature columns")
    reader = {"format": fmt, "options": options, "target_column": contract.get("target_column"),
              "id_columns": ids, "feature_columns": columns, "declared_types": kinds}
    return reader, files
