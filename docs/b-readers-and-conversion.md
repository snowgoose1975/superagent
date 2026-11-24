# B input readers and conversion

B turns task files into the existing training and validation manifests. It selects readers from file roles, declared feature types, and where necessary the training data format. All readers and conversions run as fixed local code. The source directory stays read-only. Official test data is opened only after validation has selected and frozen a model.

## Supported layouts

| Layout | Input | Development labels | Feature-only test |
| --- | --- | --- | --- |
| `csv` | CSV with a header | Target column | Omit the target column |
| `npy` | One NumPy feature array or supported numeric and categorical arrays | Separate label array | Omit the test label array |
| `tsv` | UTF-8 TSV with a header | Target column | Omit the target column |
| `text_lines` | One UTF-8 text sample per line | Separate label file with one label per line | Omit the test label file |
| `text_directory` | One `.txt` sample per file | Class subdirectory names | Put test files directly in the test directory |
| `image_directory` | One JPEG or PNG sample per file | Class subdirectory names | Put test images directly in the test directory |
| `image_csv` | CSV with an image path column, optionally alongside table or text fields | Target column | Omit the target column |
| `convert` | Flat JSON records or delimited text | Mapped target field | Omit the target field |

Conventional CSV, NPY, and TSV partition names support rule-based discovery. Other supported layouts can declare `reader_format`, `reader_options`, and `partitions` in `info.json`. For tasks with documented file roles and nonstandard names, run `bash run-task.sh ../task/my_task --b-analysis llm`. When preflight cannot recognize the task identity, add a stable `--task-budget-id`. [LLM-assisted data preparation](b-llm-upgrades.md)

### Image index example

Use an `info.json` like this when each sample has an image path, a review, and a price in the same CSV row.

```json
{
  "metric": "accuracy",
  "reader_format": "image_csv",
  "reader_options": {
    "image_column": "photo",
    "auxiliary_types": {"review": "text", "price": "numeric"}
  },
  "target_column": "label",
  "id_columns": ["id"],
  "partitions": {
    "train": {"features": ["development.csv"], "labels": null},
    "validation": {"features": ["holdout.csv"], "labels": null},
    "test": {"features": ["sealed.csv"], "labels": null}
  }
}
```

Image paths are relative to the task directory by default. `image_base` sets a different relative directory. B applies EXIF orientation, decodes JPEG and PNG files as RGB, and preserves the resulting height and width. All images must match the shape learned from training. The reader rejects corrupt images and inconsistent dimensions.

Without `auxiliary_types`, B supplies an image tensor in TaskSpec 2.0. With `auxiliary_types`, B flattens each image in height, width, channel order and scales the RGB values by `1/255`. It combines those numeric values with the declared text, numeric, and categorical fields. The resulting data uses the existing TaskSpec 1.0 table interface. The B preprocessor fits any imputation, category encoding, or text vectorization on training rows and reuses those transforms and the pixel column order in other partitions.

### Flat record example

For a JSON array such as `[{"body":"Excellent","class":"positive"}, {"body":"Poor","class":"negative"}]`, declare the source fields and the names the model receives.

```json
{
  "reader_format": "convert",
  "reader_options": {
    "source_format": "json",
    "field_paths": {"text": ["body"], "label": ["class"]}
  },
  "target_column": "label",
  "feature_columns": ["text"],
  "feature_types": {"text": "text"},
  "metric": "accuracy",
  "partitions": {"train": {"features": ["source.json"], "labels": null}}
}
```

For delimited records, use `"source_format": "delimited"` and a single-character `delimiter`. Conversion supports top-level fields, including a field-name mapping. B derives or verifies the reader rules from training input and applies those rules to subsequent partitions. It rejects nested JSON, external JSONL, and unsupported encodings or formats. Class-directory layouts require a single supported media type. The image index is a single table with one row per sample. Current file, sample, array, and dense matrix limits are in `data_b/input_readers.py` and the B preprocessor. Inputs beyond those limits produce an error.

## Split shards

CSV, TSV, NPY, line-based text, and flat records support ordered split shards. Specify the per-shard feature and label sources in task metadata or documented layout instructions. B checks each shard's alignment, schema, and IDs, then merges its rows while preserving their source file and row index. Large primitive NPY inputs may be opened as read-only memory maps. The combined data and preprocessing are still subject to the configured shape and allocation limits.

## Test coverage and outputs

For the readers with row filtering, empty text fields and missing image references can remove entire development rows. Each removal keeps features, labels, and identifiers aligned. Numeric missing values go through preprocessing. Inspect the session's `task_input/discovery_report.json` for input, retained, and dropped counts, reasons, and original row numbers. File corruption, ambiguous layouts, and contradictory labels can stop the task.

An official test partition requires full coverage. If the reader would drop rows, final evaluation fails and reports the row counts. The existing frozen bundle remains available. A labeled test produces its declared metric. A feature-only partition runs the same frozen predictor and produces predictions for every input row, using original sample IDs when supplied. The session report records whether the test was scored, generated predictions, or failed.

## Frozen raw input

Bundles trained with an additional input reader include `reader_rules.json` and its checksum. The normalized-array inference command still accepts `--features <array.npy>`. With a bundle that has frozen reader rules, use `--raw-input` to apply those rules to raw files or a class-directory reader's flat input directory.

```bash
python predict.py --bundle submission/my_task/final_run/bundle.json \
  --features new_data/samples.csv --raw-input --source-root new_data \
  --output predictions.csv
```

Supply new features using the training columns and image path rules. Standalone prediction allows label-free input and can filter missing text and image references. The adjacent `converted_*` directory holds the normalized array, `sample_index.csv`, and `filtering_report.json` for matching the retained rows back to their sources. Use the corresponding reader implementation when loading a frozen raw-input bundle.
