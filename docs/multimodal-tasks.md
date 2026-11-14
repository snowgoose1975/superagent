# TaskSpec 2.0: text and image classification

TaskSpec 1.0 remains the table route. TaskSpec 2.0 adds bounded raw-text and image-array inputs while keeping the same train/validation-only search, C executor, D scorer, metric selection, and frozen-prediction lifecycle. A task declares exactly two partitions, `train` and `validation`; it must not include a `test` field. Optional final test inputs live in a separate test config and are read only after the validation-selected bundle is frozen; see [`final-test.md`](final-test.md).

## Shared TaskSpec 2.0 fields

```json
{
  "schema_version": "2.0",
  "task_id": "text_demo",
  "data_version": "text_demo_v1",
  "source_format": "text_array_package",
  "source_root": "../data/text_demo",
  "modality": {
    "kind": "text",
    "text_columns": ["title", "body"],
    "max_characters": 10000
  },
  "label_values": [0, 1],
  "label_names": ["negative", "positive"],
  "primary_metric": {"name": "f1_macro", "greater_is_better": true},
  "partitions": {
    "train": {"features_path": "X_train.npy", "labels_path": "y_train.npy", "n_samples": 1000},
    "validation": {"features_path": "X_valid.npy", "labels_path": "y_valid.npy", "n_samples": 200}
  },
  "train_policy": {"max_trials": 12, "random_seed": 42, "timeout_seconds": 120}
}
```

`source_root` is resolved relative to the TaskSpec file. Partition paths are relative to that directory. The labels in the validation partition are exported to D's separate evaluation package; C receives no validation labels. B signs the source files and relevant preparation code into a versioned immutable data package. If an input or preparation implementation changes, use a new `data_version`; old packages remain intact.

## Text arrays

Use `source_format: "text_array_package"` and `modality.kind: "text"`. Each feature file is an NPY matrix with one column per declared text field. A single text column may also use a one-dimensional NPY array. Primitive Unicode/byte arrays and safely parsed object arrays are accepted; each element must decode to a string. The reader limits each element to `max_characters`, allows at most eight columns, and caps a partition at eight million total characters. B republishes text as bounded JSONL, avoiding NPY fixed-width string expansion.

The supported text classifier is LogisticRegression with a registered TF-IDF vectorizer. Word or character analysis, n-gram bounds, document frequency, and vocabulary size have finite registry limits. C fits the vectorizer on train only and keeps it inside the frozen model pipeline. No user code or dynamic sklearn class names are accepted.

## Image arrays

Use `source_format: "image_array_package"` and `modality.kind: "image"`. Each feature file is an NPY tensor shaped `(N, H, W)` for `HW`, or `(N, H, W, C)` for `HWC`. `image_shape` declares one sample's dimensions, and `value_range` declares inclusive finite pixel bounds. A sample is limited to one million values; NPY reader limits also bound total array size.

The registered image baseline scales the declared pixel interval to `[0, 1]`, flattens each image, and then runs a whitelisted classifier. This is a stable classical baseline, not a CNN and not a claim that flattening is best for visual tasks. Supported image classifiers are RandomForest, ExtraTrees, HistGradientBoosting, and LogisticRegression. Optional feature selection is the registered `VarianceThreshold` component.

## Plan and prediction behavior

Plan 2.0 carries an explicit `representation` and a finite `pipeline` object. C rejects model/representation mismatches and unknown components before fitting. Plan/manifest 1.0 remain exact-schema compatible. The frozen bundle uses schema 2.0 for text/image tasks and includes the modality declaration plus the whole fitted C pipeline; `predict.py` accepts a new NPY feature array of the declared modality and emits encoded class IDs and probabilities. It does not fit, score, or select a model.

## Example runs

For the table configuration, omit a search config for the original short legacy profile. To run the longer configured search:

```bash
../.venv/bin/python main.py run --config configs/coupon.json \
  --search-config configs/search-defaults.json
```

A text or image TaskSpec 2.0 works through the same `--config` argument. School planning is optional; offline planning is the default and is sufficient to verify B/C/D. Keep `data_version` unique when replacing source arrays.
