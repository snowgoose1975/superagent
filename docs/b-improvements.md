# Data preparation improvements

This records the initial B changes and their artifact contracts. [LLM-assisted preparation](b-llm-upgrades.md) describes the later opt-in stage.

- A supplied validation partition is retained. B can create a deterministic stratified holdout from training data when necessary. Test remains optional. A run with no test partition reports `not_requested`.
- Explicit CSV sample IDs keep their original text and leading zeroes. Composite IDs are losslessly represented as JSON arrays. Generated IDs derive from source partition and original row offset. Missing or duplicate IDs within a partition are rejected. Overlap across official partitions is reported without silently removing data.
- Numeric, categorical, and text table columns are prepared independently. `info.json.feature_types` may name all input columns and their declared types. Pure text and NPY image inputs retain their TaskSpec 2.0 flows.
- Training-only profiles summarize missingness, constant columns, cardinality, duplicate feature rows, text length, and image pixel ranges. A receives bounded aggregates. Full details are recorded in the prepared profile.
- Automatic prepared version signatures include preparation code digests. Budget identity remains tied to original training content.

## Sample index

The optional `partitions.train.sample_ids_path` and `partitions.validation.sample_ids_path` identify UTF-8 CSV files inside `source_root`. The index requires a unique nonempty `sample_id`, with optional `source_file`, `source_row`, and `original_id`. Source row offsets are zero-based and exclude a CSV header. A recorded source path is provenance text.

Prepared data combines train and validation provenance in `metadata/sample_index.csv`. C receives validation IDs through its existing manifest path. Frozen test materialization writes `sample_index_test.csv` and records its checksum in `test_source.json`. Ordered prediction IDs can be mapped back through this index.

## Mixed tables

The initial mixed-table transformer used training medians, train-fit category imputation and one-hot encoding, and per-field TF-IDF for text. Each text field has a bounded vocabulary and shares the remaining feature budget. The resulting finite dense matrix is capped at 4096 features and 256 MiB. These are feature-output bounds. See the runtime capability registry for the current optional native and sparse candidates. Frozen prediction reuses the fitted preprocessing.

## Original verification

The first B verification ran 50 synthetic regressions. 48 passed and two Windows symlink permission cases were skipped. The runs checked original ID round trips, holdout source rows, reusable data checksums, training-only preprocessing, mixed tables, text and images, a fresh-process frozen prediction, and deferred test reads. [Recorded test results](b-validation-results.json)
