# Classifier implementation handoff

Historical work breakdown for the v2 classifier registry, executor, and module integration. Current operation is documented in the [README](../README.md). Dated evidence is in [validation](VALIDATION.md).

| Stage | Change | Acceptance |
| --- | --- | --- |
| H0 | Preserve the initial baseline and artifacts | Record environment, data version, metrics, seeds, and known failure paths |
| H1 | Canonical registry and plan normalization | Export one capability list and reject unsupported model names, parameters, and code |
| H2 | Four dense classifier families | Train and verify binary and multiclass prediction probabilities |
| H3 | Structured diagnostics | Keep failure status, convergence, dimensions, runtime, and artifact evidence with each trial |
| H4 | Planner search integration | Cover families, deduplicate entire recipes, apply patience, preserve finalize reserve, and support optional confirmation seeds |
| H5 | Task metrics | Propagate accuracy, macro F1, and log loss and their selection direction through profile and frozen evaluation |
| H6 | Typed text and image pipelines | Validate shape, feature order, and modality, then load frozen models in a new process |
| H7 | Reference-price task budget | Reserve atomically, distinguish known and unknown usage, reuse historical accounting, and emit deterministic reports |
| H8 | Sealed final test | Assert freeze precedes test reads, verify full rows and probabilities, and retain the bundle on a test error |
| H9 | Automatic tasks | Discover conventional CSV/NPY layouts and forward bounded metadata and one data identity |
| H10 | General text portfolio and baseline | Verify independent field representations, recipe coverage, and majority/prior baseline diagnostics |

## Responsibility boundaries

A proposes bounded model and pipeline choices based on training summaries and validation feedback. B declares the prepared representation and owns train-fitted classic table preprocessing. C trains and predicts from the manifest with isolated workers. D aligns labels by sample ID, scores the declared metric, and runs budget accounting. The runner owns version, experiment identity, search policy, and finalization. Test data are materialized after freeze.

The legacy table schema 1.0 path uses the same experiment directory, metadata, and probability conventions. Schema 2.0 records raw text or image tensor modality and the corresponding pipeline. Frozen bundles must preserve the fitted artifacts, exact class order, approved preprocessing path, and resource information required by a fresh process. Worker input and output checksums support reproducibility.

Model family coverage and compatible advanced models are data-dependent. Include pipeline choices in the normalized recipe key. Treat convergence messages as feedback. Stop only on explicitly recorded search, resource, accounting, or failure conditions. Confirmations use configuration-defined seed offsets and add no paid calls.

## Development verification

```bash
python -m compileall -q main.py agent_a data_b executor_c evaluation_d scripts
python -m pytest
ruff check .
git diff --check
```

Use fresh run roots, retain generated evidence, and use mock transport for school gateway verification. Regression checks should cover copied or stale payloads, missing fields, malformed numeric values, repeated proposals, stable IDs, unknown usage, exhausted call budgets, worker errors, artifact changes, and freeze-before-test order. Compare fixed train/validation partitions before drawing conclusions. [Interfaces](interfaces.md) · [Budget](task-budget.md)

Library references: [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html), [histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Recorded environment versions govern reproduction.
