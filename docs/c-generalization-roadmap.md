# General classification roadmap

The same search and evaluation path should serve unseen tabular, text, and NPY image tasks. Use compatible candidates, stable validation measurements, bounded experiments, and a durable task budget.

## Candidate families

| Modality | Candidate | Comparison |
| --- | --- | --- |
| Table | CPU XGBoost on prepared numeric features | Classic trees and linear classifier |
| Table | CatBoost on raw categorical columns | One-hot or train-fitted frequency alternatives |
| Text | Per-field TF-IDF, calibrated LinearSVC, hashing and SGD | Word and character reference recipes |
| Text | Frozen Qwen and Qwen plus lexical features | Sparse baseline and semantic-only features |
| Image | Color, gradients, and texture summaries | Flattened pixel baseline |
| Compatible portfolios | Aligned probability averages | Best single validation candidate |

Use declared types and the training size, feature dimensions, cardinality, missingness, text field structure, and label distribution. An unavailable dependency, weight, or device needs a runtime catalog decision independent of task name. Verify native CatBoost inference with missing values and unseen categories. Validate text calibration on training folds and avoid calibration candidates when class counts are too small.

## Search and scoring

1. Measure a training-prior validation baseline and a capable classic model.
2. Cover genuinely different model families and feature representations, then refine candidates with promising validation performance.
3. Use normalized recipe keys to avoid duplicates. Reuse embeddings for matching data, fields, encoder release, transform, device, and dtype.
4. Compare small probability ensembles only with identical sample IDs and class order. Keep learned weights, thresholds, and stacking on train folds.
5. Select by the declared validation objective. Freeze the complete model and preprocessing before opening test.

Report data fingerprints, candidate outcomes, improvements over the baseline, selected recipe, resource failures, wall time, and the reason for stopping. If more trials fail to help, patience and finite candidate space should bound the run. Report historical test scores as observations from frozen runs. New recipe decisions use development partitions and independent tasks.

## Budget and release

Keep local trial count separate from paid planner calls. Persist one task identity and preserve settled usage and reservations across restarts. Use Decimal accounting and reserve verified upper bounds before each request. Known call limits can yield to local search. Unknown usage must retain its reservation and halt the active session. Display billed call counts, settled cost, pending or unknown reservations, and reference price source.

The archived public prices do not establish the institution's invoice or eligibility of local model usage. A formal cross-machine evaluation needs a shared budget policy and confirmed pricing under the [course rules](../AIAA3102%20-%20Final%20Project%20-%20Full%20Assignment%20Requirements%20%281%29.md).

Release checks compare both published tasks on matched development partitions and exercise another representative task without task-specific code branches. Confirm old bundle compatibility, fresh-process inference, freeze-before-test, and the candidate failure paths. Require defensible, reproducible measurements before promising a gain. [Validation records](VALIDATION.md)
