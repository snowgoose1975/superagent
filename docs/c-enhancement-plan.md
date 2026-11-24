# Classifier enhancement plan archive

This records the design that preceded the current registry and performance portfolio. See [implementation handoff](c-implementation-handoff.md) and [validation](VALIDATION.md) for later changes and measured results.

## Course requirements

Use a single general-purpose classifier workflow across supported tasks. Respect the supplied train, validation, and test partitions and each task's metric. Keep test and final submission scoring outside search. Submit reproducible model, prediction, and accounting artifacts with the declared task policy. Eligibility and actual price verification remain separate from the published-price reference ledger. [Original course requirements](../AIAA3102%20-%20Final%20Project%20-%20Full%20Assignment%20Requirements%20%281%29.md)

## Initial design

- Use one model registry to generate capability descriptors and validate canonical parameter values in both planner and executor. Preserve compatible old plans and frozen bundles.
- Expand the table baseline from random forests and logistic regression to ExtraTrees and histogram boosting. Bound tree count, depth, and training iterations. Retain class ordering, probability validation, and train-only preprocessing.
- Emit structured diagnostics about convergence, duration, model size, and feature shape. Retain safe identifiers and error codes for failed trials. Limit resources in isolated workers, preserve artifacts on failure, and make duplicate recipes detectable.
- Compare model families first, then refine promising configurations and optionally confirm seeds. Use the registered primary metric and direction when comparing candidates. Prefer diverse candidates over repeated minor variants.
- Stage native categorical models, bounded text and image transforms, and aligned probability ensembles behind compatible interfaces. For learned stacking, use out-of-fold training predictions.
- Keep cost accounting independent of trial count. Reserve before requests, settle known usage, and retain reservations when usage is unknown.

A historical planning example allocated 30 local experiments across four baselines, eight exploration trials, 12 refinements, and six seed confirmations. It illustrated resource allocation. It was neither a required stopping policy nor evidence of achieved accuracy.

## Verification

Compare candidates on the same source bytes, validation partition, metric, seed, and search budget. Report failures and resources along with measured validation deltas. Keep even a stronger historical model as a fixed development reference. Freeze the validation winner before scoring test. Use pinned or recorded dependencies. Report only verified results.

References: [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) · [Histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) · [CatBoost categories](https://catboost.ai/docs/en/features/categorical-features)
