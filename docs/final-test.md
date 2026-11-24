# Final test evaluation

SuperAgent selects the best candidate by the task's declared validation metric. It freezes that model, its fitted preprocessing, and artifact checksums in `submission/<task>/<final-id>/`. The final evaluator opens the sealed test partition after the bundle exists. It scores the frozen predictions, using the original training label mapping, and writes `test_predictions.csv` and `test_evaluation.json` in the session. A task with no labeled test partition can still train, score on validation, and freeze.

The automatic directory runner resolves the test paths in advance but delays array and CSV content reads. A legacy configured task can supply a separate test config with schema version 1.0 and fields `task_id`, `source_root`, `features_path`, `labels_path`, and `n_samples`. Paths resolve inside `source_root`. Test labels must match declared training classes. Scoring uses the task's accuracy, macro F1, or log loss policy.

To evaluate an existing final bundle:

```bash
python main.py evaluate-test --bundle submission/<task>/<final-id>/bundle.json --test-config <test-config.json>
```

This command runs frozen inference and evaluation. A test error is saved separately and leaves the bundle in place. Use an intact bundle, its matching descriptor or test config, and the same encoder revision if it uses local embeddings. Test results are final reporting outputs. Search decisions, profiles, class mappings, and feature fitting use training or validation inputs according to their module contracts.
