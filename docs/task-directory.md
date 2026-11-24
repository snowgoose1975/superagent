# Automatic task directories

Pass a task directory to `main.py run --task` or `bash run-task.sh`. B reads task documentation and trains only on the declared development data. It normalizes features and labels into the session's `task_input/`. Run with `--planner school --b-analysis llm` to use the documented layouts and requirements in [LLM-assisted B](b-llm-upgrades.md).

| Input | Supported form |
| --- | --- |
| Tabular or text CSV | Train and optional validation/test CSV with labeled samples |
| NPY table or image | Numeric or categorical feature arrays, or image tensors, plus a one-dimensional label array |
| Task metadata | README and `info.json` for conventional layouts |

Examples of conventional names are `train.csv`, `val.csv`, `test.csv`, or `X_train.npy` with `y_train.npy`. Separate `N_*.npy` and `C_*.npy` arrays support numeric and categorical NPY tables. A supplied validation split is preserved. Otherwise a bounded deterministic stratified split is made from training. Explicit metadata and documentation can identify target, ignored identifier columns, metric, and feature kinds. Unknown or ambiguous target and metric declarations produce errors. A supplied test partition is described lazily and materialized after freeze. If the task supplies no test partition, the report marks final test `not_requested`.

`task.discovery.json` records the resolved format, source hashes for development inputs, chosen metric, sample provenance, and the task budget identity. C receives a validated manifest and D receives a separate label manifest. School planning uses bounded profile information and score feedback. [Module interfaces](interfaces.md) · [Final test](final-test.md)
