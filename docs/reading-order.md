# Reading order

1. Read the [README](../README.md) for setup, execution modes, and artifacts.
2. Follow `main.py` through discovery, plan selection, execution, and freeze.
3. Follow `data_b/task_directory.py` and `data_b/prepare.py` through input discovery and train-only preparation. Inspect the task discovery report and B manifests together.
4. Read `agent_a/controller.py` and the upstream planner to see how bounded capabilities and validation scores produce a candidate.
5. Read `executor_c/executor.py`, `executor_c/train.py`, and the model registry to follow worker validation, training, and prediction.
6. Read `evaluation_d/evaluate.py` and `evaluation_d/final_test.py` to follow validation scoring and frozen final evaluation.

Useful debugging breakpoints are the call to `prepare_data`, the first invocation of the controller, the call to `execute`, and the invocation of the scorer in `main.py`. Review `history.jsonl` and `events.jsonl` alongside the final session summary. [Module contracts](interfaces.md)
