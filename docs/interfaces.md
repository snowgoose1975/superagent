# SuperAgent module interfaces

`main.py` coordinates discovery, preparation, planning, training, validation, and freeze. This document uses A for planning, B for data preparation, C for execution, and D for evaluation and accounting.

## Planning

A implements `decide(context)`. The result is either `{"action":"stop","reason":"..."}` or a run proposal with `action`, `model`, `params`, and `reason`, plus an optional `pipeline` for schema 2.0. A receives the declared task metric and compatible capabilities, bounded training statistics, plan history, and validation feedback. It may use the school gateway or local rules. The main runner assigns the experiment ID, seed, timeout, and version and validates the complete plan before execution.

Enhanced search applies family coverage, duplicate recipe checks, patience, and experiment limits. A configured finalization reserve bounds school request time. Optional seed confirmations consume local experiment slots and add no school requests. Unknown provider usage stops the session. A previously successful experiment may be frozen as a partial result when a fatal interruption prevents normal finalization.

## Data preparation, execution, and evaluation

| Module | Input | Output |
| --- | --- | --- |
| B | A task directory or validated TaskSpec | Immutable prepared data, training profile, separate evaluation manifest, preprocessor where applicable |
| C | Validated plan, prepared data manifest, and run directory | Model, ordered validation predictions, structured diagnostics, and result manifest |
| D | C result manifest and B evaluation manifest | Task metric on validation, keyed by sample ID |
| Final inference | Frozen bundle and features | Ordered class IDs and probabilities |

Opt-in B analysis adds declarative layout and preprocessing proposals through the same school gateway. B quality evidence and profiles come from task documentation and training data. Test materialization happens after validation-based freeze. [B analysis contracts](b-llm-upgrades.md) · [Final test](final-test.md)

Table data use schema 1.0, with B's fitted feature conversion and an optional native categorical path for CatBoost. Schema 2.0 carries a typed text or image representation and a registered C pipeline. Every worker validates shapes, types, row alignment, and artifacts. C receives validation features and sample IDs. D receives validation labels separately. The task metrics are accuracy, macro F1, and log loss, with declared optimization directions.

## Main entry points

```text
main.py run --task <directory> [--planner offline|school]
main.py run --config <task-spec.json> [--test-config <test-config.json>]
main.py evaluate-test --bundle <bundle.json> --test-config <test-config.json>
predict.py --bundle <bundle.json> --features <file> --output <predictions.csv>
```

Automatic tasks enable enhanced search. The legacy config interface can opt in with `--search-config configs/search-defaults.json`. The Linux/WSL launcher selects school performance mode. Session event logs are time-ordered indexes. Trial history, result manifests, and gateway records retain the detailed evidence.

## Frozen portfolios

Performance search may compare the validation winner with compatible probability ensembles. An ensemble bundle contains an ordered `ensemble` list of relative member bundle paths. The parent stores member manifest checksums. Each member verifies its own artifacts and predicts in the shared class order. Inference averages member probabilities and emits the highest-probability class. Frozen test evaluation accepts the top-level bundle.
