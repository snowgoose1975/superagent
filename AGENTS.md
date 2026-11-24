# SuperAgent Team v2 contributor instructions

## Modules and entry points

- `main.py` coordinates the modules. `predict.py` handles inference with a frozen bundle.
- `agent_a/` selects plans. `data_b/` resolves input layouts and prepares artifact manifests. `executor_c/` trains models and manages workers. `evaluation_d/` scores results and manages the school API gateway and budget.
- `configs/` holds nonsecret configurations. `docs/` holds project documentation. `scripts/` holds synthetic-data utilities. `tests/` holds verification. `.vscode/` holds project-only editor settings.
- Record the purpose of a new source directory in these instructions or an approved plan before creating it. Use lowercase English names with underscores for source directories.
- Manual school mode requires the project `.env` and the `SCHOOL_API_KEY` entry. The command-line launcher defaults to school mode. Direct `main.py` runs default to offline mode. Automated checks use synthetic credentials and mock transport. Preserve the user's real credentials and environment configuration.
- Optional `--b-analysis llm` uses the D gateway and persistent task ledger to resolve declared layouts, review training statistics, and propose declarative preprocessing. Gateway access requires a task identity before the first billed request. B uses validated configuration with fixed local code.

## Data and artifacts

- Keep original task files read-only. Model and rule selection must be independent of specific task names.
- B may reuse an immutable complete prepared version when source and artifact checksums match. Preserve incomplete versions as evidence and reject attempts to overwrite them.
- Use unique IDs for sessions, experiments, and frozen bundles. Preserve historical artifacts. Every experiment artifact belongs to its owning run and data version.
- Fit transforms and classifiers on training rows. B exports validation labels for D. A and C can access the validation scores or features defined by their interfaces.
- Select models by validation metrics. Keep test paths and contents out of the planner and search artifacts. Discover test paths without opening the test data. Materialize and score test only after the validation winner has been frozen.
- Default finalization freezes the trained model and its transformers. Preserve the fit partition and class mappings in the bundle.

## School requests and encoder resources

- Route every school API request through D. Reserve before sending a request and settle using provider token values. Unknown usage retains the reservation and stops the session. Retain API and task ledger history across restarts.
- Use the existing project reference prices for estimates. Keep credentials, authorization headers, raw prompts, and raw response bodies out of logs and the repository.
- Direct legacy school runs default to one call. Automatic tasks default to 20 calls and a task budget of 20 RMB at reference prices. Reuse the ledger associated with the same training input.
- Keep Qwen downloads anonymous and pinned to a public revision. Verify the weights before local inference. Weights live in the project cache. Content-addressed train and validation embeddings live under the run root's development cache. Final test embeddings are never cached there.
- Exclude credentials, caches, virtual environments, and raw test files from delivery archives.

## Verification and delivery

- Preserve the interfaces in `docs/interfaces.md` and the C/D artifact contracts when replacing modules. Document copied code and integration changes in `docs/provenance.md`.
- Verify with pytest, Ruff, fresh-process model loading, and the relevant development partitions. Use fresh temporary directories and keep run evidence.
- Obtain authorization before deleting files, changing credentials, modifying system or global settings, changing CI or databases, rewriting history, pushing, or publishing externally.
