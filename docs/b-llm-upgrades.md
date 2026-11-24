# LLM-assisted data preparation

B can resolve task files, review training-data quality, and configure preprocessing with structured decisions from the school planner. This mode uses the same D gateway, call limit, and durable task budget as A.

## Run with school planning

```bash
bash run-task.sh ../task/my_task --b-analysis llm
```

Established layouts can reuse the ledger identified during deterministic preflight. For a layout that requires the LLM to locate the training files, supply the task's stable ledger ID before the first billed request:

```bash
bash run-task.sh ../task/my_task --b-analysis llm --task-budget-id course-task-03
```

Keep that ID across resumed runs, directory moves, and renames. If the LLM locates different training inputs from preflight or resolves a task with a conflicting historical identity, the run stops before quality review. Any billed attempt stays in its original ledger. Layout and quality review usually consume one request each. Filename-based document selection can add a request when conventional docs are unavailable. Layout resolution allows up to four attempts. Each other requested B stage allows one correction for an invalid response or an empty reply with settled usage. Corrections use the same gateway and budget. Other gateway errors propagate to the session. Unknown usage retains its reservation. Local rule-based discovery remains the default.

## Layout resolution

B inventories up to 256 visible file and directory entries. Class directories represent groups of individual sample files. B checks conventional Markdown and RST instructions first. When it requests document selection, the planner sees candidate paths before any candidate contents. Use repeatable `--task-doc` to specify Markdown, RST, TXT, or JSON instructions relative to the task root and skip that request. Supply one to four documents, up to 8,000 bytes each and 9,000 bytes total. B validates that data paths are distinct and inside the task directory. The LLM supplies exact document excerpts supporting the paths, target column when applicable, and metric. Unknown or conflicting requirements produce a diagnostic.

The planner describes source files, partition roles, features, label location, and the metric. Program code chooses the reader and validates its parameters. CSV and TSV support label columns. NPY supports one feature array or separate numeric and categorical arrays with a label vector. Numeric-looking CSV categories retain their lexical form, including leading zeroes. Fixed adapters also handle line text, text and image directories, image references, and bounded record conversion. See the reader guide for declarations and unlabeled test formats. [Input readers](b-readers-and-conversion.md)

## Quality review

B computes training-only statistics such as missingness, cardinality, low-frequency values, constant columns, duplicate feature rows, and modality-specific summaries. The LLM receives bounded statistics and excerpts from task documentation. A concise response can suggest just a numeric imputation strategy, for example `{"preprocessing":{"numeric_imputation":"median"}}`. The school review can also record findings and unresolved questions. Its applicable configuration is limited to numeric imputation. Feature types come from the layout and reader, and explicit preprocessing settings remain supported for manually configured tasks. Artifacts live under the session's `b_analysis/` directory. Test content stays sealed until after validation-based selection and freeze.

## Declarative preprocessing

TaskSpec 1.0 accepts a bounded `preprocessing` object such as:

```json
{
  "numeric_imputation": "median",
  "add_missing_indicator": true,
  "categorical_min_frequency": 3,
  "missing_values": {"age": [-999]}
}
```

Numeric filling supports median, mean, and most frequent. Missingness indicator placement and low-frequency grouping are fit on training rows. The frequency threshold is from 2 to 1000, with up to eight declared missing sentinels per numeric feature. A newly observed category goes to the infrequent bucket when that bucket exists. The corrected types and preprocessing enter the version signature and applicable frozen transforms. Text and image quality review leaves classifier pipelines under C's capability registry.

## Interfaces and evidence

`data_b.llm_analysis.resolve_layout` returns layout declarations and excerpts. `discover_task_directory(..., resolution=...)` validates and stages the layout. `review_quality` returns the checked TaskSpec and review artifacts. Programmatic school runs use `run_project(..., b_analysis="llm", task_documents=...)` with a persistent budget through D. B's original regression used a mock HTTP transport with synthetic tasks and exercised sealed test input, sample IDs, training-only fitting, unseen values, and frozen predictions. Deployment acceptance needs task-specific measurements. [B verification record](b-llm-validation-results.json)

If both layout replies are empty with settled usage, a successful deterministic preflight can supply the layout and A can continue using the same gateway. Two settled empty quality replies retain the validated preprocessing defaults. An unrecognized layout still requires a valid layout reply.
