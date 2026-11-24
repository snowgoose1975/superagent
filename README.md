# SuperAgent Team v2

SuperAgent searches registered classifiers for a supplied classification task. A proposes model and feature recipes, B prepares training and validation data, C fits models and produces predictions, and D scores and freezes the selected model. Final test evaluation starts after the validation winner is frozen.

## Run on Linux or WSL

Clone the repository from a Linux or WSL terminal:

```bash
git clone https://github.com/snowgoose1975/superagent.git
cd superagent
test -e .env || cp .env.example .env
mkdir -p ../task
```

Open `.env` in a local editor and set `SCHOOL_API_KEY` to your authorized key. Obtain the complete task directory through the course channel and place it under the sibling `task/` directory. From the cloned repository, run:

```bash
bash run-task.sh ../task/Released_Task2-Headlines_CTR
```

Replace the task directory name for a different task. Keep each task's instructions, data files, and original subdirectory structure together.

The launcher creates an isolated Python 3.12 environment, installs the CPU classifier dependencies, adds the school gateway to the proxy bypass list, and enables performance search with the school planner. Set `SCHOOL_API_KEY` in the project `.env` using `.env.example` as the template. Keep the key local. Pass `--planner offline` to use local planning, or append other `main.py run` options. The launcher marks runs with failed selected experiments as `degraded` and exits with code 1 while preserving completed artifacts.

Rule-based B discovers established CSV, NPY, and TSV layouts. It also reads declared layouts for line-based text, class directories, image indexes, and flat records. Supported formats also include sharded splits. Add `--b-analysis llm` to use the school gateway for layout discovery, quality review, and declarative preprocessing:

```bash
bash run-task.sh ../task/Released_Task2-Headlines_CTR --b-analysis llm
```

A recognizable layout uses the automatically identified task ledger. An unrecognized layout needs a stable `--task-budget-id` before the gateway sends a request. Use `--task-doc` for a supported document that automatic discovery would miss. B and A share the configured request limit. [Task directory contract](docs/task-directory.md) · [B analysis](docs/b-llm-upgrades.md) · [Input readers](docs/b-readers-and-conversion.md)

## School planning and budget

The planner receives a compact training profile, validation feedback, and the compatible capability registry. It submits structured plans with registered model names and bounded parameters. The school API gateway reserves and settles each billed request, including B requests when enabled. Task ledgers persist in `sessions/task-budgets/` across runs. The automatic default is 20 school requests and a 20 RMB task budget at the configured reference prices. The performance launcher allows up to 80 local experiments. These controls can be adjusted with `--llm-max-calls`, `--llm-budget-rmb`, `--task-budget-rmb`, and `--max-trials`.

Performance search resumes school planning on the next round after a single settled empty reply. A second consecutive empty reply switches the remaining search to local rules. Unknown provider usage stops the session and retains the reservation. Summary and event records include the actual decision source. [School API](docs/school-api.md) · [Task budget](docs/task-budget.md)

## Models and data

Table tasks support numeric and categorical features with linear, tree, and boosting classifiers. CPU XGBoost and native categorical CatBoost are available with the performance dependencies. Text supports field-level word, character, and hybrid TF-IDF, hashing, linear and naive Bayes classifiers, and local Qwen embeddings. Image-array tasks support pixel, color, and gradient features. The runtime catalog filters choices by input representation and installed resources.

For an eligible text task, the launcher can prepare the public Qwen3-Embedding-0.6B model after the planner selects it. On first use this downloads CPU dependencies and about 1.19 GB of pinned weights. The checksum is verified before loading. Later runs with the same output root can reuse the weights and matching development embeddings. Reuse checks the data digest, partition, field order, encoder version, and runtime. `--pretrained cache-only` uses existing resources. `--pretrained disabled` disables encoder choices. [Multimodal tasks](docs/multimodal-tasks.md)

Performance mode offers a signed pairwise text pipeline when the task identifies a two-field binary preference problem. It compares shared word/character TF-IDF with an optional Qwen embedding difference. Size-gated calibrated RBF classifiers can compete on table data and bounded image features. Image tasks can also use the spatial pyramid. The local planner prioritizes promising validation recipes, and both planners use the same validated candidate catalogue.

## Direct Python entry points

Install the base requirements in a virtual environment. Add `requirements-quality.txt` for optional CPU classifiers and `requirements-dev.txt` for local verification. Run `main.py` with that interpreter.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py run --task ../task/in_vehicle_coupon_recommendation --planner offline
```

For PowerShell, use `py -3.12 -m venv .venv` and `.\.venv\Scripts\python.exe`. Legacy tasks can supply `--config configs/coupon.json` and an explicit `--test-config`. Standalone frozen evaluation uses `main.py evaluate-test --bundle <bundle.json> --test-config <test.json>`. Use `predict.py` for standalone frozen inference. [Final test](docs/final-test.md)

Bundles with frozen B reader rules also support standalone `predict.py --raw-input --source-root` on new data in the training format. [Frozen raw input](docs/b-readers-and-conversion.md#frozen-raw-input)

## Outputs

- `data/prepared/` holds immutable, reusable prepared data versions.
- `sessions/` holds session summaries, trial history, event logs, and API usage records.
- `runs/` holds model training artifacts and worker logs.
- `submission/` holds the frozen bundles. Bundle manifests include artifact checksums.

The evaluation module keeps validation labels outside planning and training. Test features and labels stay outside search and are read after the final bundle is frozen. Session reports distinguish actual school requests from mock or offline runs. Validation measurements use the supplied partitions and are development evidence. [Interfaces](docs/interfaces.md) · [Reading order](docs/reading-order.md) · [Validation](docs/VALIDATION.md)

The final evaluation scores every row of a labeled test partition. A supported feature-only test partition produces predictions with original sample IDs when provided, and no metric. If input filtering would remove official test rows, the test stage fails and the frozen bundle remains available. [Test coverage and outputs](docs/b-readers-and-conversion.md#test-coverage-and-outputs)
