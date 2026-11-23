# SuperAgent

SuperAgent prepares classification data, searches registered models, and freezes the validation winner before evaluating the test split. It supports tabular, text, and image-array tasks.

## Run in Linux or WSL

```bash
git clone https://github.com/snowgoose1975/superagent.git
cd superagent
test -e .env || cp .env.example .env
mkdir -p ../task
```

Set your authorized school key as `SCHOOL_API_KEY` in the local `.env`. Put the task directory and its instructions in the sibling `task/` folder. Run from the cloned repository:

```bash
bash run-task.sh "../task/your-task" --b-analysis llm
```

Replace `your-task` with the actual directory name. The launcher prepares Python dependencies and configures the school-domain proxy bypass. It uses school planning for experiments. The optional B analysis shares the same task ledger. For an offline run, omit `--b-analysis llm` and pass `--planner offline`. Unrecognized layouts need a stable `--task-budget-id` before billable discovery.

Results are written to `sessions/`, `runs/`, and `submission/`. Keep task data, credentials, and run artifacts local.
