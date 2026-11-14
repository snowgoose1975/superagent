# Formal task-wide budget ledger

`evaluation_d.task_budget.TaskBudgetLedger` is an opt-in, restart-safe accounting ledger for one explicitly named task budget. `main.py` wires it into `SchoolGateway` only when both `--billing-profile` and `--task-budget-id` are supplied; otherwise the existing per-session development ledger is used. The task budget ID is an accounting key only and does not affect A's model or search decisions.

The ledger lives at `<root>/<budget_id>.json` and uses a stable lock file plus an atomic JSON replacement. Reopening an existing ID preserves its history and rejects a changed budget or price profile. Concurrent callers serialize reserve/settle operations. A pending reservation remains charged after a process restart; `mark_unknown` also retains the full reservation, so an ambiguous request is never counted as zero or retried through the ledger.

Minimal synthetic use:

```python
from evaluation_d.task_budget import TaskBudgetLedger

ledger = TaskBudgetLedger(
    "sessions/task-budgets",
    "course-project-2026",
    budget_rmb="20",
    profile="deepseek-v4.1-flash-peak-2026-10-21",
)
reservation = ledger.reserve(
    "api-session-1:1",
    session_id="api-session-1",
    max_input_tokens=16_384,
    max_output_tokens=1_024,
)
# After receiving valid provider usage:
ledger.settle("api-session-1:1", input_tokens=1200, output_tokens=300)
# If transport or usage becomes unknown instead:
# ledger.mark_unknown("api-session-1:1", reason_code="TIMEOUT")
print(ledger.snapshot())
```

The selected reference profile is intentionally conservative and explicit. DeepSeek's published peak/cache-miss standard rates are USD 0.30 per million input tokens and USD 1.20 per million output tokens. Multiplying by the 2026-10-21 RMB central parity rate of 6.7192 gives CNY 2.015760 and CNY 8.063040 per million. The profile uses `Decimal`, preserves the source URLs and effective date, and does not apply cached-token, off-peak, free-quota, or platform discounts. Sources: [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/), [DeepSeek updates](https://api-docs.deepseek.com/updates/), [CFETS central parity notice for 2026-10-21](https://www.shibor.sh.cn/chinese/ccprnoticecontent/index.html?searchDate=2026-10-21).

This is a conservative conversion of a published DeepSeek reference price. It does not prove that the school's endpoint uses that provider, that a displayed model alias maps to the cited model, or that the school's actual invoice uses these rates. The existing school API JSON remains a development configuration; this module does not change it or load credentials.

## Deterministic session report

From the project root, provide a summary and a new, unused destination:

```bash
../.venv/bin/python scripts/report_session.py \
  --summary sessions/<session-id>/summary.json \
  --output reports/<unique-session-id>
```

The script reads history, optional fixed-name C diagnostics, D scores, and the gateway usage summary. It writes `leaderboard.csv`, `best_so_far.json`, and a deterministic `report.md`; it never calls an LLM or reads raw gateway request/response content. Missing diagnostics and history stay visibly missing. Session-level gateway costs are labeled as estimates under that gateway's configured development prices. To include a task-wide formal ledger snapshot, pass `--task-ledger path/to/<budget-id>.json`; the report keeps settled charges and pending/unknown reservations separate.

The ledger remains opt-in. To activate it for a school-planner run, explicitly provide both the profile and a stable budget ID. The search configuration is also explicit; it enables the expanded, bounded search policy without changing the task's `max_trials`:

```bash
no_proxy=aigc-api.hkust-gz.edu.cn \
../.venv/bin/python main.py run \
  --config configs/coupon.json \
  --planner school \
  --search-config configs/search-defaults.json \
  --llm-max-calls 20 \
  --llm-budget-rmb 20 \
  --billing-profile deepseek-v4.1-flash-peak-2026-10-21 \
  --task-budget-id course-project-final \
  --task-budget-rmb 20
```

The gateway reserves the worst-case input/output amount in this task ledger before transport, settles actual usage when reported, and retains an unknown reservation if usage is ambiguous. The task ledger is separate from the existing per-session development ledger and does not change the configured endpoint/model alias. The profile is a conservative published-price reference; it does not confirm the school's actual provider or invoice rate. `20` is a cap, not a spend target: A may stop when additional experiments are unlikely to help, and unknown usage blocks further calls.
