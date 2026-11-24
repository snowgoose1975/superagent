# School API operation

Set `SCHOOL_API_KEY` in the project `.env` using [the template](../.env.example). The launcher starts school planning and appends the gateway host to `no_proxy` and `NO_PROXY`. Keep the real key on the authorized machine.

```bash
bash run-task.sh ../task/in_vehicle_coupon_recommendation
```

For a direct development run:

```bash
no_proxy=aigc-api.hkust-gz.edu.cn \
.venv/bin/python main.py run --config configs/coupon.json --planner school \
  --llm-config configs/school-api.json --llm-max-calls 1 --llm-budget-rmb 0.10
```

The gateway validates the HTTPS host and endpoint, attaches the key for the request, and accounts for calls in a per-session ledger and the persistent task ledger, when configured. Authoring or reviewing a task never requires sending the test partition. Optional B analysis shares the same gateway and budget with A. [B analysis](b-llm-upgrades.md) · [Budget](task-budget.md)

## Response handling

- A valid bounded plan can produce a validation experiment. Rejected structured plans may use a local fallback for the affected round.
- Performance mode can resume school planning after one settled empty reply. If the fallback sequence continues, the controller may switch to local search as recorded in the session summary.
- Unknown provider usage stops the session and keeps its reservation charged. The gateway checks call limits, remaining budget, provider token bounds, and successful ledger settlement.
- A rules decision, mock response, and actual school request have distinct sources in the summary and history.

The nonsecret endpoint, model alias, reference price and bounds live in [school-api.json](../configs/school-api.json). Session metadata and `calls.jsonl` record request attempts, token counts, and estimated cost while keeping raw prompts, response bodies, and credentials out of the ledger. Reports use project reference prices. The service alias and the provider's actual invoice require separate verification by the institution.

Set `max_input_bytes` for UTF-8 payload preflight and `max_input_tokens` for the reservation and reported input token ceiling. A configuration with only the legacy token setting uses it as its default byte limit as well.
