"""decide(): the single decision point of the agent loop.

Input : task facts, experiment history, remaining budget.
Output: {"action": "run", "plan": {...}, "reason": ...}
     or {"action": "stop", "reason": ...}.

All stop conditions live here so the loop itself stays a dumb executor:

1. budget_exhausted    - remaining budget is zero (course limit RMB 20/task).
2. max_experiments     - the configured experiment cap is reached.
3. no_improvement      - the last `patience` runs did not beat the best
                         validation score seen so far.
4. search_space        - the planner has no untried configuration left.
"""

from __future__ import annotations

from typing import Any


def _beats(value: float | None, best: float | None, greater: bool) -> bool:
    if value is None or best is None:
        return value is not None
    return value > best if greater else value < best


def decide(context: dict[str, Any]) -> dict[str, Any]:
    history = context["history"]
    greater = bool(context.get("greater_is_better", True))
    remaining_raw = context["remaining_budget_rmb"]
    if remaining_raw is None and context.get("planning_mode") == "offline_rules":
        remaining = None
    else:
        remaining = float(remaining_raw)
    max_experiments = int(context["max_experiments"])
    patience = int(context.get("patience", 3))
    search = context.get("search") or {}
    expanded = context.get("search_profile") == "expanded"
    pending = search.get("pending_model_ids", [])
    pending_recipes = search.get("pending_text_recipes", [])
    coverage_size = len(pending_recipes) if pending_recipes else len(pending)
    coverage_active = (
        expanded
        and search.get("phase") == "coverage"
        and isinstance(pending, list)
        and isinstance(pending_recipes, list)
        and len(history) + coverage_size <= max_experiments
    )

    if remaining is not None and remaining <= 0.0:
        return {
            "action": "stop",
            "reason": f"budget_exhausted (remaining {remaining:.4f} RMB)",
        }
    if len(history) >= max_experiments:
        return {
            "action": "stop",
            "reason": f"max_experiments_reached ({len(history)}/{max_experiments})",
        }

    consecutive_failures = 0
    for entry in reversed(history):
        if entry.get("status") == "success":
            break
        consecutive_failures += 1
    failure_limit = search.get("max_consecutive_failures")
    if expanded and type(failure_limit) is int and consecutive_failures >= failure_limit:
        return {"action": "stop", "reason": "consecutive_failure_limit_reached"}

    min_trials = search.get("min_exploration_trials", 8) if expanded else 0
    best: float | None = None
    streak = 0
    if expanded:
        delta = float(search.get("min_delta", 0.0))
        initial_values = [
            entry["metric_value"]
            for entry in history[:min_trials]
            if entry.get("status") == "success"
            and type(entry.get("metric_value")) in (int, float)
        ]
        if initial_values:
            best = max(initial_values) if greater else min(initial_values)
        for entry in history[min_trials:]:
            if entry.get("status") != "success":
                continue
            value = entry.get("metric_value")
            improves = (
                value is not None
                and (best is None or (value > best + delta if greater else value < best - delta))
            )
            if improves:
                best = value
                streak = 0
            elif value is not None:
                streak += 1
        if coverage_active or len(history) < min_trials:
            streak = 0
        active_patience = int(search.get("patience", patience))
    else:
        for entry in history:
            value = entry.get("metric_value")
            if _beats(value, best, greater):
                best = value
                streak = 0
            else:
                streak += 1
        active_patience = patience
    if best is not None and streak >= active_patience:
        return {
            "action": "stop",
            "reason": (
                f"no_improvement for {streak} consecutive experiments "
                f"(patience={active_patience}, best val={best})"
            ),
        }

    planned = context["planner"].next_plan(context)
    if planned is None:
        return {"action": "stop", "reason": "search_space_exhausted"}
    if "stop" in planned:
        return {"action": "stop", "reason": str(planned["stop"])}
    return {"action": "run", "plan": planned["plan"], "reason": planned["reason"]}
