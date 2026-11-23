"""Legacy rule planner implementing the A interface."""

from __future__ import annotations

from typing import Any


class OfflineController:
    """Plan at most three trials with offline rules."""

    mode = "offline_rules"

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        """Choose whether to run or stop using prior D scores."""
        remaining = context.get("remaining_trials", 0)
        if type(remaining) is not int or remaining <= 0:
            return {"action": "stop", "reason": "The trial limit has been reached."}

        history = context.get("history", [])
        if not isinstance(history, list):
            return {"action": "stop", "reason": "The trial history has an invalid format."}
        if len(history) >= 3:
            return {"action": "stop", "reason": "The three legacy candidates have been exhausted."}

        # Each attempt advances the history, including failed trials.
        candidates = [
            ("logistic_regression", {"C": 1.0, "max_iter": 1000}),
            ("random_forest", {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2}),
        ]
        if len(history) == 0:
            model, params = candidates[0]
            reason = "Trial 1 baseline: logistic regression with C=1.0."
        elif len(history) == 1:
            model, params = candidates[1]
            reason = "Trial 2: random forest with 100 trees, depth 12, and at least two samples per leaf."
        else:
            scores: dict[str, list[float]] = {"random_forest": [], "logistic_regression": []}
            for item in history:
                if not isinstance(item, dict) or item.get("status") != "success":
                    continue
                evaluation = item.get("evaluation")
                plan = item.get("plan")
                if not isinstance(evaluation, dict) or not isinstance(plan, dict):
                    continue
                value = evaluation.get("value")
                model_name = plan.get("model")
                if (
                    model_name in scores
                    and type(value) in (int, float)
                    and 0.0 <= float(value) <= 1.0
                ):
                    scores[model_name].append(float(value))
            rf_best = max(scores["random_forest"], default=float("-inf"))
            lr_best = max(scores["logistic_regression"], default=float("-inf"))
            if rf_best > lr_best:
                model = "random_forest"
                params = {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 1}
                reason = (
                    f"Trial 3: improve the leading random forest ({rf_best:.6g}) by adding trees."
                )
            else:
                model = "logistic_regression"
                params = {"C": 0.1, "max_iter": 1000}
                if lr_best == float("-inf") and rf_best == float("-inf"):
                    reason = "Trial 3: use stronger logistic regularization because no comparable score is available."
                else:
                    reason = f"Trial 3: use stronger logistic regularization (RF={rf_best:.6g}, LR={lr_best:.6g})."
        return {"action": "run", "model": model, "params": params, "reason": reason}
