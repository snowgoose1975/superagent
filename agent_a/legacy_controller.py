"""明确标记为离线规则的实验规划器，可由正式 A 按接口替换。"""

from __future__ import annotations

from typing import Any


class OfflineController:
    """用固定规则生成最多三轮计划，不调用 LLM，也不伪造调用记录。"""

    mode = "offline_rules"

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        """根据轮次和 D 的真实评分返回 run/stop 决策。"""
        remaining = context.get("remaining_trials", 0)
        if type(remaining) is not int or remaining <= 0:
            return {"action": "stop", "reason": "剩余尝试次数为 0，停止实验。"}

        history = context.get("history", [])
        if not isinstance(history, list):
            return {"action": "stop", "reason": "历史记录格式无效，无法安全规划下一轮。"}
        if len(history) >= 3:
            return {"action": "stop", "reason": "离线规则最多执行三轮，候选已用完。"}

        # 前两轮固定候选；失败也占用一轮，因此可按历史长度推进并避免自动重试。
        candidates = [
            ("logistic_regression", {"C": 1.0, "max_iter": 1000}),
            ("random_forest", {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2}),
        ]
        if len(history) == 0:
            model, params = candidates[0]
            reason = "第 1 轮固定基线：逻辑回归 C=1.0。"
        elif len(history) == 1:
            model, params = candidates[1]
            reason = "第 2 轮固定候选：随机森林 100 棵树、深度 12、叶节点至少 2。"
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
                    f"第 3 轮根据真实 accuracy 反馈：随机森林领先（{rf_best:.6g}），扩大森林容量。"
                )
            else:
                model = "logistic_regression"
                params = {"C": 0.1, "max_iter": 1000}
                if lr_best == float("-inf") and rf_best == float("-inf"):
                    reason = "第 3 轮此前没有可比较的成功评分，尝试较强正则化的逻辑回归。"
                else:
                    reason = f"第 3 轮根据真实 accuracy 反馈：随机森林未领先（RF={rf_best:.6g}, LR={lr_best:.6g}），加强逻辑回归正则化。"
        return {"action": "run", "model": model, "params": params, "reason": reason}
