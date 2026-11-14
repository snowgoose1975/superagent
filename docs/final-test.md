# 冻结后的最终 test 评估

主搜索仍只使用 train/validation。A 只接收 validation 评分，test 文件不会进入 TaskSpec、历史记录或选模过程。统一自动入口 `--task <目录>` 会自动生成惰性 test 描述，无需人工 `--test-config`。main 先按 validation 选择模型并冻结 bundle，再用冻结的预处理和模型预测 test、计算最终指标。

自动发现阶段只枚举 test 文件名，CSV 的 test 转换、数组拼接、行数及内容摘要也都推迟到冻结之后。读取规则已在发现 train 时固定，test 不会再次决定特征类型或选择标签列。特征与标签按原始行序一一对应，标签按冻结 `label_mapping.json` 映射到模型类别 ID。

带学校 API 的完整命令可在 README 中找到。离线模式也可以同样添加：

```bash
../.venv/bin/python main.py run \
  --task ../task/in_vehicle_coupon_recommendation
```

最终 bundle 冻结后，终端会先打印选中的 experiment 和 validation 指标，然后打印 test 指标、样本数及报告路径。每个 run session 写入：

- `test_predictions.csv`：冻结 bundle 对 test 的类别预测和每类概率。
- `test_evaluation.json`：primary metric、accuracy、样本数、bundle/test 输入摘要和评估状态。

若 test 评估失败，session 状态为 `failed`，错误阶段为 `final_test`；已冻结的 final bundle 和 validation 结果保留，不会重复训练或另存 partial bundle。旧 `--config` 入口没有传 `--test-config` 时仍维持开发流程，summary 标记 `test_evaluation.status` 为 `not_requested`。

若 run 已经冻结 bundle 但还没有 test 报告，可单独补评：

```bash
../.venv/bin/python main.py evaluate-test \
  --bundle submission/in_vehicle_coupon_recommendation/final_<id>/bundle.json \
  --test-config sessions/<session_id>/task_input/test_descriptor.json
```

该命令只接受 `final_*` bundle，不调用 planner、不训练、不改 bundle。它在 `sessions/test_evaluation_<id>/` 建立新的报告目录，避免覆盖已有结果。课程 test 分数是最终评估结果，不得再拿来选择模型或调参数。
