# v2 老师任务真实开发示例

本例接入同学 A 的规则 planner；B/C/D 真实预处理、训练、预测及计算验证 accuracy。真实 LLM 调用数为 0。未读取或复制老师 test。

## 从这些文件追踪

- `example-run.json`：本例路径索引与三轮参数。
- `sessions/session_20261112T145948Z_d83ebc17/summary.json`：最终选优与冻结结果。
- `sessions/session_20261112T145948Z_d83ebc17/history.jsonl`：逐轮决策、C 执行状态、D 实际评分。
- `sessions/session_20261112T145948Z_d83ebc17/decision_001.json` 至 `decision_003.json`：A→C 的规范化计划。
- `submission/in_vehicle_coupon_recommendation/final_d83ebc17/bundle.json`：冻结模型说明。

## 三轮结果

| 轮次 | 模型 | 参数 | validation accuracy |
| --- | --- | --- | --- |
| 1 | `random_forest` | `{"n_estimators": 100, "max_depth": null, "min_samples_leaf": 1}` | 0.740886699507 |
| 2 | `logistic_regression` | `{"C": 1.0, "max_iter": 1000}` | 0.663054187192 |
| 3 | `random_forest` | `{"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1}` | 0.716748768473 |

第三轮由 A 消费前两轮的真实反馈后选择。第三轮分数下降，所以主程序选第一轮，而不是默认冻结最后一轮。训练行数为 8117，验证行数为 2030；21 个原始类别列只在 train 拟合，生成 111 个编码特征。各轮 seed 固定为 42，最多 120 秒，不合并 train/validation 重训。

冻结模型已在新进程中重新预测这 2030 行 validation：样本 ID、预测类别完全相同，概率最大绝对误差为 0.0。这是工件可重载验证，不是新的训练或 test 成绩。

学校模式从 `.env` 加载 key 后使用同一条 B→A→C→D 流程；本例没有冒充真实 API 运行。
