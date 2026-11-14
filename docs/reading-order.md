# 阅读顺序

顺着一次真实执行的模块边界阅读。建议先用默认离线模式，在 VS Code 中逐步设置断点；看清楚一份计划怎样进入 C，再继续追评分如何返回 A。

1. **`main.py`：看组装流程。** 从 `run_project()` 进入，先找到 B 的 `prepare_data(...)`、A 的控制器、计划校验与 `execute(...)`。重点看 main 如何在把计划交给 C 之前补齐 experiment ID、seed 和 timeout。这里是最快看到“模型计划传给 C”的位置。之后再看 D 的 `evaluate(...)` 和 history 如何交给下一次 A 决策。
2. **`data_b/prepare.py`：看 B 先准备了什么。** B 读取声明的 train/validation NPY，只用 train 拟合转换，输出 C 的数据清单、验证 ID，并另存供 D 使用的验证标签。确认 C 的 manifest 不包含验证标签路径。
3. **`agent_a/controller.py` 与 `agent_a/upstream/`：看 A 怎样决策。** `PackageAController` 把 main 的训练概况、能力清单、历史和剩余实验数转换成上游 `decide()` 所需上下文。离线模式使用同学 A 的规则 planner；school 模式可将同学包 `LLMPlanner` 接到相同决定接口。A 的输出只包含 action、model、params 和 reason，不生成或执行 Python 代码。
4. **`executor_c/executor.py`：看 C 的父进程。** C 校验计划和数据清单、创建唯一实验目录、启动固定 worker、管理日志/时间限制，并核验 success 或失败状态。
5. **`executor_c/train.py`：看模型训练。** 只支持固定白名单中的随机森林和逻辑回归。worker 使用训练数据拟合，读取验证特征生成预测和概率；不读验证标签，也不计算分数。
6. **`evaluation_d/evaluate.py`：看 D 独立评分。** D 消费 C 的 success 结果，从 B 提供的独立标签文件按 `sample_id` 对齐后计算 accuracy。这个结果写入会话 history，再作为下一轮 A 的反馈。
7. **`evaluation_d/school_gateway.py`：最后看学校服务调用边界。** school 模式的 HTTP 请求和调用账本由 D 网关管理，不能把 API key 或对话正文传入 A 的日志。按用户最新选择，key 从项目根 `.env` 的 `SCHOOL_API_KEY` 加载；自动化验证只用 mock。

这条链可以概括为：`main → B → A → C → D → A`。A 无论来自离线规则还是学校模型，都只提出配置；C 仍使用相同任务清单和训练入口，D 仍使用相同的独立评分实现。

## 初学者建议断点

- 在 `main.py` 找到调用 `planner.decide(...)` 后的 `decision`，看 A 返回的模型和参数。
- 在 `main.py` 的 `validate_plan({...})` 后检查补齐了 task/version/experiment、seed 和 timeout 的完整计划。
- 在同一轮的 `execute(plan, ...)` 设置断点，确认 C 收到的就是这份计划。
- 在 `evaluation_d/evaluate.py` 看预测与标签按 ID 对齐的位置，再回到 main 看分数怎样进入下一次 history。

建议先读离线模式。学校模式涉及真实外部请求；先查看 [`school-api.md`](school-api.md) 的运行和停机行为，再决定是否在本地调用。
