# 模块接口

新增显式 B LLM 接口与 TaskSpec 1.0 可选 `preprocessing` 见 [B 三项升级](b-llm-upgrades.md)。它复用 D 网关，不改变 C/D 数据清单。`run_project(..., b_analysis='llm')` 仅用于 school + task 目录，需解析前可用的持久预算账本；默认规则路径不变。

main 负责组装 A/B/C/D。A 的实现可以替换，B 的准备数据协议、C 的执行协议和 D 的独立评分流程保持稳定。学校模式只替换 A 的决策来源，并通过 D 的网关发请求与记账。

## A 的控制器

main 调用控制器 `decide(context: dict) -> dict`。上下文包含训练概况、任务指标、训练多数类 validation baseline、C 能力清单、先前 history 和 `remaining_trials`；自动任务还包含 README/info 的有界 `task_context` 摘要，其属于不可信任务数据。A 不接收验证标签或 test 数据。每条 history 项记录 experiment ID、decision/plan、C 结果路径和状态；成功评分项还可包含相对 baseline 的方向化差值。

A 返回：

```python
{"action": "run", "model": "random_forest", "params": {"n_estimators": 100}, "reason": "..."}
```

对 Plan 2.0 的文本/图像任务，A 还可返回注册表内的 `pipeline`；文本 pipeline 会随模型和参数一起去重、保存并冻结。表示选择仍是有限白名单，不生成或执行代码。

或者明确停止：

```python
{"action": "stop", "reason": "..."}
```

`PackageAController` 适配同学包的 `planner` 与 `decide`，把 main 的历史字段映射到上游规划器字段。它只提取模型和参数，不把上游包的原运行入口、数据读取、评分器或 budget 接进来。实验 ID、seed 和 timeout 由 main 按任务政策补齐，C 再进行白名单参数校验。

## 两种规划模式

- `offline` 是默认模式，使用同学 A 的规则 planner；不联网、不需要 key，summary 的真实 LLM 调用数为 0。
- `school` 复用同学 A 的 `LLMPlanner` 生成模型、参数和可选的声明式 pipeline。它只接收训练概况、任务说明摘要、validation baseline、C 能力和历史 validation 分数，不读取验证标签或 test，不生成或执行任意代码。请求经 `evaluation_d/school_gateway.py` 发送并记账，A 不直接访问 HTTP。

- 可选 `--test-config` 只在 validation 最优 bundle 冻结后进入 `evaluation_d/final_test.py`。它调用冻结预测器并按任务 primary metric 评分；test 结果只写最终 session 报告和终端，不回流 A、history 或模型选择。已冻结 bundle 可通过 `evaluate-test` 子命令独立补评。

school 模式下 `--llm-max-calls` 限制本次主会话的请求尝试数量。兼容 `--config` 默认 1 次，自动 `--task` 默认 20 次。请求超时、HTTP 错误或 usage 缺失都记为未知并停止，不回退、不重试。若 usage 已知但回复不符合 A/C 方案协议，系统才可记录原因并由规则 planner 产生备用方案。summary 记录规划模式、`last_decision_source`、每轮 `decision_sources` 和 `real_llm_calls`，不会把 mock 记成真实调用。

自动 `--task` 默认启用增强搜索；兼容 `--config` 需显式加 `--search-config configs/search-defaults.json`。`confirmation_seed_offsets` 默认是空列表；只有配置了正偏移、且任务轮数与 session 时间仍有余量时，主程序才按 validation 分数稳定排序最多两个候选执行 C-only seed 复核。确认轮不发新的 LLM 请求，实际 seed 写入 Plan/history。网关 usage unknown 时不做确认。API 在已有成功评分后失败时，summary 保持 `failed` 并保存错误；若能安全冻结先前最佳成功实验，会提供 `partial_bundle_path`，不会继续训练或调用 API。

## B/C/D 保持的接口

B 的可选样本索引、混合特征处理和无 test 开发模式见 [B 优化交接](b-improvements.md)。`discover_task_directory()` 返回的 `test_descriptor_path` 可能为 `None`，调用方仅在该路径存在时进入最终 test。

- **B**：`prepare_data(task_spec, data_root) -> dict`。B 读任务配置声明的 train/validation，拟合预处理器仅使用 train；向 C 提供 data manifest，向 D 单独提供 validation labels 和 evaluation manifest。
- **C**：`execute(plan, data_manifest_path, runs_root) -> result_relpath`。接收 main 校验后的固定计划，返回相对 runs root 的结果 JSON 路径。Plan/Manifest 1.0 保持表格协议；2.0 携带文本或图像表示与注册 pipeline。C 只读 train 标签、验证特征和验证 ID。
- **D 评分**：`evaluate(result_path, evaluation_manifest_path, output_path=None) -> dict`。只读 C 的 success 和 B 的验证标签，按 `sample_id` 对齐并计算任务声明的 accuracy、macro-F1 或 log loss；不回写 C 结果。
- **D 学校网关**：`SchoolGateway.call_llm(messages) -> {text, input_tokens, output_tokens, cost_rmb}`，独立负责学校 HTTP 请求、限额、预算预留和账本。key 从 `evaluation_d/env_config.py` 加载后仅注入网关，不进入 A 的提示词；网关在内存中处理消息，账本只写元数据。

TaskSpec 2.0 文本/图像文件格式、资源上限、模型兼容表和预测行为见 [`multimodal-tasks.md`](multimodal-tasks.md)。新增的全流程只使用固定 TF-IDF 或图像展平缩放组件；C 不执行 LLM 生成代码。

增强文本搜索会在有足够实验轮数时先覆盖 LogisticRegression、MultinomialNB、ComplementNB 与共享/逐字段的 word/character TF-IDF 组合，再进入参数细化。validation baseline 只用训练类别频率建立多数类预测或先验概率基线；它帮助发现明显弱于简单参考的方案，不作为自动拒绝规则。

## CLI

全自动任务目录入口（项目目录执行）：

```bash
no_proxy=aigc-api.hkust-gz.edu.cn \
../.venv/bin/python main.py run --task ../task/in_vehicle_coupon_recommendation --planner school
```

只需给路径；`data_b.task_directory.discover_task_directory()` 自动生成 TaskSpec、说明摘要和惰性 test 描述。只有冻结完成后，D 才调用 `materialize_final_test()`。不需要用户编写数据集配置。输入范围见 [`task-directory.md`](task-directory.md)。

默认离线：

```powershell
python main.py run --config configs/coupon.json
```

明确调用学校 API：

```powershell
python main.py run --config configs/coupon.json --planner school --llm-config configs/school-api.json --llm-max-calls 1 --llm-budget-rmb 0.10
```

school key 从项目根 `.env` 的 `SCHOOL_API_KEY` 读取。复制 `.env.example` 为 `.env` 并在其中设置 key；URL、model、价格与请求限制在 `configs/school-api.json`。费用按开发配置估算，不等于已确认的课程正式价格。详细使用与停机语义见 [`school-api.md`](school-api.md)。

自动 `--task` 学校模式启用每任务 20 元参考价账本，训练输入摘要相同时复用；已匹配到旧配置运行的账本也会沿用。兼容 `--config` 的跨会话账本仍需明确提供 `--billing-profile deepseek-v4.1-flash-peak-2026-10-21 --task-budget-id <稳定编号> --task-budget-rmb 20`。该账本使用保守公开参考价；不会更改 `school-api.json` 中的 API endpoint/model，也不代表学校 API 端点的实际扣费。账单记录与确定性报告见 [`task-budget.md`](task-budget.md)。

完整验证记录、当前测试工具可用性和实际 smoke 范围见 [`VALIDATION.md`](VALIDATION.md)。

## 替换 A/D

离线或学校 A 都适配 main 的同一控制器接口；保留 A 返回的 action/model/params/reason 结构即可替换实现，不需改 B/C。若更换 D 评分器，仍须接受同一 result/evaluation manifest 路径，并返回身份相符的 accuracy、样本数和 greater-is-better 字段。不要把 API 网关职责放进 B、C 或评分函数。
