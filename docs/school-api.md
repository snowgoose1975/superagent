# 学校 API 本地调用说明

学校模式向学校 API 发送模型规划请求，从项目根 `.env` 加载 key。默认模式是 offline。Codex 自动验证使用模拟响应；真实学校调用由你本地运行。

## 安装和运行

在 VS Code 打开 `artifacts/superagent-team-v2/` 文件夹，再在项目目录打开 PowerShell 终端：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

默认离线运行：

```powershell
.\.venv\Scripts\python.exe main.py run --config configs/coupon.json
```

学校 API 模式是显式开启的：

```powershell
.\.venv\Scripts\python.exe main.py run --config configs/coupon.json --planner school --llm-config configs/school-api.json --llm-max-calls 1 --llm-budget-rmb 0.10
```

不需要激活虚拟环境，不要更改 PowerShell 执行策略。`.venv` 只保存项目依赖。首次使用时，在项目根目录复制模板并填写 key：

```powershell
Copy-Item .env.example .env
```

然后编辑项目根 `.env`，仅设置这一项：

```dotenv
SCHOOL_API_KEY=实际学校API密钥
```

不要把 key 放入命令行、聊天、`configs/school-api.json`、源码或其他配置字段。`.env` 应由 `.gitignore` 排除，也不会放入交付包；只提交无密钥的 `.env.example` 模板。运行 CLI 时，程序从项目根 `.env` 加载 `SCHOOL_API_KEY`。

`configs/school-api.json` 存放 URL、模型名、输入/输出单价和请求上限，不含 key。它复用项目已有的学校 DeepSeek 开发参数。本例命令的预算为 RMB 0.10；配置限制输入消息 UTF-8 字节数加模板预留不超过 16,384，最大输出 1,024 tokens。输入字节限制是开发阶段的保守检查，实际 token 以供应商 usage 为准。费用按配置单价估算；这不是已核验的课程正式价格。

## 请求会做什么

School A 使用同学包的 `LLMPlanner` 提出模型与参数。提示只含 B 的训练概况、C 的能力和已有实验/validation metric 历史；不含验证标签、test、任意 Python 代码或执行指令。A 的输出还要通过固定模型/参数校验，main 补齐实际 experiment ID、seed、timeout，再交给 C 的固定训练 worker。

在 VS Code 调试时，可先在 `main.py` 中 A 返回计划后、`execute(plan, ...)` 之前设置断点。终端也会打印 `A→C` 参数，让你先看清学校模型建议的配置，再观察 C 实际收到的计划。模型只选择受限配置，不会生成或运行代码。

## 默认一次调用和预算行为

命令中的 `--llm-max-calls 1` 限制本次主会话最多一次请求。若供应商返回有效 usage，且计划内容通过校验，主循环可训练并评分这一轮；之后 A 会因调用上限停止，不会再发送第二次请求。`--llm-budget-rmb 0.10` 是本次独立开发会话的预算上限，网关会在请求前为配置的最大输入/输出预留费用。

异常时的行为有意保持严格：

- HTTP 错误、连接/请求超时、响应无法解析或 usage 缺失：记录为 `unknown`，保留预算预留并立即停止本次会话；不重试。
- usage 已知，但模型回复文本不能解析、缺少模型/参数、提出不支持参数或重复计划：记录已知 token 和估算费用后，允许规则 planner 为当前轮提供备用计划。这个备用来源记录为 `rules_fallback`；请求仍计入 `real_llm_calls` 和调用上限。供应商响应缺少文本或 token 超过预留限制时，会登记已知用量并停止。
- 有效学校计划：决策来源记录为 `llm`。后续遇到一请求上限时会停止，并在 summary 中显示规划模式、最后决策来源与真实调用数。

默认网关账本覆盖本次独立开发会话；未知用量不会视为零，也不会冒充正式课程任务总账。若需跨 session 记账，必须显式传入 `--billing-profile` 与稳定的 `--task-budget-id`，具体用法见 [`task-budget.md`](task-budget.md)。参考价格 profile 不会覆盖 `school-api.json` 的 endpoint/model，也不证明学校端实际扣费。

## 调用账本与隐私

每次运行都会在 `sessions/api-<timestamp>-<uuid>/` 独立创建账本，包括 `metadata.json` 与 `calls.jsonl`。文件记录请求状态、真实调用次数、供应商 token usage、估算费用、预算预留和运行配置中的模型/URL等必要元数据。账本不保存 API key、Authorization 头、完整消息、提示词、原始请求体或响应正文。失败与 unknown 记录会保留，不覆盖或自动清理。

## 真实调用与自动验证的区别

Codex 自动验证使用假 HTTP transport/mock usage，不会读取或取得你的 key，也不向学校服务发送请求。本轮 Codex 没有发送真实学校 API 请求；真实响应由你在本地配置 `.env` 并运行上述命令后验证。遇到真实调用异常时，先查看 summary 的 `llm_gateway` 状态与对应账本；不要通过删除记录或重置预算来继续。

无论 offline 还是 school，模型选择和评分只使用 train/validation；test 不进入默认任务配置和闭环。只有你之后明确要求独立最终用例时，才会在选模冻结后另行使用 test。
