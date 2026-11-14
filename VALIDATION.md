# v2 验证记录

这是 2026-11-12 的历史记录。2026-11-14 的 H0–H9 改动及当前验证结果以 [`docs/VALIDATION.md`](docs/VALIDATION.md) 为准；下述旧版 pytest/Ruff 通过记录不代表新改动已运行完整 suite。

2026-11-12。环境为项目既有 `.venv-demo`：Python 3.12.13、numpy 2.5.4、scikit-learn 1.9.1、pandas 2.3.3、joblib 1.6.0；具体版本另见 `tested-environment.json`。没有安装全局依赖或修改系统设置。

## 已验证

- **98 项测试分批通过**：B/C、主循环、旧控制器对照、同学 A 适配、`.env` 加载、学校网关及模拟 API 完整训练流程。
- 首轮完整执行 97 项时，85 项通过、12 项失败。已修正 Windows 下 API 会话及临时文件名过长导致的写入错误，并修正一项测试误把合法方案当成非法方案的样例；21 项 API 相关测试复验通过。随后增加中断请求不能继续消费的用例，最终 22 项 API 相关测试全部通过；与首轮通过项重叠 9 项，合计 98 个不同用例。
- `.env` 解析用合成 key 测试：支持引号/BOM/注释，拒绝占位值、空值、重复和错误格式；不执行命令、不做变量插值、不搜索父目录、不改变系统环境。没有读取或写入真实项目 `.env`。
- 学校 API 由假 HTTP transport 验证完整请求格式、Bearer 注入、JSON 解析、已知 usage 计费、调用/预算限制、未知 usage 停止、失败不重试、禁止重定向，以及凭据和原始消息不进入账本。模拟调用单独记为 mock，真实请求次数为 0。
- 模拟 API 成功方案经过同学 A 的 `LLMPlanner`，进入现有 C 的真实 scikit-learn 训练、D 独立评分并冻结模型。已知 usage 但方案无效时明确标记 `rules_fallback`；连接失败/usage 未知直接停止。
- `ruff check` 与 `ruff format --check` 通过。B/C 原训练协议保持不变。
- 老师任务以同学 A 的规则规划真实跑完三轮 train/validation；记录见 `example-run.json`、`docs/example-run.md` 及对应 `sessions/`、`runs/`、`submission/`。最优 validation accuracy 为 0.7408866995073892，是第一轮随机森林；这是开发分数。
- 新进程重新加载冻结工件，对 2030 行 validation 预测：样本 ID、预测类别完全一致，概率最大绝对误差为 0。未合并 validation 重训。
- 交付检查按 v1 的 `SHA256SUMS.txt` 逐个核对旧文件，保留 v1 原样；v2 ZIP 包含 `.env.example`，排除真实 `.env`、缓存和虚拟环境，附自身 SHA256 清单。

## 复现测试

在项目目录安装 `requirements-dev.txt` 后，可使用新的临时目录运行，避免复用会被 pytest 清理的旧 basetemp：

```powershell
$testRoot = Join-Path (Get-Location) ('.cache/test-temp/' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest --basetemp $testRoot
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

本次原始验证及失败记录保存在工作区 `.cache/test-temp/team-v2-20261112-a04b6c/`，不清理；最终 API 测试摘要另附 `docs/pytest-api.log`，机器可读证据见 `docs/verification-results.json`。

## 范围

Codex 没有发出真实学校模型请求。真实 key 由用户自行放入 `.env` 后运行 school 模式，不能把模拟通过理解为真实 API 已验证。按用户最新要求不再排查 TLS。

没有读取或复制老师 test 数组，没有正式课程 test 成绩或正式价格/总预算合规声明。API 是独立开发会话账本，未清空或解除其他历史任务的未知费用预留。
