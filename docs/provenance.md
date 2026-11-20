# 来源与整合说明

v2 是独立整合副本。v1 与前轮交付保留原样；本目录不复制历史 runs、旧会话账本或冻结模型。

## 来源

- **B/C/D 评分**：从 `artifacts/superagent-team-v1/` 复制到 v2，并按 v2 的模块接线需要调整。当前 B 支持表格、原始文本和图像张量的声明式 train/validation；C 使用固定注册表模型与有限 Pipeline；D 独立按 sample ID 计算任务声明的 accuracy、macro-F1 或 log loss。
- **A 上游规划器**：选择性接入同学提供的 `a-agent-v1(1).zip` 中 planner/decide 和 LLM planner 部分。`agent_a/upstream/` 保存被纳入的源文件；接入时的归档与源码 SHA256、来源和适配记录应以该目录的来源记录为准。
- **学校 HTTP/记账**：由 v2 的 `evaluation_d/school_gateway.py` 提供。它不复用上游包的 `run.py`、`task_adapter.py`、`evaluate.py` 或 `budget.py`，也不接入其第二套 B/D 流程。
- **任务数据**：来自用户提供的课程任务本地目录。项目原副本仅包含允许使用的 train/validation 与 Readme/info；自动入口可以读取外部 NPY/CSV 任务，原目录保持只读。test 不进入搜索、A 规划、C 训练或验证评分，只在显式 `--task` 或 `--test-config` 的模型冻结后最终评估中读取。自动实现验证采用合成 test。

## 整合改动

- 为支持独立启动，将原项目 `superagent/src/superagent/adapters/npy_reader.py` 的静态受限读取器原样纳入 `data_b/npy_reader.py`，替代依赖外层目录的导入包装；来源是用户提供的 Windows 原项目，未改动读取逻辑。
- 使用 `PackageAController` 把同学的 `planner/decide` 适配到现有 main `decide(context)` 接口，保留 main 生成实验编号、随机种子和 timeout 的责任。
- 学校模式只接入同学 A 的 `LLMPlanner`；D 网关负责学校 API 请求、调用限额、预算预留和元数据账本。
- 学校 key 由 `evaluation_d/env_config.py` 从项目根 `.env` 的 `SCHOOL_API_KEY` 加载；`.env.example` 是无密钥模板，真实 `.env` 不随源码版本控制或打包。
- 对未知 usage、超时和 HTTP 失败采取停止策略；不因异常自动重试或退回离线规划继续调用。只有 usage 已知而方案内容不合规时，才允许有记录的规则备用决定。
- 保持 B 的预处理和 C/D 训练评分协议；不接同学包其他运行、数据读取、评分或预算子系统。
- v1 的 `OfflineController` 原实现保存在 `agent_a/legacy_controller.py`，只作对照，默认入口已使用同学 A。
- 保留每次开发会话和 API 调用元数据，不清空历史或声称其为正式课程总账。
- C 注册表、白名单工厂、训练诊断、A 搜索接线、Plan/Manifest/Bundle 2.0 模态路由、任务指标和报告均属于 v2 的增量实现；兼容表格的旧 1.0 计划和冻结包仍有独立路径。
- H7 的跨会话任务预算账本在兼容配置模式下显式启用，自动任务模式默认启用并关联同一训练输入已有账本；报价配置采用 DeepSeek 公布的峰时、未命中缓存参考价换算，不能证明学校 endpoint 的模型别名或实际账单。
- H8/H9 的最终 test evaluator、任务目录发现器及 README/info 摘要是 v2 增量实现。发现器基于训练 schema 和说明生成内部契约，不按任务名选择模型；test 读取规则先固定，冻结后才实际读取。A 上游适配增加有界说明与 primary metric，不接入其他上游数据执行器。

复制的上游代码和改动摘要可通过 `agent_a/upstream/` 的来源记录复核。来源文件哈希应以最终交付文件中的记录为准；本文不推断未记录的源码哈希或未完成验证的结果。

## 声明

2026-11-20 增量：`data_b/llm_analysis.py`、可选预处理配置、CLI 早期网关接线及 `test_b_llm.py` 由 Codex 辅助实现，复用现有 D 网关与 sklearn 组件，没有复制外部 Agent 框架。真实模型调用未由自动验证执行；测试均为合成数据与 mock transport。

2026-11-14 B 第一轮增量：在现有集成代码上增加可选样本来源索引、无 test 开发模式、混合表格的有界 TF-IDF 分支和训练画像；相应适配 A 上游 `_compact_profile` 以接收有界统计，不替换其规划或调用网关。实现由 Codex 辅助编写，未复制额外第三方 Agent 代码；范围与验证见 `b-improvements.md`。

这是课程项目的集成实现，不表示课程已认可某个完整 Agent 底座。任务预算 profile 是可追溯的公开参考价换算，不是学校端正式账单核验。自动验证使用 mock，不代表发生过真实学校 API 请求。v2 的验证范围和当前限制见 [`VALIDATION.md`](VALIDATION.md)。
