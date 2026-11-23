# SuperAgent Team v2 规则

## 当前范围

- 项目用于阅读四人模块如何连接。B/C 为实际数据准备与训练实现；A 接入同学包的规则规划和可选 LLM 规划；D 真实计算 accuracy，新增显式学校 API 网关。默认离线。用户指定学校模式从本项目根 `.env` 加载 `SCHOOL_API_KEY`，真实 `.env` 由用户自行创建，Codex 不写入或读取真实 key、不继续 TLS 排查。
- `main.py` 是集成入口，`predict.py` 是冻结后新特征预测入口。
- B 的显式 `--b-analysis llm` 复用 D 网关和持久任务预算，先选说明，再解析文件用途与类型，程序据此选择读取器；缺失值建议仅限数值填补。`data_b/input_readers.py` 提供 TSV、逐行文本、分类目录、图片索引和有界平面 JSON/分隔记录转换，不支持嵌套或外部 JSONL，不执行生成代码。图片保留原尺寸，混合输入保留全部 RGB 像素并与文本/表格特征拼接，沿用 TaskSpec 1.0；纯文本/图片沿用 TaskSpec 2.0。缺失文本/图片的整行样本同步移除并记录原行号、数量和原因，数值缺失继续填补。A/C 训练接口保持。`reader_rules.json` 随模型冻结，`predict.py --raw-input --source-root` 复用规则。详细限制见 `docs/b-readers-and-conversion.md`。
- `data_b/task_directory.py` 自动解析 `--task` 指定目录的说明和开发分区；`sessions/<session>/task_input/` 保存自动生成的内部任务契约、开发数据规范化结果及惰性 test 描述，`test_input/` 只在冻结后创建。任务源目录只读；不按任务名选择模型或规则分支。
- `agent_a/` 放决策模块；`data_b/` 放任务读取、预处理和清单导出；`executor_c/` 放训练和进程管理；`evaluation_d/` 放评分及预留调用接口。各目录使用英文小写和下划线。
- `configs/` 是不含密钥的任务配置；`docs/` 是流程、接口、来源和阅读说明；`scripts/` 是合成数据工具；`tests/` 是项目测试；`.vscode/` 放项目调试入口与缓存显示规则，不修改全局设置。
- `agent_a/upstream/` 只保存选择性接入的 A 原包代码和来源摘要，不接其数据读取、评分、运行入口或 budget；`evaluation_d/school_gateway.py` 是实际 HTTP 及调用账本；`docs/school-api.md` 是学校 API 操作说明。新增目录先在本文件或已批准方案写用途，再创建。
- 根 `.env.example` 只放 key 字段占位示例；真实 `.env`、其备份或其他密钥文件禁止进入版本控制或 ZIP。`evaluation_d/env_config.py` 只解析本项目明确指定的 `.env`，不执行 shell、不插值、不搜索父目录、不修改系统环境；只在显式 school 启动时读取，测试使用合成占位 key。

## 数据及工件

- `data/raw/<task>/` 保存用户允许的训练/验证输入及说明，不改原始来源文件。
- `data/prepared/<task>/<version>/` 是 B 冻结输出，只有校验摘要一致的完整版本允许复用；不完整版本保留并拒绝覆盖。
- `sessions/<session_id>/` 保存决策、历史、评分和 summary；`runs/<task>/<version>/<experiment>/` 保存 C 日志、模型和预测；`submission/<task>/<final_id>/` 保存冻结工件。final_id 使用短编号，bundle 内记录完整 session_id。
- `sessions/api-<timestamp>-<uuid>/` 保存 API 调用前预留与完成/未知 usage 元数据，禁止写入 key、认证头、原始请求或响应正文；保留，不覆盖或自动清理。每次 school 启动都写明独立的开发会话范围，不冒充正式课程总账或清除旧账。
- `.cache/`、`__pycache__/`、`.pytest_cache/`、`.venv/` 是本地工具产物，不收入交付 zip，不自动清理历史。
- 每次主循环和每次实验都使用唯一编号，不覆盖，不自动删除或重试失败。
- 搜索只读 train/validation；validation 标签只在 B 导出和 D 独立评分使用，不传给 A/C。test 不进入默认 TaskSpec、搜索历史或 A prompt；显式 `--task` 的惰性 test 描述或独立 test config 只在 validation 最优 bundle 冻结后评估。发现阶段不打开 test CSV/NPY 或统计其内容。原始 test 也不打包。
- 预处理只在训练行拟合，按任务声明类型处理特征。模型选择只根据验证分数，禁止用 test 调参。
- 不默认合并 train+validation 重训；冻结时保留选定训练模型和对应预处理器。

## 工程约定

- A/D 替换遵循 docs/interfaces.md，保留主程序与 B/C 文件协议。
- 所有真实模型调用经 D 网关记账；请求前写 reserved，请求失败、超时、usage 缺失时写 unknown 并立即停止，不重试或自动降级继续消耗。供应商 token 使用原值，开发费用按用户已有配置计算，未知不能记为零。兼容 `--config` 的 school 默认最多调用一次；自动 `--task` 默认最多 20 次调用和每任务 20 元参考价账本，同一训练输入复用已有预算。Codex 不修改真实 .env、密钥或全局环境；自动验证只用 mock，不取得用户 key 或发真实请求。
- 复制代码须在 docs/provenance.md 记录来源与整合改动，不声称原创第三方实现或课程已认可某框架。
- 验证使用 pytest、ruff、新进程模型加载和真实开发分区；测试临时目录显式指定新目录并保留。
- 删除文件/目录/git 历史，修改密钥、.env、CI、数据库、全局依赖或系统配置，以及 push/rebase/reset/公开发布，均须先询问用户。
