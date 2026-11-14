# SuperAgent 分类执行模块 C：优化实施交接总纲

版本：2026-11-13；实施状态更新：2026-11-14。H0–H10 已在当前 v2 工作区实现；本文前半保留最初规格与决策依据，最终实现状态、执行证据和未覆盖项见 [`VALIDATION.md`](VALIDATION.md) 及文末“当前实施状态”。

本总纲把 `docs/c-enhancement-plan.md` 细化为工作包。阶段、接口和验收按本文执行；课程原文与用户后续指令优先。当前阶段不引入运行时任意代码生成，不重写整个项目。

## 1. 接手任务与成功标准

目标：在现有 SuperAgent v2 上，把 C 扩展为稳定的通用分类实验执行器。采用“固定执行框架 + 白名单模型/组件 + A 自主选择方案 + D 独立评分”的路线。

第一批必须完成：统一注册表、四类表格分类模型、可靠参数校验、训练诊断、A 真实接通新能力、可配置搜索停止、独立冻结预测和回归验证。四模型是 RF、LR、ExtraTrees、HistGradientBoosting，均支持二分类和多分类。

H5–H7 覆盖任务指标、可组合 Pipeline 与文本/图像输入、正式参考价格账本和报告交付；H8–H9 加入冻结后的独立 test 和全自动任务目录；H10 将 A 的完整声明式文本方案贯穿 C、history、冻结预测，并让多字段文本覆盖多个 classifier/vectorizer recipe。CatBoost、概率融合、跨会话成功结果缓存、深度预训练图像模型、任意未知任务格式仍不在当前能力承诺内。

交付应证明系统实际完成了不同方案的自主实验和反馈优化。不能以模型类数量、代码文件数量或是否花满 20 元作为成功标准；不承诺超过当前分数。

## 2. 必读文件与已核实状态

项目根：`/home/czc12/HKUST(GZ)/python4ai/superagent-team-v2`。下文路径均相对此根。

按顺序读：

1. `AGENTS.md`。
2. `AIAA3102 - Final Project - Full Assignment Requirements (1).md`。
3. `docs/interfaces.md`、`docs/c-enhancement-plan.md`、本文。
4. `main.py`、`predict.py`。
5. `executor_c/contracts.py`、`executor.py`、`train.py`、`consumer.py`、`capabilities.json`。
6. `agent_a/controller.py`、`agent_a/upstream/{planner,decide,llm_planner}.py`。
7. `data_b/{prepare,preprocessing,raw_reader,npy_reader}.py`、`evaluation_d/evaluate.py`。
8. 受影响测试、`pytest.ini`、依赖文件。

已核实事实，接手时仍需检查有无后续变化：

- C 已有独立子进程、超时 kill、线程限制、唯一目录拒覆盖、日志、总耗时、源码/环境记录、输入摘要和产物校验、原子结果发布。不要重新造一套替代它。
- 已有 joblib 同进程重载探测与跨进程预测集成测试。增加模型时扩展这些检查。
- 模型参数范围散落在 C 的 Python/JSON、A 的提示词/校验/规则空间，当前只接 RF/LR。
- A 已有当前会话模型/参数去重，patience 默认为 4。main 实际使用任务 random_seed，上游临时生成的 seed 不成为实际训练 seed。
- B 仅在 train 拟合缺失处理和独热编码，输出 dense float64。C 只收到 train X/y、validation X/ID，不收到 validation 标签。
- B、D 和 A/main 当前固定 accuracy，选模默认取最大值。支持其他指标需要联动。
- `configs/coupon.json` 当前 data_version 为 teacher_v2、max_trials 为 20。不要把配置值写死为程序通用规则。
- 原读取器已经复制到 `data_b/npy_reader.py`，当前项目不应再依赖外层 Windows 源码目录。

开发参考会话：`sessions/session_20261113T145539Z_efd01c58/summary.json`，5 轮、最佳 accuracy 0.7512315270935961、patience=4 停止。历史 0.0060775 元是配置开发价估算，不是课程正式核价。不要修改旧记录。

## 3. 课程约束与模块边界

### 必须保留

- 分类范围为二分类/多分类；任务可能是表格、文本、图像，不宣称仅完成表格就覆盖全部 unseen 风险。
- 根据数据类型、规模、类别数、指标、资源和历史实验选择方案；task_id 可用于隔离/账本/路径，不可用于调用人工特制算法。
- 正式运行中不得人工改下一轮方案、调停止条件或挑最终模型。可以在开发阶段设计通用候选空间。
- 搜索阶段不读指定 test。最终自动选模并冻结后，独立预测阶段才读取允许的最终评估特征；不能根据 test 结果重新选择。
- validation 标签仍只在 B 导出和 D 评分。训练内早停/CV 可以使用 train 内部划分，不可借用 D 的标签。
- 不默认合并 train+validation 重训；按已选择的 train 模型及其预处理器冻结。
- 正式报告结果需来自同一最终提交版本的完整自主运行；冻结后 unseen 使用同一代码版本，整体期限 24 小时。

### 职责分工

| 模块 | 应负责 | 不应承担 |
| --- | --- | --- |
| A | 从合法能力选择方案、读反馈、决定搜索方向和停止 | HTTP 直连、读 validation 标签、任意执行代码 |
| B | 读取正式声明的数据格式、训练拟合转换、标签映射、发布版本化表示 | 用 validation 拟合预处理 |
| C | 校验计划、构造模型/Pipeline、训练预测、资源控制、诊断与可加载工件 | 外部 validation 评分、决定最终模型、另开 LLM 调用链 |
| D | 独立评分、LLM 网关、token 与费用记录 | 回写 C 结果伪造成功 |
| main/predict | 串联流程、预算/时间约束、选优冻结、最终输入输出适配 | 根据 task 名称特制方法 |

## 4. 接口兼容策略：先决定再编码

核心公开函数第一批保持：

```python
execute(plan: dict, data_manifest_path: Path | str,
        runs_root: Path | str) -> str  # result.json 相对路径
decide(context: dict) -> dict
evaluate(result_path, evaluation_manifest_path, output_path=None) -> dict
predict_bundle(bundle_path, features_path, output_path, id_prefix="prediction") -> Path
```

第一批 C 计划仍使用现有 1.0 的精确顶层字段：

```json
{
  "schema_version": "1.0",
  "task_id": "demo_task",
  "data_version": "demo_v1",
  "experiment_id": "exp_unique_001",
  "model": "extra_trees",
  "params": {"n_estimators": 200},
  "random_seed": 42,
  "timeout_seconds": 120
}
```

`params` 缺省项由统一注册表补齐。LLM 的 proposal 继续只给 model/params/reason 或 stop；task_id、experiment_id、seed、timeout 由可信编排器补齐。不要让模型随意指定文件路径或环境。

版本分别管理，不要把 `contracts.py` 的全局 SCHEMA_VERSION 一改导致所有文件失效：

| 对象 | 第一批策略 | 后续策略 |
| --- | --- | --- |
| Plan | 保持 1.0 精确顶层字段，扩展合法模型/参数 | 可组合组件新增字段时另引 Plan 2.0，保留 1.0 parser |
| B data manifest | 保持现有 1.0 dense-only | 引入 sparse/raw-text 等表示时设计 Manifest 2.0 |
| C result / worker metadata | 保留核心结构和旧含义 | 诊断通过独立 sidecar，必要的路径引用由明确兼容的 reader 接收 |
| capabilities | 单独采用新版能力 schema，建议 2.0 | A 与 C 同批接通；不能借此升级其他 schema |
| bundle | 第一批保持旧工件路径与 1.0 读取 | 新表示/自定义模型封装需要版本化读取并保留旧包兼容 |
| 任务/evaluation 配置 | 第一批继续 accuracy 1.0 | 指标参数与新格式按各自版本解析，不向旧严格 schema 塞字段 |

若保留旧符号 MODEL_PARAMETERS，只能由注册表派生作为兼容视图，不能再维护第二份常量表。旧输入补新默认值后必须保持 RF/LR 原先实际行为。

旧模型重载兼容主要针对同一受支持依赖环境；不要承诺 sklearn joblib 文件跨任意依赖版本二进制兼容。正式交付记录并复现已验证依赖版本。

## 5. 工作包 H0：基线和变更边界

先完成以下记录，再动实现：

- 实际 Python/依赖版本，当前 CLI、关键 schema 和读写方。
- 现有测试实际状态；有失败则区分基线问题和本次引入，不能只报告绿灯。
- `data_b/prepare.py::_source_signature` 当前哈希覆盖 prepare.py/raw_reader.py/preprocessing.py/npy_reader.py，任一变化都可能使旧 B 版本拒绝复用。
- C 现有源码快照只有 train/contracts/executor；新增注册表、factory 和诊断代码也需要覆盖。
- 历史 runs/sessions/prepared/submission 只读保留；验证使用新唯一输出根。

通过条件：写出变更文件清单和兼容决策，能说明哪些修改会触发新数据版本。不得删除旧版本、手改旧哈希或关闭摘要检查来绕过错误。

## 6. 工作包 H1：唯一模型注册源

建议新增文件，均位于已有目录：

- `executor_c/model_registry.py`：纯描述和校验，不 import heavy sklearn estimator、不训练、不读数据。
- `executor_c/model_factory.py`：按内部 builder ID 构造 estimator/Pipeline，延迟加载需要的依赖。
- `scripts/export_capabilities.py`：从 registry 生成 JSON 清单；开发者显式运行，不在 import 时改文件。

建议 API，可按现有风格命名，但职责不变：

```python
get_model_spec(model_id: str) -> dict
normalize_params(model_id: str, params: dict) -> dict
get_capabilities() -> dict
build_estimator(model_id: str, normalized_params: dict, seed: int) -> object
```

注册描述至少包括：model_id、builder_id、参数 schema/default、适用 representation、支持二/多分类、predict/predict_proba 能力、固定设置、默认 baseline 和有限搜索候选、registry_version。factory 的白名单由内部注册项关联，拒绝外部 Python 类路径。

参数校验要求：

- 支持 int、number、bool、enum、nullable、有限的联合类型和必要条件约束。
- 拒绝未知字段、NaN/Infinity、bool 冒充数值、字符串数字、未声明类型和越界值；错误不回显原始回复。
- 默认值补全、浮点规范化、排序序列化的结果作为计划和去重的共同基础。
- max_features 的数字只开放“比例”含义，规范为 float，避免把整数 1 和比例 1.0 混淆；不开放 count-int 模式。
- 不暴露 n_jobs=-1、动态脚本、callback、warm_start、输出路径和其他无法稳定复现的自由配置。
- `contracts.validate_plan` 保留现有错误类型并调用 registry，A 使用同一规范化接口；避免循环 import。
- capabilities JSON 与运行时值相等仍须有测试，生成清单不能变成另一手工来源。

通过条件：变动一个参数定义会一致影响 A 能力、C 预检和 JSON 清单；非法参数不能启动 fit；读能力不会创建文件或读密钥。

## 7. 工作包 H2：四类模型和参数的明确范围

以下是项目有意收窄的首批支持范围，不是 sklearn 的所有能力；硬上限与搜索候选分开管理。

### RandomForest / ExtraTrees

| 参数 | 默认 | 合法范围 |
| --- | --- | --- |
| n_estimators | 100 | 整数 1..500 |
| max_depth | null | null 或整数 1..50 |
| min_samples_leaf | 1 | 整数 1..100 |
| min_samples_split | 2 | 整数 2..100 |
| max_features | "sqrt" | "sqrt" / "log2" / null / 比例 (0,1] |
| criterion | "gini" | "gini" / "entropy" |
| class_weight | null | null / "balanced" |

RF 固定 bootstrap=True，ExtraTrees 固定 bootstrap=False，均固定 n_jobs=1、random_state=实际 seed。默认 RF 旧三参数输入补齐后应与原配置行为一致。

### LogisticRegression

| 参数 | 默认 | 合法范围 |
| --- | --- | --- |
| C | 1.0 | 有限数，0 < C <= 100，统一转 float |
| max_iter | 1000 | 整数 1..2000 |
| class_weight | null | null / "balanced" |

保留现有 StandardScaler + LogisticRegression Pipeline、lbfgs 和兼容版本下的默认 L2 行为。第一批不新增 solver/penalty 自由组合；不要为了消除 warning 擅自改现有模型语义。

### HistGradientBoosting

| 参数 | 默认 | 合法范围 |
| --- | --- | --- |
| learning_rate | 0.1 | 有限数，0.001..1.0 |
| max_iter | 100 | 整数 1..500 |
| max_leaf_nodes | 31 | 整数 2..127 |
| max_depth | null | null 或整数 1..50 |
| min_samples_leaf | 20 | 整数 1..100 |
| l2_regularization | 0.0 | 有限数，0..100 |

第一批固定 categorical_features=None，使用 B 已经生成的数值特征；固定 early_stopping=False，使初始行为明确且不依赖 `auto` 的样本数阈值。记录实际 n_iter_。

内部早停属于后续扩展：仅从 train 划分，显式设置 fraction、seed、n_iter_no_change，检查小样本/少数类限制，不能使用 D 的 validation 标签。不能把早停检查集当作外部独立分数。

默认候选示例：树数量 100/200/400，树深 null/8/16/32，叶子样本数 1/2/5/10，max_features sqrt/0.5/1.0；LR 的 C 0.03/0.1/0.3/1/3/10/30；HGB 学习率 0.03/0.1/0.2，迭代 100/200/400，叶子数 15/31/63，L2 0/1/10。不要直接穷举全部笛卡尔积。

通过条件：4 模型 × 二/多分类，均能训练、保存、重载、按正确类别顺序生成概率，并经 D 独立评分。A mock 必须至少实际选择一次 ET/HGB，直接调用 C 通过不算端到端完成。

## 8. 工作包 H3：结构化诊断与失败反馈

新增 `executor_c/diagnostics.py`，每实验独占发布 `diagnostics.json`，该 sidecar 自己使用 schema_version=1.0。固定命名约定：worker 写 `worker_diagnostics.json`，父进程结合退出结果原子发布同目录下最终 `diagnostics.json`。不要让可能被 kill 的 worker 和 parent 竞争写同一个终态文件。main 在验证 result 路径后只读取固定同目录文件，做路径/版本/字段校验；第一批无需为此改变 result 的核心结构。旧历史没有该文件时明确显示诊断缺失，不伪造数值。

建议字段：

```json
{
  "schema_version": "1.0",
  "experiment_id": "exp_unique_001",
  "status": "success",
  "phase": "complete",
  "timings_seconds": {
    "input": 0.02, "fit": 1.25, "predict": 0.01,
    "serialize": 0.03, "reload_check": 0.02, "total": 1.50
  },
  "model_bytes": 123456,
  "input_shape": {"train": [120, 30], "validation": [40, 30]},
  "input_matrix_bytes": 38400,
  "random_seed": 42,
  "effective_iterations": 100,
  "warnings": [],
  "warnings_truncated": false,
  "failure": null,
  "experiment_signature": "sha256-hex"
}
```

字段语义：示例数字仅表示格式；timings 用 monotonic/perf_counter，分阶段时间不是端到端总耗时的严格求和；input_matrix_bytes 声明是否包含 y/ID，统一按已加载 X_train+X_valid 的字节计。model_bytes 为实际文件大小，非内存峰值。

warning 记录 category、phase、限长清洗后的 message，最多保留 20 条，超出明确截断。未知/未完成字段为 null，不能写 0 冒充测量结果。普通 ConvergenceWarning 记录后允许完成，但非有限预测或训练异常仍失败。

effective_iterations 对 RF/ET 表示实际树数，对 HGB 表示 n_iter_，对 LR 表示 n_iter_ 的最大值，模型类型和定义必须可查。完成或失败都尝试发布诊断；身份非法、还未安全建立实验目录的 InvocationError 不要求创建目录或 sidecar。

保留 rejected/failed/timeout/success 既有状态。诊断细分 input/fit/predict/artifact 阶段。父进程 timeout 只能确定最近已知阶段，不把过期 worker 状态误称精确终止位置。不要把任意负退出码都断言为 OOM。诊断失败不能掩盖原始训练失败；第一批可将诊断缺失标记为缺失并使报告显式显示，不能伪造完整记录。

扩展环境/源码审计时使用字段白名单，不 dump os.environ。源码摘要覆盖本次实际执行的 registry/factory/diagnostics 及现有关键模块；只保存源码和必要依赖信息。

接线必改：main 读取 sidecar→history；`PackageAController._map_history` 保留安全诊断摘要；`LLMPlanner._user_message` 将该摘要纳入下一轮。不能只在 runs 下落盘却不送回 A。

通过条件：一次成功、一次收敛警告、一次真实 timeout 和一次预测失败都可追踪；mock LLM 的第二轮上下文能看到第一轮诊断；其中不含 validation 标签或原始回复。

## 9. 工作包 H4：A 接线、阶段搜索和时间/种子

### A 的最小必改

- 移除 SYSTEM_PROMPT 中只允许 RF/LR 的硬编码，改为通用固定规则 + 当前 capability schema。
- `validate_model_params` 委托统一 validator，接受缺省参数并规范化后去重，不保留旧的 exact 三/两参数集合限制。
- 规则备用 planner 的 FAMILIES/候选源由 registry 派生，不能保留“非 RF 就当 LR”的分支。
- 离线与 school 使用同一合法候选空间，记录实际来源 llm/rules/known-usage fallback，不把规则基线伪称真实 LLM 方案。
- LLM 无效/重复输出仍使用已有网关限制：usage 已知才可有记录地用规则给当前决策备用；transport/timeout/unknown usage 立即停止，不重试或降级继续消费。

### 配置和状态机

新增不含密钥的 `configs/search-defaults.json` 和拟新增 `--search-config`。不要把 A 的策略字段直接塞进 B 严格 task schema。max_trials 仍由任务 train_policy 给出，max API calls 仍由网关限制，不能维护三份互相覆盖的轮数。

建议配置：

```json
{
  "schema_version": "1.0",
  "require_family_coverage": true,
  "min_exploration_trials": 8,
  "patience": 8,
  "min_delta": 0.0,
  "max_consecutive_failures": 3,
  "max_session_seconds": 1800,
  "finalize_reserve_seconds": 60,
  "confirmation_seed_offsets": []
}
```

该配置已实装为 `configs/search-defaults.json`，通过 `--search-config` 显式启用。省略该选项时保留 legacy 短会话行为；增强配置从 registry 的 profile/tag 读取候选，不再手写第二份 RF/LR 参数表。确认 seed 偏移默认关闭；显式配置时，确认运行受剩余主实验轮数、session 时间预留和 API usage 状态约束，实际 seed 写入 Plan/history。

阶段：

1. coverage：A 在当前数据表示和资源下合法的模型族中选择未尝试的基线。硬预算/调用数/时间上限始终优先；不会为了覆盖强制越界。一次基线失败也算尝试并反馈，不无限卡在该族。
2. exploration/refinement：达到最低覆盖后，基于历史分数和诊断决定广度探索或局部改进。只放宽“无提升”停止，不取消所有停止条件。
3. confirmation：配置非空、时间/轮数允许时，对至多两个预先按规则选出的候选做 seed 复核；否则跳过。
4. finalize：自动选择成功且评分有效的最佳单次模型并冻结；没有成功模型则保留失败会话和原因。

最低探索轮数不能突破总轮数、时间或调用/预算上限，未完成覆盖时明确记录原因。进入 refinement 时以此前有效最佳分数为 best、把 patience 计数器置零，之后仅有效评分的“未超过 best + min_delta”（越小越好的指标反向）推进计数，实质提升重置。失败尝试占总轮数但不伪造分数、也不推进有效评分的 patience。增强策略中，C rejected/failed/timeout 或 D 评分失败计入 max_consecutive_failures，成功评分重置，达到阈值转 finalize；网关 unknown 不等此阈值，立即停止消费。最终选优仍按指标实际值，而不是 min_delta 决定；相同分数保留更早成功项作为确定性 tie-break。

如果 LLM 自主明确 stop，记录来源与原因；最低覆盖/探索是普通搜索的门槛，不强迫系统否认一个有意义的自主 stop。需在报告解释未覆盖原因；不能无条件把所有 stop 改成继续。

### 种子和签名

正常搜索沿用 task random_seed，让方案对比保持一致。去重键包括规范化 model/params、实际 seed、数据/预处理指纹与实现版本；预算/timeout 记录在实际计划中。增加 timeout 的同方案仍是有记录的新资源尝试，不复用上一次失败。

确认阶段 seed 由 main 按固定 offset 规则产生，通过 controller 上下文传递并最终写入 C plan；不要在 A 中生成一个实际被 main 丢弃的 seed。是否允许失败方案再次尝试须有明确资源/seed 变化和记录，不自动重试原计划。

已有 params_key 的“策略候选键”和“实际执行指纹”要区分命名用途：LLM 的 proposal 尚无 seed 时使用本轮有效 seed 规范化后判重；旧 history 1.0 可从其 plan 恢复实际 seed。不能把不同 seed 的确认实验排除。

时间限制由 main 持续计算剩余时间；启动新 worker 前将有效 timeout 设为 min(计划 timeout, 剩余搜索时间)，且仍 <= C 的 600 秒上限。时间不足时不再开新实验，保留成功候选用于 finalize。当前一次 LLM 请求也要受剩余时间约束，不能只限制训练。

确认实验先仅用于稳定性报告；冻结仍按最佳单次验证分数，保持当前选模语义。若以后按均值/投票选择，必须另定义并测试整套政策，不能报告均值却输出人工挑出的模型。

终态语义明确区分：正常达到轮数/已知预算/调用上限且已有有效模型，可以 success；没有有效模型则 failed。网关 transport/unknown 等错误保留 failed 与原始错误，可另外冻结此前已经独立评分成功的最佳模型，标为 partial 工件并记录路径，不因此把会话改成 success、不再训练或调用 LLM。正常结束需要保存的工件无法完整冻结时仍是失败，不能只因有过高分就写 success。

通过条件：增强配置下 4 模型有机会被探索，patience 不在第 5 轮机械截断；旧无 config 行为有兼容测试；硬限制仍成立；指定 seed 与实际 seed 一致；known/unknown 网关路径均不漏账。

## 10. 工作包 H5：任务指标和目标方向（团队联动）

本包是正式任务通用性的必要项；C 开发者提供输出能力，D/A/B/main 一同落地，不能把评分器塞进 C。

第一批可支持 accuracy、f1_macro、log_loss，覆盖“越大越好”和“越小越好”。ROC-AUC 等按正式任务需要增加；不提前宣称全指标支持。

- 主指标由任务声明，包含名字、方向、必要参数（正类/average 等）；方向可从内部 metric spec 验证，拒绝用户配置与定义矛盾。
- B 导出对应 evaluation manifest；更改 B preparation 代码或任务语义时创建新的显式 data_version，不能改旧 teacher_v2 摘要。
- D 校验适用条件、固定 labels 顺序，缺类/非法概率等行为明确定义并测试；不按分数好坏挑 metric。
- f1_macro 使用任务完整 class_ids 作为 labels、zero_division=0，避免验证缺类时指标定义漂移；log_loss 使用顺序一致的完整概率列及显式 labels，允许 validation 缺类但不允许未知类，不能以预测类别替代概率。其他特殊口径以任务明文定义为准。
- A `_map_history` 的 `[0,1]` 检查按 metric domain 修改。log_loss 可以 >1，不能拒绝合法值或把失败记为 0。
- 修改 main `_check_score`、最终 max/min 选优、输出文案、freeze selection，以及 A 的 greater_is_better=true 假设。
- 旧 summary 的 best_validation_accuracy 只在 accuracy 任务填写；新通用字段建议 best_validation_metric={name,value,greater_is_better}。不是 accuracy 时旧字段为 null，不塞入其他指标的值。
- 保留旧 accuracy 1.0 输入 parser。升级任务/evaluation schema 时提供独立的版本路由，不强迫旧 B manifest 全部改版。

通过条件：同一组受控预测能证明 accuracy/f1 的最大值选优和 log_loss 的最小值选优；实际小数据完整跑通，不仅单测一个 compare 函数。

## 11. 工作包 H6：可组合组件与非表格路径（第二阶段）

本包之前，系统应准确声明“已支持 dense 表格二/多分类”。不能靠在 capabilities 里写 text/image 就称已支持。

### 受限 Pipeline

新增 Plan 2.0 parser，顶层新增 `pipeline` 的合法结构，显式按版本分派，保留 Plan 1.0 解释为默认 Pipeline。首先只提供 C 可独立实现的 train-only 特征选择（none/去常量），以及模型专属缩放策略；A 从组合规则选，C 拒绝非法组合，整个 sklearn Pipeline 一起保存。selector 只在 train fit，validation 和冻结预测只 transform/predict；编码器、selector、模型及其顺序必须全部冻结，不能预测时重拟合。

示意：`pipeline.feature_selector={name:"variance_threshold", params:{threshold:0.0}}`。不开放任意 Python 类路径；若全部特征被移除，明确失败。B 原有 imputer/OHE 仍只拟合 train，C 不再重复拟合另一份不一致的 B 预处理器。

### 文本

1. B 新增声明式文本数据读取和标签/ID协议；Manifest 2.0 区分 raw_text、sparse_numeric、dense_numeric，不能向 1.0 添加不识别字段。
2. 建立 train-only 文本向量化 + LR 分类的完整可序列化 Pipeline。可将词/字符 TF-IDF 作为有限候选，依据语言和数据特点选择，而非任务名。
3. 若 B 发布冻结 sparse 特征，C 的 sparse loader、hash、模型兼容过滤及 predict 必须配套；HGB 不应接收 unsupported sparse 输入或暗中全量转 dense。
4. 原始文本进入谁的转换器、词表由谁拟合必须只选一处负责；同一词表/特征处理必须保存并用于最终预测。
5. 规定 max_features、文本长度/样本规模和内存策略；禁止联网临时下载不声明的 embedding 或依赖。

### 图像

根据已发布任务明确 shape/channel/label/资源约定后选择通用基线，训练和推理共享处理。额外预训练视觉/embedding 模型的许可范围先按课程要求核验，不默认任意第三方权重获准；不要把展平图像后运行 RF 的 smoke test 宣称高质量图像支持。

### 其他后置项

- CatBoost 需要 B 原生类别视图与冻结预测路由；不直接把当前独热矩阵当作原生类别方案。
- 概率融合先做至多两模型、固定受限权重、类别对齐和全部依赖工件冻结；不能让 C 根据 D 标签拟合融合权重。
- 严格 train 内 CV 的预处理需逐折拟合；直接对 B 全 train 已拟合特征切折不宣传为严格无泄漏 CV。
- 跨会话成功缓存默认关闭。启用前校验同数据/预处理/seed/完整参数/实现和依赖，保留来源记录并由 D 独立评分；不得引入人工挑选的旧任务方案。

## 12. 工作包 H7：正式计费、版本与交付（团队联动）

开发模式默认保持原有每 session 账本。正式模式另显式设计，不可把 existing summary 的开发价格改名即声称符合课程。

已核验 DeepSeek 公开参考价和 2026-10-21 汇率，并实现为独立价格 profile；学校 endpoint 的真实 provider、model alias 和实际账单仍未核验。`school-api.json` 仍是调用配置，不自动替换 endpoint/model，也不把参考价声称为学校正式扣费。

任务费用计算：`sum(input_tokens * official_input_price_per_million + output_tokens * official_output_price_per_million) / 1000000`，使用 Decimal，保留原始 token 数与价格来源/生效日。不使用平台优惠、免费或缓存折扣；细分 token 与 total 的包含关系要核对，避免重复/漏算。

每任务、所有 agent、所有相关 session 关联到统一预算范围。任务预算 ID 只用于记账，不参与算法路由。并发调用需有原子 reserve/settle 或明确单写锁，重启不重置 20 元；unknown 用量继续保守占用，禁止记零或删除历史后再跑。未确定的开发/正式尝试归属在记录中说明，不能自行把每次启动等同新任务额度。

正式 release 记录全系统源码摘要、依赖锁定/已验证版本、TaskSpec、策略配置和工件关联。所有正式报告结果用同一最终程序完整自主重跑。保留有效 partial 结果和失败解释；API 失败后的 session 失败不能掩盖之前已成功实验的存在。

最终预测包至少包含：模型/组合工件、B/C 所需预处理、原始标签映射、特征或模态约定、实际参数/seed、依赖和源码版本、输入/工件摘要、任务要求的输出格式。sample ID 按输入指定顺序/规则保持；输出原始标签还是编码类别必须由任务协议决定。

报告辅助输出（无需新前端）：`scripts/report_session.py --summary ... --output ...`，生成唯一目录下 leaderboard.csv、report.md、best-so-far 曲线或可嵌入报告数据。不得调用 LLM 写摘要或直接改写 report.pdf；自动摘要可使用确定性模板。

## 13. 文件改动地图

| 文件 | 工作内容 |
| --- | --- |
| executor_c/model_registry.py（新增） | schema/default、能力、有限候选、规范化 |
| executor_c/model_factory.py（新增） | 内部白名单构造4模型/Pipeline |
| executor_c/diagnostics.py（新增） | 分阶段诊断、清洗warning、签名 |
| executor_c/contracts.py | 保留计划/数据/路径校验，委托参数规则，分离schema版本 |
| executor_c/train.py | registry/factory执行、阶段计时、实际参数与工件检查 |
| executor_c/executor.py | 保留隔离，扩展源码快照与最终诊断，处理timeout缺失信息 |
| executor_c/capabilities.json | 生成产物，不手工维护参数 |
| executor_c/consumer.py | 按需支持新artifact引用；保留路径/概率/行校验 |
| agent_a/controller.py | 目标/seed/策略上下文、诊断透传、共用规范化 |
| agent_a/upstream/*.py | 去硬编码、阶段策略、schema校验/候选空间；记录来源改动 |
| main.py | --search-config、实际seed/时间控制、history、按目标选优、freeze |
| predict.py | 新模型重载回归；H6新增表示/组件时版本化路由 |
| configs/search-defaults.json（新增） | 显式启用增强搜索，不改真实.env |
| scripts/export_capabilities.py（新增） | 确定性导出和校验capabilities |
| scripts/report_session.py（新增） | 合并C诊断/D评分/调用用量生成报告 |
| data_b/*、evaluation_d/evaluate.py | H5/H6所需协议/指标协作，第一批避免无关改动 |
| evaluation_d/school_gateway.py等 | H7课程计费工作包，保留unknown停止和旧账 |
| tests/* | 参数化四模型、协议边界、闭环和故障测试 |
| docs/interfaces.md、README.md、docs/provenance.md、VALIDATION.md | 接口/执行说明/来源/实际验证结果 |

不新增顶层框架、数据库、Web UI 或散乱目录。现有目录足够；确需新目录先按 AGENTS.md 记录用途。移植 upstream 的变更在 provenance/source_record 中可追溯，不抹去原作者信息。

## 14. 验收矩阵：实际行为，不只检查文件存在

| ID | 场景 | 必须证明 |
| --- | --- | --- |
| T01 | 旧RF/LR计划 | 原参数语义、合法默认值和旧bundle在受支持环境可用 |
| T02 | 统一能力 | 生成JSON与runtime一致；A/C按同一范围接受或拒绝 |
| T03 | 非法参数 | bool冒int、NaN/Inf、额外字段、越界、非法枚举在fit前拒绝 |
| T04 | 4模型×2/3类 | 实际训练、保存、完整预测、概率类别顺序、独立评分 |
| T05 | 缺失/未知类别 | B仅train拟合，validation新类别保持协议，无标签泄漏 |
| T06 | 故障 | 真实短timeout清理进程；训练/预测/保存失败不发布success |
| T07 | 工件与路径 | 不覆盖、路径逃逸、输入变更/损坏模型被拒绝，旧结果字节不变 |
| T08 | 重载 | 新进程及不同cwd从冻结包预测，与冻结前预测一致 |
| T09 | A接线 | mock学校规划器实际提出ET/HGB并运行，规则备用也可选新模型 |
| T10 | 反馈 | 第二轮mock上下文看到第一轮耗时/warning/失败阶段，无validation标签 |
| T11 | 搜索停止 | 默认兼容；增强配置覆盖门槛/patience/min_delta/硬限制可解释 |
| T12 | 去重/seed | 缺省参数等价命中；不同seed可复核；main真正使用记录seed |
| T13 | 指标方向（H5） | accuracy/f1最大和log_loss最小端到端选优；合法loss>1不被拒 |
| T14 | 网关 | mock known usage准确计数，unknown/失败立即停止无重试，不记零 |
| T15 | 通用性 | 改task_id/数据集路径不改变算法规则；不同列/类别数能同入口运行 |
| T16 | 多表示（H6） | unsupported表示在训练前拒绝；text变换仅train拟合；不暗中dense化 |
| T17 | 正式预算（H7） | 跨session累加、价格来源、restart/unknown/concurrent预留模拟一致 |
| T18 | 正式交付 | 最终源码版本一致、统一入口、自主停止选模、冻结后最终预测，test不影响选择 |

测试原则：用小合成数据验证协议、失败和工件；时间数字仅检验有限非负与阶段语义，不断言精确耗时或新模型必定优于旧模型。不要写复刻实现的测试来制造数量。已有3轮合成测试不应因默认配置变化被无理由改成“反正返回成功即可”。

## 15. 验证命令与工件保留

以下使用项目根为 cwd、父目录现有 `.venv`；若其他机器路径不同先定位解释器，不创建全局环境修改。

```bash
../.venv/bin/python -m pytest tests/test_executor_c.py tests/test_main.py \
  --basetemp=".cache/test-temp/c-core-$(date -u +%Y%m%dT%H%M%S)-$$"
../.venv/bin/python -m pytest tests/test_package_a.py tests/test_llm_controller.py \
  tests/test_school_flow.py tests/test_school_gateway.py \
  --basetemp=".cache/test-temp/c-integration-$(date -u +%Y%m%dT%H%M%S)-$$"
../.venv/bin/python -m ruff check .
```

每次 basetemp 必须不存在，pytest 可能删除已存在的指定目录，禁止复用固定名字。新增测试加入相应定向命令；受影响定向检查通过后，交付前完整 pytest 回归一次，仍使用新唯一 basetemp。不得删历史测试目录。

pytest/ruff 缺失时可按 requirements-dev.txt 安装到已选项目虚拟环境，遵循执行环境权限；不改全局依赖。环境/网络阻塞需准确报告，不能跳过后声称通过。

真实开发验证使用新output-root，例如：

```bash
../.venv/bin/python main.py run --config configs/coupon.json \
  --output-root ".cache/c-benchmark-$(date -u +%Y%m%dT%H%M%S)-$$" \
  --planner offline
```

增强配置验证通过 `--search-config configs/search-defaults.json` 显式运行。新增输出根只包含本次生成的工件，原始数据按任务config来源读取；不复制/读取原始test。

校验配置使受控四模型基线能够覆盖，不以这条命令的高低分判断测试通过。真实学校API验证只由用户本地显式启动；自动开发验证使用mock，不读取key、不代发模型请求。

## 16. 实施节奏、完成定义与禁止事项

实施顺序为 H0→H1→H2→H3→H4→H5→H6→H7；当前工作区按该顺序完成实现。交付状态和验证证据以 `VALIDATION.md` 为准，不能把缺失 pytest/ruff 的环境记录成完整测试通过。

每次阶段交付提供：改了什么/为什么、实际命令与pass/fail、生成工件链接、兼容性与已知局限、下一工作包。不把真实API未测说成已测，不把mock当正式实验，不为了更高分触碰test。

禁止：任意LLM生成代码执行、默认自动下载外部权重、自动更换真实模型/计费价格、读取/打印/修改真实.env、清空历史、覆写teacher_v2、改旧摘要伪装复用、全局依赖/代理/TLS修改、擅自push/发布。用户需要这些不同范围的行动时按已有授权与AGENTS规则判断，不能从本文推断全部已授权。

第一批完成定义：四模型真实闭环 + 同源能力校验 + 可见反馈 + 有界搜索 + 原有安全执行/保存不退化 + 新进程推理 + 有证据的回归结果。准确率和API消费作为实验结果报告，不作为保证。

## 17. 可直接转给接手模型的提示词

```text
请在现有 superagent-team-v2 上实施 C 模块增强，沿用固定执行框架与受限可组合方案，不引入任意LLM代码执行。

先读 AGENTS.md、课程完整要求、docs/c-implementation-handoff.md 和 docs/interfaces.md，再核对当前源码。以交接总纲为实施规格，先完成 H0-H4 第一批全部工作：统一模型注册表和参数校验；RF/LR扩参；新增ExtraTrees和HistGradientBoosting；结构化诊断；A提示词/校验/规则搜索接线；显式增强搜索配置、真实seed和时间限制；模型冻结和新进程预测兼容。

保留 C execute 接口、Plan/DataManifest 1.0兼容和已有子进程/超时/不覆盖/哈希/工件校验。capabilities单独版本化。不要只改C而让A仍只认两模型。不要改B预处理来绕过第一批问题。

不读取或修改真实.env/key，不调用真实模型API，不清理历史或修改旧数据摘要；自动验证用离线/mock，新工件存唯一目录。按工作包完成定向测试并在最终交付前完整回归，缺失依赖只处理项目环境。禁止把失败、未测或mock说成通过/真实调用。

按 H0-H7 完成实现，不引入运行时任意代码生成。先读课程要求和交接总纲，再检查当前实现与验证记录。用户已授权连续推进，不要因常规实现选择中途停下来询问。
```

如需委托全系统增强，把上段“先完成 H0-H4 第一批”改成“按顺序完成 H0-H7，并先检查任务/模态与正式价格依据等前置资料；能独立完成的工作持续推进，未知范围明确记录，不猜测”，并与各模块负责人统一修改范围。

## 18. 参考和未核验事项

- 课程来源：`AIAA3102 - Final Project - Full Assignment Requirements (1).md`；允许受限方案路线，未强制任意代码生成。
- [ExtraTrees官方参数](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)：核对默认bootstrap、特征采样及概率接口。
- [HGB官方参数](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)：核对输入、early_stopping和n_iter_。
- [LR官方参数](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)：核对固定solver与依赖版本兼容。
- 官方库文档描述的是库能力，本文的参数限制、文件分工与阶段安排是本项目设计。当前requirements允许较宽版本范围，实施应选定并记录经测试版本，不拿stable文档默认值代替当前环境核验。
- DeepSeek 公开参考价和 2026-10-21 汇率来源已记录，学校 endpoint 的 provider/model alias 和实际 invoice 未核验；账本是显式可选参考价核算。
- 文本与图像执行路径已按 TaskSpec/Manifest 2.0 实现为受限基线；不宣称其精度或性能超过针对具体任务设计的视觉/文本方法。
- 当前依赖范围仍允许多个版本；验证证据只适用于 `VALIDATION.md` 记录的实际环境。pytest/ruff 无法运行时不得声称完整自动回归通过。

## 19. 当前实施状态（H0–H10）

- 当前固定模型白名单包括四种 dense/image 表格分类器（RandomForest、LogisticRegression、ExtraTrees、HistGradientBoosting）和两种 raw-text 稀疏分类器（MultinomialNB、ComplementNB）；LogisticRegression 同时支持 raw text。文本默认起始 portfolio 为三个模型族 × 最多四种文本表示，共 12 个方案；按验证分数和 patience 继续有限搜索。
- CSV/Numpy 表格、文本和 NPY 图像分类通过统一 `run --task <目录>` 接入；支持 accuracy、macro-F1、log loss。训练多数类/先验概率 validation baseline 用于诊断。test 只在 validation 选优并冻结后独立读取。
- pipeline 选择属于方案身份并进入 history 与冻结 plan；文本逐字段 TF-IDF 的自动 final-test 回归覆盖 freeze→predict，旧 text plan 缺少 `field_mode` 时维持 shared 默认值。
- `docs/VALIDATION.md` 记录精确运行命令、33 项定向 unittest、Headlines train/validation 搜索审计、没能执行的 pytest/Ruff 以及安全边界。全量 pytest 与 Ruff 当前仍受开发环境缺少工具限制；不能声明全量测试通过。
- 本轮实现检查点与最初自动目录版本分别保留于 Git；没有提交真实 `.env`、课程任务目录或 API key，也没有调用真实学校 API。

本文第 14 节的 T01–T18 是需求验收矩阵，不代表逐项 pytest 全绿。当前有限注册方案旨在支持同一通用流程与可追溯改进，不保证所有未知任务格式或分布都能获得高分；新数据格式应扩展通用发现/读取器，不加 task-name 专用算法分支。
