# SuperAgent H0–H10 实施与验证记录

## B 三项升级验证（2026-11-20）

- `test_b_llm.py` **18/18**：真实 D 网关实现 + fake HTTP，覆盖任意命名 CSV/NPY、文档依据、类型/缺失标记、低频合并、前导零、CLI 早期预算初始化、三阶段共用网关、usage unknown 停止、冻结后新进程预测和不向 A 转发 test 文件路径。
- 既有 B 回归 **17/17**、任务发现 **17 通过/2 权限跳过**、自动任务整合 **14/14**。
- 共 **68 项：66 通过、2 跳过**。新增和原有测试均使用合成数据；学校 API transport/key 全为 mock，未读取真实 `.env`、未调用真实 LLM。GitHub PR 操作与课程模型调用是不同流程。
- Python 3.10 语法解析与 `git diff --check` 通过。pytest/Ruff 仍未安装，未绕过 TLS 或修改全局依赖；这不是完整 pytest suite 验收，也不是实际模型理解能力或课程数据性能验证。
- 本地证据：`.cache/b-three-upgrades-4e1ea443/`，其中 `test_b_llm.final.log` 是最终针对新模式及冻结后上下文隔离的复核。原三组回归路径未改变，结果分别留存。机器可读结果见 `b-llm-validation-results.json`。

实现、运行命令、预算编号要求与当前输入边界见 [B 三项升级](b-llm-upgrades.md)。

## B 第一轮优化补充（2026-11-14）

本节是克隆仓库后的本地增量验证，不覆盖下方历史结果。范围与接口见 [B 优化交接](b-improvements.md)，机器可读结果见 [b-validation-results.json](b-validation-results.json)。

- 新增 B 回归 **17 项全部通过**：可选 test、原始/多列 ID、来源行号、缓存版本、混合特征、训练画像和冻结后新进程预测。独立预测类别及概率与 C 验证预测一致（atol=1e-12）。
- 原发现器 **19 项：17 通过、2 跳过**；跳过原因是 Windows 未授予符号链接权限，测试会明确说明，不修改系统权限。
- 原自动任务整合 **14 项全部通过**：表格、纯文本、图像、指标、冻结后 test、预算复用和 mock 学校 API。
- 合计 **50 项，48 通过、2 跳过**。使用标准库 unittest，未运行完整 pytest/Ruff；本机缺少这两项工具，未继续 TLS 排查或修改全局依赖。
- 修改代码通过 Python 3.10 语法检查、空白检查与 `git diff --check`。日志保留于 `.cache/b-opt-validation-5782323e/`。
- 全部输入为合成数据；真实 `.env`/key 未读取，真实 LLM 调用为 0。B 的 LLM 语义任务解析、组/时间划分及新文件读取器尚未实现，不能把本轮称为任意输入支持或真实课程性能验收。

更新时间：2026-11-14（Asia/Shanghai）。验证范围是当前工作区的离线实现与合成数据，不代表学校 API 的真实端到端运行。

## 实施范围

- **H0–H2：** `executor_c/model_registry.py` 是模型与参数的规范来源；固定白名单工厂支持 RandomForest、LogisticRegression、ExtraTrees 和 HistGradientBoosting。能力清单由注册表导出，Plan/Manifest/result/bundle 1.0 的表格路径保留。
- **H3–H4：** C 发布受限训练诊断；A 能读到历史诊断；增强搜索可显式开启模型覆盖、patience、失败上限和 session 时间预留。学校请求 timeout 会受 finalize reserve 限制。确认 seed 默认关闭，开启时按 validation 排序候选、使用配置的 seed offset，且不额外调用 LLM。API usage 未知时停止。
- **H5：** B/D/A/main/bundle 已联动 `accuracy`、`f1_macro` 和 `log_loss`；最大化/最小化方向按任务配置选择。
- **H6：** TaskSpec/Plan/Manifest/Bundle 2.0 为 raw text 和 image tensor 提供受限输入路由。文本为 train-only TF-IDF Pipeline；图像为固定 flatten/scale Pipeline；输入大小、列数、文本长度和图像样本大小均有限制。C 不执行 LLM 生成代码。兼容预测路径保留。
- **H7：** 学校网关可选接入 Decimal 任务级跨会话预算账本。发送前预留最坏费用；已知 usage 结算；unknown 保留预留。离线报告生成 leaderboard、best-so-far 和 Markdown。

## 当前环境与验证

记录环境：Python 3.12.3、NumPy 2.5.3、pandas 3.1.1、scikit-learn 1.9.1。项目虚拟环境没有 pytest；系统路径也找不到 pytest。`../.venv/bin/python -m pytest --version` 返回 `No module named pytest`。系统路径与项目环境均没有 Ruff 可执行文件。因此当前工作区改动的 pytest suite 和 Ruff 检查**没有运行**；历史 `verification-results.json` 中 2026-11-12 的旧版测试记录与本次改动分开保留。

本次代码更新后执行：

```bash
../.venv/bin/python -m compileall -q main.py agent_a data_b executor_c evaluation_d scripts
```

编译、JSON 解析/能力注册表一致性、CLI help 和本次涉及 Python/Markdown 文件的行尾空白扫描均通过。另有以下离线 smoke 证据：

- 四种表格模型各自覆盖二分类和三分类的 8 个 C→D 合成实验全部成功；显式增强搜索 8 轮覆盖四模型族。`f1_macro` 与 `log_loss` 的合成 main 循环成功，log loss 按较小值冻结。
- H6 raw-text LogisticRegression 和 image-tensor RandomForest 流程分别通过 B→A→C→D→Bundle 2.0→冻结预测；旧表格 bundle 1.0 也完成冻结预测兼容 smoke。
- 当前追加的确认轮 smoke 在一次基线后执行两个配置 seed offset；历史记录中的 seed 为 42、43、44，确认轮不请求 LLM。
- 当前追加的失败恢复 smoke 通过 mock 让第一轮 API usage 已知并成功训练，第二轮模拟超时和 unknown usage。session 保持 `failed`，不继续训练，summary 指向已评分最佳模型的 `partial_...` bundle。
- H7 mock 网关账本检查了 usage settle、unknown reserve、重新打开账本和报告生成；未接触外网。

## H8 冻结后最终 test

- 新增 `--test-config` 后置阶段：只在 validation 最佳方案被冻结后，调用 `predict_bundle()` 推理并按冻结 bundle 的 primary metric 评分，同时报告 accuracy。test 结果不进 A、history 或选模。`evaluate-test` 子命令可从现有 `final_*` bundle 补评。
- coupon test 配置只保存 `C_test.npy` / `y_test.npy` 的位置和 2537 样本数，不复制原始 test 文件。实现代码先以合成数据验证；完成审查后，再对已冻结的 `final_641b5e45` 独立执行一次最终 test，准确率为 `0.7745368545526212`（77.45%）。这次不训练、不调用真实学校 API，bundle SHA-256 前后相同。
- 该次正式 test 报告为 `sessions/test_evaluation_20261114T030404Z_744eb71e/test_evaluation.json`，预测为 `sessions/test_evaluation_20261114T030404Z_744eb71e/test_predictions.csv`。test 分数没有反馈给 A 或用于模型选择。
- 离线合成冒烟：表格分类 success/failure 两条路径通过，含顺序断言、bundle 不变、standalone 补评及错误标签时保留 final bundle 且不额外生成 partial bundle；raw-text 和 image-tensor 的 final-test 路由各通过一次。
- `compileall` 与 `main.py --help`、`run --help`、`evaluate-test --help` 通过。pytest suite 未运行，因为项目虚拟环境报 `No module named pytest`；Ruff 未安装。新增 pytest 回归用例已提交到 `tests/test_main.py` 和 `tests/test_multimodal_flow.py`，完整自动检查仍需在开发工具可用的环境执行。

自动 pytest 和 Ruff 检查缺席意味着上面这些 smoke 不能代替完整回归。可在网络可用时用 `uv pip install --python ../.venv/bin/python -r requirements-dev.txt` 将开发工具装入现有项目虚拟环境，再按 `docs/c-implementation-handoff.md` 第 15 节执行 pytest 与 Ruff；不要装入系统 Python。当前验证环境无法连接 PyPI，因此这一步没有完成。

## H9 全自动任务目录入口

`run --task <目录>` 自动读取 README/info 和 train/validation，生成 session 内部配置、任务摘要和惰性 test 描述。学校模式默认 20 次调用、20 次实验和每任务 20 元参考价账本；兼容旧 `--config` 入口。自动模式只在 validation 选优冻结后读取 test，终端输出最终指标。

以下标准库 unittest 已实际运行通过，不依赖 pytest：

```bash
../.venv/bin/python -m unittest discover -s tests -p test_task_directory.py -v
../.venv/bin/python -m unittest discover -s tests -p test_task_run.py -v
```

- **19 项发现器测试通过**：NPY 的数值/类别组合、文本/图像、CSV 标签及显式输入列、未知指标、分层 validation 留出、类别映射、元数据一致性、源目录只读、路径边界、test 延后读取、输入变更检测及缓存版本。
- **10 项集成测试通过**：任意改名的 NPY 与 CSV 文本任务、图像和多指标、freeze-before-test 顺序、最终评分失败保留模型、无效任务在网关初始化前失败、旧配置兼容、旧账本复用及冲突停止。学校 CLI 使用真实 SchoolGateway 实现配合假 transport 和合成 key，确认自动默认值及任务摘要接线；没有真实请求。
- 优惠券 NPY 与 Headlines CTR CSV 两个实际任务仅做开发分区发现与内部 TaskSpec 校验；审计明确禁止打开 `test.csv` 和 `*_test.npy`。两者识别成功。优惠券自动预算匹配到原有 `course-project-2026`，没有创建新账本或修改旧账本。
- A 的 10 个现有控制器断言函数通过直接合成执行复核（含新任务摘要有界与注入隔离断言）；这不是完整 pytest 运行。代码编译、3 个 CLI help、配置 JSON 和行尾空白扫描通过。

本阶段未读取真实 `.env` 或 key、未调用学校 API、未重新评估真实 test。完整 pytest suite 和 Ruff 仍未运行；H9 的 29 项及 H10 的 33 项标准库 unittest 和上述离线验证不能替代这两项检查。

## H10 通用配方搜索与 validation 基线

- A 的 run 决策现在可携带白名单 `pipeline`；文本表示随模型/参数贯穿 Plan 2.0、C 训练、history、冻结的 `plan.json` 与独立 `predict_bundle()`。缺失新字段的旧文本计划默认按原共享词表处理。能力清单仍由唯一模型/表示注册表生成。
- 多字段文本默认策略覆盖共享/逐字段独立词表、word unigram/character 2–5 gram，与 LogisticRegression、MultinomialNB、ComplementNB 组成 12 个起始配方。参数细化只在可用预算与实验轮数内继续；完整配方去重防止不同文本表示误认成重复试验。
- B 的训练类别计数用于产生多数类预测基线；`log_loss` 额外使用训练先验概率。D 只在 validation labels 上评分；A 获得有限摘要与方向化 baseline 差值，不接触 labels 或 test。baseline 与 manifest 身份不匹配时拒绝评分。baseline 用于诊断，不硬编码任务专属的方案或成功阈值。
- 自动任务 integration 新增 4 项通用配方用例，验证逐字段信号、12 种文本配方覆盖、LLM 通过不同 pipeline 区分同模型配置、validation baseline 和任务身份检查。连同 19 项 task-directory 测试，共 33 项定向 unittest 全部通过：

```bash
../.venv/bin/python -m unittest tests.test_task_run tests.test_task_directory -v
```

- 在 `Released_Task2-Headlines_CTR` 上执行了离线 train/validation 搜索审计，未启用 test config；Python 文件打开审计会阻止访问原始 `test.csv`。16 个试验成功，覆盖 12 个模型×文本表示组合后进行 LogisticRegression 参数细化；最佳 validation accuracy 为 `0.678723`，训练多数类 baseline 为 `0.511702`。这是开发分区搜索机制检查，不是 test 结果，也不能证明对其他任务的性能保证。产物位于 `/tmp/superagent-h10-headlines-output/`。
- 用户询问通用文本改动是否影响 vehicle 表格任务后，又以当前离线规则规划器对 `in_vehicle_coupon_recommendation` 做 train/validation 回归审计；Python audit hook 拦截 `*_test.npy`/`test.csv` 打开，run 使用人工配置入口且不带 test config。自动识别为表格，4 个现有 dense 模型族都成功；16 轮按 patience 停止，validation accuracy `0.757635`，training-majority baseline `0.568473`，summary 的 `test_evaluation.status` 为 `not_requested`。这说明 dense 路由可用，不是与旧 school-LLM 决策的严格 A/B，也不能推断最终 test 或保证学校 LLM 轨迹相同。产物位于 `/tmp/superagent-h10-vehicle-output/`。
- 额外运行 `unittest discover -s tests -v` 时，33 个定向用例通过；另 11 个 pytest 风格模块因环境缺少 `pytest` 而导入失败（`ModuleNotFoundError`），并非这些测试已通过。项目 `.venv` 同样没有 Ruff；因此完整 pytest 和 Ruff 仍待开发依赖可用的环境运行。没有安装测试工具或更改项目依赖。
- 本阶段未读取真实 `.env`/API key、未调用学校 API、未打开 Headlines `test.csv`，也没有按该任务名称加入模型分支。仍支持任务受现有读入器与分类模态范围约束；通用支持范围不等于对所有未知文件格式或任务都保证高分。

## 预算价格和运行边界

参考 profile `deepseek-v4.1-flash-peak-2026-10-21` 使用官方峰时、未命中缓存标准价 USD 0.30/M 输入和 USD 1.20/M 输出，按 2026-10-21 人民币中间价 6.7192 换算为 RMB 2.015760/M 和 RMB 8.063040/M。来源：[DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/)、[DeepSeek updates](https://api-docs.deepseek.com/updates/)、[CFETS 2026-10-21 central parity](https://www.shibor.sh.cn/chinese/ccprnoticecontent/index.html?searchDate=2026-10-21)。这只是可审计的参考价 profile；尚未证明学校 endpoint 的 provider、model alias、缓存/折扣政策或实际账单与该 profile 一致。

下面的自动入口启用增强搜索和任务预算 ledger；同一训练输入会自动关联原账本。`20` 是费用上限，不是必须花完的目标。LLM 调用数还受 `--llm-max-calls` 限制；A 可根据历史停止，未知 usage 会停止调用并保留预算占用。

```bash
no_proxy=aigc-api.hkust-gz.edu.cn \
../.venv/bin/python main.py run \
  --task ../task/in_vehicle_coupon_recommendation \
  --planner school
```

运行不读取 `test` 来选模型。没有 pytest/Ruff、没有真实 endpoint 身份/账单核验、文本和图像基线不等于任务定制的高性能模型，均为当前明确限制。
