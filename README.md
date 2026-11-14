# SuperAgent Team v2

将老师的完整任务目录放在项目同级的 `task/` 下，从本项目目录运行：

```bash
no_proxy=aigc-api.hkust-gz.edu.cn \
.venv/bin/python main.py run \
  --task ../task/in_vehicle_coupon_recommendation \
  --planner school
```

换任务只需替换 `--task` 后的目录。程序自动读取 README/info 和训练数据，识别特征、标签、模态及指标，生成 session 内部契约并搜索；按 validation 选优冻结后再读 test、打印最终指标。无需编写或修改数据集 JSON。学校模式沿用项目 `.env` 的 key，自动任务默认最多 20 次模型调用、20 次实验和每任务 20 元参考价账本；同一训练输入重复启动复用预算。省略 `--planner school` 可离线运行相同数据流程。

目前自动入口支持课程现有的 NPY 数组包和 CSV 分区（表格/纯文本），以及 NPY 图像张量。增强搜索按识别出的模态和训练画像评估注册模型族；多字段文本可比较共享/逐字段 word/character TF-IDF 以及三种稀疏文本分类器。程序还会报告训练多数类 validation 基线，便于判断最终方案是否优于简单参考。模型与预处理依然由有限注册表实现，A 负责选择配方并读取 validation 反馈，不会生成可执行代码。它会在质量、patience、时间、调用数、实验数或费用上限允许的范围内探索；20 元是上限，不要求花完。说明提取规则、输入边界和输出见 [`docs/task-directory.md`](docs/task-directory.md)。下方 `--config` 命令保留为开发兼容入口。

Table tasks keep TaskSpec 1.0. Bounded text and image-array classification use TaskSpec 2.0; see [`docs/multimodal-tasks.md`](docs/multimodal-tasks.md). For a configured multi-trial offline search, add `--search-config configs/search-defaults.json` to `main.py run`.

这是一个供学生在 VS Code 阅读和运行的四模块整合项目：B 准备训练/验证数据，A 选择模型和参数，C 训练并生成验证预测，D 独立评分，再把真实验证分数交回 A。A 复用同学提供的 planner/decide 逻辑并通过适配器接入；学校模型 API 是可选模式，默认运行不会发送网络请求。

## 先运行离线模式

Linux/WSL 克隆项目后，在项目目录创建独立环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py run --task ../task/<任务目录>
```

在 VS Code 使用 **File → Open Folder** 打开 `git clone` 生成的项目目录。在该目录的 Windows PowerShell 终端中执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py run --config configs/coupon.json
```

不需要激活虚拟环境，也不要修改 PowerShell 执行策略。测试依赖是可选的：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

离线模式使用同学 A 的规则规划器，不需要 API key。它查看训练概况、C 的模型能力和历史验证评分，再返回模型与参数；main 将补齐任务编号、随机种子和时间限制，并把计划交给 C。可以在 `main.py` 调用 `execute(...)` 前设断点，直接检查这个最终计划。

本任务的老师开发分区保存在 `data/raw/in_vehicle_coupon_recommendation/`，包括 train/validation 与 Readme/info；没有复制 test。默认离线政策最多三轮，失败尝试也占一轮并写入历史。

## 选择学校 API 模式

只有需要实际调用学校模型服务时，才显式传 `--planner school`：

```powershell
.\.venv\Scripts\python.exe main.py run --config configs/coupon.json --planner school --llm-config configs/school-api.json --llm-max-calls 1 --llm-budget-rmb 0.10
```

school 模式从项目根目录 `.env` 读取 `SCHOOL_API_KEY`。Linux/WSL 可运行 `cp .env.example .env`；Windows PowerShell 可运行 `Copy-Item .env.example .env`。然后在 `.env` 中填写自己获准使用的 `SCHOOL_API_KEY`。不要共享真实 `.env` 或 API key；`.env` 不进入版本控制或交付包。`configs/school-api.json` 保存服务 URL、模型和开发单价，不含 key；价格只是开发估算依据，尚未核验为课程正式价格。

上述兼容配置命令最多发出一次真实模型请求；本文开头的自动 `--task` 命令默认最多 20 次调用。合规方案触发真实训练和评分。若请求超时、HTTP 失败或供应商 usage 缺失，调用用量记为未知并立即停止，不重试、不回退离线规划继续运行。只有 usage 已知而模型回复内容不合规时，才允许记录后使用规则规划器处理当前决策。summary 会写明规划模式、最后决策来源和真实调用次数。

学校 API 真实响应必须由你在本地终端运行后验证。Codex 的自动验证使用 mock，不会取得你的 key 或发真实请求。调用账本保存在独立的 `sessions/api-<timestamp>-<uuid>/` 会话中，只记录状态、token 和费用估算等元数据，不保存 key、认证头、原始请求或响应正文。这是独立开发会话，不会重置或替代正式任务预算、旧调用账本，也不会把未知 usage 记为零。

详细操作和停止规则见 [`docs/school-api.md`](docs/school-api.md)。预算参数与跨 session 账本见 [`docs/task-budget.md`](docs/task-budget.md)；20 元是上限，不是必须花完的目标。实际验证环境和限制见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。

## 如何阅读

按 [`docs/reading-order.md`](docs/reading-order.md) 从 `main.py` 开始。第一步先看 main 如何构造实验计划并传给 C，然后沿 B → A → C → D → 反馈 A 追完整轮次。接口和替换点见 [`docs/interfaces.md`](docs/interfaces.md)；上游来源与改动边界见 [`docs/provenance.md`](docs/provenance.md)。

在 VS Code 运行 **Python: Select Interpreter**，选择 `.venv\Scripts\python.exe`。离线模式可用 `.vscode/launch.json` 启动调试；需要本机已可用的 VS Code Python 扩展。学校模式从项目根 `.env` 读取 key；项目不会自动安装 VS Code 扩展或更改全局设置。

## 数据与结果

任务搜索只包含 train 和 validation。B 只在 train 上拟合预处理，并另外生成供 D 读取的验证标签；C 只收到训练特征/标签、验证特征和验证样本 ID。A 看到训练概况、任务说明摘要、模型能力和历史验证分数。`--task` 自动包含最终 test 阶段；开发兼容入口 `--config` 仍需显式添加 `--test-config`。程序先按 validation 选出并冻结方案，再读取 test、打印指标，并将报告保存在该 session。test 不进入搜索或评分反馈，原始任务目录只读。

在 Linux/WSL 终端中，学校 API 搜索结束后评估老师提供的优惠券 test 集：

```bash
no_proxy=aigc-api.hkust-gz.edu.cn \
.venv/bin/python main.py run \
  --task ../task/in_vehicle_coupon_recommendation \
  --planner school \
  --llm-config configs/school-api.json \
  --search-config configs/search-defaults.json \
  --llm-max-calls 20 \
  --llm-budget-rmb 20 \
  --billing-profile deepseek-v4.1-flash-peak-2026-10-21 \
  --task-budget-id course-project-2026 \
  --task-budget-rmb 20
```

若 bundle 已经冻结，可以使用 `main.py evaluate-test --bundle <bundle.json> --test-config configs/coupon-test.json` 补跑一次最终评估，不会调用 A、训练或重新选择模型。任务边界和恢复入口说明见 [`docs/final-test.md`](docs/final-test.md)。

项目输出位于 `data/prepared/`、`sessions/`、`runs/` 和 `submission/`。完整 B 版本在源内容与产物摘要一致时复用；原始数据改变时应使用新的 `data_version`。每个主会话、实验和冻结 bundle 都用新编号。失败证据和历史数据会保留，不自动覆盖或清理。

当前选模分数是由 D 根据真实 validation 标签计算的 accuracy。它可以是同学评估的模型质量反馈，但不能称为正式课程成绩。只有实际运行学校 API 的本地会话才算真实 LLM 规划；mock 或离线运行都不是。

验证范围与真实结果见 [`docs/VALIDATION.md`](docs/VALIDATION.md)，示例运行的说明见 [`docs/example-run.md`](docs/example-run.md)。文档不会代替真实运行证据。
