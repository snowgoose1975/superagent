# B 三项升级：任务解析、质量检查、可配置预处理

2026-11-23 更新：本文记录最初三项升级；后续新增读取器、声明式格式转换、混合输入特征融合及原始输入预测，当前能力以 [B 读取与转换说明](b-readers-and-conversion.md) 为准。

最新范围收敛：LLM 质量检查仅建议数值缺失值的 median / mean / most_frequent 填补，输出可省略诊断字段、忽略额外字段，不要求 evidence，一般疑问只记录。下文类型覆盖、低频合并、缺失指示及特殊标记建议属于历史方案，当前质量检查不会应用；原有显式预处理配置能力保留。

读取流程也已统一：CSV/NPY、文本、图片和 CSV 索引混合输入均由初始 LLM 解析文件角色和类型，再由 B 程序选择读取器、生成配置；新增格式不再多调用一轮读取确认。当前语义接口和调用轮次见上方链接文档。

日期：2026-11-20。基于既有 B 数据准备接口增量实现；不新增现成 Agent 服务，不执行 LLM 生成的代码，不替代 A 的模型搜索或 C 的训练实现。

## 改前与改后

| 项目 | 改前 | 显式开启 B LLM 后 |
|---|---|---|
| 文件命名 | 自动入口主要匹配 train.csv / N_train.npy 等名字 | LLM 根据文件清单和任务说明生成路径映射，CSV/NPY 文件可以任意命名并在子目录中 |
| 说明文件 | README / info.json | 任意命名 Markdown/RST；TXT/JSON 通过 --task-doc 明确指定 |
| 质量判断 | 缺失率、类型、基数等程序统计 | LLM 结合说明解释统计；保留 findings、原文依据和未解决问题 |
| 预处理 | 中位数填补和固定 One-hot 等默认配置 | 可选数值填补策略、缺失指示、低频类别合并及有证据的数值缺失标记；类型覆盖需文档依据 |
| 调用与计费 | 学校模式主要由 A 调用 | B 解析、B 质量检查、A 规划共用 D 的同一网关、次数限制和持久任务账本 |

默认仍是离线规则模式。选择 LLM 模式需要获准的模型配置，所有调用必须计费；本功能测试只使用 mock transport 和合成 key，没有真实模型质量或费用验收。

## 如何开启

从项目目录执行：

```bash
python main.py run --task ../task/my_task --planner school --b-analysis llm --task-budget-id course-task-03 --llm-max-calls 20
```

TXT/JSON 说明不是按扩展名直接自动打开，因为它也可能是测试数据。可指定任意名称的说明路径：

```bash
python main.py run --task ../task/my_task --planner school --b-analysis llm --task-doc 任务说明.txt --task-budget-id course-task-03
```

`--task-doc` 可以重复，路径相对于任务目录。原来的 `--task` 和 `--config` 规则入口保持兼容。

`--task-budget-id` 必须是这项任务稳定复用的账本编号；续跑、改名或移动目录不能改编号重置费用。解析发生在训练文件已知之前，因此 LLM 模式不能等解析完才初始化预算。若发现同训练输入已关联其他账本，程序停止后续调用与训练；已发生的解析费用仍保留在当次指定账本，必须核对，不能当作零或自动转移/清账。程序化调用也要求注入带持久任务账本的 D 网关。

一次正常 B 分析需要 2 次调用（layout、quality），A 规划另需调用。JSON/结构校验失败可限次纠正一次；网络错误、超时、usage 不明不重试。预算或调用上限不足时停止并保留失败与费用记录。

## 新接口

```python
from data_b.llm_analysis import resolve_layout, review_quality

resolution = resolve_layout(task_dir, gateway, audit_dir, document_paths=None)
discovered = discover_task_directory(task_dir, staging_dir, resolution=resolution)
spec, quality = review_quality(discovered['task_spec'], gateway, audit_dir, resolution['documents'])
prepared = prepare_data(spec, data_root)
```

主入口额外支持：

```python
run_project(..., planner_mode='school', gateway=gateway,
            task_directory=task_dir, b_analysis='llm', task_documents=None)
```

gateway_factory 在此显式模式会先收到 `spec=None`；CLI 已适配。规则模式仍在任务识别完成后初始化网关。所有失败走已有 summary 与 D metadata，不会在 B 失败后偷偷退回规则流程继续花费。

## 1. 通用任务解析

文件清单支持嵌套目录，最多 256 个可见文件，不跟随目录符号链接。先提供文件名和限定长度的说明，不能把任意数据内容作为“说明”自动读取。默认自动读取非 test 命名的 Markdown/RST；其他纯文本说明由调用方显式指定。

LLM 返回声明式 layout：格式、各分区特征/标签路径、标签列、ID、指标、可选特征声明、证据及未解决项。CSV 支持嵌入标签，NPY 支持独立标签和单特征数组或 N+C 两组。当前没有 Parquet、图片目录、任意混合文件格式或无标签 test 提交适配器。

程序检查：

- 路径必须属于目录清单、无越界/隐藏路径；分区不能复用同一文件。
- 说明引用必须是确切原文；分区文件路径、目标列和指标必须有对应证据。
- 格式、指标和字段受已有读取器/TaskSpec 能力约束；未知或冲突信息不能静默变成 accuracy 或别的标签。
- 仅训练/验证进入内容读取，test 保持惰性 descriptor，冻结后使用相同 reader。
- CSV 数值候选在新 LLM 路径保留词法表示，避免类型修正前把城市码 `001` 提前压成数值 `1`。真正数值转换仍在训练处理器中执行。

证据字符串校验能拒绝虚构引用，但不构成对自然语言语义的形式化证明。实际模型对陌生说明的理解仍需真实 API 和更多合成变体验收，不宣称支持任何输入。

## 2. 数据质量检查

程序先计算训练统计，包括缺失、唯一值、低频取值数量、常量列、重复特征行和文本/图像统计。LLM 只接收有界统计与说明，不接收原始训练行、验证标签或 test 内容。

quality 的固定字段：`findings`、`type_overrides`、`preprocessing`、`evidence`、`unresolved_questions`。诊断与配置保存在 `sessions/<id>/b_analysis/`，原始 API 请求/响应正文不写进 D 的账本。

类型修正需要对应字段的文档原文；特殊缺失标记需要字段和值的原文，不能因为某个数值“看着异常”就删掉它。未知字段、修改标签/指标/划分/样本行、任意 Python 组件均被拒绝。文本/图像任务只输出质量诊断，不在 B 重写 C 的流水线。

## 3. 可配置预处理

TaskSpec 1.0 增加可选字段 `preprocessing`，老配置省略时保持默认行为：

```json
{
  "numeric_imputation": "median",
  "add_missing_indicator": true,
  "categorical_min_frequency": 3,
  "missing_values": {"age": [-999]}
}
```

- 数值填补允许 median / mean / most_frequent。
- 缺失指示在训练时学习缺失列；不会用验证集决定增加哪些列。
- 低频阈值是 2–1000 的整数。频次只来自训练集；存在 infrequent 分组时，新类别进入该组，否则按 One-hot 的兼容行为处理。
- 特殊缺失标记首版仅支持数值特征，每列最多 8 个。预处理器先将标记转为缺失，再填补。
- 配置和修正后的类型参与不可变版本签名，随 fitted preprocessor 保存，冻结推理完整复用；不合并验证集重训。

这些是一次任务准备时的有界配置。若要比较多种 B 处理配方，需要 A 调度不同数据版本，B 不独立搜索选优。

## 示例

任务目录：

```text
使用说明.md
records/development.csv
records/holdout.csv
records/sealed.csv
```

说明明确指定各文件用途、`clicked` 是标签、accuracy 是指标、`city_code` 是类别、`age=-999` 表示未知。旧规则入口找不到 train.csv；新模式先生成文件映射，再识别城市码语义、验证缺失标记，并按配置填补、增加缺失指示和合并低频类别。数据交给原有 C/D，sealed.csv 在 B 阶段保持未读取。

回归使用真实 D 网关实现配合假 transport（不是实际 API），验证总账结算、usage unknown 立即停止、最多一次纠正、任意 CSV/NPY 名称、TXT 显式说明、标签/指标保护、前导零、训练独立拟合、CLI 共用一次网关初始化，以及 C 训练后新进程冻结预测。
