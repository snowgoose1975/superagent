# B 原始数据读取、格式转换与特征融合

本地增量改动，2026-11-23。保留 CSV/NPY 读取分支及 A/C/D 数据接口；不按任务名选择处理方案。

## 数据流

1. 程序先扫描文件清单，不读候选文件正文。新增一轮 LLM 调用，从文件名和目录结构选择最多四个说明或元信息文件；不要求叫 README/info。程序排除明显的数据及密钥文件，检查路径、类型和大小后再读取，形成 documents。显式指定 `--task-doc` 时跳过这一轮。然后 LLM 根据文件清单、documents 和输出结构选择数据角色、读取器和参数；原 CSV/NPY 规则路径继续兼容。
2. 对新增格式，只打开已确认的训练源，提供至多 1600 个字符的片段或训练类别目录名；LLM 确认配置。不会抽样验证/测试内容。
3. 已有读取器优先。平面记录字段改名、非标准分隔符使用 `convert` 声明式转换规则；不再处理嵌套 JSON，也不让 LLM 重写全部样本。
4. B 校验、整理为 NPY/TaskSpec，调用原有预处理；训练拟合只使用训练行。
5. 冻结前不读取 test；冻结后按相同规则转换并评价。原始输入不修改。

## 支持的读取方式

| format | 输入组织 | 标签 |
|---|---|---|
| csv / npy | 原有格式 | 原有协议 |
| tsv | 带表头的 UTF-8 TSV | 指定目标列 |
| text_lines | UTF-8，每行一条文本 | 独立文件，每行一个标签 |
| text_directory | 分区目录/类别/*.txt | 类别目录名 |
| image_directory | 分区目录/类别/*.{jpg,jpeg,png} | 类别目录名 |
| image_csv | CSV 含图片路径、标签、可选其他字段 | 指定目标列 |
| convert | 根级 JSON 记录数组、自定义分隔文本 | 顶层字段映射 |

TSV 使用常规 train/val/test 名称时可离线自动发现。其他格式可用 `info.json` 显式声明 `reader_format`、`reader_options` 和 `partitions`；省略 partitions 时按常规分区名匹配。任意文件名/目录通过原有 `--b-analysis llm` 解析。`--task-doc` 用法不变。

LLM 模式默认先选择说明文件，候选支持 Markdown/RST/TXT/JSON。候选用途不明确时报告歧义；文件大小仍受单文件 8000 字节、合计 9000 字节限制。选中文件的内容只在下一轮解析任务时提供。CSV/NPY、文本、图片和现有 CSV 索引混合输入均采用三轮（找说明、解析文件用途/类型、缺失值建议），不再额外调用 LLM 确认读取配置。显式 --task-doc 可省去找说明一轮。每轮校验失败最多纠正一次，网络错误或未知 usage 停止。所有轮次复用原 D 网关和任务预算。

外部 JSONL 输入及其转换路径已移除。纯文本的内部数据工件仍使用既有 JSONL 编码，这是 B/C 文件协议，不属于外部读取器。

## 提示词与校验约定

任务解析只要求 LLM 识别文件用途、数据类型、标签和指标，不再要求输出 format、reader_options 或 feature_prefixes。程序根据训练源扩展名、训练目录中的文件类型和 feature_types 选择读取器；image_path 字段自动生成图片列配置，其他 numeric/categorical/text 字段自动生成混合读取配置。NPY 可选用 feature_kinds 描述各特征文件的 numeric/categorical/text/image 含义，程序生成内部代号；单数组不提供类型或旧代号不匹配时走现有自动识别。两数组仍需明确哪个是数值、哪个是类别，程序统一排序为 N+C。兼容旧版读取配置。

可省略 id_columns、feature_columns、feature_types、reader_options、unresolved_questions 和不存在的验证/测试分区，程序补默认值；单个 features 文件也可直接返回字符串。额外说明字段不传入执行配置。CSV/NPY 的 reader_options 仍须为 `{}`；NPY 标签仍须给出独立、非空的文件路径。分区路径须逐字匹配 files 中的实际相对路径，不能使用通配符、占位名或新名称；说明中的通配符只用于 evidence，关键文件/标签/指标的原文依据仍校验。

文件角色、路径和原文依据校验通过后，程序仅探测训练源：分隔记录检查最多 65536 字符的片段，识别逗号、制表符、分号或竖线；平面 JSON 按既有限额检查，必要时生成声明式转换规则。文档明确的字段改名由 column_mapping 描述，程序生成 field_paths。验证、测试和冻结预测复用训练时确定的规则，不单独推断分隔符或字段映射。目录读取器按训练类别目录中的 JPG/PNG 或 TXT 扩展名选择；不混读不同类型的类别目录。

校验失败时，纠正上下文包含上一轮失败答复（受上下文预算限制）与具体字段错误，要求重新确认文件角色和类型，而不要求 LLM 生成底层读取参数。最多纠正一次，文件/标签/原文依据检查不放宽。普通 CSV 预算预检查也记录训练特征列明确引用的已有图片文件摘要，与语义图片索引读取保持同一训练输入身份；不读取测试资源。读取器错误属于任务发现错误，预检查不完整时允许进入正式 LLM 解析，正式解析后的数据错误仍停止。

例如 LLM 可以返回以下完整语义配置；B 自动选择 image_csv 并生成图片列、辅助字段等内部设置：

```json
{
  "data_type": "mixed",
  "partitions": {
    "train": {"features": ["train.csv"], "labels": null},
    "validation": null,
    "test": null
  },
  "target_column": "label",
  "id_columns": ["id"],
  "feature_types": {"photo": "image_path", "review": "text", "price": "numeric"},
  "metric": "accuracy",
  "evidence": [
    {"field": "train", "document": "notes.txt", "quote": "Train is train.csv."},
    {"field": "target_column", "document": "notes.txt", "quote": "Target column is label."},
    {"field": "metric", "document": "notes.txt", "quote": "Primary metric accuracy."}
  ],
  "unresolved_questions": []
}
```

上述引文须存在于实际提供的 notes.txt 中。这里的字段类型是路径/内容含义，图片仍由程序解码，不交给 LLM 逐张读取。

质量检查现仅让 LLM 建议数值缺失值填补方式，例如 `{"preprocessing":{"numeric_imputation":"median"}}`。允许 median / mean / most_frequent，兼容大小写和首尾空格。findings 可省略或为字符串；额外字段忽略，一般 unresolved_questions 保留为提示，evidence 不再是本阶段的必需项。缺字段、空配置或无效策略保留已有填补配置（未配置时默认为中位数），不会仅因此再次调用 LLM 或停止训练。

本阶段不应用类型覆盖、低频类别合并、缺失指示或特殊标记推断；有限数值 -999 不会自动视为缺失。已有 TaskSpec 显式预处理配置仍保留，仅覆盖合法的数值填补策略；纯文本、纯图像及无数值列的表格不应用该建议。实际填补仍由 B 预处理器在训练行拟合，其他分区复用。任务解析的文件、标签、划分和指标检查及 D 网络/预算规则保持。非 JSON 或空回复仍按既有有界纠正规则处理，不把网络失败视为有效默认回复。

## 图片与混合输入示例

```json
{
  "metric": "accuracy",
  "reader_format": "image_csv",
  "reader_options": {
    "image_column": "photo",
    "auxiliary_types": {"description": "text", "price": "numeric"}
  },
  "target_column": "label",
  "id_columns": ["id"],
  "partitions": {
    "train": {"features": ["development.csv"], "labels": null},
    "validation": {"features": ["holdout.csv"], "labels": null},
    "test": {"features": ["sealed.csv"], "labels": null}
  }
}
```

CSV 中 photo 是相对于任务根目录的路径；也可显式配置 image_base。JPEG/PNG 按 EXIF 方向解码并统一 RGB，保留方向修正后的原始宽高，不做 resize，也不跳过坏图片。image_size 选项已移除；旧配置需去掉这一项。首张训练图片的形状记录为 decoded_image_shape，其他训练图片、验证集、冻结后测试及原始输入预测均须相同尺寸，否则明确报错。通道转换为 RGB 不改变空间分辨率。

不配置 auxiliary_types 时输出原有图像 TaskSpec 2.0，图片数组保留 `(N,H,W,3)` 原始分辨率。配置后，混合模态也按原尺寸读取，按 HWC 顺序展开全部 `H×W×3` 个 RGB 值，仅做除以 255 的数值缩放，与同一行文本/表格字段拼接；不做 4×4 网格、池化、裁剪或像素下采样。像素列命名为 __image_pixel_0 等。B 的声明式 NPY 列宽和预处理预算允许这些像素列超过普通表格的 4096 列限制，保留全部像素；现有元素数量、文件大小、内存上限仍检查，超限报错，不自动缩图。缺失值建议的 LLM 输入只包含原有文本/表格列，不发送内部像素列。

原有 B 预处理器只在训练行拟合文本 TF-IDF、填补和类别编码，输出原有 dense_numeric/TaskSpec 1.0。像素是数值特征，全部保留并冻结列顺序；A/C 的模型接口不变。冻结后的原始输入预测复用原尺寸和相同像素顺序。

这是基础的特征级融合，不是图文深度模型。当前要求同一图片索引表提供各模态字段；尚不支持任意多张表按 ID 自动 join、音频、视频或端到端图文微调。

## 转换后备示例

原始 JSON 为平面记录数组 `[{"body":"很好用","class":"positive"}]`：

```json
{
  "reader_format": "convert",
  "reader_options": {
    "source_format": "json",
    "field_paths": {"text": ["body"], "label": ["class"]}
  },
  "target_column": "label",
  "feature_columns": ["text"],
  "feature_types": {"text": "text"},
  "metric": "accuracy",
  "partitions": {"train": {"features": ["source.json"], "labels": null}}
}
```

程序根据训练源和初始语义配置选择 convert，处理文档明确的顶层字段改名或分隔符，不增加 LLM 确认轮次；分区、ID、目标字段、指标不允许变更。field_paths 的每个值仅允许包含一个顶层字段名，records_path 不再支持，遇到嵌套对象/数组明确报错。规则不接受 Python、导入、网络或文件写入语句。任意二进制格式、复杂计算和未实现解码器明确失败。

格式不兼容可通过规则适配；保留样本中的缺标签、重复 ID、分区源重用、越界路径、损坏图片等数据错误停止，不调用转换来掩盖。文本/图片内容缺失按下节的显式规则删除并记录。UTF-8、文件/文本/数组规模有界，见 input_readers.py。

## 缺失文本与图片的删行规则

任一输入文本字段为 None、NaN、空字符串或纯空白时，删除整个样本；图片路径为空或指向不存在的文件时同样处理。数值字段缺失继续保留并走既有填补流程。字符串字面量 "None"、"null" 等不自动当作空文本；非法路径、损坏图片和尺寸不一致仍报错。图片数组中整张图片全为缺失值时删除，正常全黑/全零图片有效。

程序同步筛选特征、标签、样本 ID，并保留原始 source_row，避免删除后标签错位或重新编号丢失对应关系。规则用于训练、验证、冻结后测试，以及支持的原始输入预测；没有剩余样本或不足以完成分类/划分时明确失败。筛选不依赖标签值或评分结果。

discovery_report.json 的 sample_filtering 按分区记录 input_count、kept_count、dropped_count、reason_counts 和前 1000 条删除明细。声明的原始数据行数按删除前的数量核对，交给 C/D 的清单按保留样本数生成。冻结后 test_source.json 同时记录原始与保留测试数量，评分只覆盖保留样本，不冒充完整原测试集分数。原始输入预测的 converted_* 目录额外保存 sample_index.csv 和 filtering_report.json，供预测输出按顺序映射回原始样本。

读取限制属于当前实现的资源保护值，不是课程或 B/C 协议强制值：文件、数组元素和密集矩阵分别有大小上限，以防整体加载、像素展开和中间副本耗尽内存。当前超过限制会报错，不自动降分辨率、压缩图片或删掉额外有效样本；本次未调整这些数值。

## 冻结与新数据预测

新格式的 `reader_rules.json` 保存到 staging，随最终模型复制并记录校验和；源代码版本、转换配置也参与数据版本。保持原 `predict.py --features 标准数组.npy` 用法。对无标签原始新数据可使用：

```bash
python predict.py --bundle PATH_TO_BUNDLE/bundle.json \
  --features NEW_DATA/new.csv --raw-input --source-root NEW_DATA \
  --output predictions.csv
```

新文件沿用训练时的列名与图片引用规则，但无需标签。目录模式也允许扁平的无标签图片/TXT 目录。转换出的 X.npy 保留在输出附近的唯一 converted_* 目录，便于复查，不自动删除。读入代码与冻结规则版本不一致时停止。

## 验证与边界

`tests/test_input_readers.py` 包括 TSV 多行字段、TXT+标签、文本目录、不同模式/尺寸图像、图片索引、平面 JSON/自定义分隔转换、拒绝嵌套及外部 JSONL 输入，以及混合输入 C→D→冻结→新进程标准/原始预测。

LLM 用例使用模拟回复，验证训练探测与惰性 test；真实 API 验收需要项目本地配置且照常记录费用。旧 CSV/NPY 测试一并回归，不把合成流程成功等同于真实分类效果。
