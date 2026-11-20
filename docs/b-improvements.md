# B 模块第一轮优化与交接

本文件保留第一轮的范围记录。2026-11-20 新增的 LLM 解析、质量检查和可配置预处理见 [B 三项升级](b-llm-upgrades.md)，下文“尚未实现”的历史边界不表示该新功能不存在。

日期：2026-11-14。基于仓库提交 `3d6c2c0` 的本地增量修改。没有提交/push，没有读取真实 `.env` 或调用真实模型 API。测试结果以本节验证记录为准。

## 本轮行为

1. 自动发现仅要求 train；validation 可选，缺少时保持原有分层留出；test 可选。没有 test 时仍搜索、评分和冻结，summary 标记 `test_evaluation.status=not_requested`，不伪造 test 结果。
2. CSV 原始单列 ID 保持原样，包括前导零。多列 ID 使用 JSON 数组组合，避免拼接冲突。无 ID 的 CSV/NPY 使用稳定来源分区与原始行号。内部留出后仍能追溯原训练文件和行号。
3. 同一分区 ID 为空或重复时拒绝。不同官方分区的 ID 可以是各自局部编号，发现报告记录 `overlapping_partition_ids`，不据此静默删除或重新划分数据。该值不是实体泄漏已被排除的证明。
4. 混合表格分别处理 numeric/categorical/text，不再因为一个文本字段就把所有列交给文本向量器。纯文本任务继续走原 TaskSpec 2.0 的稀疏 TF-IDF 注册流程，图像张量流程保留。
5. `info.json.feature_types` 可显式声明全部输入列的 numeric/categorical/text 类型，优先于启发式推断；声明不完整或类型非法时拒绝，不静默猜测。
6. 表格画像增加训练缺失率、常量列、基数、重复特征行与文本长度分布；纯文本增加长度/空文本统计，图像增加训练像素范围和均值。全部基于训练数据，画像不含原始行、ID 或验证标签。
7. A 的 LLM 提示词接收有限数量的统计摘要。详细字段写到 profile；正常提示词最多取 8 列，缩略时取 4 列。所有真实 LLM 调用仍经 D，未新增第二套网关。
8. 自动版本包含相关 B 准备代码摘要，数据或处理代码变化得到新版本；预算身份仍只取原始训练输入内容，不能借改代码或版本重置账本。

## 输入与返回接口

主接口保持：

```python
discover_task_directory(task_dir, workspace_dir, ...) -> dict
prepare_data(task_spec, data_root) -> dict
```

`discover_task_directory` 的 `test_descriptor_path` 现在允许为 `None`。整合入口已适配。`prepare_data` 成功返回新增 `sample_index_path`，其余 C/D 路径及失败语义保留。

TaskSpec 1.0/2.0 的 `partitions.train/validation` 增加可选字段：

```json
{
  "features_path": "X_train.npy",
  "labels_path": "y_train.npy",
  "n_samples": 100,
  "sample_ids_path": "sample_index_train.csv"
}
```

旧任务省略该字段仍生成原有 `train_...` / `validation_...` ID。指定时路径必须在 source_root 内；其内容和准备代码都进入冻结签名，修改索引后不能原地复用旧版本。

索引为 UTF-8 CSV，至少含唯一非空字符串 `sample_id`，可选 `source_file,source_row,original_id`。目录发现器会输出全部四列。`source_row` 是**从零开始的数据行/数组样本位置，不包含 CSV 表头**。source_file 只是来源记录，不会被索引读取器作为可执行路径再打开。

数据包中 `metadata/sample_index.csv` 合并训练和验证来源，增加 `split` 列。该文件属于版本校验产物。C 清单仍严格使用原有字段，验证 ID 通过原来的 `validation.sample_ids_path` 传给 C；验证标签只交 D。

最终 test 冻结后物化时保存 `sample_index_test.csv`，摘要记录在 `test_source.json`。为兼容 D 当前最终评价器，test 预测仍用 `test_...` 顺序编号；需要原始 ID 的提交程序按此索引恢复，不应重新排序猜测。

## 混合表格的有界基线

- numeric：训练中位数填补。
- categorical：训练缺失填补与 OneHotEncoder，未知验证类别忽略。
- text：逐字段训练 TF-IDF，每字段最多 512 个特征；总特征预算不足时均分剩余额度。空白或无词字符输入使用稳定占位表示。
- 输出仍是 C v1 支持的有限二维稠密 float64，最多 4096 特征。训练估算与验证转换超过 256 MiB 时明确失败；这是输出矩阵边界，不是整个进程的硬内存限制。
- 全部学习步骤只拟合训练行，预处理器随 bundle 冻结，独立预测只 transform。

这不是混合模态深度融合或动态代码生成；纯文本任务不强制稠密化。图像目录、Parquet/JSONL 任务发现、按组/时间留出、B 的 LLM 语义解析仍未实现。无验证分区时不要把随机分层留出当成适合所有实体/时间数据的策略。

## 验证

运行合成数据回归：

```bash
python -m unittest discover -s tests -p test_b_improvements.py -v
python -m unittest discover -s tests -p test_task_directory.py -v
python -m unittest discover -s tests -p test_task_run.py -v
```

测试包括 C 输入契约、原始 ID、多列 ID、留出来源、缓存摘要、训练统计、未知验证词不入词表、填补器不使用验证极值、纯文本与图像兼容、混合表格 C→D→冻结→新进程预测，以及 test 延迟读取。新进程预测的类别和完整概率与 C 验证预测逐行比较，误差上限 1e-12。

最终三组共运行 **50 项，48 项通过、2 项跳过**：新增 B 回归 17/17、原发现器 17 通过与 2 个 Windows 符号链接权限跳过、原整合回归 14/14。Python 3.10 语法检查与 `git diff --check` 通过。机器可读结果和源码摘要见 `b-validation-results.json`；日志保存在 `.cache/b-opt-validation-5782323e/`。

本机未安装 pytest/Ruff；上一轮工具安装因 TLS 错误中止，本轮不继续 TLS 排查或修改全局依赖。标准库测试不是完整 pytest/Ruff 验收，真实学校 API 和课程任务性能也不在本轮验证范围内。
