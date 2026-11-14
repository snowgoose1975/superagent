# 从任务目录全自动运行

统一入口为 `main.py run --task <目录>`。选择学校模型时加 `--planner school`；任务相关的 schema、列名、标签、指标、样本数及 test 配置不由操作者填写。

```bash
no_proxy=aigc-api.hkust-gz.edu.cn \
../.venv/bin/python main.py run --task ../task/in_vehicle_coupon_recommendation --planner school

no_proxy=aigc-api.hkust-gz.edu.cn \
../.venv/bin/python main.py run --task ../task/Released_Task2-Headlines_CTR --planner school
```

程序只根据输入格式、任务说明和训练数据特征选择数据路由，不按任务名选择模型。自动任务默认启用 `search-defaults.json`、最多 20 次实验；文本任务在轮数足够时优先覆盖三个兼容模型族和共享/逐字段 word/character 表示，其他表示先覆盖可用模型族，再进行验证集驱动的细化。学校模式默认最多 20 次模型调用、单会话 20 元上限和按训练输入内容识别的每任务预算账本。账本采用现有官方参考价 profile，其与学校端实际收费的差异仍见预算文档。省略学校模式时运行离线规划。

同一训练输入换目录名或再次启动不会重置预算；如果旧配置运行已经使用任务账本，会根据训练文件摘要自动关联原账本。发现多个相冲突的旧账本时停止，避免重新从零计算。

支持输入：

- NPY 数组包：`C_train.npy` 类别特征、`N_train.npy` 数值特征（可组合），或 `X_train.npy` 通用特征；对应 `y_train.npy` 和 val/valid/validation 分区。字符串长文本可路由到文本 Pipeline，图像数值张量路由到图像 Pipeline。
- CSV：`train.csv`、`val.csv` 或其 validation 别名、`test.csv`。标签列从 README/info 的目标声明或唯一常见标签列识别；文本列与数值/类别列根据训练数据判断。文本列进入 TF-IDF Pipeline，标签不进入特征。
- README 文件名大小写不敏感；`info.json` 可选。说明中的受支持指标自动归一为 accuracy、macro-F1 或 log-loss。缺少明确指标时会记录采用的默认值，明确要求未知指标时会停止并记录原因。

CSV 中 README/info 明确列出的输入列优先于自动猜测，额外列记录在发现报告中并排除；目标列和 ID 不能混入输入。没有 validation 时，程序只从 train 中按类别划出 20% 留出集，类别样本不足则明确报错。

发现阶段只读取说明、训练和验证分区；类别全集只从 train 建立。test CSV/NPY 不打开、不转换、不计算内容摘要。生成的 `task_input/`、`task.snapshot.json`、`task.discovery.json` 和有界 `task_context.json` 记录输入解释与来源；README 摘要作为任务数据传给 A，不能更改执行权限或运行任意代码。

搜索完成后，程序先冻结 validation 最优模型，再依照先前确定的读取规则转换 test 并评分。`test_evaluation.json` 记录原始输入与派生输入摘要，`test_predictions.csv` 记录顺序一致的预测；test 分数不反馈给 A。最终评分失败保留 frozen bundle。

主程序同时输出由训练类别频率得到的 validation 多数类 baseline 及最终方案相对提升；这项基线不会读取 test，也不会替代真实验证评分。搜索有可能在达到 20 轮或用完费用前按 patience、实验时间或其他硬限制停止；上限不是消费/调用目标。

自动接入仍有输入格式边界：图片目录、任意压缩包、未知表格布局、歧义目标列或不支持的评价指标会在日志中明确报告失败，不会在无人运行过程中要求操作者写配置或偷偷换成别的目标。新格式需要扩展通用读取器，不能按某个任务名定制分支。
