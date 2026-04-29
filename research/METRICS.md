# 指标语义映射

最后更新：2026-04-28

## 目的

当前实验批次已经结束，指标命名和 per-client 轮级指标导出已经开始按语义化名称迁移。

这份文档只负责记录：

- 当前 CSV / WandB / 日志中的旧名字
- 它们在代码里的真实计算方式
- 更符合语义、后续可考虑采用的新名字
- 当前新代码中的落盘字段

旧 CSV 仍通过汇总脚本兼容读取，避免历史结果失效。

## 总原则

- 新实验使用语义化字段名
- 不改写已有历史 CSV
- 汇总脚本需要兼容旧 CSV 字段名
- per-client 指标单独写入 `per_client_round_metrics.csv`

## 指标映射表

### 1. `server_test_acc`

- 当前名字：`server_test_acc`
- 当前真实含义：服务器模型在测试集上的 overall accuracy
- 计算方式：标准分类 accuracy
- 是否有歧义：基本没有
- 建议新名字：保留不变

### 2. `server_test_balanced_acc`

- 当前名字：`server_test_balanced_acc`
- 当前真实含义：服务器测试集上的 macro recall
- 计算方式：
  - 先按类别计算 recall
  - 再对所有类别取平均
- 代码来源：
  - `common.get_metrics()` 中的 `recall_mean`
  - `server.py` 把它写入 `server_test_balanced_acc`
- 是否有歧义：有
  - 名字像 `balanced accuracy`
  - 实际实现是 `macro recall`
- 新名字：`server_test_macro_recall`

### 3. `server_test_macro_f1`

- 当前名字：`server_test_macro_f1`
- 当前真实含义：服务器测试集上的 macro F1
- 计算方式：各类 F1 的平均
- 是否有歧义：基本没有
- 建议新名字：保留不变

### 4. `global_disamb_acc`

- 当前名字：`global_disamb_acc`
- 当前真实含义：所有客户端的 q-vector 消歧 accuracy 的简单平均
- 计算方式：
  - 每个 client 用 `argmax(q)` 得到消歧预测标签
  - 与该 client 的真实训练标签比较，得到 client-level disambiguation accuracy
  - 服务器再对所有 client 的该指标取平均
- 是否有歧义：有
  - `global` 容易让人误解为“全局样本统一计算”
  - 实际上是 `mean over clients`
- 新名字：`mean_client_disamb_q_acc`

### 5. `global_disamb_balanced_acc`

- 当前名字：`global_disamb_balanced_acc`
- 当前真实含义：所有客户端的 q-vector 消歧 macro recall 的简单平均
- 计算方式：
  - 每个 client 先计算 q-vector 消歧混淆矩阵
  - `client.py` 中把 `balanced_acc` 直接设为 `recall_mean`
  - 服务器再对所有 client 的该值取平均
- 是否有歧义：有，且歧义最大
  - `global` 不是真正 global
  - `balanced_acc` 实际实现是 `macro recall`
- 新名字：`mean_client_disamb_q_macro_recall`

### 6. `global_disamb_macro_f1`

- 当前名字：`global_disamb_macro_f1`
- 当前真实含义：所有客户端的 q-vector 消歧 macro F1 的简单平均
- 计算方式：先算每个 client 的 macro F1，再对 client 取平均
- 是否有歧义：有
  - 主要是 `global` 一词不准确
- 新名字：`mean_client_disamb_q_macro_f1`

### 7. `client_disamb_std`

- 当前名字：`client_disamb_std`
- 当前真实含义：各客户端 q-vector 消歧 macro recall 的标准差
- 计算方式：
  - 取每个 client 的 `balanced_acc`（实际是 macro recall）
  - 再对这些 client-level 值计算标准差
- 是否有歧义：有
  - 没写清是对什么指标求标准差
  - 也没体现它基于 q-vector
- 新名字：`std_client_disamb_q_macro_recall`

### 8. `client_test_acc_mean`

- 当前名字：`client_test_acc_mean`
- 当前真实含义：所有客户端个性化测试精度的平均值
- 计算方式：
  - 每个 client 在自己的 local test set 上测试
  - 服务器对所有 client test accuracy 取平均
- 是否有歧义：中等
  - 没显式体现这是 personalized / local test
- 新名字：`mean_client_personalized_test_acc`

### 9. `client_test_acc_std`

- 当前名字：`client_test_acc_std`
- 当前真实含义：所有客户端个性化测试精度的标准差
- 计算方式：对所有 client local test accuracy 求标准差
- 是否有歧义：中等
- 新名字：`std_client_personalized_test_acc`

### 9.1. `min_client_personalized_test_acc`

- 当前名字：`min_client_personalized_test_acc`
- 当前真实含义：所有客户端个性化测试精度中的最低值
- 计算方式：对所有 client local test accuracy 取最小值
- 用途：观察最弱客户端是否被方法伤害或改善

### 9.2. `p10_client_personalized_test_acc`

- 当前名字：`p10_client_personalized_test_acc`
- 当前真实含义：所有客户端个性化测试精度的第 10 百分位数
- 计算方式：对所有 client local test accuracy 计算 10th percentile
- 用途：观察低表现客户端群体，而不只看均值

### 10. `client_vote_pseudo_acc_mean`

- 当前名字：`client_vote_pseudo_acc_mean`
- 当前真实含义：被选中客户端在本轮训练中，vote pseudo labels 的平均准确率
- 计算方式：
  - 只统计 selected clients
  - 从这些 client 的 `last_train_metrics` 中读取 `vote_pseudo_acc`
  - 再取平均
- 是否有歧义：有
  - 名字没体现是 selected clients
  - 名字没体现是 per-round training metric
- 新名字：`mean_selected_client_vote_pseudo_acc`

### 11. `client_vote_confidence_mean`

- 当前名字：`client_vote_confidence_mean`
- 当前真实含义：被选中客户端在本轮训练中，vote pseudo labels 的平均置信度
- 计算方式：
  - 只统计 selected clients
  - 读取每个 client 的 `vote_confidence`
  - 再取平均
- 是否有歧义：有
  - 名字没体现 selected clients
- 新名字：`mean_selected_client_vote_confidence`

### 12. `mean_selected_client_vote_confidence_p10` / `p50` / `p90`

- 当前名字：`mean_selected_client_vote_confidence_p10`、`mean_selected_client_vote_confidence_p50`、`mean_selected_client_vote_confidence_p90`
- 当前真实含义：被选中客户端在本轮训练中，vote pseudo label 置信度分布的客户端级分位数均值
- 计算方式：
  - 每个 selected client 收集本轮全部 vote 样本的 `max(vote_soft_label)`
  - 对该 client 的置信度列表分别取 10/50/90 分位数
  - server 再对 selected clients 的这些分位数做简单平均
- 用途：区分“均值高”与“只有少量高置信样本拉高均值”

### 13. `mean_selected_client_vote_high_conf_error_rate`

- 当前名字：`mean_selected_client_vote_high_conf_error_rate`
- 当前真实含义：被选中客户端中，高置信 vote 样本的错误率均值
- 计算方式：
  - 高置信阈值固定为 `0.8`
  - 对每个 selected client，只在 `vote_confidence >= 0.8` 的样本中统计
  - 错误率 = `high_conf_wrong_count / high_conf_sample_count`
  - 若该 client 没有高置信样本，则记为 `0`
  - server 再对 selected clients 的该比率做简单平均
- 用途：定位“高置信但错”的危险 bad case

### 14. `mean_selected_client_vote_low_conf_correct_rate`

- 当前名字：`mean_selected_client_vote_low_conf_correct_rate`
- 当前真实含义：被选中客户端中，低置信 vote 样本的正确率均值
- 计算方式：
  - 低置信阈值固定为 `0.5`
  - 对每个 selected client，只在 `vote_confidence <= 0.5` 的样本中统计
  - 正确率 = `low_conf_correct_count / low_conf_sample_count`
  - 若该 client 没有低置信样本，则记为 `0`
  - server 再对 selected clients 的该比率做简单平均
- 用途：定位“低置信但其实正确”的保守 corner case

### 15. `mean_selected_client_vote_sample_coverage`

- 当前名字：`mean_selected_client_vote_sample_coverage`
- 当前真实含义：被选中客户端中，本轮被 vote 机制处理的样本覆盖率均值
- 计算方式：
  - 对每个 selected client，`vote_sample_coverage = vote_samples / len(local_train_dataset)`
  - 当前实现里，若启用 vote，通常近似为本地 epoch 数倍覆盖后的统计口径；该值主要用于确认 vote 是否实际参与训练
- 用途：区分“vote 没起作用”和“vote 起作用但质量差”

### 16. `mean_selected_client_candidate_size_mean` / `p90` / `ambiguity_rate`

- 当前名字：`mean_selected_client_candidate_size_mean`、`mean_selected_client_candidate_size_p90`、`mean_selected_client_candidate_ambiguity_rate`
- 当前真实含义：被选中客户端的候选标签难度统计均值
- 计算方式：
  - 对每个 client 的全部训练样本，先计算 `candidate_size = 候选标签数`
  - `candidate_size_mean`：该 client 所有样本的候选集大小均值
  - `candidate_size_p90`：该 client 候选集大小的 90 分位数
  - `candidate_ambiguity_rate`：`candidate_size > 1` 的样本占比
  - server 再对 selected clients 的这些 client-level 统计做简单平均
- 用途：衡量某个 client 或某轮所处的数据歧义强度

## 新增 per-client 轮级指标

新实验每轮额外写入：

- 文件：`per_client_round_metrics.csv`
- 粒度：每个 `round`、每个 `client_id` 一行
- 用途：支持统计最终 10 轮每个客户端的个性化测试表现与 q-vector 消歧表现

字段：

- `round`
- `client_id`
- `selected`
- `client_disamb_q_acc`
- `client_disamb_q_macro_recall`
- `client_disamb_q_macro_precision`
- `client_disamb_q_macro_f1`
- `client_personalized_test_acc`
- `client_personalized_test_macro_recall`
- `client_personalized_test_macro_precision`
- `client_personalized_test_macro_f1`
- `client_vote_confidence_mean`
- `client_vote_confidence_p10`
- `client_vote_confidence_p50`
- `client_vote_confidence_p90`
- `client_vote_high_conf_error_rate`
- `client_vote_low_conf_correct_rate`
- `client_vote_sample_coverage`
- `client_candidate_size_mean`
- `client_candidate_size_std`
- `client_candidate_size_p50`
- `client_candidate_size_p90`
- `client_candidate_ambiguity_rate`

其中：

- `client_vote_*` 只依赖该 client 本轮训练阶段产生的 vote 伪标签
- `client_candidate_*` 是该 client 的训练集静态统计，但为了和轮级表现对齐，也按轮重复写入

## 新增 bad-case 样本导出

新实验每个 client 额外写入：

- 文件：`client_{client_id}_vote_bad_cases.csv`
- 路径：`csv_logs/{exp_id}_{exp_name}/`
- 粒度：每个触发 bad-case / corner-case 规则的训练样本一行

字段：

- `Round`
- `Epoch`
- `Index`
- `TrueLabel`
- `Candidates`
- `CandidateSize`
- `QPseudoLabel`
- `VotePseudoLabel`
- `VoteConfidence`
- `ModelPred`
- `IsVoteCorrect`
- `IsModelCorrect`
- `CaseType`

当前导出规则：

- `high_conf_wrong`
  - `VoteConfidence >= 0.8`
  - 且 `VotePseudoLabel != TrueLabel`
- `low_conf_correct`
  - `VoteConfidence <= 0.5`
  - 且 `VotePseudoLabel == TrueLabel`

用途：

- `high_conf_wrong`：定位最危险的错误监督样本
- `low_conf_correct`：定位投票其实有效但置信度保守的 corner case

## 新增混淆矩阵导出

新实验每轮额外写入混淆矩阵 CSV：

- 目录：`csv_logs/{exp_id}_{exp_name}/confusion_matrices/`
- server 测试集：`round_{round}_server.csv`
- client 个性化测试集：`round_{round}_client_{client_id}_personalized_test.csv`

CSV 行表示真实类别，列表示预测类别。日志中同时记录每个矩阵路径，以及 recall 最低的若干类别和它们最常被误分到的类别。

## 实验命名规则

新实验的 WandB name 使用固定结构，避免只靠随机 `exp_id` 区分 run：

`fedpll_{dataset}_{model}_{vote}_s{seed}_r{rounds}_le{local_epochs}_noise{noise}_{partition}_{optimizer}_lr{lr}_{loss}`

示例：

- `fedpll_svhn_resnet18_vote0_s42_r60_le5_noise0.3_noniid_p0.7_alpha0.5_adam_lr0.001_lc`
- `fedpll_svhn_resnet18_vote10_s3407_r60_le5_noise0.3_noniid_p0.7_alpha0.5_adam_lr0.001_lc`

WandB group 使用：

- `vote_{dataset}_{model}_noise{noise}`

WandB tags 包含：

- `seed:{seed}`
- `vote:{vote_num_models}`
- `rounds:{rounds}`
- `partition:{partition}`

## 新增汇总脚本

- `scripts/summarize_vote_results.py`
  - 汇总新 `round_metrics.csv`
  - 兼容旧 CSV 字段名
  - 输出弱客户端指标：`min_client_personalized_test_acc`、`p10_client_personalized_test_acc`
- `scripts/summarize_per_client_metrics.py`
  - 读取 `per_client_round_metrics.csv`
  - 默认统计最终 10 轮
  - 输出 `per_client_final10_run_summary.csv`
  - 输出 `per_client_final10_group_summary.csv`

## 特别说明：`disambiguation` 指标与测试精度不是一回事

### q-vector 消歧指标

- 预测标签来源：`argmax(q)`
- 数据来源：客户端训练集样本
- 作用：衡量 partial-label 消歧效果

### client 测试指标

- 预测标签来源：`argmax(model logits)`
- 数据来源：每个 client 的 local test set
- 作用：衡量个性化联邦学习场景下的客户端泛化表现

### server 测试指标

- 预测标签来源：`argmax(global model logits)`
- 数据来源：服务器测试集
- 作用：衡量全局模型表现

## 已执行迁移

- `server.py` 中的 `round_metrics.csv` 表头已改为语义化名称
- WandB 指标名称已改为语义化名称
- `scripts/summarize_vote_results.py` 已兼容新旧字段
- 新增 `scripts/summarize_per_client_metrics.py`
- 新增 `per_client_round_metrics.csv` 轮级客户端指标导出

## 后续计划

- 用新日志格式重新运行需要 per-client final-10 分析的实验。
- 若需要纳入旧实验，需要从 W&B 历史文件恢复 per-client 每轮记录；旧 `round_metrics.csv` 本身无法恢复该信息。
