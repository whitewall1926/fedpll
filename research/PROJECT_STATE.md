# 项目状态

最后更新：2026-04-28

## 当前分支

- `feature/vote-pseudo-label`

## 当前主问题

这是一个个性化联邦学习实验仓库。

当前主问题不是单纯提升全局模型，而是理解 `vote pseudo label` 为什么有效，以及它带来的本地消歧提升能否在个性化场景下稳定转化为更好的客户端表现，并与 `server test acc` 形成一致收益。

## 当前假设

- 提升可能来自外部伪标签本身。
- 提升可能来自 `vote_num_models` 带来的投票规模变化。
- 提升可能主要来自 `q vector` 更新更加稳定。
- 本地 `disambiguation` 指标提升不一定等价于个性化测试收益提升。
- 个性化收益提升也不一定等价于全局 `server test acc` 提升。

## 当前重点

- 比较 `vote0`、`vote1`、`vote3`、`vote5`、`vote10`。
- 同时跟踪本地消歧指标、`client_test_acc_mean` 和 `server test acc`。
- 结合 `round_metrics.csv` 和最近 commit 历史判断实验进展。
- 新实验开始记录 per-client 轮级指标，支持最终 10 轮客户端级分析。
- local test set 现在按 local train class-count 比例划分，避免只匹配类别集合而不匹配数量分布。

## 当前阶段性结论

- 从当前已汇总的结果看，`vote pseudo label` 能明显提升本地消歧指标和 `client_test_acc_mean`。
- 这说明它在个性化联邦学习语境下，至少对客户端侧表现有明显帮助。
- 但这种提升目前没有稳定转化为更高的 `server test acc`。
- 当前已完成 run 中，`vote0` 的 `server_test_acc` 最高，`vote1` 最差。
- `vote5` 和 `vote10` 在本地指标上明显优于 `vote0`，说明投票规模可能确实在增强本地消歧。
- 当前本地汇总中每个 `vote_tag` 已有 3 个 run，但仍需要用新日志格式重跑以获得 per-client final-10 指标。

## 当前代码脉络

- [main.py](/home/yxf/proj/main.py)：加载配置和数据集，启动 `Server`
- [server.py](/home/yxf/proj/server.py)：联邦轮次、客户端采样、模型聚合、评估与日志
- [client.py](/home/yxf/proj/client.py)：候选标签生成、`q` 更新、vote/prototype/FedSA 逻辑
- [common.py](/home/yxf/proj/common.py)：配置模型、划分、指标、画图
- [model.py](/home/yxf/proj/model.py)：模型定义与工厂函数

## 已知风险

- `common.non_iid_partition()` 可能返回少于 `num_clients` 的子集，但 `server.py` 默认按完整客户端数索引。

## 下一步

- 然后继续深入分析 `client.py`，重点解释 `q` 更新与 `vote pseudo label` 的作用机制。
- 继续积累更多 seed 的完整结果，验证个性化收益和全局收益之间的偏差是否稳定存在。
- 用新日志格式重新运行需要 per-client final-10 分析的 vote 实验。
- 运行单 seed `noise_level × {vote0, vote5, vote10}` 扫描：
  - 脚本：`scripts/run_vote_noise_single_seed.sh`
  - 默认设定：`seed=42`、`rounds=30`、`noise_level={0.3,0.4,0.5,0.6}`
- 跑完后优先比较弱客户端是否被 `vote5/10` 修复：
  - 脚本：`scripts/compare_weak_clients.py`
  - 重点观察 `client 2`、`client 5`、`client 1`
  - 核心指标：`client_personalized_test_acc`、`client_disamb_q_macro_f1`
- 判断高噪声下多模型投票是否更鲁棒：
  - 主指标：`mean_client_disamb_q_macro_recall`
  - 辅助指标：`mean_client_disamb_q_acc`、`mean_client_personalized_test_acc`
  - 诊断指标：`mean_selected_client_vote_high_conf_error_rate`
- 运行 `scripts/summarize_per_client_metrics.py`，统计每个客户端最终 10 轮：
  - q-vector 消歧指标：accuracy / macro recall / macro precision / macro f1
  - 个性化测试指标：accuracy / macro recall / macro precision / macro f1
