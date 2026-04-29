# 实验记录

每个实验或一组紧密相关的实验写一条记录。

---

## EXP-2026-04-16-VOTE-FAST-BATCH

### 目标

快速、可重复地比较 `vote0`、`vote1`、`vote3`、`vote5`、`vote10`。

### 相关提交

- `871f433` 新增快速投票实验批量运行脚本
- `78032b8` 整理项目目录结构并归档实验配置
- `9ba2b6e` 完善投票实验的可观测性并补充消融配置

### 相关配置与脚本

- `configs/vote/fast/`
- `scripts/run_vote_fast.sh`
- `scripts/summarize_vote_results.py`

### 做了什么

- 增加了快速批量执行 vote 消融实验的脚本。
- 规范了 vote 实验配置。
- 增加了轮级指标导出，便于后续聚合分析。

### 重点指标

- `server_test_acc`
- `global_disamb_acc`
- `global_disamb_balanced_acc`
- `client_test_acc_mean`
- `client_vote_pseudo_acc_mean`
- `client_vote_confidence_mean`

### 当前结论

- 比较不同 `vote_num_models` 的实验基础设施已经齐备。
- 当前未解决的问题是：收益到底来自投票规模、伪标签机制，还是更稳定的 `q` 更新。
- 根据当前已汇总结果：
  - `vote pseudo label` 明显提升了 `global_disamb_balanced_acc` 和 `client_test_acc_mean`
  - 但当前 `server_test_acc` 最高的是 `vote0`
  - `vote1` 的全局表现明显最差
  - `vote5` 和 `vote10` 在本地指标上优于 `vote0`
- 由于这是个性化联邦学习场景，`client_test_acc_mean` 不能被当成次要指标，它本身就是核心目标之一。
- 当前每个 `vote_tag` 只有 1 个 run，因此这些判断仍是阶段性观察。

### 下一步

- 分析已完成实验，在不同 vote 规模下对比本地消歧与全局精度的联动关系。
- 等更多 seed 跑完后，再判断 `vote` 是否真的改善全局泛化，还是主要改善本地消歧。
- 当前批次实验结束后，已新增 per-client metrics 导出。下一批新实验可观察：
  - 哪些客户端从 vote 中受益最大
  - 哪些客户端消歧提升但测试表现没有同步提升
  - vote 是否缩小或扩大客户端之间的性能差异

### 2026-04-19 更新

- 新增 `per_client_round_metrics.csv`，每轮记录每个客户端的 q-vector 消歧指标和 personalized test 指标。
- 新增 `scripts/summarize_per_client_metrics.py`，默认统计最终 10 轮 per-client 指标。
- 旧实验的 `round_metrics.csv` 无法恢复逐客户端逐轮测试指标；需要用新日志格式重跑，或从 W&B 历史文件单独恢复。

---

## EXP-2026-03-29-VOTE-SOFT-Q

### 目标

验证使用 vote 得到的 soft labels 更新 `q vector` 是否能提升本地消歧能力。

### 相关提交

- `dc8b687` update: 使用投票数得到的软标签更新置信度向量
- `4db002c` exp: 验证模型投票是否能够增加本地消歧准确率

### 相关文件

- `client.py`
- `server.py`
- `main.py`
- `common.py`

### 做了什么

- 把基于投票的 soft pseudo labels 接入客户端 `q` 更新逻辑。

### 当前结论

- 这条实验线建立了现在 vote 消融实验的核心机制基础。

### 下一步

- 用新的可观测性管线，对比 no-vote 与不同投票规模的差异。
