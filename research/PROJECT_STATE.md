# 项目状态

最后更新：2026-06-05

## 当前分支

- `feature/vote-pseudo-label`

## 当前主问题

这是一个个性化联邦学习实验仓库。

当前主问题不再只是解释 `vote pseudo label` 为什么有效，而是要把方法线拆分清楚：

- `FedVotePLL`：带多客户端投票伪标签的动态消歧版本
- `FedAvg-PLL`：不使用投票，但保留本地动态 `q` 更新的 PLL 版本
- `FedAvg`：固定候选标签 soft-label、训练过程中不更新 `q` 的联邦基线

当前重点是保证这三类方法在同一联邦训练框架、相同数据划分和相同评估口径下可以公平比较，从而把性能差异解释为“是否使用动态消歧 / 是否使用投票增强”的方法差异，而不是工程设置差异。
## 当前假设

- `FedVotePLL > FedAvg-PLL`：说明多客户端投票伪标签能提升动态消歧质量。
- `FedAvg-PLL > FedAvg`：说明本地动态 `q` 更新本身就比固定候选 soft-label 更有效。
- 如果 `FedVotePLL > FedAvg`，则整体收益来自“动态消歧 + 投票共识”共同作用。
- 个性化本地测试准确率、`q` 消歧准确率和 `server test acc` 可能并不完全同步，需要拆开解释。
- 当前 `train_acc` 只是训练态 batch 上的即时准确率，不能直接等同于标准训练集评估精度。
## 当前重点

- 明确三类方法的算法定义与实验口径：`FedAvg`、`FedAvg-PLL`、`FedVotePLL`。
- 继续使用当前统一日志格式跟踪：
  - `server test acc`
  - `client personalized test acc`
  - `disamb q acc`
  - `train_acc`
- 使用新 notebook / 脚本从日志提取每轮 10 个客户端的平均：
  - 测试准确率
  - 训练准确率
  - 消歧准确率
- 维护 `research/` 中的状态记录，保证新窗口打开后可以快速恢复上下文。
## 当前阶段性结论

- 当前 `vote pseudo label` 路线已具备较完整的实验配置与日志记录能力。
- 已新增训练准确率日志与 CSV 输出，便于区分训练态与测试态表现。
- 已实现“固定 `q` 的 FedAvg 基线”开关：通过 `update_q: false` 可以关闭训练过程中的 `q` 更新。
- 这样可以在同一代码框架中比较：
  - 固定候选标签 soft-label（FedAvg）
  - 本地动态消歧（FedAvg-PLL）
  - 多客户端投票增强消歧（FedVotePLL）
- 目前 `feature/vote-pseudo-label` 分支已经包含：
  - 训练准确率输出
  - vote 相关实验配置
  - fixed-q FedAvg 基线配置
## 当前代码脉络

- [main.py](/home/yxf/proj/main.py)：加载配置和数据集，启动 `Server`
- [server.py](/home/yxf/proj/server.py)：联邦轮次、客户端采样、模型聚合、评估与日志
- [client.py](/home/yxf/proj/client.py)：候选标签生成、`q` 更新、vote/prototype/FedSA 逻辑
- [common.py](/home/yxf/proj/common.py)：配置模型、划分、指标、画图
- [model.py](/home/yxf/proj/model.py)：模型定义与工厂函数

## 已知风险

- `common.non_iid_partition()` 可能返回少于 `num_clients` 的子集，但 `server.py` 默认按完整客户端数索引。

## 下一步

- 运行并比较以下配置：
  - `configs/vote/full/config_fedavg_r101_s42.yaml`
  - `configs/vote/full/config_vote0_r101_s42.yaml`
  - `configs/vote/full/config_vote10_r200_s42.yaml` 或对应 101 轮对齐版本
- 明确中期汇报中的方法命名与问题定义：
  - `FedAvg`：固定 `q`
  - `FedAvg-PLL`：动态 `q`
  - `FedVotePLL`：投票更新 `q`
- 继续完善 notebook，把日志提取、均值统计和论文风格绘图统一起来。
- 如果后续要切到新窗口，优先阅读：
  - `research/PROJECT_STATE.md`
  - `research/WORKLOG.md`
  - 最近两个 commit：`000a902`、`d2b8d91`
