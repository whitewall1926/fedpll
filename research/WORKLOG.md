# 工作记录

## 2026-04-28

### 本次目标

为 bad case / corner case 分析补充更细的客户端级和样本级统计，并记录其计算方式。

### 本次完成

- 修改 `client.py`：
  - 新增客户端候选标签难度统计：
    - `candidate_size_mean`
    - `candidate_size_std`
    - `candidate_size_p50`
    - `candidate_size_p90`
    - `candidate_ambiguity_rate`
  - 新增 vote 置信度分位数统计：
    - `vote_confidence_p10`
    - `vote_confidence_p50`
    - `vote_confidence_p90`
  - 新增 vote bad-case 统计：
    - `vote_high_conf_error_rate`
    - `vote_low_conf_correct_rate`
    - `vote_sample_coverage`
  - 新增样本级 `client_{id}_vote_bad_cases.csv` 导出，记录：
    - `high_conf_wrong`
    - `low_conf_correct`
- 修改 `server.py`：
  - 将上述 client 侧统计汇总到 `round_metrics.csv`
  - 将上述 client 侧统计写入 `per_client_round_metrics.csv`
  - 每轮日志新增：
    - vote 置信度 `p10/p50/p90`
    - 高置信错误率
    - 低置信正确率
    - vote 样本覆盖率
    - 候选集大小与歧义率
- 更新 `research/METRICS.md`：
  - 明确记录新增字段语义、计算方式、阈值和 bad-case CSV 结构
- 新增 `scripts/run_vote_noise_single_seed.sh`：
  - 单 seed 扫描 `noise_level × {vote0, vote5, vote10}`
  - 默认 `seed=42`、`rounds=30`
  - 运行时从现有 fast 模板生成临时配置，避免手工维护 12 份 YAML
  - 结束后自动运行 `summarize_vote_results.py` 和 `summarize_per_client_metrics.py`
- 新增 `scripts/compare_weak_clients.py`：
  - 按 `noise_level` 对齐比较同一 client 在 `vote0/vote5/vote10` 下的最终窗口表现
  - 默认按 `vote0` 的 personalized acc 选出每个噪声下最弱的 `top-k` 客户端
  - 输出 `weak_client_comparison_final{N}.csv`

### 本次涉及文件

- `client.py`
- `server.py`
- `scripts/run_vote_noise_single_seed.sh`
- `scripts/compare_weak_clients.py`
- `research/METRICS.md`
- `research/WORKLOG.md`

### 验证方式

- 已运行 `python3 -m py_compile client.py server.py`，语法通过。
- 已运行 `python3 -m py_compile scripts/compare_weak_clients.py`，语法通过。
- 待用下一次实验确认：
  - `round_metrics.csv` 新字段正常落盘
  - `per_client_round_metrics.csv` 新字段正常落盘
  - `client_{id}_vote_bad_cases.csv` 正常生成

## 2026-04-20

### 本次目标

让每个客户端的 local test set 类别数量分布尽量匹配其 local train set，而不是只匹配类别集合。

### 本次完成

- 修改 `common.split_testset_by_distribution()`：
  - 输入从二值 `phi_matrix` 改为通用 `distribution_matrix`
  - 对每个类别按 `distribution_matrix[:, class_id]` 的比例分配测试样本
  - 使用训练集 class-count 矩阵时，local test 分布会尽量贴近 local train 分布
  - 不同客户端之间不再重复使用同一个测试样本
- 修改 `server.py`：
  - 先用 `compute_client_class_counts_from_subsets()` 计算训练集每客户端类别计数
  - 用该 `counts` 调用 `split_testset_by_distribution()`
  - IID 和 non-IID 路径都会初始化 `client_test_datasets`
- 新增 personalized FL 弱客户端汇总指标：
  - `min_client_personalized_test_acc`
  - `p10_client_personalized_test_acc`
- 更新 `scripts/summarize_vote_results.py`，让 vote 汇总包含上述弱客户端指标。
- 更新 `research/METRICS.md`，记录弱客户端指标语义。
- 将每轮完整汇总指标写入 logger，便于直接从日志追踪 server/client/disamb/vote 指标。
- 新增混淆矩阵 CSV 导出：
  - server test：`confusion_matrices/round_{round}_server.csv`
  - client personalized test：`confusion_matrices/round_{round}_client_{client_id}_personalized_test.csv`
  - 日志中记录每个混淆矩阵路径和 recall 最低的类别摘要。
- 精简日志输出：
  - 移除每轮每客户端 optimizer、norm entropy、逐客户端 disamb/test、epoch train acc、Q-vector inspection 等重复信息
  - 每轮保留 selected clients、汇总指标、worst clients、混淆矩阵目录和 server worst classes
  - 增加中文自然语言解读，解释 server-client 差距、弱客户端表现、q 消歧和 vote 伪标签质量
  - 增加 per-client 指标行，逐客户端展示 personalized acc、macro recall、macro F1、q acc 和 q macro F1
  - 将原本过长的单行日志拆成多行短日志，避免查看日志时需要横向拖动
  - 完整 per-client 指标仍保留在 `per_client_round_metrics.csv` 和 W&B

### 本次涉及文件

- `common.py`
- `server.py`
- `client.py`
- `main.py`
- `scripts/summarize_vote_results.py`
- `research/METRICS.md`
- `research/WORKLOG.md`

### 验证方式

- 已运行 `python3 -m py_compile common.py server.py scripts/summarize_vote_results.py`，语法通过。
- 已用 `Server.__new__` 最小实例测试混淆矩阵 CSV 写入和最差类别摘要 helper。

## 2026-04-19

### 本次目标

在实验批次结束后，按 `research/METRICS.md` 的语义映射统一更新指标命名，并新增每轮每客户端指标导出，支持最终 10 轮 per-client 测试准确率统计。

### 本次完成

- 修改 `server.py` 中 `round_metrics.csv` 的字段名，使用语义化指标名。
- 修正 W&B 中 `sevrer_test/acc` 的拼写，并将 server/client/disamb/vote 指标改为语义化路径。
- 新增 `per_client_round_metrics.csv`，每轮记录每个客户端的：
  - q-vector 消歧 accuracy / macro recall / macro precision / macro f1
  - personalized test accuracy / macro recall / macro precision / macro f1
- 更新 `scripts/summarize_vote_results.py`，兼容旧 CSV 字段名并输出新字段。
- 新增 `scripts/summarize_per_client_metrics.py`，默认统计每个客户端最终 10 轮指标。
- 修改 `main.py` 的实验命名规则，使 run name 显式包含 vote、seed、rounds、local epochs、noise、partition、optimizer、lr 和 loss 开关。
- 更新 `research/METRICS.md`，记录新字段和迁移状态。

### 本次涉及文件

- `server.py`
- `client.py`
- `main.py`
- `scripts/summarize_vote_results.py`
- `scripts/summarize_per_client_metrics.py`
- `research/METRICS.md`
- `research/WORKLOG.md`

### 验证方式

- 已运行 `python3 -m py_compile` 检查修改后的 Python 文件，语法通过。
- 未运行 pytest：当前环境未安装 `pytest` / `python3 -m pytest`。

### 下一步

- 用新日志格式重新运行需要 per-client final-10 分析的 vote 实验。
- 运行 `python scripts/summarize_per_client_metrics.py` 生成 per-client final-10 汇总。
- 如需分析旧实验的 per-client final-10，需要额外从 W&B 历史文件恢复；旧 `round_metrics.csv` 无法提供逐客户端逐轮数据。

## 2026-04-17

### 本次目标

建立一套可持久化的科研工作流，让未来的 Codex 会话能通过仓库文件和 git 历史恢复上下文。

### 本次完成

- 阅读仓库结构，确认主执行链路。
- 阅读 `README.md`、`main.py`、`server.py`、`client.py`、`common.py`、`model.py`。
- 阅读最近 git 历史，确认当前研究主线。
- 阅读 `tmux` 中的 `vote_fast` 会话，确认上次运行的是 `scripts/run_vote_fast.sh`。
- 阅读 `csv_logs/vote_run_summary.csv` 和 `csv_logs/vote_group_summary.csv`。
- 确认当前分支为 `feature/vote-pseudo-label`。
- 确认当前主要研究线是 `vote pseudo label` 消融与可观测性。
- 新增科研状态管理文档，用于后续会话恢复。

### 本次涉及文件

- `AGENTS.md`
- `PROJECT_STATE.md`
- `WORKLOG.md`
- `EXPERIMENT_LOG.md`
- `DECISIONS.md`

### 关键发现

- 最近 commit 明确围绕 `vote_num_models` 比较、观测指标增强、批量实验运行展开。
- 当前最大的研究问题不是泛泛的代码整理，而是解释 `vote pseudo label` 的收益来源。
- 仓库本身已经有连续的 git 历史，但之前缺少“面向恢复”的文档层。
- 用户明确说明该项目的研究设定是个性化联邦学习，不应把它按普通联邦学习理解。
- `tmux` 当前运行的是 `vote_fast` 会话，对应 `scripts/run_vote_fast.sh` 的批量实验。
- `fast` vote 配置里的 `proto` 都是 `false`，日志里出现 `using prototype_guidance` 只是当前代码打印条件不够严谨。
- 当前汇总结果显示：vote 能提升本地消歧和客户端指标，但目前没有超过 `vote0` 的全局测试精度，因此需要区分个性化收益与全局收益。

### 验证方式

- 通过阅读主代码文件确认训练流程。
- 通过阅读最近 `git log --oneline` 和 `git log --stat` 确认当前研究方向。
- 通过阅读 `tmux capture-pane` 确认当前实验会话状态。
- 通过阅读 `csv_logs/vote_run_summary.csv` 与 `csv_logs/vote_group_summary.csv` 确认阶段性结果。

### 下一步

- 后续每次进入仓库，先读这些状态文件，再读最近 commit。
- 下一轮分析时重点拆解 `client.py` 中的 `q` 更新、vote 伪标签与指标变化。
- 持续跟踪 `run_vote_fast.sh` 的输出，并在更多 run 完成后重新判断 `vote` 对全局泛化的真实影响。
- 当前实验全部跑完后，实现 per-client 轮级指标导出，补充客户端级：
  - disambiguation acc / macro recall / macro precision / macro f1
  - personalized test acc / macro recall / macro precision / macro f1
