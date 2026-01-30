import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib.pyplot as plt

import wandb
import numpy as np

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict


import yaml
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # --- 基础训练参数 ---
    seed: int
    local_epochs: int
    batch_size: int
    optimizer: str  # 或者用 Literal['sgd', 'adam']
    
    # [关键] 这里定义了类型，Pydantic 会自动把 yaml 里的数字转成 float
    lr: float 
    momentum: float
    
    # [自动修补] 你的 YAML 里没写这个，但我给了默认值
    # 这样读取配置时，它会自动补上 5e-4，不用改 YAML 也能跑
    weight_decay: float = 5e-4 

    # --- 模型与数据 ---
    model_name: str
    dataset: str
    num_classes: int
    
    # --- 联邦学习设置 ---
    rounds: int
    num_clients: int
    ratio: float       # 采样比例
    partition: str     # non_iid
    
    # --- 噪声与异构参数 ---
    noise_level: float
    p: float           # 可能是噪声概率或划分参数
    alpha_dir: float   # Dirichlet alpha
    
    # --- 算法开关 (Bool) ---
    mixup_alpha: float
    lc: bool           # Label Correction
    mix: bool
    ga: bool           # Gradient Alignment
    uniform: bool
    proto: bool        # Prototype

    exp_id: str = ""   
    exp_name: str = ""
    
    # 显卡
    device: str = ""
    
    warmup: int = 5
    
    # 筛选上传原型的样本的标准
    mask_mode: str = "entropy"
    
    # 是否共享模型
    upmodel: bool = True

    # --- [工业级] 校验逻辑 ---
    @field_validator('lr')
    def check_lr_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Learning rate must be positive, got {v}")
        return v

    @classmethod
    def from_yaml(cls, path: str):
        """工厂方法：从 YAML 文件直接读取并实例化"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # 1. 读取原始字典
                raw_data = yaml.safe_load(f)
                
            # 2. 实例化 (这里会自动进行类型检查和默认值填充)
            return cls(**raw_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")


class GlobalConfig:
    lr: float
    momentum: float
    local_epochs: int
    noise_level: float

class PLLDataset(Dataset):
    def __init__(self, base_dataset,candidate_labels, num_classes=10, rho=0.0):
        self.base_dataset = base_dataset
        self.candidate_labels = candidate_labels
        self.num_classes = num_classes

        
    def __len__(self):
        return len(self.base_dataset)
    
    
    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        canditates = self.candidate_labels[idx].detach().clone()
        # canditates = torch.tensor(self.candidate_labels[idx])
        return data, target, canditates, idx


svhn_mean = [0.4377, 0.4438, 0.4728]
svhn_std = [0.1980, 0.2010, 0.1970]
def denormalize_image(tensor):
    """
    使用 SVHN 特定的均值和标准差反归一化图像张量。
    """
    img = tensor.cpu().clone()
    
    # 转换为 torch.tensor 并调整形状以便广播
    # [3] -> [3, 1, 1]
    mean = torch.tensor(svhn_mean).view(3, 1, 1)
    std = torch.tensor(svhn_std).view(3, 1, 1)
    
    # 反归一化: (img * std) + mean
    img = img * std + mean
    
    # 裁剪到 [0, 1] 范围
    img = torch.clamp(img, 0, 1)
    
    # [C, H, W] -> [H, W, C] 以便绘图
    return img.permute(1, 2, 0).numpy()


def iid_partition(dataset, num_clients):
    perm = torch.randperm(len(dataset))
    new_perms = np.array_split(perm.numpy(), num_clients)
    client_train_dataset = [Subset(dataset, torch.tensor(p)) for p in new_perms]
    return client_train_dataset


def seed_everything(seed=42):
    import random
    random.seed(seed)               # Python 内置随机数
    np.random.seed(seed)            # Numpy 随机数
    torch.manual_seed(seed)         # CPU 上的随机数
    torch.cuda.manual_seed(seed)    # 当前 GPU
    torch.cuda.manual_seed_all(seed)  # 所有 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker_with_seed(seed):
    def seed_worker(worker_id):
        import random, numpy as np
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker


def get_g(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def non_iid_partition(dataset, num_clients, num_classes, p=0.5, alpha_dir=0.5):
    """
    非IID数据划分函数
    
    参数:
    - dataset: 完整数据集
    - num_clients: 客户端数量N
    - num_classes: 类别数量M
    - p: 伯努利分布参数，控制每个客户端是否包含某个类别
    - alpha_dir: 狄利克雷分布参数，控制每个客户端在某个类别上的样本分布
    
    返回:
    - client_datasets: 每个客户端的数据子集列表
    - phi_matrix: 生成的指示矩阵Φ
    """
    # 第一步：生成N×M的指示矩阵Φ
    phi_matrix = np.random.binomial(1, p, size=(num_clients, num_classes))
    
    # 确保每个类别至少被一个客户端包含
    for j in range(num_classes):
        if np.sum(phi_matrix[:, j]) == 0:
            # 随机选择一个客户端包含这个类别
            i = np.random.randint(0, num_clients)
            phi_matrix[i, j] = 1
    
    # 按类别组织样本索引
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # 初始化每个客户端的数据索引
    client_indices = [[] for _ in range(num_clients)]
    
    # 对每个类别进行处理
    for j in range(num_classes):
        # 获取包含类别j的客户端索引
        clients_with_j = np.where(phi_matrix[:, j] == 1)[0]
        v_j = len(clients_with_j)
        
        if v_j == 0:
            continue
            
        # 从狄利克雷分布采样概率向量
        q_j = np.random.dirichlet(np.repeat(alpha_dir, v_j))
        
        # 获取类别j的所有样本索引
        indices_j = class_indices.get(j, [])
        n_j = len(indices_j)
        
        if n_j == 0:
            continue
            
        # 打乱样本顺序
        np.random.shuffle(indices_j)
        
        # 计算每个客户端应得的样本数量
        proportions = (q_j * n_j).astype(int)
        total_allocated = np.sum(proportions)
        
        # 处理由于取整可能导致的样本数量不匹配
        if total_allocated < n_j:
            # 随机选择一些客户端增加样本
            extra = n_j - total_allocated
            extra_indices = np.random.choice(v_j, extra, replace=True)
            for idx in extra_indices:
                proportions[idx] += 1
        elif total_allocated > n_j:
            # 随机选择一些客户端减少样本
            reduction = total_allocated - n_j
            reduction_indices = np.random.choice(v_j, reduction, replace=True)
            for idx in reduction_indices:
                if proportions[idx] > 0:
                    proportions[idx] -= 1
        
        # 分配样本给客户端
        start = 0
        for i, client_idx in enumerate(clients_with_j):
            end = start + proportions[i]
            if end > n_j:
                end = n_j
            client_indices[client_idx].extend(indices_j[start:end])
            start = end
    
    # 创建每个客户端的数据子集
    client_datasets = []
    for indices in client_indices:
        if indices:  # 只包含非空数据集
            client_datasets.append(Subset(dataset, indices))
    
    return client_datasets, client_indices, phi_matrix


def compute_client_class_counts_from_indices(client_indices, dataset, num_classes):
    """
    client_indices: list of lists, 每个子列表是该 client 的样本索引（可为空）
    dataset: 原始 dataset（dataset[idx] -> (data, label) 或 label）
    num_classes: 类别总数
    返回: numpy array shape (num_clients, num_classes) 的计数矩阵
    """
    num_clients = len(client_indices)
    counts = np.zeros((num_clients, num_classes), dtype=int)
    for i, inds in enumerate(client_indices):
        for idx in inds:
            item = dataset[idx]
            # 支持 dataset[idx] 返回 (x,y) 或 直接 y
            if isinstance(item, tuple) or isinstance(item, list):
                label = item[1]
            else:
                label = item
            if isinstance(label, torch.Tensor):
                label = label.item()
            counts[i, int(label)] += 1
    return counts
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plot_acc_counts_heatmap_blue_test(counts: np.ndarray,
                                 normalize: str = 'none',   # 'none'|'row'|'col'|'all'
                                 title: str = 'True x Pred (blue heatmap)',
                                 figsize=(10, 6),
                                 annotate: bool = True,
                                 save_path: Optional[str] = None,
                                 show_class_dist_top_n: int = 8    # 如果类太多，右上只显示前N类的占比
                                 ):
    """
    counts: 原始混淆矩阵 (num_classes x num_classes)，行=真实, 列=预测
    返回: matplotlib.figure.Figure
    """
    # defensive copy & ensure float
    mat = np.asarray(counts, dtype=float).copy()
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("counts should be a square (C x C) confusion matrix")

    # 计算指标（假设 get_metrics 已在作用域）
    metrics = get_metrics(mat)   # 需要你实现并返回 recall_class, precision_class, f1, accuracy, etc.

    # 归一化选项（用于显示）
    disp = mat.copy()
    if normalize == 'row':
        s = disp.sum(axis=1, keepdims=True); s[s == 0] = 1.0; disp = disp / s
    elif normalize == 'col':
        s = disp.sum(axis=0, keepdims=True); s[s == 0] = 1.0; disp = disp / s
    elif normalize == 'all':
        s = disp.sum(); s = s if s != 0 else 1.0; disp = disp / s
    # else 'none' -> keep raw counts in disp (we still use metrics for percentages)

    num_classes = disp.shape[0]

    # 自动字体大小
    max_dim = max(num_classes, num_classes)
    fontsize = int(np.clip(120.0 / max_dim, 4, 12))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(disp, aspect='auto', interpolation='nearest', cmap='Blues')

    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    ax.set_title(title)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))

    # 构造带百分比的刻度标签
    precs = metrics.get("precision_class")
    recs = metrics.get("recall_class")
    # 如果某些为 NaN（已用 get_metrics 处理为 0），也安全显示
    xticklabels = []
    yticklabels = []
    for i in range(num_classes):
        p = 0.0 if precs is None else float(precs[i] * 100)
        r = 0.0 if recs is None else float(recs[i] * 100)
        # 例如：Pred0\nP:76.23%
        xticklabels.append(f"{i}\nP:{p:.1f}%")
        # 例如：True0\nR:62.5%
        yticklabels.append(f"{i}\nR:{r:.1f}%")

    ax.set_xticklabels(xticklabels, fontsize=max(6, fontsize-1))
    ax.set_yticklabels(yticklabels, fontsize=max(6, fontsize-1))

    # 标注格子
    if annotate:
        for i in range(num_classes):
            for j in range(num_classes):
                # 显示策略：
                # - 如果没有归一化，显示原始计数。对角线上同时显示 count + recall%.
                # - 如果归一化，显示小数概率。
                if normalize == 'none':
                    count = int(counts[i, j])

                    txt = f"{count}"
                else:
                    val = disp[i, j]
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize, color='black')

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    # 在图右上放 summary box
    accuracy = metrics.get("accuracy")
    precision_mean = metrics.get("precision_mean")
    recall_mean = metrics.get("recall_mean")
    macro_f1 = metrics.get("f1")
    class_dist = None
    # 计算每类占比（真实类占比）
    row_sum = mat.sum(axis=1)
    total = row_sum.sum()
    if total > 0:
        class_dist = row_sum / total
    else:
        class_dist = np.zeros_like(row_sum)

    # 准备 summary 字符串（保持短小）
    summary_lines = [
        f"accuracy: {accuracy*100:.2f}%",
        f"precision (mean): {precision_mean*100:.2f}%",
        f"recall (mean): {recall_mean*100:.2f}%",
        f"macro_f1: {macro_f1*100:.2f}%",
        f"total samples: {int(total)}"
    ]
    # 如果类别不多，附上每类占比；若很多，只显示前 N
    if num_classes <= show_class_dist_top_n:
        for idx, p in enumerate(class_dist):
            summary_lines.append(f"C{idx}: {p*100:.2f}%")
    else:
        # 取占比前N
        top_idx = np.argsort(class_dist)[::-1][:show_class_dist_top_n]
        for idx in top_idx:
            summary_lines.append(f"C{idx}: {class_dist[idx]*100:.2f}%")
        summary_lines.append("...")

    summary_text = "\n".join(summary_lines)
    # 放在坐标轴的右上角（变为图内坐标）
    # ax.text(1.02, 1.0, summary_text, transform=ax.transAxes,
    #         fontsize=max(8, fontsize-1), va='top', ha='left',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85),
    #         )

    fig.text(0.89, 0.5, summary_text,
         fontsize=max(8, fontsize-1), va='center', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    plt.tight_layout()
    # if save_path:
    #     fig.savefig(save_path, dpi=200, bbox_inches='tight')
    # else:
    #     plt.show()

    return fig


import numpy as np
import torch
from torch.utils.data import Subset

def split_testset_by_distribution(global_test_dataset, phi_matrix):
    """
    根据训练数据的分布矩阵 phi_matrix，为每个客户端构建专属的测试集。
    策略：只要客户端拥有某类训练数据，就分给它该类的所有测试数据。
    
    Args:
        global_test_dataset: PyTorch Dataset (e.g., CIFAR10 test set)
        phi_matrix: [num_clients, num_classes] 矩阵，记录了每个客户端的类别分布情况
                    (可以是数量，也可以是概率，只要 >0 代表存在即可)
    
    Returns:
        test_datasets: list of Subsets, len = num_clients
    """
    
    # 1. 获取全局测试集的标签
    # 处理不同数据集格式 (CIFAR/MNIST 通常是 .targets, 自定义可能是 .labels)
    if hasattr(global_test_dataset, 'targets'):
        test_labels = np.array(global_test_dataset.targets)
    elif hasattr(global_test_dataset, 'labels'):
        test_labels = np.array(global_test_dataset.labels)
    else:
        # 如果是 TensorDataset，通常第二个元素是 label
        # 这是一个比较暴力的 fallback，视具体 dataset 实现调整
        test_labels = np.array([y for _, y in global_test_dataset])

    num_clients, num_classes = phi_matrix.shape
    client_test_datasets = []

    # 2. 建立“倒排索引”：记录每个类别对应的所有测试样本索引
    # class_id -> [idx1, idx2, idx5...]
    class_indices_map = {c: np.where(test_labels == c)[0] for c in range(num_classes)}

    print(f"Start partitioning test set for {num_clients} clients...")

    for client_idx in range(num_clients):
        # 3. 找出当前客户端拥有的类别
        # 假设 phi_matrix[k][c] > 0 表示该客户端拥有类别 c
        client_dist_vec = phi_matrix[client_idx]
        
        # 获取该客户端拥有的所有类别索引
        # 使用 > 0 判断，兼容 count 矩阵或 probability 矩阵
        target_classes = np.where(np.array(client_dist_vec) > 0)[0]
        
        # 4. 收集这些类别对应的所有测试样本索引
        client_test_indices = []
        for c in target_classes:
            if c in class_indices_map:
                client_test_indices.extend(class_indices_map[c])
        
        # 排序索引（可选，为了美观和确定性）
        client_test_indices = np.sort(client_test_indices)
        
        # 5. 创建 Subset
        if len(client_test_indices) > 0:
            client_subset = Subset(global_test_dataset, client_test_indices)
            client_test_datasets.append(client_subset)
        else:
            print(f"[Warning] Client {client_idx} has no valid classes in phi_matrix!")
            client_test_datasets.append(None) # 或者给一个空的 Subset

    print(f"Successfully created {len(client_test_datasets)} local test datasets.")
    return client_test_datasets

# --- 使用示例 ---
# 假设 test_dataset 是你加载好的 CIFAR10 测试集
# client_test_sets = split_testset_by_distribution(test_dataset, phi_matrix)

# 验证一下 Client 0
# print(f"Client 0 Test Size: {len(client_test_sets[0])}")


import torch
import numpy as np
from torch.utils.data import Subset
from collections import Counter

def print_dataset_distribution(dataset, title="Dataset Distribution"):
    """
    统计并打印 PyTorch Dataset 中各个类别的样本数量。
    支持 Subset, TensorDataset 以及 torchvision 数据集。
    
    Args:
        dataset: torch.utils.data.Dataset (或 Subset)
        title: 打印时的标题
    Returns:
        dict: {class_id: count}
    """
    if dataset is None:
        print(f"[{title}] Dataset is None (Empty).")
        return {}

    targets = []
    
    # --- 策略 A: 快速路径 (直接读取属性，不遍历) ---
    # 场景 1: 它是 Subset (最常见的情况)
    if isinstance(dataset, Subset):
        # 尝试访问底层 dataset 的 targets 或 labels
        if hasattr(dataset.dataset, 'targets'):
            # dataset.indices 是 subset 选择的索引列表
            # dataset.dataset.targets 是原始的大标签列表
            # 我们只需要挑出 subset 对应的那些
            all_targets = np.array(dataset.dataset.targets)
            targets = all_targets[dataset.indices]
        elif hasattr(dataset.dataset, 'labels'):
            all_labels = np.array(dataset.dataset.labels)
            targets = all_labels[dataset.indices]
            
    # 场景 2: 它是普通 Dataset (如 CIFAR10 原身)
    elif hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
        
    # --- 策略 B: 慢速路径 (通用遍历) ---
    # 如果上面没获取到 targets (比如是 TensorDataset)，则只能老实遍历
    if len(targets) == 0 and len(dataset) > 0:
        # 为了不拖慢速度，如果数据量巨大，可以只采样前1000个，这里默认全遍历
        print(f"[{title}] No .targets attribute found, iterating... (might be slow)")
        for _, label in dataset:
            if isinstance(label, torch.Tensor):
                targets.append(label.item())
            else:
                targets.append(label)

    # --- 统计与打印 ---
    # 使用 Counter 统计
    counter = Counter(targets)
    sorted_classes = sorted(counter.keys())
    
    print(f"\n--- {title} (Total: {len(dataset)}) ---")
    print(f"{'Class ID':<10} | {'Count':<10} | {'Proportion':<10}")
    print("-" * 36)
    
    for cls in sorted_classes:
        count = counter[cls]
        ratio = count / len(dataset)
        print(f"{cls:<10} | {count:<10} | {ratio:.2%}")
    print("-" * 36 + "\n")
    
    return dict(counter)

def compute_client_class_counts_from_subsets(client_datasets, num_classes):
    """
    client_datasets: list of torch.utils.data.Subset（可能包含空的 Subset）
    返回 counts 矩阵 (num_clients, num_classes)
    """
    num_clients = len(client_datasets)
    counts = np.zeros((num_clients, num_classes), dtype=int)
    for i, subset in enumerate(client_datasets):
        # subset is a Subset(dataset, indices)
        if isinstance(subset, Subset):
            dataset = subset.dataset
            for idx in subset.indices:
                item = dataset[idx]
                if isinstance(item, tuple) or isinstance(item, list):
                    label = item[1]
                else:
                    label = item
                if isinstance(label, torch.Tensor):
                    label = label.item()
                counts[i, int(label)] += 1
        else:
            # 如果不是 Subset，尝试按索引遍历（例如直接是 list of indices）
            for idx in subset:
                item = dataset[idx]
                if isinstance(item, tuple) or isinstance(item, list):
                    label = item[1]
                else:
                    label = item
                if isinstance(label, torch.Tensor):
                    label = label.item()
                counts[i, int(label)] += 1
    return counts
from tqdm import tqdm
def generate_candidates(model, data_loader, device, noise_rate, num_classes=10):
    model.eval()
    model.to(device)
    
    all_candidates = []
    
    print(f"\n[Phase 2] Generating Partial Labels using Instance-Dependent Logic...")
    print(f"Algorithm: Suppress GT -> Normalize Max -> Scale by Rate({noise_rate}) -> Binomial")

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Generating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. Oracle 预测
            outputs = model(inputs)
            # outputs shape: [B, 10]
            
            # 2. 构造真实标签掩码
            gt_mask = torch.zeros(inputs.size(0), num_classes, device=device)
            gt_mask.scatter_(1, targets.unsqueeze(1), 1)
            
            # 3. 获取 Softmax 概率
            probs = torch.softmax(outputs, dim=1)
            
            # 4. 【关键】抑制真值 (只看错误项)
            probs[gt_mask.bool()] = 0
            
            # 为了后面兜底用，先存一份纯净的 instance-dependent 权重
            fallback_probs = probs.clone()

            # 5. 【关键】Max 归一化
            # 让最像真的那个错误项概率变为 1.0
            max_val, _ = probs.max(dim=1, keepdim=True)
            max_val[max_val == 0] = 1.0 # 防止除零
            probs = probs / max_val
            
            # 6. 【关键】密度缩放
            # 控制整体噪音数量
            mean_val = probs.mean(dim=1, keepdim=True)
            mean_val[mean_val == 0] = 1.0
            probs = probs / mean_val * noise_rate
            
            # 7. 截断与采样
            probs[probs > 1.0] = 1.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=probs)
            # print(probs)
            noise_mask = m.sample()
            
            # 8. 合并 (真值 + 噪音)
            final_candidates = gt_mask + noise_mask
            final_candidates[final_candidates > 1.0] = 1.0
            
            # ==========================================
            # 9. [CRITICAL FIX] 强制翻转兜底 (Force Flip)
            # ==========================================
            # 检查每个样本现在的候选集大小
            candidate_counts = final_candidates.sum(dim=1)
            
            # 找到那些 "不幸" 没有生成任何噪音的样本索引 (sum == 1 说明只有 GT)
            failed_indices = (candidate_counts == 1).nonzero(as_tuple=True)[0]
            
            if len(failed_indices) > 0:
                # 既然要做 Instance-Dependent，兜底也不能瞎选。
                # 从 fallback_probs (已经去掉了真值) 中，按概率采样一个最容易混淆的类
                target_probs = fallback_probs[failed_indices]
                
                # 防止全0概率导致报错 (极少数情况)，加一个极小 epsilon
                target_probs = target_probs + 1e-6
                
                # 采样出一个补救的索引
                # multinomial 返回形状 [Num_Failed, 1]
                new_noise_indices = torch.multinomial(target_probs, 1).squeeze(1)
                
                # 强制把这个位置置为 1
                # final_candidates[行号, 列号] = 1
                final_candidates[failed_indices, new_noise_indices] = 1.0
            all_candidates.append(final_candidates.cpu())
            
    return torch.cat(all_candidates, dim=0)


def plot_counts_heatmap(counts, normalize='none', figsize=(10,6), annotate=False, cmap='viridis'):
    """
    normalize: 'none'|'row'|'col'|'all'
    返回 matplotlib Figure
    """
    mat = counts.astype(float)
    if normalize == 'row':
        s = mat.sum(axis=1, keepdims=True); s[s==0]=1.0; mat = mat / s
    elif normalize == 'col':
        s = mat.sum(axis=0, keepdims=True); s[s==0]=1.0; mat = mat / s
    elif normalize == 'all':
        total = mat.sum(); total = total if total!=0 else 1.0; mat = mat / total

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap=cmap)
    ax.set_xlabel('Class')
    ax.set_ylabel('Client')
    ax.set_title('Client x Class counts' + (f' (norm={normalize})' if normalize!='none' else ''))
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i,j]
                txt = f"{int(counts[i,j])}" if normalize=='none' else f"{v:.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")
    fig.tight_layout()
    return fig

from typing import Dict, Any

def get_metrics(conf: np.ndarray = None) -> Dict[str, Any]:
    """
    简洁版指标（基于混淆矩阵 conf）：
      - recall_class: 每类 recall（真实为 i 中被正确预测为 i 的比例）
      - precision_class: 每类 precision（被预测为 i 中真实为 i 的比例）
      - f1: 每类 F1 的平均（macro F1）
      - accuracy: 总体准确率 = sum(diag) / sum(all)
    返回字典并打印（保持简洁）。
    """
    if conf is None:
        raise ValueError("conf cannot be None")
    conf = np.asarray(conf, dtype=float)
    if conf.ndim != 2 or conf.shape[0] != conf.shape[1]:
        raise ValueError("conf must be a square (C x C) array")

    tp = np.diag(conf)
    row_sum = conf.sum(axis=1)
    col_sum = conf.sum(axis=0)
    total = conf.sum()

    # 安全计算（分母为0时设为0），并屏蔽警告输出
    with np.errstate(divide='ignore', invalid='ignore'):
        recall_class = np.where(row_sum > 0, tp / row_sum, 0.0)
        precision_class = np.where(col_sum > 0, tp / col_sum, 0.0)
        f1_class = np.where((precision_class + recall_class) > 0,
                            2 * precision_class * recall_class / (precision_class + recall_class),
                            0.0)

    f1 = float(np.mean(f1_class)) if f1_class.size > 0 else 0.0
    acc = float(tp.sum() / total) if total > 0 else 0.0

    # 更可读的打印
    # print("Recall per class:", recall_class)
    # print("Precision per class:", precision_class)
    # print("Macro F1:", f1)
    # print("Accuracy:", acc)

    return {
        "recall_class": recall_class,
        "recall_mean": float(recall_class.mean()) if recall_class.size>0 else 0.0,
        "precision_class": precision_class,
        "precision_mean": float(precision_class.mean()) if precision_class.size>0 else 0.0,
        "f1": f1,
        "accuracy": acc
    }

def plot_acc_counts_heatmap_blue(counts,
                             normalize='none',   # 'none'|'row'|'col'|'all'
                             title='True x Pred (blue heatmap)',
                             figsize=(10, 6),
                             annotate=True,
                             save_path=None    # 若提供路径则保存图像，否则 plt.show()
                             ):
    """
    counts: numpy array shape (num_clients, num_classes), 原始计数矩阵
    normalize: 是否归一化显示（'none' 保留原始计数）
    annotate: 在格子上显示数字（整数或小数）
    save_path: 若为字符串则保存到该路径（例如 'heatmap.png'），否则显示窗口
    返回: matplotlib.figure.Figure
    """
    # 复制并归一化
    mat = counts.astype(float)
    metrics = get_metrics(mat)

    if normalize == 'row':
        s = mat.sum(axis=1, keepdims=True); s[s == 0] = 1.0; mat = mat / s
    elif normalize == 'col':
        s = mat.sum(axis=0, keepdims=True); s[s == 0] = 1.0; mat = mat / s
    elif normalize == 'all':
        s = mat.sum(); s = s if s != 0 else 1.0; mat = mat / s
    # else 'none' -> keep raw counts

    num_clients, num_classes = mat.shape

    # 动态计算字体大小（防止文字太拥挤）
    # 值域：4 ~ 12，客户端或类别多时减小字体
    max_dim = max(num_clients, num_classes)
    fontsize = int(np.clip(120.0 / max_dim, 4, 12))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap='Blues')

    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    ax.set_title(title)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_clients))

    # 如果类别或客户端过多，隐藏部分刻度以免拥挤
    if num_classes > 40:
        ax.set_xticks(np.arange(0, num_classes, max(1, num_classes // 40)))
    if num_clients > 40:
        ax.set_yticks(np.arange(0, num_clients, max(1, num_clients // 40)))

    # 在每个格子上标注数值
    if annotate:
        for i in range(num_clients):
            for j in range(num_classes):
                val = mat[i, j]
                if normalize == 'none':
                    txt = f"{int(counts[i, j])}"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize, color='black')

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    fig.tight_layout()

    # if save_path:
    #     fig.savefig(save_path, dpi=200, bbox_inches='tight')
    #     print(f"Saved heatmap to {save_path}")
    # else:
    #     plt.show()

    return fig


def plot_candidates_counts_heatmap_blue(counts,
                             normalize='none',   # 'none'|'row'|'col'|'all'
                             title='Target x Candidate (blue heatmap)',
                             figsize=(10, 6),
                             annotate=True,
                             save_path=None    # 若提供路径则保存图像，否则 plt.show()
                             ):
    """
    counts: numpy array shape (num_clients, num_classes), 原始计数矩阵
    normalize: 是否归一化显示（'none' 保留原始计数）
    annotate: 在格子上显示数字（整数或小数）
    save_path: 若为字符串则保存到该路径（例如 'heatmap.png'），否则显示窗口
    返回: matplotlib.figure.Figure
    """
    # 复制并归一化
    mat = counts.astype(float)
    if normalize == 'row':
        s = mat.sum(axis=1, keepdims=True); s[s == 0] = 1.0; mat = mat / s
    elif normalize == 'col':
        s = mat.sum(axis=0, keepdims=True); s[s == 0] = 1.0; mat = mat / s
    elif normalize == 'all':
        s = mat.sum(); s = s if s != 0 else 1.0; mat = mat / s
    # else 'none' -> keep raw counts

    num_clients, num_classes = mat.shape

    # 动态计算字体大小（防止文字太拥挤）
    # 值域：4 ~ 12，客户端或类别多时减小字体
    max_dim = max(num_clients, num_classes)
    fontsize = int(np.clip(120.0 / max_dim, 4, 12))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap='Blues')

    ax.set_xlabel('Candidate')
    ax.set_ylabel('Target')
    ax.set_title(title)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_clients))

    # 如果类别或客户端过多，隐藏部分刻度以免拥挤
    if num_classes > 40:
        ax.set_xticks(np.arange(0, num_classes, max(1, num_classes // 40)))
    if num_clients > 40:
        ax.set_yticks(np.arange(0, num_clients, max(1, num_clients // 40)))

    # 在每个格子上标注数值
    if annotate:
        for i in range(num_clients):
            for j in range(num_classes):
                val = mat[i, j]
                if normalize == 'none':
                    txt = f"{int(counts[i, j])}"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize, color='black')

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    fig.tight_layout()

    # if save_path:
    #     fig.savefig(save_path, dpi=200, bbox_inches='tight')
    #     print(f"Saved heatmap to {save_path}")
    # else:
    #     plt.show()

    return fig

import logging
from logging import Logger
import os

def setup_logger(save_path, log_file_name) -> Logger:
    logger = logging.getLogger() 
    if len(logger.handlers) > 0:
        return logger
    
    logger.setLevel(logging.INFO) 
    
    formatter = logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")

    # [FIX 1]: 修正路径拼接
    full_log_file = os.path.join(save_path, log_file_name)
    
    # [FIX 2]: 确保父目录存在，否则 FileHandler 会报错
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # [FIX 3]: 显式指定 utf-8
    fh = logging.FileHandler(filename=full_log_file, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def plot_counts_heatmap_blue(counts,
                             normalize='none',   # 'none'|'row'|'col'|'all'
                             title='Client x Class (blue heatmap)',
                             figsize=(10, 6),
                             annotate=True,
                             save_path=None    # 若提供路径则保存图像，否则 plt.show()
                             ):
    """
    counts: numpy array shape (num_clients, num_classes), 原始计数矩阵
    normalize: 是否归一化显示（'none' 保留原始计数）
    annotate: 在格子上显示数字（整数或小数）
    save_path: 若为字符串则保存到该路径（例如 'heatmap.png'），否则显示窗口
    返回: matplotlib.figure.Figure
    """
    # 复制并归一化
    mat = counts.astype(float)
    if normalize == 'row':
        s = mat.sum(axis=1, keepdims=True); s[s == 0] = 1.0; mat = mat / s
    elif normalize == 'col':
        s = mat.sum(axis=0, keepdims=True); s[s == 0] = 1.0; mat = mat / s
    elif normalize == 'all':
        s = mat.sum(); s = s if s != 0 else 1.0; mat = mat / s
    # else 'none' -> keep raw counts

    num_clients, num_classes = mat.shape

    # 动态计算字体大小（防止文字太拥挤）
    # 值域：4 ~ 12，客户端或类别多时减小字体
    max_dim = max(num_clients, num_classes)
    fontsize = int(np.clip(120.0 / max_dim, 4, 12))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap='Blues')

    ax.set_xlabel('Class')
    ax.set_ylabel('Client')
    ax.set_title(title)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_clients))

    # 如果类别或客户端过多，隐藏部分刻度以免拥挤
    if num_classes > 40:
        ax.set_xticks(np.arange(0, num_classes, max(1, num_classes // 40)))
    if num_clients > 40:
        ax.set_yticks(np.arange(0, num_clients, max(1, num_clients // 40)))

    # 在每个格子上标注数值
    if annotate:
        for i in range(num_clients):
            for j in range(num_classes):
                val = mat[i, j]
                if normalize == 'none':
                    txt = f"{int(counts[i, j])}"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize, color='black')

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    fig.tight_layout()

    # if save_path:
    #     fig.savefig(save_path, dpi=200, bbox_inches='tight')
    #     print(f"Saved heatmap to {save_path}")
    # else:
    #     plt.show()

    return fig