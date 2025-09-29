import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib.pyplot as plt

import wandb
import numpy as np
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