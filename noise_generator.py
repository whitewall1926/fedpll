import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy

class InstanceDependentNoise:
    def __init__(self, model, device, num_classes=10):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.to(device)
        self.model.eval()

    def generate_candidates(self, dataset, q=0.0, k=1, norm_std=0.1):
        """
        实现引用中的 ID 噪声生成方法。
        
        Args:
            dataset: 原始数据集 (包含 clean labels)
            q (float): 噪声翻转率 (在此方法中通常通过 k 控制，这里保留接口)
            k (int): 每个样本添加 k 个错误候选标签 (Set size = k + 1)
            norm_std (float): 不需要，但在某些变体中用于控制分布平滑度
            
        Returns:
            candidates: [N, num_classes] 的 0/1 矩阵
        """
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        all_candidates = []
        
        print(f"Generating ID-N Candidates (k={k})...")
        
        with torch.no_grad():
            for data, targets in loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                batch_size = data.size(0)

                # 1. "utilizing the prediction of a neural network"
                # 获取模型对每个类的预测 logit
                logits = self.model(data) # [B, C]
                
                # 2. 屏蔽真实标签
                # 我们只关心“哪个错误标签最像真的”，所以把真实标签的概率屏蔽掉
                mask = torch.one_hot(targets, self.num_classes).bool()
                logits_masked = logits.clone()
                # 设为负无穷，Softmax后概率为0，确保不会采样到真实标签
                logits_masked = logits_masked.masked_fill(mask, -1e9)
                
                # 3. 计算非真实标签的采样概率
                # "probability ... is related to each instance itself"
                probs = F.softmax(logits_masked, dim=1) 
                
                # 4. 根据概率采样 k 个错误标签
                # 概率越大的错误类（越混淆的类），越容易被选中
                wrong_indices = torch.multinomial(probs, num_samples=k, replacement=False)
                
                # 5. 构建 One-Hot 候选矩阵
                candidates = torch.zeros(batch_size, self.num_classes, device=self.device)
                # 填入真实标签
                candidates.scatter_(1, targets.view(-1, 1), 1)
                # 填入生成的错误标签
                candidates.scatter_(1, wrong_indices, 1)
                
                all_candidates.append(candidates.cpu())
                
        return torch.cat(all_candidates)

def train_clean_model(model, dataset, device, epochs=10):
    """
    "trained with original clean labels"
    训练一个基础模型用于生成混淆矩阵。
    """
    print(f"--- Training Generator on Clean Labels for {epochs} epochs ---")
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    for ep in range(epochs):
        total = 0
        correct = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
        print(f"  Generator Epoch {ep+1}: Acc {100*correct/total:.2f}%")
        
    return model