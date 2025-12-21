import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import specific model
from model import SmallCNN 
from common import seed_everything

def get_features_hook(features_blob, name):
    def hook(model, input, output):
        # SmallCNN 的全连接层前是 flatten 后的向量
        # 这里的 input[0] 就是那个 fc1 之前的向量
        features_blob[name] = input[0].detach()
    return hook

def run_verification_v2():
    # 1. 强力配置
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 10
    NOISE_LEVEL = 0.3
    EPOCHS = 20        # SmallCNN 20轮足够了
    BATCH_SIZE = 64    # 小 Batch 梯度更新更频繁
    LR = 0.05
    
    print(f"--- PROTOCOL V2: SmallCNN + Partial Loss + CIFAR10 ---")
    
    # 2. 准备数据 (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 下载 CIFAR10
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 取 5000 个做快速验证
    subset_indices = np.random.choice(len(full_dataset), 5000, replace=False)
    
    # 构造带噪数据
    noisy_data = []
    print("Generating Synthetic PLL Noise...")
    for idx in subset_indices:
        img, true_label = full_dataset[idx]
        
        # 构造 candidate
        candidate = torch.zeros(NUM_CLASSES)
        candidate[true_label] = 1.0
        
        # 随机添加噪声候选
        for c in range(NUM_CLASSES):
            if c != true_label and np.random.random() < NOISE_LEVEL:
                candidate[c] = 1.0
        
        noisy_data.append((img, true_label, candidate))

    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch])
        candidates = torch.stack([item[2] for item in batch])
        return imgs, targets, candidates

    loader = DataLoader(noisy_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 3. 模型与优化器
    model = SmallCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    # 4. 训练循环
    print("Starting Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, targets, candidates in loader:
            imgs, candidates = imgs.to(device), candidates.to(device)
            targets = targets.to(device) # 仅用于监控准确率
            
            optimizer.zero_grad()
            output = model(imgs) # [B, 10]
            
            # --- CRITICAL FIX: Partial Cross Entropy Loss ---
            # 我们想要 maximize sum(probs * candidates)
            # 即 minimize -log( sum(probs * candidates) )
            probs = torch.softmax(output, dim=1)
            
            # 只保留候选集内的概率和
            probs_sum = (probs * candidates).sum(dim=1) 
            
            # 加个 1e-7 防止 log(0)
            loss = -torch.log(probs_sum + 1e-7).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 监控真实准确率 (God's Eye)
            preds = output.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        scheduler.step()
        acc = 100. * correct / total
        print(f"Epoch {epoch+1:02d}: Loss {total_loss/len(loader):.4f} | Real Acc: {acc:.2f}%")

    # 5. 验证原型
    print("\nExtracting Prototypes...")
    # 注册 Hook 到 SmallCNN 的 fc1 之前
    # 查看 model.py: SmallCNN 结构是 conv -> flatten -> fc1 -> fc2
    # 我们截取 fc2 (最后一层) 的输入，也就是 fc1 的输出
    features_blob = {}
    model.fc2.register_forward_hook(get_features_hook(features_blob, 'feat'))
    
    model.eval()
    
    all_feats = []
    all_targets = []
    all_confs = []
    all_preds = []
    
    with torch.no_grad():
        for imgs, targets, candidates in loader:
            imgs = imgs.to(device)
            output = model(imgs)
            
            feat = features_blob['feat'] # [B, 256]
            all_feats.append(feat.cpu())
            all_targets.append(targets)
            
            probs = torch.softmax(output, dim=1)
            max_p, preds = probs.max(dim=1)
            all_confs.append(max_p.cpu())
            all_preds.append(preds.cpu())

    all_feats = torch.cat(all_feats)
    all_targets = torch.cat(all_targets)
    all_confs = torch.cat(all_confs)
    all_preds = torch.cat(all_preds)

    # 6. 计算相似度
    CONF_THRESHOLD = 0.7 # 降低一点门槛
    
    oracle_protos = torch.zeros(NUM_CLASSES, all_feats.shape[1])
    emp_protos = torch.zeros(NUM_CLASSES, all_feats.shape[1])
    
    valid_classes = 0
    for k in range(NUM_CLASSES):
        # Oracle
        mask_gt = (all_targets == k)
        if mask_gt.sum() > 0:
            oracle_protos[k] = all_feats[mask_gt].mean(dim=0)
        
        # Empirical (High Confidence)
        mask_emp = (all_preds == k) & (all_confs > CONF_THRESHOLD)
        if mask_emp.sum() > 5: # 至少要有5个样本
            emp_protos[k] = all_feats[mask_emp].mean(dim=0)
            valid_classes += 1
        else:
            print(f"Class {k}: Not enough confident samples ({mask_emp.sum()})")

    # Normalize
    oracle_protos = F.normalize(oracle_protos, p=2, dim=1)
    emp_protos = F.normalize(emp_protos, p=2, dim=1)
    
    # Similarity
    sim_matrix = torch.mm(oracle_protos, emp_protos.t()).numpy()
    diag_sim = np.diag(sim_matrix)
    
    print("\n" + "="*30)
    print(f"Result (Threshold={CONF_THRESHOLD})")
    print(f"Avg Similarity: {np.mean(diag_sim):.4f}")
    print("="*30)
    print("Class | Sim")
    for k in range(NUM_CLASSES):
        print(f"{k:5d} | {diag_sim[k]:.4f}")

    # Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Empirical (Predicted)")
    plt.ylabel("Oracle (True)")
    plt.title(f"Prototype Verification (SmallCNN)\nAvg Sim: {np.mean(diag_sim):.4f}")
    plt.tight_layout()
    plt.savefig("prototype_v2.png")
    print("\nSaved prototype_v2.png")

if __name__ == "__main__":
    run_verification_v2()