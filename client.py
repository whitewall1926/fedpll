import torch
import wandb
import copy
from torch.utils.data import DataLoader
from common import PLLDataset, get_g, seed_worker_with_seed
import numpy as np
import torch.nn.functional as F


from torch.nn.utils import parameters_to_vector


sim = np.array([
    [1.00,0.10,0.05,0.05,0.02,0.15,0.70,0.03,0.20,0.70],
    [0.10,1.00,0.03,0.03,0.05,0.02,0.15,0.80,0.03,0.02],
    [0.05,0.03,1.00,0.60,0.03,0.05,0.02,0.03,0.60,0.10],
    [0.05,0.03,0.60,1.00,0.03,0.02,0.02,0.03,0.60,0.12],
    [0.02,0.05,0.03,0.03,1.00,0.10,0.05,0.02,0.03,0.20],
    [0.15,0.02,0.05,0.02,0.10,1.00,0.50,0.02,0.05,0.08],
    [0.70,0.15,0.02,0.02,0.05,0.50,1.00,0.05,0.02,0.75],
    [0.03,0.80,0.03,0.03,0.02,0.02,0.05,1.00,0.05,0.02],
    [0.20,0.03,0.60,0.60,0.03,0.05,0.02,0.05,1.00,0.50],
    [0.70,0.02,0.10,0.12,0.20,0.08,0.75,0.02,0.50,1.00],
], dtype=float)


probs_matrix = []




class Client:
    def __init__(self,
                 local_model, 
                 train_dataset, 
                 client_id,
                 config):
        self.config = config
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.rounds = 0

        self.train_dataset = train_dataset
        candidate_labels = []

        counts = np.zeros((self.config.num_classes, self.config.num_classes), dtype=int)

        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            candidate = torch.zeros(self.config.num_classes, dtype=int)
            candidate[label] = 1         

            # label_sim = sim[label].copy()
            # import random
            # for i in range(self.config.num_classes):
            #     if i != label and random.random() < label_sim[i]:
            #         candidate[i] = 1
            # print("true label =", label, "candidate =", candidate)

            # cnt = int(self.config.noise_level * (self.config.num_classes))
            # label_sim = sim[label].copy()
            # label_sim_idx = np.argsort(label_sim)[::-1].copy()
            # choosed_labels = label_sim_idx[:cnt]
            # candidate[choosed_labels] = 1

            # import random
            # others = []
            # for i in range(self.config.num_classes):
            #     if i not in choosed_labels:
            #         others.append(i)
            # candidate[random.choice(others)] = 1
            
            import random
            for other_label in range(self.config.num_classes):
                if other_label != label and random.random() < self.config.noise_level:
                    candidate[other_label] = 1

            for j in range(len(candidate)):
                counts[label][j] += candidate[j].item()

            candidate_labels.append(candidate)
            
        print(f'client id:{client_id}\n', counts)
        self.train_plldataset = PLLDataset(self.train_dataset, num_classes=self.config.num_classes, candidate_labels=candidate_labels, rho=self.config.noise_level)
        
        
         
        q = []
        for idx in range(len(self.train_plldataset)):
            candidates = self.train_plldataset.candidate_labels[idx]
            p = candidates.clone().float()
            p /= (candidates.sum() * 1.0)
            # if (idx < 4):
            #     print(p, "true target=", self.train_plldataset[idx][1])
            q.append(p)
        q = torch.stack(q).to(device=self.device)
        self.q = q

        

        self.local_model = copy.deepcopy(local_model).to(self.device)

        self.client_id = client_id
        
        self.train_loader = DataLoader(
            self.train_plldataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=False,
            worker_init_fn=seed_worker_with_seed(self.config.seed),
            generator=torch.Generator().manual_seed(self.config.seed)
        )

        self.optimizer = None
        self.epochs = self.config.local_epochs
        self.global_model = None

        
    def compute_flattened_local_grad(self, num_batches=1, use_global_model=False):
        """
        在本地使用当前 local_model（或 global_model，如果 use_global_model=True）对 num_batches 批次计算梯度并返回展平向量。
        - num_batches: 从 train_loader 读取几个 batch 计算平均梯度（默认 1）
        - 返回值: 1D torch.Tensor (dtype float32) 在 CPU 上（便于网络传输或 server 汇总）
        注意：不会修改模型参数（只 forward+backward，并清理 grad）
        """
        model = self.local_model if not use_global_model else self.global_model
        model.to(self.device)
        model.zero_grad()
        accum_grads = None
        batches_used = 0

        # iterate up to num_batches
        for i, batch in enumerate(self.train_loader):
            data, target, candidates, idxs = batch
            data = data.to(self.device)
            # build q_batch for this batch if necessary (self.q exists)
            if isinstance(idxs, torch.Tensor):
                idxs_dev = idxs.to(self.device)
            else:
                idxs_dev = torch.tensor(idxs, device=self.device)

            # compute loss (use same loss as training, but no create_graph)
            output = model(data)
            # 使用 pll_loss_vectorized（不改变 q permanently）
            loss = self.pll_loss_vectorized_unchanged(output=output, idxs=idxs_dev, miu=0.99)
            model.zero_grad()
            loss.backward()

            # collect grads in same order as model.parameters()
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().to(torch.float32))
                else:
                    grads.append(torch.zeros_like(p).to(torch.float32))

            # flatten
            g_vec = parameters_to_vector(grads)  # 1D tensor on current device
            g_vec = g_vec.detach().cpu()         # move to CPU for safe aggregation / transfer

            if accum_grads is None:
                accum_grads = g_vec
            else:
                accum_grads += g_vec
            batches_used += 1

            if batches_used >= num_batches:
                break

        if accum_grads is None:
            # no batches (empty dataset) -> zero vector of length equal to model param vector
            dummy = [torch.zeros_like(p) for p in model.parameters()]
            accum_grads = parameters_to_vector(dummy).detach().cpu()

        # average across used batches
        accum_grads /= float(max(1, batches_used))
        return accum_grads  # CPU 1D tensor


    
    # -------------------------
    # compute_grad_vector: 返回展平梯度向量
    # -------------------------
    def compute_grad_vector(self, model, data, q_batch, create_graph=False):
        """
        Compute gradient vector g = flatten( d Lc / d theta ) for given model & batch.
        - model: nn.Module (must be on self.device)
        - data: input tensor on device
        - q_batch: soft labels tensor [B, C], on device (detached or requires_grad False)
        - create_graph: whether to create graph for higher-order derivatives
        Returns: 1D tensor of shape [D] (same device as model)
        """
        model.zero_grad()
        # forward
        out = model(data)                         # [B, C]
        logp = torch.log_softmax(out, dim=1)
        loss = - (q_batch * logp).sum(dim=1).mean()
        # compute grads (as tuple)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=create_graph, retain_graph=True, allow_unused=True)
        # some params may be None -> replace with zeros
        grads = [g if g is not None else torch.zeros_like(p, device=self.device) for g, p in zip(grads, model.parameters())]
        # flatten to vector
        g_vec = parameters_to_vector(grads)  # returns 1D tensor
        return g_vec, loss

    # -------------------------
    # gradient_alignment_loss: 余弦相似度 negative (with stability)
    # -------------------------
    def gradient_alignment_loss(self, local_grad_vector, global_grad_vector, eps=1e-8):
        """
        local_grad_vector: 1D tensor, maybe requires_grad True (if create_graph used)
        global_grad_vector: 1D tensor, should be detached (no grad) representing g_F
        returns scalar tensor L_ga = -cos(local, global)
        """
        # ensure same device
        local = local_grad_vector
        global_v = global_grad_vector.to(local.device)

        # norms
        n1 = torch.norm(local)
        n2 = torch.norm(global_v)
        if n1.item() < eps or n2.item() < eps:
            # if any norm is too small, return zero loss (or small value)
            return torch.tensor(0.0, device=local.device, dtype=local.dtype)

        cos = (local * global_v).sum() / (n1 * n2 + eps)  # scalar
        return -cos  # we minimize negative cosine to maximize alignment

    def update_config(self, new_config):
        self.config = new_config
        self.train_loader = DataLoader(
            self.train_plldataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=False,
            worker_init_fn=seed_worker_with_seed(self.config.seed),
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        self.epochs = self.config.local_epochs


    def train(self, 
              global_model_state_dict, 
              roud,
              new_config=None,
              global_grad_vector=None):
        if new_config != None:
            self.config = None
            self.update_config(new_config)

        print(f"roud {roud} trainning..............................\n")
        self.local_model.load_state_dict(global_model_state_dict)
        self.global_model = copy.deepcopy(self.local_model)
        self.global_model.eval()
        if global_grad_vector is not None:
            g_global_vec = global_grad_vector.to(self.device).detach()  # treat as constant
        else:
            g_global_vec = None
        if self.config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.config.lr)
            print(f'client {self.client_id} using adam ')
        else:
            self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.config.lr, momentum=0.5)
            print(f'client {self.client_id} using sgd ')


        self.local_model.train()
        
        q = self.q
        avg_train_acc = 0
        avg_train_loss = 0
        # print("---", self.epochs, self.config.local_epochs)

        print(f"client{self.client_id} using lc_loss\n")
        if self.config.mix == True:
            print(f"client{self.client_id} using mix_loss\n")
        if self.config.ga == True:
            print(f"client{self.client_id} using lga\n")
        
        for epoch in range(self.epochs):
            
            total_samples= 0
            train_loss = 0
            train_acc = 0

            for data, target, candidates, idxs in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.local_model(data)
                # lc_loss = self.pll_loss_vectorized(output=output, idxs=idxs, candidates=candidates, miu=0.99) 
                lc_loss = self.pll_loss_vectorized_soft_preds(output=output, idxs=idxs, candidates=candidates, miu=0.99) 
                mix_loss = 0.0
                if self.config.mix == True:
                    mix_loss =  self.pll_mix_up_loss(data=data, idxs=idxs, global_model_state_dict=global_model_state_dict)

                lga = 0.0
                if self.config.ga == True:
                    g_local_vec, _ = self.compute_grad_vector(self.local_model, data, q_batch=self.q[idxs], create_graph=True)
                    #    g_global computed on global_model (no higher-order graph needed)
                    if g_local_vec.numel() != g_global_vec.numel():
                       raise ValueError("Dimension mismatch between local and global grad vectors")
                    lga = self.gradient_alignment_loss(g_local_vec, g_global_vec)

                loss = lc_loss + mix_loss + lga

                loss.backward()

                batch_size = data.size(0)
                train_loss += loss.detach().item() * batch_size
                total_samples += batch_size

                preds = output.argmax(dim=1)
                train_acc += (preds==target).sum().item()

                self.optimizer.step()

                


            print(q[:5], "\n")
            train_loss = train_loss /  total_samples
            train_acc = train_acc / total_samples
            
            print(f"----> client: {self.client_id} | Roud: {roud:3d} | Epoch: {epoch:3d} | "
              f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}\n")

            # wandb.log({
            # f"client_train/{self.client_id}/acc": train_acc,
            # f"client_train/{self.client_id}/loss": train_loss   
            # }, step=roud)
            avg_train_acc = avg_train_acc + train_acc
            avg_train_loss = avg_train_loss + train_loss 
        avg_train_acc = avg_train_acc * 1.0 /  self.epochs
        avg_train_loss = avg_train_loss * 1.0 / self.epochs
        wandb.log({
            f"client_train/{self.client_id}/acc": avg_train_acc,
            f"client_train/{self.client_id}/loss": avg_train_loss   
            }, step=roud)
        return self.local_model.state_dict() 
    
    
    def get_acc_matrix(self, test_loader):
        self.local_model.eval()
        # test_acc = 0
        # test_loss = 0
        
        # 初始化混淆矩阵（行：实际类别，列：预测类别）
        acc_matrix = np.zeros((self.config.num_classes, self.config.num_classes), dtype=int)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.local_model(data)
                preds = output.argmax(dim=1)  # 获取预测类别
                
                # 计算准确率
                # test_acc += (preds == target).sum().item()
                
                # 统计混淆矩阵：将张量转为numpy数组（需先移到CPU）
                target_np = target.cpu().numpy()
                preds_np = preds.cpu().numpy()
                
                # 遍历每个样本，更新混淆矩阵
                for t, p in zip(target_np, preds_np):
                    acc_matrix[t][p] += 1  # 实际类别t被预测为p，对应位置+1
        # 计算整体准确率
        # test_acc = test_acc / len(test_loader.dataset)

        # 日志记录和打印
        
        return acc_matrix  # 可选择返回混淆矩阵
    

    def test(self, test_loader, epoch, criterion=None):
        self.local_model.eval()
        test_acc = 0
        test_loss = 0
        
        acc_matrix = np.zeros((self.config.num_classes, self.config.num_classes))

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.local_model(data)
                # loss = criterion(output, target)
                # test_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                test_acc += (preds == target).sum().item()
                
        test_acc = test_acc / len(test_loader.dataset)
        # test_loss = test_loss / len(test_loader.dataset)

        wandb.log({f"client_test/{self.client_id}/acc":test_acc}, step=epoch)
        # wandb.log({"test/loss":test_loss})
        print(f"client:{self.client_id}: Epoch: {epoch:3d} | "
            #   f"Test Loss: {test_loss:.6f} | "
            f"Test Acc: {test_acc:.4f}\n")
        return test_acc
    def pll_loss_vectorized_soft_preds(self, output, idxs, candidates,  miu=0.99):
        q = self.q
        batch_size = output.size(0)
        
        # cand_mask = candidates.to(device=self.device).bool()
        # neg_inf = -1e-9
        # logits_masked = output.clone()
        # logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)
        # pred_idx = logits_masked.argmax(dim=1)  # [batch_size]
        device = output.device
        B, C = output.shape

        # ensure cand mask is boolean tensor on same device
        if not isinstance(candidates, torch.Tensor):
            # if it's list-of-lists, build mask (rare)
            cand_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
            for i, cand in enumerate(candidates):
                if len(cand) == 0:
                    cand_mask[i, :] = True
                else:
                    cand_mask[i, list(cand)] = True
        else:
            cand_mask = candidates.to(device=device).bool()

        # 1) 把非 candidate 的 logits 设为一个很小的负数，再 argmax（保证只在候选中选）
        neg_inf = -1e9
        logits_masked = output.clone()
        logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)

        # 若某行所有 candidates 都被 mask（不应该），回退为全候选以避免全 -inf
        # （上面 masked_fill 不会变成全 -inf 因为 cand_mask 行至少应有一个 True；但多保守处理）
        all_neg = (~cand_mask).all(dim=1)
        if all_neg.any():
            logits_masked[all_neg] = output[all_neg]  # 回退：不做 masked_fill

        pred_idx = logits_masked.argmax(dim=1)  # 现在一定只会在候选中选

        p = torch.zeros_like(output)     # [batch_size, num_classes]
        p[torch.arange(batch_size), pred_idx] = 1.0
        
        p_soft = torch.softmax(logits_masked.detach(), dim=1)
        q[idxs] = q[idxs] * miu + (1 - miu) * p_soft


        log_probs = torch.log_softmax(output, dim=1)
        loss = -(q[idxs] * log_probs).sum(dim=1).mean()
        return loss
    def pll_loss_vectorized(self, output, idxs, candidates,  miu=0.99):
        q = self.q
        batch_size = output.size(0)
        
        # cand_mask = candidates.to(device=self.device).bool()
        # neg_inf = -1e-9
        # logits_masked = output.clone()
        # logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)
        # pred_idx = logits_masked.argmax(dim=1)  # [batch_size]
        device = output.device
        B, C = output.shape

        # ensure cand mask is boolean tensor on same device
        if not isinstance(candidates, torch.Tensor):
            # if it's list-of-lists, build mask (rare)
            cand_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
            for i, cand in enumerate(candidates):
                if len(cand) == 0:
                    cand_mask[i, :] = True
                else:
                    cand_mask[i, list(cand)] = True
        else:
            cand_mask = candidates.to(device=device).bool()

        # 1) 把非 candidate 的 logits 设为一个很小的负数，再 argmax（保证只在候选中选）
        neg_inf = -1e9
        logits_masked = output.clone()
        logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)

        # 若某行所有 candidates 都被 mask（不应该），回退为全候选以避免全 -inf
        # （上面 masked_fill 不会变成全 -inf 因为 cand_mask 行至少应有一个 True；但多保守处理）
        all_neg = (~cand_mask).all(dim=1)
        if all_neg.any():
            logits_masked[all_neg] = output[all_neg]  # 回退：不做 masked_fill

        pred_idx = logits_masked.argmax(dim=1)  # 现在一定只会在候选中选

        p = torch.zeros_like(output)     # [batch_size, num_classes]
        p[torch.arange(batch_size), pred_idx] = 1.0

        q[idxs] = q[idxs] * miu + (1 - miu) * p


        log_probs = torch.log_softmax(output, dim=1)
        loss = -(q[idxs] * log_probs).sum(dim=1).mean()
        return loss
    def pll_loss_vectorized_unchanged(self, output, idxs, miu=0.99):
        q = self.q
        batch_size = output.size(0)
        pred_idx = output.argmax(dim=1)  # [batch_size]
        p = torch.zeros_like(output)     # [batch_size, num_classes]
        p[torch.arange(batch_size), pred_idx] = 1.0

        # q[idxs] = q[idxs] * miu + (1 - miu) * p

        log_probs = torch.log_softmax(output, dim=1)
        loss = -(q[idxs] * log_probs).sum(dim=1).mean()

        return loss
    
    def pll_mix_up_loss(self, data, idxs, global_model_state_dict):
        """
        Compute adaptive MixUp PLL loss on a batch.

        Args:
            data: tensor, shape [B, ...], already on CPU or device (we'll move to self.device)
            idxs: tensor of dataset indices for this batch, shape [B]
            global_model_state_dict: state_dict of the global model (CPU tensors or device)

        Returns:
            loss (torch.scalar)
        """
        # 1) prepare global model: copy local model architecture and load global weights
        #    (we don't want to modify self.local_model here)
        global_model = copy.deepcopy(self.local_model)
        global_model.load_state_dict(global_model_state_dict)
        global_model.to(self.device)
        global_model.eval()

        # Ensure data & idxs on correct device
        data = data.to(self.device)
        idxs = idxs.to(self.device)

        batch_size = data.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)

        # 2) shuffle within batch to get pairing for MixUp
        perm = torch.randperm(batch_size, device=self.device)
        shuffled_data = data[perm]                       # [B, ...]
        # get q for original and shuffled (self.q indexed by dataset idx)
        q_orig = self.q[idxs]                            # [B, C]
        q_shuf = self.q[idxs[perm]]                      # [B, C]

        # 3) sample gamma (MixUp coefficient). sample scalar per-batch (or sample per-sample if wanted)
        alpha = float(self.config.mixup_alpha) if hasattr(self.config, "mixup_alpha") else float(self.config.get("mixup_alpha", 0.4))
        if alpha > 0:
            gamma_val = np.random.beta(alpha, alpha)
            # optional: make gamma >= 0.5 to keep dominant contribution (paper sometimes does)
            # gamma_val = max(gamma_val, 1.0 - gamma_val)
            gamma = torch.tensor(gamma_val, device=self.device, dtype=torch.float32)
        else:
            # no mixup
            gamma = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        # 4) build mixed inputs and mixed q (soft labels)
        mix_data = gamma * data + (1.0 - gamma) * shuffled_data      # [B, ...]
        mix_q = gamma * q_orig + (1.0 - gamma) * q_shuf              # [B, C]

        # 5) compute global-model confidences needed for adaptive weights
        #    use no_grad because we don't train global_model here
        with torch.no_grad():
            # probs for original and shuffled
            probs_orig = torch.softmax(global_model(data), dim=1)       # [B, C]
            probs_shuf = torch.softmax(global_model(shuffled_data), dim=1)  # [B, C]
            probs_mix  = torch.softmax(global_model(mix_data), dim=1)   # [B, C]

            # per-sample max-confidence
            max_orig = probs_orig.max(dim=1).values                     # [B]
            max_shuf = probs_shuf.max(dim=1).values                     # [B]
            max_mix  = probs_mix.max(dim=1).values                      # [B]

        # w1 = gamma * max(F(x_i)) + (1-gamma) * max(F(x_j))
        # w2 = max(F(mix))
        # w = w1 * w2   (elementwise)
        # gamma is scalar -> broadcast to [B]
        w1 = gamma * max_orig + (1.0 - gamma) * max_shuf   # [B]
        w2 = max_mix                                       # [B]
        weight = w1 * w2                                   # [B]

        # 6) forward with local model on mix_data and compute weighted PLL cross-entropy
        mix_output = self.local_model(mix_data)            # [B, C]
        log_probs = torch.log_softmax(mix_output, dim=1)   # [B, C]

        # per-sample PLL loss: - q_mix^T log p
        sample_loss = - (mix_q * log_probs).sum(dim=1)     # [B]

        # weighted sum, then average by batch size (paper used 1/n_m)
        loss = (weight * sample_loss).sum() / float(batch_size)

        return loss

    