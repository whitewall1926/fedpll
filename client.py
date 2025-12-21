# import torch
# import wandb
# import copy
# from torch.utils.data import DataLoader
# from common import PLLDataset, get_g, seed_worker_with_seed
# import numpy as np
# import torch.nn.functional as F
# import random

# from torch.nn.utils import parameters_to_vector


# sim = np.array([
#     [1.00,0.10,0.05,0.05,0.02,0.15,0.70,0.03,0.20,0.70],
#     [0.10,1.00,0.03,0.03,0.05,0.02,0.15,0.80,0.03,0.02],
#     [0.05,0.03,1.00,0.60,0.03,0.05,0.02,0.03,0.60,0.10],
#     [0.05,0.03,0.60,1.00,0.03,0.02,0.02,0.03,0.60,0.12],
#     [0.02,0.05,0.03,0.03,1.00,0.10,0.05,0.02,0.03,0.20],
#     [0.15,0.02,0.05,0.02,0.10,1.00,0.50,0.02,0.05,0.08],
#     [0.70,0.15,0.02,0.02,0.05,0.50,1.00,0.05,0.02,0.75],
#     [0.03,0.80,0.03,0.03,0.02,0.02,0.05,1.00,0.05,0.02],
#     [0.20,0.03,0.60,0.60,0.03,0.05,0.02,0.05,1.00,0.50],
#     [0.70,0.02,0.10,0.12,0.20,0.08,0.75,0.02,0.50,1.00],
# ], dtype=float)


# probs_matrix = []




# class Client:
#     def __init__(self,
#                  local_model, 
#                  train_dataset, 
#                  client_id,
#                  config):
#         self.client_id = client_id
#         self.config = config
#         self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#         self.rounds = 0

#         self.train_dataset = train_dataset
#         candidate_labels = []

#         counts = np.zeros((self.config.num_classes, self.config.num_classes), dtype=int)

#         # [!! 新代码 !!] 
#         # 初始化用于追踪“每类前3个”的工具
#         self.debug_indices = []
#         class_counts_tracker = np.zeros(self.config.num_classes, dtype=int)
#         num_per_class_to_track = 3
#         # [!! 新代码结束 !!]
        
#         # self.start_table = wandb.Table(columns=["Index", "True Label", "Candidates", "Q-Vector (Epoch 0)"], log_mode="MUTABLE")
#         # self.end_table   = wandb.Table(columns=["Index", "True Label", "Candidates", f"Q-Vector (Epoch {self.config.local_epochs - 1})"], log_mode="MUTABLE")

#         self.start_table = wandb.Table(columns=["Index", "True Label", "Candidates", "Q-Vector (Epoch 0)"])
#         self.end_table   = wandb.Table(columns=["Index", "True Label", "Candidates", f"Q-Vector (Epoch {self.config.local_epochs - 1})"])


#         for i in range(len(train_dataset)):
#             _, label = train_dataset[i]
#             candidate = torch.zeros(self.config.num_classes, dtype=int)
#             candidate[label] = 1         

#             # [!! 新代码 !!] 检查是否需要追踪这个索引
#             if class_counts_tracker[label] < num_per_class_to_track:
#                 self.debug_indices.append(i)
#                 class_counts_tracker[label] += 1
#             # [!! 新代码结束 !!]


#             # label_sim = sim[label].copy()
#             # import random
#             # for i in range(self.config.num_classes):
#             #     if i != label and random.random() < label_sim[i]:
#             #         candidate[i] = 1
#             # print("true label =", label, "candidate =", candidate)

#             # cnt = int(self.config.noise_level * (self.config.num_classes))
#             # label_sim = sim[label].copy()
#             # label_sim_idx = np.argsort(label_sim)[::-1].copy()
#             # choosed_labels = label_sim_idx[:cnt]
#             # for idx in choosed_labels:
#             #     if random.random() < label_sim[idx]:
#             #         candidate[idx] = 1

#             # candidate[choosed_labels] = 1
#             # extra_k = 2 # 可配置，默认 2
#             # if extra_k > 0:
#             #     # 排除已选与真实标签
#             #     excluded = set(choosed_labels.tolist())
#             #     excluded.add(label)   # 如果不想把真实标签剔除，把这一行注释掉
#             #     pool = [i for i in range(self.config.num_classes) if i not in excluded]
#             #     if len(pool) > 0:
#             #         extra = np.random.choice(pool, size=min(extra_k, len(pool)), replace=False)
#             #         candidate[extra] = 1

#             # import random
#             # others = []
#             # for i in range(self.config.num_classes):
#             #     if i not in choosed_labels:
#             #         others.append(i)
#             # candidate[random.choice(others)] = 1
            
#             import random
#             for other_label in range(self.config.num_classes):
#                 if other_label != label and random.random() < self.config.noise_level:
#                     candidate[other_label] = 1

#             for j in range(len(candidate)):
#                 counts[label][j] += candidate[j].item()

#             candidate_labels.append(candidate)
            
#         print(f'client id:{client_id}\n', counts)
#         self.train_plldataset = PLLDataset(self.train_dataset, num_classes=self.config.num_classes, candidate_labels=candidate_labels, rho=self.config.noise_level)
        
        
         
#         q = []
#         for idx in range(len(self.train_plldataset)):
#             candidates = self.train_plldataset.candidate_labels[idx]
#             p = candidates.clone().float()
#             p /= (candidates.sum() * 1.0)
#             # if (idx < 4):
#             #     print(p, "true target=", self.train_plldataset[idx][1])
#             q.append(p)
#         q = torch.stack(q).to(device=self.device)
#         self.q = q

#         # [!! 修改后的打印 !!] 
#         print(f"Client {self.client_id} tracking {len(self.debug_indices)} fixed indices (up to {num_per_class_to_track} per class):")
#         print(f"  {self.debug_indices}")
#         # [!! 修改结束 !!]

#         self.local_model = copy.deepcopy(local_model).to(self.device)

        
        
#         self.train_loader = DataLoader(
#             self.train_plldataset,
#             batch_size=self.config.batch_size,
#             shuffle=True,
#             num_workers=1,
#             drop_last=False,
#             worker_init_fn=seed_worker_with_seed(self.config.seed),
#             generator=torch.Generator().manual_seed(self.config.seed)
#         )

#         self.optimizer = None
#         self.epochs = self.config.local_epochs
#         self.global_model = None

        
#     def compute_flattened_local_grad(self, num_batches=1, use_global_model=False):
#         """
#         在本地使用当前 local_model（或 global_model，如果 use_global_model=True）对 num_batches 批次计算梯度并返回展平向量。
#         - num_batches: 从 train_loader 读取几个 batch 计算平均梯度（默认 1）
#         - 返回值: 1D torch.Tensor (dtype float32) 在 CPU 上（便于网络传输或 server 汇总）
#         注意：不会修改模型参数（只 forward+backward，并清理 grad）
#         """
#         model = self.local_model if not use_global_model else self.global_model
#         model.to(self.device)
#         model.zero_grad()
#         accum_grads = None
#         batches_used = 0

#         # iterate up to num_batches
#         for i, batch in enumerate(self.train_loader):
#             data, target, candidates, idxs = batch
#             data = data.to(self.device)
#             # build q_batch for this batch if necessary (self.q exists)
#             if isinstance(idxs, torch.Tensor):
#                 idxs_dev = idxs.to(self.device)
#             else:
#                 idxs_dev = torch.tensor(idxs, device=self.device)

#             # compute loss (use same loss as training, but no create_graph)
#             output = model(data)
#             # 使用 pll_loss_vectorized（不改变 q permanently）
#             loss = self.pll_loss_vectorized_unchanged(output=output, idxs=idxs_dev, miu=0.99)
#             model.zero_grad()
#             loss.backward()

#             # collect grads in same order as model.parameters()
#             grads = []
#             for p in model.parameters():
#                 if p.grad is not None:
#                     grads.append(p.grad.detach().clone().to(torch.float32))
#                 else:
#                     grads.append(torch.zeros_like(p).to(torch.float32))

#             # flatten
#             g_vec = parameters_to_vector(grads)  # 1D tensor on current device
#             g_vec = g_vec.detach().cpu()         # move to CPU for safe aggregation / transfer

#             if accum_grads is None:
#                 accum_grads = g_vec
#             else:
#                 accum_grads += g_vec
#             batches_used += 1

#             if batches_used >= num_batches:
#                 break

#         if accum_grads is None:
#             # no batches (empty dataset) -> zero vector of length equal to model param vector
#             dummy = [torch.zeros_like(p) for p in model.parameters()]
#             accum_grads = parameters_to_vector(dummy).detach().cpu()

#         # average across used batches
#         accum_grads /= float(max(1, batches_used))
#         return accum_grads  # CPU 1D tensor


    
#     # -------------------------
#     # compute_grad_vector: 返回展平梯度向量
#     # -------------------------
#     def compute_grad_vector(self, model, data, q_batch, create_graph=False):
#         """
#         Compute gradient vector g = flatten( d Lc / d theta ) for given model & batch.
#         - model: nn.Module (must be on self.device)
#         - data: input tensor on device
#         - q_batch: soft labels tensor [B, C], on device (detached or requires_grad False)
#         - create_graph: whether to create graph for higher-order derivatives
#         Returns: 1D tensor of shape [D] (same device as model)
#         """
#         model.zero_grad()
#         # forward
#         out = model(data)                         # [B, C]
#         logp = torch.log_softmax(out, dim=1)
#         loss = - (q_batch * logp).sum(dim=1).mean()
#         # compute grads (as tuple)
#         grads = torch.autograd.grad(loss, model.parameters(), create_graph=create_graph, retain_graph=True, allow_unused=True)
#         # some params may be None -> replace with zeros
#         grads = [g if g is not None else torch.zeros_like(p, device=self.device) for g, p in zip(grads, model.parameters())]
#         # flatten to vector
#         g_vec = parameters_to_vector(grads)  # returns 1D tensor
#         return g_vec, loss

#     # -------------------------
#     # gradient_alignment_loss: 余弦相似度 negative (with stability)
#     # -------------------------
#     def gradient_alignment_loss(self, local_grad_vector, global_grad_vector, eps=1e-8):
#         """
#         local_grad_vector: 1D tensor, maybe requires_grad True (if create_graph used)
#         global_grad_vector: 1D tensor, should be detached (no grad) representing g_F
#         returns scalar tensor L_ga = -cos(local, global)
#         """
#         # ensure same device
#         local = local_grad_vector
#         global_v = global_grad_vector.to(local.device)

#         # norms
#         n1 = torch.norm(local)
#         n2 = torch.norm(global_v)
#         if n1.item() < eps or n2.item() < eps:
#             # if any norm is too small, return zero loss (or small value)
#             return torch.tensor(0.0, device=local.device, dtype=local.dtype)

#         cos = (local * global_v).sum() / (n1 * n2 + eps)  # scalar
#         return -cos  # we minimize negative cosine to maximize alignment

#     def update_config(self, new_config):
#         self.config = new_config
#         self.train_loader = DataLoader(
#             self.train_plldataset,
#             batch_size=self.config.batch_size,
#             shuffle=True,
#             num_workers=1,
#             drop_last=False,
#             worker_init_fn=seed_worker_with_seed(self.config.seed),
#             generator=torch.Generator().manual_seed(self.config.seed)
#         )
#         self.epochs = self.config.local_epochs


#     # def train(self, 
#     #           global_model_state_dict, 
#     #           roud,
#     #           new_config=None,
#     #           global_grad_vector=None):
#     #     if new_config != None:
#     #         self.config = None
#     #         self.update_config(new_config)

#     #     print(f"roud {roud} trainning..............................\n")
#     #     self.local_model.load_state_dict(global_model_state_dict)
#     #     self.global_model = copy.deepcopy(self.local_model)
#     #     self.global_model.eval()
#     #     if global_grad_vector is not None:
#     #         g_global_vec = global_grad_vector.to(self.device).detach()  # treat as constant
#     #     else:
#     #         g_global_vec = None
#     #     if self.config.optimizer.lower() == "adam":
#     #         self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.config.lr)
#     #         print(f'client {self.client_id} using adam ')
#     #     else:
#     #         self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.config.lr, momentum=0.5)
#     #         print(f'client {self.client_id} using sgd ')


#     #     self.local_model.train()
        
#     #     q = self.q
#     #     avg_train_acc = 0
#     #     avg_train_loss = 0
#     #     # print("---", self.epochs, self.config.local_epochs)

#     #     print(f"client{self.client_id} using lc_loss\n")
#     #     if self.config.mix == True:
#     #         print(f"client{self.client_id} using mix_loss\n")
#     #     if self.config.ga == True:
#     #         print(f"client{self.client_id} using lga\n")
        
#     #     for epoch in range(self.epochs):
            
#     #         total_samples= 0
#     #         train_loss = 0
#     #         train_acc = 0

#     #         for data, target, candidates, idxs in self.train_loader:
#     #             data = data.to(self.device)
#     #             target = target.to(self.device)

#     #             self.optimizer.zero_grad()
#     #             output = self.local_model(data)
#     #             lc_loss = self.pll_loss_vectorized(output=output, idxs=idxs, candidates=candidates, miu=0.99) 
#     #             # lc_loss = self.pll_loss_vectorized_soft_preds(output=output, idxs=idxs, candidates=candidates, miu=0.99) 
#     #             mix_loss = 0.0
#     #             if self.config.mix == True:
#     #                 mix_loss =  self.pll_mix_up_loss(data=data, idxs=idxs, global_model_state_dict=global_model_state_dict)

#     #             lga = 0.0
#     #             if self.config.ga == True:
#     #                 g_local_vec, _ = self.compute_grad_vector(self.local_model, data, q_batch=self.q[idxs], create_graph=True)
#     #                 #    g_global computed on global_model (no higher-order graph needed)
#     #                 if g_local_vec.numel() != g_global_vec.numel():
#     #                    raise ValueError("Dimension mismatch between local and global grad vectors")
#     #                 lga = self.gradient_alignment_loss(g_local_vec, g_global_vec)

#     #             loss = lc_loss + mix_loss + lga

#     #             loss.backward()

#     #             batch_size = data.size(0)
#     #             train_loss += loss.detach().item() * batch_size
#     #             total_samples += batch_size

#     #             preds = output.argmax(dim=1)
#     #             train_acc += (preds==target).sum().item()

#     #             self.optimizer.step()

                


#     #         # print(q[:5], "\n")
#     #         # [!! 修改后的日志 !!] 
#     #         # 只在第一个和最后一个epoch打印固定的索引
#     #         if epoch == 0 or epoch == self.epochs - 1:
#     #             print(f"--- Client {self.client_id} Epoch {epoch} Q-Vector Inspection ---")
                
#     #             # 1. 使用固定的 self.debug_indices
#     #             for idx in self.debug_indices:
#     #                 if idx >= len(self.train_dataset): continue # 安全检查
                    
#     #                 # 2. 循环打印
#     #                 q_vector = self.q[idx].cpu().numpy()
#     #                 true_label = self.train_dataset[idx][1] 
#     #                 candidate_vector = self.train_plldataset.candidate_labels[idx]
#     #                 candidate_indices = torch.where(candidate_vector == 1)[0].cpu().numpy()
                    
#     #                 # 格式化q向量，保留2位小数
#     #                 q_formatted = ", ".join([f"{x:.2f}" for x in q_vector])
                    
#     #                 print(f"  [Tracked Index {idx}]:")
#     #                 print(f"    True Label:   {true_label}")
#     #                 print(f"    Candidates:   {candidate_indices}")
#     #                 print(f"    Q-Vector:     [{q_formatted}]")
                
#     #             print("--------------------------------------------------\n")
#     #         # [!! 日志修改结束 !!]

#     #         train_loss = train_loss /  total_samples
#     #         train_acc = train_acc / total_samples
#     #         print(f"----> client: {self.client_id} | Roud: {roud:3d} | Epoch: {epoch:3d} | "
#     #           f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}\n")

#     #         # wandb.log({
#     #         # f"client_train/{self.client_id}/acc": train_acc,
#     #         # f"client_train/{self.client_id}/loss": train_loss   
#     #         # }, step=roud)
#     #         avg_train_acc = avg_train_acc + train_acc
#     #         avg_train_loss = avg_train_loss + train_loss 
#     #     avg_train_acc = avg_train_acc * 1.0 /  self.epochs
#     #     avg_train_loss = avg_train_loss * 1.0 / self.epochs
#     #     wandb.log({
#     #         f"client_train/{self.client_id}/acc": avg_train_acc,
#     #         f"client_train/{self.client_id}/loss": avg_train_loss   
#     #         }, step=roud)
#     #     return self.local_model.state_dict() 
    
#     def train(self, 
#             global_model_state_dict, 
#             roud,
#             new_config=None,
#             global_grad_vector=None):
#         if new_config != None:
#             self.config = None
#             self.update_config(new_config)

#         print(f"roud {roud} trainning..............................\n")
#         self.local_model.load_state_dict(global_model_state_dict)
#         self.global_model = copy.deepcopy(self.local_model)
#         self.global_model.eval()
#         if global_grad_vector is not None:
#             g_global_vec = global_grad_vector.to(self.device).detach()  # treat as constant
#         else:
#             g_global_vec = None
#         if self.config.optimizer.lower() == "adam":
#             self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.config.lr)
#             print(f'client {self.client_id} using adam ')
#         else:
#             self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.config.lr, momentum=0.5)
#             print(f'client {self.client_id} using sgd ')


#         self.local_model.train()
        
#         q = self.q
#         avg_train_acc = 0
#         avg_train_loss = 0

#         # [!! 新代码 !!] 初始化用于 wandb.Table 的列表
#         start_epoch_data = []
#         end_epoch_data = []

#         print(f"client{self.client_id} using lc_loss\n")
#         if self.config.mix == True:
#             print(f"client{self.client_id} using mix_loss\n")
#         if self.config.ga == True:
#             print(f"client{self.client_id} using lga\n")
        
#         for epoch in range(self.epochs):
            
#             total_samples= 0
#             train_loss = 0
#             train_acc = 0

#             for data, target, candidates, idxs in self.train_loader:
#                 data = data.to(self.device)
#                 target = target.to(self.device)

#                 self.optimizer.zero_grad()
#                 output = self.local_model(data)
#                 lc_loss = self.pll_loss_vectorized(output=output, idxs=idxs, candidates=candidates, miu=0.99) 
#                 # lc_loss = self.pll_loss_vectorized_soft_preds(output=output, idxs=idxs, candidates=candidates, miu=0.99) 
#                 mix_loss = 0.0
#                 if self.config.mix == True:
#                     mix_loss =  self.pll_mix_up_loss(data=data, idxs=idxs, global_model_state_dict=global_model_state_dict)

#                 lga = 0.0
#                 if self.config.ga == True:
#                     g_local_vec, _ = self.compute_grad_vector(self.local_model, data, q_batch=self.q[idxs], create_graph=True)
#                     #     g_global computed on global_model (no higher-order graph needed)
#                     if g_local_vec.numel() != g_global_vec.numel():
#                         raise ValueError("Dimension mismatch between local and global grad vectors")
#                     lga = self.gradient_alignment_loss(g_local_vec, g_global_vec)

#                 loss = lc_loss + mix_loss + lga

#                 loss.backward()

#                 batch_size = data.size(0)
#                 train_loss += loss.detach().item() * batch_size
#                 total_samples += batch_size

#                 preds = output.argmax(dim=1)
#                 train_acc += (preds==target).sum().item()

#                 self.optimizer.step()

            
#             # [!! 修改后的日志 !!] 
#             # 捕获第一个和最后一个epoch的数据
#             if epoch == 0 or epoch == self.epochs - 1:
                
#                 # [!! 新代码 !!] 选择要填充的列表
#                 current_data_list = start_epoch_data if epoch == 0 else end_epoch_data
                
#                 print(f"--- Client {self.client_id} Epoch {epoch} Q-Vector Inspection (for console) ---")
                
#                 for idx in self.debug_indices:
#                     if idx >= len(self.train_dataset): continue 
                    
#                     q_vector = self.q[idx].cpu().numpy()
#                     true_label = self.train_dataset[idx][1] 
#                     candidate_vector = self.train_plldataset.candidate_labels[idx]
#                     candidate_indices = torch.where(candidate_vector == 1)[0].cpu().numpy()
#                     q_formatted = ", ".join([f"{x:.2f}" for x in q_vector])
                    
#                     # [!! 新代码 !!] 填充列表
#                     current_data_list.append([
#                         idx, 
#                         true_label, 
#                         str(candidate_indices),  # wandb.Table 需要字符串
#                         q_formatted
#                     ])

#                     # 保留控制台打印
#                     print(f"  [Tracked Index {idx}]:")
#                     print(f"    True Label:   {true_label}")
#                     print(f"    Candidates:   {candidate_indices}")
#                     print(f"    Q-Vector:     [{q_formatted}]")
                
#                 print("--------------------------------------------------\n")
#             # [!! 日志修改结束 !!]
            

#             train_loss = train_loss /  total_samples
#             train_acc = train_acc / total_samples
#             print(f"----> client: {self.client_id} | Roud: {roud:3d} | Epoch: {epoch:3d} | "
#                 f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}\n")

#             avg_train_acc = avg_train_acc + train_acc
#             avg_train_loss = avg_train_loss + train_loss 
        
#         # ----- [!! WANDB LOGGING !!] -----
#         avg_train_acc = avg_train_acc * 1.0 /  self.epochs
#         avg_train_loss = avg_train_loss * 1.0 / self.epochs

#         # [!! 新代码 !!] 创建并填充 wandb 表格
#         logs_to_wandb = {
#             f"client_train/{self.client_id}/acc": avg_train_acc,
#             f"client_train/{self.client_id}/loss": avg_train_loss  
#         }

#         # 1. 创建 Epoch 0 表格
#         if start_epoch_data:
#             # start_table = wandb.Table(columns=["Index", "True Label", "Candidates", f"Q-Vector (Epoch 0)"], log_mode="INCREMENTAL")
#             start_table = self.start_table
#             for row in start_epoch_data:
#                 start_table.add_data(*row)
#             logs_to_wandb[f"client_train/{self.client_id}/q_vectors_start"] = start_table

#         # 2. 创建最后一个 Epoch 表格
#         if end_epoch_data:
#             # end_table = wandb.Table(columns=["Index", "True Label", "Candidates", f"Q-Vector (Epoch {self.epochs - 1})"])
#             end_table = self.end_table
#             for row in end_epoch_data:
#                 end_table.add_data(*row)
#             logs_to_wandb[f"client_train/{self.client_id}/q_vectors_end"] = end_table

#         # 统一发送所有日志
#         wandb.log(logs_to_wandb, step=roud)
#         # ----- [!! WANDB LOGGING 结束 !!] -----

#         return self.local_model.state_dict()
    

#     def get_acc_matrix(self, test_loader):
#         self.local_model.eval()
#         # test_acc = 0
#         # test_loss = 0
        
#         # 初始化混淆矩阵（行：实际类别，列：预测类别）
#         acc_matrix = np.zeros((self.config.num_classes, self.config.num_classes), dtype=int)

#         with torch.no_grad():
#             for data, target in test_loader:
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 output = self.local_model(data)
#                 preds = output.argmax(dim=1)  # 获取预测类别
                
#                 # 计算准确率
#                 # test_acc += (preds == target).sum().item()
                
#                 # 统计混淆矩阵：将张量转为numpy数组（需先移到CPU）
#                 target_np = target.cpu().numpy()
#                 preds_np = preds.cpu().numpy()
                
#                 # 遍历每个样本，更新混淆矩阵
#                 for t, p in zip(target_np, preds_np):
#                     acc_matrix[t][p] += 1  # 实际类别t被预测为p，对应位置+1
#         # 计算整体准确率
#         # test_acc = test_acc / len(test_loader.dataset)

#         # 日志记录和打印
        
#         return acc_matrix  # 可选择返回混淆矩阵
    

#     def test(self, test_loader, epoch, criterion=None):
#         self.local_model.eval()
#         test_acc = 0
#         test_loss = 0
        
#         acc_matrix = np.zeros((self.config.num_classes, self.config.num_classes))

#         with torch.no_grad():
#             for data, target in test_loader:
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 output = self.local_model(data)
#                 # loss = criterion(output, target)
#                 # test_loss += loss.item() * data.size(0)
#                 preds = output.argmax(dim=1)
#                 test_acc += (preds == target).sum().item()
                
#         test_acc = test_acc / len(test_loader.dataset)
#         # test_loss = test_loss / len(test_loader.dataset)

#         wandb.log({f"client_test/{self.client_id}/acc":test_acc}, step=epoch)
#         # wandb.log({"test/loss":test_loss})
#         print(f"client:{self.client_id}: Epoch: {epoch:3d} | "
#             #   f"Test Loss: {test_loss:.6f} | "
#             f"Test Acc: {test_acc:.4f}\n")
#         return test_acc
#     def pll_loss_vectorized_soft_preds(self, output, idxs, candidates,  miu=0.99):
#         q = self.q
#         batch_size = output.size(0)
        
#         # cand_mask = candidates.to(device=self.device).bool()
#         # neg_inf = -1e-9
#         # logits_masked = output.clone()
#         # logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)
#         # pred_idx = logits_masked.argmax(dim=1)  # [batch_size]
#         device = output.device
#         B, C = output.shape

#         # ensure cand mask is boolean tensor on same device
#         if not isinstance(candidates, torch.Tensor):
#             # if it's list-of-lists, build mask (rare)
#             cand_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
#             for i, cand in enumerate(candidates):
#                 if len(cand) == 0:
#                     cand_mask[i, :] = True
#                 else:
#                     cand_mask[i, list(cand)] = True
#         else:
#             cand_mask = candidates.to(device=device).bool()

#         # 1) 把非 candidate 的 logits 设为一个很小的负数，再 argmax（保证只在候选中选）
#         neg_inf = -1e9
#         logits_masked = output.clone()
#         logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)

#         # 若某行所有 candidates 都被 mask（不应该），回退为全候选以避免全 -inf
#         # （上面 masked_fill 不会变成全 -inf 因为 cand_mask 行至少应有一个 True；但多保守处理）
#         all_neg = (~cand_mask).all(dim=1)
#         if all_neg.any():
#             logits_masked[all_neg] = output[all_neg]  # 回退：不做 masked_fill

#         pred_idx = logits_masked.argmax(dim=1)  # 现在一定只会在候选中选

#         p = torch.zeros_like(output)     # [batch_size, num_classes]
#         p[torch.arange(batch_size), pred_idx] = 1.0
        
#         p_soft = torch.softmax(logits_masked.detach(), dim=1)
#         q[idxs] = q[idxs] * miu + (1 - miu) * p_soft


#         log_probs = torch.log_softmax(output, dim=1)
#         loss = -(q[idxs] * log_probs).sum(dim=1).mean()
#         return loss
#     def pll_loss_vectorized(self, output, idxs, candidates,  miu=0.99):
#         q = self.q
#         batch_size = output.size(0)
        
#         # cand_mask = candidates.to(device=self.device).bool()
#         # neg_inf = -1e-9
#         # logits_masked = output.clone()
#         # logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)
#         # pred_idx = logits_masked.argmax(dim=1)  # [batch_size]
#         device = output.device
#         B, C = output.shape

#         # ensure cand mask is boolean tensor on same device
#         if not isinstance(candidates, torch.Tensor):
#             # if it's list-of-lists, build mask (rare)
#             cand_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
#             for i, cand in enumerate(candidates):
#                 if len(cand) == 0:
#                     cand_mask[i, :] = True
#                 else:
#                     cand_mask[i, list(cand)] = True
#         else:
#             cand_mask = candidates.to(device=device).bool()

#         # 1) 把非 candidate 的 logits 设为一个很小的负数，再 argmax（保证只在候选中选）
#         neg_inf = -1e9
#         logits_masked = output.clone()
#         logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)

#         # 若某行所有 candidates 都被 mask（不应该），回退为全候选以避免全 -inf
#         # （上面 masked_fill 不会变成全 -inf 因为 cand_mask 行至少应有一个 True；但多保守处理）
#         all_neg = (~cand_mask).all(dim=1)
#         if all_neg.any():
#             logits_masked[all_neg] = output[all_neg]  # 回退：不做 masked_fill

#         pred_idx = logits_masked.argmax(dim=1)  # 现在一定只会在候选中选

#         p = torch.zeros_like(output)     # [batch_size, num_classes]
#         p[torch.arange(batch_size), pred_idx] = 1.0

#         q[idxs] = q[idxs] * miu + (1 - miu) * p


#         log_probs = torch.log_softmax(output, dim=1)
#         loss = -(q[idxs] * log_probs).sum(dim=1).mean()
#         return loss
#     def pll_loss_vectorized_unchanged(self, output, idxs, miu=0.99):
#         q = self.q
#         batch_size = output.size(0)
#         pred_idx = output.argmax(dim=1)  # [batch_size]
#         p = torch.zeros_like(output)     # [batch_size, num_classes]
#         p[torch.arange(batch_size), pred_idx] = 1.0

#         # q[idxs] = q[idxs] * miu + (1 - miu) * p

#         log_probs = torch.log_softmax(output, dim=1)
#         loss = -(q[idxs] * log_probs).sum(dim=1).mean()

#         return loss
    
#     def pll_mix_up_loss(self, data, idxs, global_model_state_dict):
#         """
#         Compute adaptive MixUp PLL loss on a batch.

#         Args:
#             data: tensor, shape [B, ...], already on CPU or device (we'll move to self.device)
#             idxs: tensor of dataset indices for this batch, shape [B]
#             global_model_state_dict: state_dict of the global model (CPU tensors or device)

#         Returns:
#             loss (torch.scalar)
#         """
#         # 1) prepare global model: copy local model architecture and load global weights
#         #    (we don't want to modify self.local_model here)
#         global_model = copy.deepcopy(self.local_model)
#         global_model.load_state_dict(global_model_state_dict)
#         global_model.to(self.device)
#         global_model.eval()

#         # Ensure data & idxs on correct device
#         data = data.to(self.device)
#         idxs = idxs.to(self.device)

#         batch_size = data.size(0)
#         if batch_size == 0:
#             return torch.tensor(0.0, device=self.device)

#         # 2) shuffle within batch to get pairing for MixUp
#         perm = torch.randperm(batch_size, device=self.device)
#         shuffled_data = data[perm]                       # [B, ...]
#         # get q for original and shuffled (self.q indexed by dataset idx)
#         q_orig = self.q[idxs]                            # [B, C]
#         q_shuf = self.q[idxs[perm]]                      # [B, C]

#         # 3) sample gamma (MixUp coefficient). sample scalar per-batch (or sample per-sample if wanted)
#         alpha = float(self.config.mixup_alpha) if hasattr(self.config, "mixup_alpha") else float(self.config.get("mixup_alpha", 0.4))
#         if alpha > 0:
#             gamma_val = np.random.beta(alpha, alpha)
#             # optional: make gamma >= 0.5 to keep dominant contribution (paper sometimes does)
#             # gamma_val = max(gamma_val, 1.0 - gamma_val)
#             gamma = torch.tensor(gamma_val, device=self.device, dtype=torch.float32)
#         else:
#             # no mixup
#             gamma = torch.tensor(1.0, device=self.device, dtype=torch.float32)

#         # 4) build mixed inputs and mixed q (soft labels)
#         mix_data = gamma * data + (1.0 - gamma) * shuffled_data      # [B, ...]
#         mix_q = gamma * q_orig + (1.0 - gamma) * q_shuf              # [B, C]

#         # 5) compute global-model confidences needed for adaptive weights
#         #    use no_grad because we don't train global_model here
#         with torch.no_grad():
#             # probs for original and shuffled
#             probs_orig = torch.softmax(global_model(data), dim=1)       # [B, C]
#             probs_shuf = torch.softmax(global_model(shuffled_data), dim=1)  # [B, C]
#             probs_mix  = torch.softmax(global_model(mix_data), dim=1)   # [B, C]

#             # per-sample max-confidence
#             max_orig = probs_orig.max(dim=1).values                     # [B]
#             max_shuf = probs_shuf.max(dim=1).values                     # [B]
#             max_mix  = probs_mix.max(dim=1).values                      # [B]

#         # w1 = gamma * max(F(x_i)) + (1-gamma) * max(F(x_j))
#         # w2 = max(F(mix))
#         # w = w1 * w2   (elementwise)
#         # gamma is scalar -> broadcast to [B]
#         w1 = gamma * max_orig + (1.0 - gamma) * max_shuf   # [B]
#         w2 = max_mix                                       # [B]
#         weight = w1 * w2                                   # [B]

#         # 6) forward with local model on mix_data and compute weighted PLL cross-entropy
#         mix_output = self.local_model(mix_data)            # [B, C]
#         log_probs = torch.log_softmax(mix_output, dim=1)   # [B, C]

#         # per-sample PLL loss: - q_mix^T log p
#         sample_loss = - (mix_q * log_probs).sum(dim=1)     # [B]

#         # weighted sum, then average by batch size (paper used 1/n_m)
#         loss = (weight * sample_loss).sum() / float(batch_size)

#         return loss

## ----------------分割线

import torch
import wandb
import copy
from torch.utils.data import DataLoader
from common import PLLDataset, get_g, seed_worker_with_seed
import numpy as np
import torch.nn.functional as F
import random
import os
from torch.nn.utils import parameters_to_vector
import matplotlib.pyplot as plt
from common import denormalize_image, svhn_mean, svhn_std


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
        self.client_id = client_id
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rounds = 0

        self.train_dataset = train_dataset
        candidate_labels = []

        counts = np.zeros((self.config.num_classes, self.config.num_classes), dtype=int)

        # [!! 新代码 !!] 
        # 初始化用于追踪“每类前3个”的工具
        self.debug_indices = []
        class_counts_tracker = np.zeros(self.config.num_classes, dtype=int)
        num_per_class_to_track = 3

        
        exp_id = str(self.config.get("exp_id", "unknown_id"))
        exp_name = str(self.config.get("exp_name", "unknown_run"))

        self.log_dir = os.path.join("csv_logs", f"{exp_id}_{exp_name}")
        os.makedirs(self.log_dir, exist_ok=True) # 创建这个文件夹


        # [!! 新代码结束 !!]
        
        # [!! 修改点 !!] 
        # 1. 为 wandb.Table 添加 "Round" 列
        # 2. 初始化 CSV 日志文件并写入表头
        
        # 为 wandb.Table 更新列名
        self.start_table = wandb.Table(columns=["Round", "Index", "True Label", "Candidates", "Q-Vector (Epoch 0)"])
        self.end_table   = wandb.Table(columns=["Round", "Index", "True Label", "Candidates", f"Q-Vector (Epoch {self.config.local_epochs - 1})"])

        # [!! 新 CSV INIT !!]
        self.q_log_filename = os.path.join(self.log_dir, f'client_{self.client_id}_q_log.csv')
        self.csv_header = "Round,Epoch,Index,TrueLabel,Candidates,QVector\n"
        try:
            # 'w' 模式会覆盖旧实验的日志，这通常是期望的行为
            with open(self.q_log_filename, 'w') as f: 
                f.write(self.csv_header)
            print(f"Client {self.client_id} logging Q-vectors to {self.q_log_filename}")
        except IOError as e:
            print(f"Warning: Could not write CSV header to {self.q_log_filename}: {e}")
        # [!! CSV INIT 结束 !!]


        for i in range(len(train_dataset)):
            data, label = train_dataset[i]

            # if self.client_id == 6 and i == 183:
            #     img_to_plot = denormalize_image(data)
        
            #     plt.figure(figsize=(4, 4))
            #     plt.imshow(img_to_plot)
            #     plt.title(f"Client {self.client_id} / Sample {183}\n / True Label: {label})")
            #     plt.axis('off')
            #     filename = f"client_{self.client_id}_sample_{183}_label_{label}.png"
            #     plt.savefig(filename)
            #     plt.show() # 在 Jupyter 中显示
            #     print(f"图像已保存为: {filename}")
            
            candidate = torch.zeros(self.config.num_classes, dtype=int)
            candidate[label] = 1        

            # [!! 新代码 !!] 检查是否需要追踪这个索引
            if class_counts_tracker[label] < num_per_class_to_track:
                self.debug_indices.append(i)
                class_counts_tracker[label] += 1
            # [!! 新代码结束 !!]


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
            # for idx in choosed_labels:
            #     if random.random() < label_sim[idx]:
            #         candidate[idx] = 1

            # ... (其他候选标签生成逻辑) ...

            # 根据噪声等级随机生成
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

        # [!! 修改后的打印 !!] 
        print(f"Client {self.client_id} tracking {len(self.debug_indices)} fixed indices (up to {num_per_class_to_track} per class):")
        print(f"  {self.debug_indices}")
        # [!! 修改结束 !!]

        self.local_model = copy.deepcopy(local_model).to(self.device)

        
        
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
        ... (此函数内容不变) ...
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
        ... (此函数内容不变) ...
        """
        model.zero_grad()
        # forward
        out = model(data)                      # [B, C]
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
        ... (此函数内容不变) ...
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

        # [!! 新代码 !!] 初始化用于 wandb.Table 的列表
        start_epoch_data = []
        end_epoch_data = []

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

                #使用硬标记代替如果存在某个置信度超过0.5
                lc_loss = self.pll_loss_vectorized_hard(output=output, idxs=idxs, candidates=candidates, miu=0.99, roud=roud)

                mix_loss = 0.0
                if self.config.mix == True:
                    mix_loss =  self.pll_mix_up_loss(data=data, idxs=idxs, global_model_state_dict=global_model_state_dict)

                lga = 0.0
                if self.config.ga == True:
                    g_local_vec, _ = self.compute_grad_vector(self.local_model, data, q_batch=self.q[idxs], create_graph=True)
                    #     g_global computed on global_model (no higher-order graph needed)
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

                
            # [!! 修改后的日志 !!] 
            # 捕获第一个和最后一个epoch的数据
            if epoch == 0 or epoch == self.epochs - 1:
                
                # [!! 新代码 !!] 选择要填充的列表
                current_data_list = start_epoch_data if epoch == 0 else end_epoch_data
                
                print(f"--- Client {self.client_id} Epoch {epoch} Q-Vector Inspection (for console) ---")
                
                for idx in self.debug_indices:
                    if idx >= len(self.train_dataset): continue 
                    
                    q_vector = self.q[idx].cpu().numpy()
                    true_label = self.train_dataset[idx][1] 
                    candidate_vector = self.train_plldataset.candidate_labels[idx]
                    candidate_indices = torch.where(candidate_vector == 1)[0].cpu().numpy()
                    q_formatted = ", ".join([f"{x:.2f}" for x in q_vector])
                    
                    # [!! 修改点 !!] 填充列表, 加入 roud
                    current_data_list.append([
                        roud, # <--- [!! ADDED ROUD !!]
                        idx, 
                        true_label, 
                        str(candidate_indices),  # wandb.Table 需要字符串
                        q_formatted
                    ])

                    # 保留控制台打印
                    print(f"  [Tracked Index {idx}]:")
                    print(f"    True Label:   {true_label}")
                    print(f"    Candidates:   {candidate_indices}")
                    print(f"    Q-Vector:     [{q_formatted}]")
                
                print("--------------------------------------------------\n")
            # [!! 日志修改结束 !!]
            

            train_loss = train_loss /  total_samples
            train_acc = train_acc / total_samples
            print(f"----> client: {self.client_id} | Roud: {roud:3d} | Epoch: {epoch:3d} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}\n")

            avg_train_acc = avg_train_acc + train_acc
            avg_train_loss = avg_train_loss + train_loss 
        
        # ----- [!! WANDB LOGGING & CSV LOGGING !!] -----
        avg_train_acc = avg_train_acc * 1.0 /  self.epochs
        avg_train_loss = avg_train_loss * 1.0 / self.epochs

        # [!! 新代码 !!] 创建并填充 wandb 表格
        logs_to_wandb = {
            f"client_train/{self.client_id}/acc": avg_train_acc,
            f"client_train/{self.client_id}/loss": avg_train_loss  
        }

        # [!! 新 CSV 逻辑 !!]
        # 在 wandb.log 之前，增量式写入 CSV 文件
        try:
            with open(self.q_log_filename, 'a') as f: # 'a' for append
                
                # 1. 处理 Epoch 0 data
                if start_epoch_data:
                    for row in start_epoch_data:
                        # row is [roud, idx, true_label, candidates_str, q_formatted]
                        self.start_table.add_data(*row) # 添加到 wandb table
                        
                        # 写入 CSV
                        q_csv_safe = f'"{row[4]}"' # 带引号，防止q_formatted中的逗号
                        candidates_csv = row[3].replace(" ", "") # e.g., "[0,6,9]"
                        # 格式: Round,Epoch,Index,TrueLabel,Candidates,QVector
                        f.write(f"{row[0]},0,{row[1]},{row[2]},{candidates_csv},{q_csv_safe}\n")
                        
                    # logs_to_wandb[f"client_train/{self.client_id}/q_vectors_start"] = self.start_table

                # 2. 处理最后一个 Epoch data
                if end_epoch_data:
                    for row in end_epoch_data:
                        self.end_table.add_data(*row) # 添加到 wandb table
                        
                        # 写入 CSV
                        q_csv_safe = f'"{row[4]}"'
                        candidates_csv = row[3].replace(" ", "")
                        # 格式: Round,Epoch,Index,TrueLabel,Candidates,QVector
                        f.write(f"{row[0]},{self.epochs - 1},{row[1]},{row[2]},{candidates_csv},{q_csv_safe}\n")
                        
                    # logs_to_wandb[f"client_train/{self.client_id}/q_vectors_end"] = self.end_table
                    
        except IOError as e:
            print(f"Error writing to CSV {self.q_log_filename}: {e}")
        # [!! CSV 逻辑结束 !!]

        # 统一发送所有日志
        wandb.log(logs_to_wandb, step=roud)
        # ----- [!! WANDB LOGGING 结束 !!] -----

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

    def get_adaptive_threshold_static(self, current_epoch, total_epochs, T_start=0.9, T_end=0.4):
           t = 0.4
           return t
    
    def get_adaptive_threshold_power(self, current_epoch, total_epochs, T_start=0.9, T_end=0.4):
        # 【方案二：幂函数衰减 (Power Decay)】
        # 特点：前期下降极快，迅速逼近 T_end。
        # 适合：数据质量较好(noise=0)，模型学得很快，不想在前面浪费时间。
        
        e = current_epoch
        E = total_epochs
        
        # power=2.0 是平方衰减，power=3.0 会更激进
        power = 6
        
        # 公式逻辑：随着 e 增大，(1 - e/E) 变小，平方后变得更小
        decay_factor = (1 - e / E) ** power
        
        T = T_end + (T_start - T_end) * decay_factor
        
        return T
    # 余弦退火比较缓和的方式
    def get_adaptive_threshold(self, current_epoch, total_epochs, T_start, T_end):
        import math
        e = current_epoch
        E = total_epochs
        T = T_end + 0.5 * (T_start - T_end) * (1 + math.cos( (e / E) * math.pi ))
        return T
    def get_uncertainty_entropy_masked(self, logits, candidates):
        device = logits.device
        
        # 1. 先把非候选集的 logits 屏蔽掉 (Masking)
        # 这样 Softmax 之后，非候选集的概率就是 0
        neg_inf = -1e9
        masked_logits = logits.clone()
        if isinstance(candidates, torch.Tensor):
            candidates = candidates.to(device)
        else:
            # 如果 candidates 还是 list 格式，先转 tensor 再移设备
            candidates = torch.tensor(candidates).float().to(device)
        # ~candidates.bool() 表示非候选位置
        masked_logits = masked_logits.masked_fill(~candidates.bool(), neg_inf)
        
        # 2. 在屏蔽后的 logits 上做 Softmax
        # 此时概率只会分布在候选标签上，和为 1
        probs = torch.softmax(masked_logits, dim=1)
        
        # 3. 计算熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
        
        # 4. 计算最大熵 log(|Yi|)
        cand_sizes = candidates.sum(dim=1).float().to(device)
        max_entropy_local = torch.log(cand_sizes + 1e-7)
        
        # 5. 归一化
        normalized_entropy = entropy / (max_entropy_local + 1e-7)
        
        return normalized_entropy
    
    def get_uncertainty_entropy(self, logits):
        """
        计算归一化熵 (0~1之间)
        0 = 非常确定 (Sniper)
        1 = 非常困惑 (Soldier/Noise)
        """
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.size(1)
        
        # 1. 计算熵: -sum(p * log(p))
        # +1e-7 是为了防止 log(0) 导致 NaN
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
        
        # 2. 归一化: 除以最大可能的熵 log(C)
        # 这样无论有10类还是100类，数值都在 0~1 之间
        max_entropy = torch.log(torch.tensor(num_classes).float().to(logits.device))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def get_uncertainty_entropy_threshold(self, logits,  candidates, alpha=0.6, T_base = 0.9):
        # information = self.get_uncertainty_entropy(logits=logits)
        information = self.get_uncertainty_entropy_masked(logits=logits, candidates=candidates)
        T = T_base - alpha * information
        return T
    

    def pll_loss_vectorized_hard(self, output, idxs, candidates, roud, miu=0.99):
        q = self.q
        batch_size = output.size(0)
        
        device = output.device
        B, C = output.shape # B=批量, C=类别数

        # 确保 cand mask 是布尔张量，在同一设备上
        if not isinstance(candidates, torch.Tensor):
            # (如果它是 list-of-lists, 构建 mask - 罕见)
            cand_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
            for i, cand in enumerate(candidates):
                if len(cand) == 0:
                    cand_mask[i, :] = True
                else:
                    cand_mask[i, list(cand)] = True
        else:
            cand_mask = candidates.to(device=device).bool()

        # 1) 把非 candidate 的 logits 设为一个很小的负数，再 argmax
        #    （保证 p 向量只在候选中选）
        neg_inf = -1e9
        logits_masked = output.clone()
        logits_masked = logits_masked.masked_fill(~cand_mask, neg_inf)

        # 若某行所有 candidates 都被 mask（不应该），回退为全候选以避免全 -inf
        all_neg = (~cand_mask).all(dim=1)
        if all_neg.any():
            logits_masked[all_neg] = output[all_neg] 

        pred_idx = logits_masked.argmax(dim=1) 

        # p 是用于 EMA 更新的硬标签 (基于当前 logits 的最佳候选)
        p = torch.zeros_like(output)
        p[torch.arange(batch_size), pred_idx] = 1.0

        # --- 【修改点：计算损失】 ---
        
        # 2) 获取当前的 q[idxs]（用于计算损失）
        #    我们使用 .detach().clone() 来确保不会意外修改原始 q
        current_q = q[idxs].detach().clone() 
        
        # 3) 检查置信度是否 > 0.5
        max_conf, max_idx = torch.max(current_q, dim=1)
        # T_dynamic = self.get_adaptive_threshold(current_epoch=roud, total_epochs=self.config.rounds, T_start=0.95, T_end=0.51)

        # 更加激进的下降方式
        # T_dynamic = self.get_adaptive_threshold_power(current_epoch=roud, total_epochs=self.config.rounds, T_start=0.9, T_end=0.4)

        # T_dynamic = self.get_adaptive_threshold_static(current_epoch=roud, total_epochs=self.config.rounds, T_start=0.9, T_end=0.4)

        T_dynamic = self.get_uncertainty_entropy_threshold(logits=output, candidates=candidates)
        print(f'current T_dynamic:{T_dynamic[:5]}')
        mask = max_conf > T_dynamic


        # mask = max_conf > 0.4
        # 4) 默认使用 current_q (软标签) 作为损失目标
        loss_target = current_q
        
        # 5) 【新逻辑】如果掩码 (mask) 中有 True (即存在 >0.5 的置信度)
        if mask.any():
            # 创建 one-hot 硬标签
            hard_labels = F.one_hot(max_idx, num_classes=C).float().to(device)
            
            # 使用 torch.where 进行条件替换：
            # - 如果 mask 为 True (即置信度>0.5), 使用 hard_labels
            # - 否则, 保持原来的 current_q
            loss_target = torch.where(mask.unsqueeze(1), hard_labels, current_q)
        
        # 6) 使用 (可能) 锐化后的 loss_target 计算损失
        log_probs = torch.log_softmax(output, dim=1)
        loss = -(loss_target * log_probs).sum(dim=1).mean()
        
        # --- 【修改点结束】 ---

        # 7) EMA 更新：
        #    注意，这里我们仍然使用 *原始*的 q[idxs] (即 current_q) 来进行 EMA，
        #    而不是使用被锐化过的 loss_target，以防止过早过拟合。
        with torch.no_grad(): # 确保 EMA 更新不计算梯度
            q[idxs] = current_q * miu + (1 - miu) * p

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
        ... (此函数内容不变) ...
        """
        # 1) prepare global model: coepy local model architecture and load global weights
        #   (we don't want to modify self.local_model here)
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
        shuffled_data = data[perm]                      # [B, ...]
        # get q for original and shuffled (self.q indexed by dataset idx)
        q_orig = self.q[idxs]                           # [B, C]
        q_shuf = self.q[idxs[perm]]                     # [B, C]

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
        #   use no_grad because we don't train global_model here
        with torch.no_grad():
            # probs for original and shuffled
            probs_orig = torch.softmax(global_model(data), dim=1)       # [B, C]
            probs_shuf = torch.softmax(global_model(shuffled_data), dim=1)  # [B, C]
            probs_mix  = torch.softmax(global_model(mix_data), dim=1)   # [B, C]

            # per-sample max-confidence
            max_orig = probs_orig.max(dim=1).values                 # [B]
            max_shuf = probs_shuf.max(dim=1).values                 # [B]
            max_mix  = probs_mix.max(dim=1).values                  # [B]

        # w1 = gamma * max(F(x_i)) + (1-gamma) * max(F(x_j))
        # w2 = max(F(mix))
        # w = w1 * w2   (elementwise)
        # gamma is scalar -> broadcast to [B]
        w1 = gamma * max_orig + (1.0 - gamma) * max_shuf   # [B]
        w2 = max_mix                                        # [B]
        weight = w1 * w2                                    # [B]

        # 6) forward with local model on mix_data and compute weighted PLL cross-entropy
        mix_output = self.local_model(mix_data)           # [B, C]
        log_probs = torch.log_softmax(mix_output, dim=1)  # [B, C]

        # per-sample PLL loss: - q_mix^T log p
        sample_loss = - (mix_q * log_probs).sum(dim=1)     # [B]

        # weighted sum, then average by batch size (paper used 1/n_m)
        loss = (weight * sample_loss).sum() / float(batch_size)

        return loss
    

