
# -*- coding: utf-8 -*-


import wandb
from client import Client
from typing import List, Dict, Optional
import model


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

from common import (
    iid_partition, 
    non_iid_partition, 
    plot_counts_heatmap_blue, 
    compute_client_class_counts_from_subsets, 
    plot_candidates_counts_heatmap_blue, 
    plot_acc_counts_heatmap_blue,
    plot_acc_counts_heatmap_blue_test
)
import numpy as np

import copy

import random

class Server:
    def __init__(self, config, train_dataset, test_dataset):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.global_model = None
        self.clients = []
        self.weights =  []
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.global_gradients = None
        self.global_grad_vector = None

    def aggregate_flattened_gradients(self, clients, num_batches_per_client=1):
        """
        要求每个 client 提供一个展平后的梯度向量（CPU tensor），函数返回按样本数加权的全局梯度向量（1D torch.Tensor on server.device）
        - clients: list of Client objects (selected or all, 根据你的设计)
        - num_batches_per_client: 每个 client 用几批来估算 local gradient
        返回: global_grad_vector (1D torch.Tensor) 在 self.device（GPU 或 CPU）
        """
        import torch

        flattened_list = []
        weights = []
        total_samples = 0

        for client in clients:
            # 客户端返回的是 CPU tensor
            g_vec_cpu = client.compute_flattened_local_grad(num_batches=num_batches_per_client)
            n_m = len(client.train_dataset)  # 样本数权重
            flattened_list.append(g_vec_cpu)
            weights.append(float(n_m))
            total_samples += n_m

        if total_samples == 0:
            # 防御性处理：返回零向量（长度取第一个的长度或构造空）
            if len(flattened_list) == 0:
                return None
            zero_vec = torch.zeros_like(flattened_list[0], device=self.device)
            return zero_vec

        # 把所有 CPU 向量堆叠到一个矩阵 (num_clients x D)
        stacked = torch.stack(flattened_list, dim=0)  # shape [K, D] on CPU
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)  # [K,1]
        weights_norm = weights_tensor / float(total_samples)  # [K,1]

        # 加权平均（在 CPU 上），然后移动到 server.device（可能是 GPU）
        global_vec_cpu = (weights_norm.to(stacked.device) * stacked).sum(dim=0)  # [D] CPU
        global_vec = global_vec_cpu.to(self.device).detach()                     # move to server.device

        # 保存到 server 便于后续使用
        self.global_grad_vector = global_vec
        return global_vec


    def pre(self):
        # torch.manual_seed(42)
        # class_idxs = {}
        # splits = {}
        # for i in range(len(self.train_dataset)):
        #     _, target = self.train_dataset[i]
        #     if not class_idxs.has_key(target):
        #         class_idxs[target] = []
        #     class_idxs[target].append(i)
        
        
        # for key in class_idxs:
        #     perm = torch.randperm(len(class_idxs[key]))    # �����������
        #     shuffled = class_idxs[key][perm]               # �����������������ţ�
        #     class_idxs[key] = shuffled

        #     splits[key] = np.array_split(class_idxs[key], self.config.num_clients)

        self.global_model = model.get_model(self.config.model_name).to(self.device)
        client_train_dataset = []

        if self.config.partition.lower() == "iid":
           client_train_dataset = iid_partition(dataset=self.train_dataset, num_clients=self.config.num_clients)
        else:
            
            client_train_dataset,_, phi_matrix,  = non_iid_partition(dataset=self.train_dataset,
                                                    num_classes=self.config.num_classes, 
                                                     num_clients=self.config.num_clients,
                                                     p=self.config.p,
                                                     alpha_dir=self.config.alpha_dir)
            
            counts = compute_client_class_counts_from_subsets(client_datasets=client_train_dataset, num_classes=self.config.num_classes)
            
            fig = plot_counts_heatmap_blue(counts=counts, normalize='none', figsize=(12, 6))
            
            wandb.log({"clients_class_distribution": wandb.Image(fig)}, commit=False)

            print("class distribution: \n",phi_matrix)
        # perm = torch.randperm(len(self.train_dataset))
        # new_perms = np.array_split(perm.numpy(), self.config.num_clients)
        # client_train_dataset = [Subset(self.train_dataset, torch.tensor(p)) for p in new_perms]
        for i in range(self.config.num_clients):
            client = Client(local_model=copy.deepcopy(self.global_model),
                           train_dataset=client_train_dataset[i],
                           client_id=i,
                           config=self.config)
           
            self.clients.append(client)
        
        label_noise_server_matrix = np.zeros([self.config.num_classes, self.config.num_classes])
        label_clients = []
        for idx, client in enumerate(self.clients):
            plldatasets = client.train_plldataset
            label_noise_client_confusion_matrix = np.zeros([self.config.num_classes, self.config.num_classes])
            candidats_list = plldatasets.candidate_labels
            for _, (_, target, candidates, _) in enumerate(plldatasets):
                label_noise_server_matrix[target] += candidates.cpu().numpy()
                label_noise_client_confusion_matrix[target] += candidates.cpu().numpy()
            label_clients.append(label_noise_client_confusion_matrix)
        label_noise_server_confusion_figure = plot_candidates_counts_heatmap_blue(counts=label_noise_server_matrix, normalize='none', figsize=(12, 6))
        wandb.log({"labels_candidates_server_distribution": wandb.Image(label_noise_server_confusion_figure)}, commit=False)
        

        for idx, label_client in enumerate(label_clients):
            fig = plot_candidates_counts_heatmap_blue(counts=label_client,  
                                                      title=f'Client{idx}: Target x Candidate (blue heatmap', normalize='none', figsize=(12, 6))
            wandb.log({f"client{idx}/labels_candidates__client_distribution": wandb.Image(fig)}, commit=False)


    def start(self):
        self.pre()
        if len(self.weights) == 0:
            for client in (self.clients):
                self.weights.append(len(client.train_dataset) * 1.0)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=128,
            shuffle=False,
        )

        for r in range(self.config.rounds):
            print(f'current step: {wandb.run.step}, current roud: {r}')

            state_dicts = list()
            k = int(len(self.clients) * self.config.ratio)
            selected_clients = random.sample(self.clients, k)
            print(f"number: {k}, slected clients: {[self.clients.index(client) for client in selected_clients]}")

            selected_weights = [self.weights[self.clients.index(client)] for client in selected_clients]

            global_grad_vector = self.aggregate_flattened_gradients(selected_clients, num_batches_per_client=1)
            
            

            for client in selected_clients:
                state_dict = client.train(global_model_state_dict=self.global_model.state_dict(), 
                                          roud = r, 
                                          global_grad_vector=global_grad_vector)
                state_dicts.append(state_dict)
            if r == 0 or (r + 1) % 10 == 0:
                for client in self.clients:
                    client.test(test_loader=test_loader, 
                                epoch = r,
                                )
                    
            new_state_dict = self.fed_avg_simple(state_dicts=state_dicts, weights=selected_weights) 
            self.global_model.load_state_dict(new_state_dict)
            test_acc = self.eval(test_loader=test_loader)
            
            wandb.log({"sevrer_test/acc":test_acc}, step=r)
            # wandb.log({"test/loss":test_loss})
            print(f"Server ----> Round: {r:3d} | "
                #   f"Test Loss: {test_loss:.6f} | "
                f"Test Acc: {test_acc:.4f}\n")
            
            if (r + 1)%100 == 0:
                client_acc_matrix = [client.get_acc_matrix(test_loader=test_loader) for client in self.clients]
                server_acc_matrix = self.get_acc_matrix(test_loader=test_loader)

                for idx, acc_matrix in enumerate(client_acc_matrix):
                    fig = plot_acc_counts_heatmap_blue_test(counts=acc_matrix,  
                                                            normalize='none', figsize=(20, 10))
                    wandb.log({f"client{idx}/True_Pred": wandb.Image(fig)}, commit=False)


                fig = plot_acc_counts_heatmap_blue_test(counts=server_acc_matrix,  
                                                            normalize='none', figsize=(20, 10))
                wandb.log({f"Server/True_Pred": wandb.Image(fig)}, commit=False)
        wandb.log({}, commit=True)

    def get_acc_matrix(self, test_loader):
        self.global_model.eval()
        # test_acc = 0
        # test_loss = 0
        
        # 初始化混淆矩阵（行：实际类别，列：预测类别）
        acc_matrix = np.zeros((self.config.num_classes, self.config.num_classes), dtype=int)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
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
    
    def fed_avg_simple(self, state_dicts: List[Dict[str, torch.Tensor]],
                   weights: Optional[List[float]] = None,
                   device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
      
        if not state_dicts:
            raise ValueError("state_dicts is empty")

        n = len(state_dicts)
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError("weights length must match state_dicts length")
        total_w = float(sum(weights))
        keys = list(state_dicts[0].keys())

        for sd in state_dicts[1:]:
            if list(sd.keys()) != keys:
                raise ValueError("All state_dicts must have the same keys and order")

        avg = {}
        for k in keys:
            
            tensors = [sd[k].detach().to(device) for sd in state_dicts]
            t0 = tensors[0]

            if t0.dtype.is_floating_point:
                acc = torch.zeros_like(t0, dtype=torch.float64, device=device)
                for t, w in zip(tensors, weights):
                    acc += t.to(torch.float64) * (w / total_w)
                avg[k] = acc.to(dtype=t0.dtype)  
            else:
                avg[k] = t0.clone()

        return avg

        

    def eval(self, test_loader):
        
        
        self.global_model.eval()
        
        test_acc = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                # loss = criterion(output, target)
                # test_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                test_acc += (preds == target).sum().item()
        test_acc = test_acc / len(test_loader.dataset)
        # test_loss = test_loss / len(test_loader.dataset)

        return test_acc



    