
# -*- coding: utf-8 -*-


import wandb
import csv
from client import Client
from typing import List, Dict, Optional
import model


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

import common
from common import (
    iid_partition, 
    non_iid_partition, 
    plot_counts_heatmap_blue, 
    compute_client_class_counts_from_subsets, 
    plot_candidates_counts_heatmap_blue, 
    plot_acc_counts_heatmap_blue,
    plot_acc_counts_heatmap_blue_test,
    split_testset_by_distribution
)
import numpy as np

import copy

import random
from common import setup_logger, ExperimentConfig 
from logging import Logger
from datetime import datetime
import os 
class Server:
    def __init__(self, config: ExperimentConfig, train_dataset, test_dataset, logger: Logger):
        self.clients: List[Client] = []
        self.config = config
        self.device = self.config.device
        self.global_model = None
        self.weights =  []
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.global_gradients = None
        self.global_grad_vector = None
        self.logger = logger
        # [FedODP 新增] 全局原型库 {class_id: tensor}
        self.global_prototypes = {}
        self.semantic_anchors = None
    
        # 配置本地测试集
        self.client_test_datasets = None
        self.metrics_dir = os.path.join("csv_logs", f"{self.config.exp_id}_{self.config.exp_name}")
        self.round_metrics_path = os.path.join(self.metrics_dir, "round_metrics.csv")
        self.per_client_round_metrics_path = os.path.join(self.metrics_dir, "per_client_round_metrics.csv")
        self.confusion_matrix_dir = os.path.join(self.metrics_dir, "confusion_matrices")

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _get_classifier_input_dim(self) -> int:
        model_ref = self.global_model

        if hasattr(model_ref, 'fc') and isinstance(model_ref.fc, torch.nn.Linear):
            return model_ref.fc.in_features
        if hasattr(model_ref, 'fc2') and isinstance(model_ref.fc2, torch.nn.Linear):
            return model_ref.fc2.in_features
        if hasattr(model_ref, 'fc3') and isinstance(model_ref.fc3, torch.nn.Linear):
            return model_ref.fc3.in_features
        if hasattr(model_ref, 'classifier') and isinstance(model_ref.classifier[-1], torch.nn.Linear):
            return model_ref.classifier[-1].in_features

        raise AttributeError(f"Unsupported classifier head for model type: {type(model_ref)}")

    def initialize_semantic_anchors(self):
        feature_dim = self._get_classifier_input_dim()
        anchors = torch.randn(
            self.config.num_classes,
            feature_dim,
            device=self.device,
        ) * self.config.fedsa_anchor_init_std
        self.semantic_anchors = torch.nn.functional.normalize(anchors, p=2, dim=1)

    def compute_anchor_margin(self, anchors: Optional[torch.Tensor]) -> float:
        if anchors is None or anchors.size(0) < 2:
            return 0.0

        normalized = torch.nn.functional.normalize(anchors, p=2, dim=1)
        pairwise_dist = torch.cdist(normalized, normalized, p=2)
        valid_mask = ~torch.eye(pairwise_dist.size(0), dtype=torch.bool, device=pairwise_dist.device)
        if valid_mask.sum() == 0:
            return 0.0
        return float(pairwise_dist[valid_mask].mean().item())

    def _write_confusion_matrix_csv(self, path: str, matrix: np.ndarray):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        matrix = np.asarray(matrix, dtype=int)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label\\pred_label"] + [f"pred_{i}" for i in range(matrix.shape[1])])
            for true_label, row in enumerate(matrix):
                writer.writerow([f"true_{true_label}"] + row.tolist())

    def _format_worst_confusion_classes(self, matrix: np.ndarray, metrics: Dict, top_k: int = 3) -> str:
        matrix = np.asarray(matrix)
        recalls = np.asarray(metrics["recall_class"])
        supports = matrix.sum(axis=1)
        valid_classes = np.where(supports > 0)[0]
        if len(valid_classes) == 0:
            return "no supported classes"

        worst_classes = sorted(valid_classes, key=lambda class_id: recalls[class_id])[:top_k]
        parts = []
        for class_id in worst_classes:
            row = matrix[class_id].copy()
            correct = row[class_id]
            row[class_id] = -1
            confused_pred = int(np.argmax(row)) if row.size > 0 and row.max() > 0 else None
            if confused_pred is None:
                parts.append(
                    f"class {class_id}: recall={recalls[class_id]:.4f}, "
                    f"support={int(supports[class_id])}, correct={int(correct)}"
                )
            else:
                parts.append(
                    f"class {class_id}: recall={recalls[class_id]:.4f}, "
                    f"support={int(supports[class_id])}, correct={int(correct)}, "
                    f"most_confused_as={confused_pred} ({int(matrix[class_id, confused_pred])})"
                )
        return "; ".join(parts)

    def _format_worst_clients(self, client_metrics: List[Dict], top_k: int = 3) -> List[str]:
        if not client_metrics:
            return ["no client metrics"]
        worst_items = sorted(
            enumerate(client_metrics),
            key=lambda item: item[1]["accuracy"],
        )[:top_k]
        return [
            f"client {client_id}: acc={metrics['accuracy']:.4f}, "
            f"macro_recall={metrics['recall_mean']:.4f}, macro_f1={metrics['f1']:.4f}"
            for client_id, metrics in worst_items
        ]

    def _format_client_metric_lines(self, disamb_metrics: List[Dict], test_metrics: List[Dict]) -> List[str]:
        if not disamb_metrics or not test_metrics:
            return ["no client metrics"]
        lines = []
        for client_id, (disamb, test) in enumerate(zip(disamb_metrics, test_metrics)):
            lines.append(
                f"client {client_id}: acc={test['accuracy']:.4f}, "
                f"macro_recall={test['recall_mean']:.4f}, "
                f"macro_f1={test['f1']:.4f}, "
                f"q_acc={disamb['accuracy']:.4f}, "
                f"q_macro_f1={disamb['f1']:.4f}"
            )
        return lines

    def _log_round_interpretation(
        self,
        round_id: int,
        server_acc: float,
        mean_client_personalized_test_acc: float,
        min_client_personalized_test_acc: float,
        p10_client_personalized_test_acc: float,
        mean_client_disamb_q_macro_f1: float,
        mean_selected_client_vote_pseudo_acc: float,
        mean_selected_client_vote_confidence: float,
        mean_selected_client_vote_confidence_p90: float,
        mean_selected_client_vote_high_conf_error_rate: float,
        mean_selected_client_candidate_ambiguity_rate: float,
    ):
        gap = mean_client_personalized_test_acc - server_acc
        gap_text = (
            f"客户端个性化均值比 server 高 {gap:.4f}"
            if gap >= 0
            else f"客户端个性化均值比 server 低 {-gap:.4f}"
        )
        vote_text = (
            "本轮没有 vote 伪标签样本"
            if mean_selected_client_vote_pseudo_acc == 0 and mean_selected_client_vote_confidence == 0
            else f"vote 伪标签准确率/置信度为 {mean_selected_client_vote_pseudo_acc:.4f}/{mean_selected_client_vote_confidence:.4f}"
        )
        self.logger.info(f"Round {round_id} 解读 | {gap_text}。")
        self.logger.info(
            f"Round {round_id} 解读 | 弱客户端 min/p10="
            f"{min_client_personalized_test_acc:.4f}/{p10_client_personalized_test_acc:.4f}；"
            f"q 消歧 macro_f1={mean_client_disamb_q_macro_f1:.4f}。"
        )
        self.logger.info(f"Round {round_id} 解读 | {vote_text}。")
        self.logger.info(
            f"Round {round_id} 解读 | vote_conf_p90={mean_selected_client_vote_confidence_p90:.4f}；"
            f"高置信错误率={mean_selected_client_vote_high_conf_error_rate:.4f}；"
            f"候选集歧义率={mean_selected_client_candidate_ambiguity_rate:.4f}。"
        )
        self.logger.info(
            f"Round {round_id} 查看错误 | 混淆矩阵 CSV 中每一行是真实类别，"
            f"非对角线的大值表示该类经常被错分到对应预测类别。"
        )

    def _log_round_metric_summary(
        self,
        round_id: int,
        server_metrics: Dict,
        mean_client_disamb_q_acc: float,
        mean_client_disamb_q_macro_recall: float,
        mean_client_disamb_q_macro_precision: float,
        mean_client_disamb_q_macro_f1: float,
        std_client_disamb_q_macro_recall: float,
        mean_client_personalized_test_acc: float,
        std_client_personalized_test_acc: float,
        min_client_personalized_test_acc: float,
        p10_client_personalized_test_acc: float,
        mean_selected_client_vote_pseudo_acc: float,
        mean_selected_client_vote_confidence: float,
        mean_selected_client_vote_confidence_p10: float,
        mean_selected_client_vote_confidence_p50: float,
        mean_selected_client_vote_confidence_p90: float,
        mean_selected_client_vote_high_conf_error_rate: float,
        mean_selected_client_vote_low_conf_correct_rate: float,
        mean_selected_client_vote_sample_coverage: float,
        mean_selected_client_candidate_size_mean: float,
        mean_selected_client_candidate_size_p90: float,
        mean_selected_client_candidate_ambiguity_rate: float,
    ):
        self.logger.info(
            f"Round {round_id} Server | acc={server_metrics['accuracy']:.4f}, "
            f"macro_recall={server_metrics['recall_mean']:.4f}, macro_f1={server_metrics['f1']:.4f}"
        )
        self.logger.info(
            f"Round {round_id} Client Overall | "
            f"acc_mean={mean_client_personalized_test_acc:.4f}, "
            f"acc_std={std_client_personalized_test_acc:.4f}, "
            f"acc_min={min_client_personalized_test_acc:.4f}, "
            f"acc_p10={p10_client_personalized_test_acc:.4f}"
        )
        self.logger.info(
            f"Round {round_id} Disamb Q | acc={mean_client_disamb_q_acc:.4f}, "
            f"macro_recall={mean_client_disamb_q_macro_recall:.4f}, "
            f"macro_precision={mean_client_disamb_q_macro_precision:.4f}, "
            f"macro_f1={mean_client_disamb_q_macro_f1:.4f}, "
            f"std_macro_recall={std_client_disamb_q_macro_recall:.4f}"
        )
        self.logger.info(
            f"Round {round_id} Vote | pseudo_acc={mean_selected_client_vote_pseudo_acc:.4f}, "
            f"confidence={mean_selected_client_vote_confidence:.4f}, "
            f"conf_p10/p50/p90={mean_selected_client_vote_confidence_p10:.4f}/"
            f"{mean_selected_client_vote_confidence_p50:.4f}/{mean_selected_client_vote_confidence_p90:.4f}, "
            f"high_conf_err={mean_selected_client_vote_high_conf_error_rate:.4f}, "
            f"low_conf_correct={mean_selected_client_vote_low_conf_correct_rate:.4f}, "
            f"coverage={mean_selected_client_vote_sample_coverage:.4f}"
        )
        self.logger.info(
            f"Round {round_id} Candidate | size_mean={mean_selected_client_candidate_size_mean:.4f}, "
            f"size_p90={mean_selected_client_candidate_size_p90:.4f}, "
            f"ambiguity_rate={mean_selected_client_candidate_ambiguity_rate:.4f}"
        )

                    

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
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.confusion_matrix_dir, exist_ok=True)
        with open(self.round_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round",
                "server_test_acc",
                "server_test_macro_recall",
                "server_test_macro_f1",
                "mean_client_disamb_q_acc",
                "mean_client_disamb_q_macro_recall",
                "mean_client_disamb_q_macro_precision",
                "mean_client_disamb_q_macro_f1",
                "std_client_disamb_q_macro_recall",
                "mean_client_personalized_test_acc",
                "std_client_personalized_test_acc",
                "min_client_personalized_test_acc",
                "p10_client_personalized_test_acc",
                "mean_selected_client_vote_pseudo_acc",
                "mean_selected_client_vote_confidence",
                "mean_selected_client_vote_confidence_p10",
                "mean_selected_client_vote_confidence_p50",
                "mean_selected_client_vote_confidence_p90",
                "mean_selected_client_vote_high_conf_error_rate",
                "mean_selected_client_vote_low_conf_correct_rate",
                "mean_selected_client_vote_sample_coverage",
                "mean_selected_client_candidate_size_mean",
                "mean_selected_client_candidate_size_p90",
                "mean_selected_client_candidate_ambiguity_rate",
                "selected_clients",
            ])
        with open(self.per_client_round_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round",
                "client_id",
                "selected",
                "client_disamb_q_acc",
                "client_disamb_q_macro_recall",
                "client_disamb_q_macro_precision",
                "client_disamb_q_macro_f1",
                "client_personalized_test_acc",
                "client_personalized_test_macro_recall",
                "client_personalized_test_macro_precision",
                "client_personalized_test_macro_f1",
                "client_vote_confidence_mean",
                "client_vote_confidence_p10",
                "client_vote_confidence_p50",
                "client_vote_confidence_p90",
                "client_vote_high_conf_error_rate",
                "client_vote_low_conf_correct_rate",
                "client_vote_sample_coverage",
                "client_candidate_size_mean",
                "client_candidate_size_std",
                "client_candidate_size_p50",
                "client_candidate_size_p90",
                "client_candidate_ambiguity_rate",
            ])


        self.global_model = model.get_model(self.config.model_name).to(self.device)
        if self.config.fedsa:
            self.initialize_semantic_anchors()
        client_train_dataset = []

        if self.config.partition.lower() == "iid":
           client_train_dataset = iid_partition(dataset=self.train_dataset, num_clients=self.config.num_clients)
        else:
            client_train_dataset,_, phi_matrix,  = non_iid_partition(dataset=self.train_dataset,
                                                    num_classes=self.config.num_classes, 
                                                     num_clients=self.config.num_clients,
                                                     p=self.config.p,
                                                     alpha_dir=self.config.alpha_dir)
            self.logger.info(
                f"Data partition | non_iid p={self.config.p}, alpha={self.config.alpha_dir}, "
                f"phi_shape={phi_matrix.shape}"
            )

        counts = compute_client_class_counts_from_subsets(client_datasets=client_train_dataset, num_classes=self.config.num_classes)
        self.client_test_datasets = split_testset_by_distribution(global_test_dataset=self.test_dataset, distribution_matrix=counts)

        fig = plot_counts_heatmap_blue(counts=counts, normalize='none', figsize=(12, 6))
        
        wandb.log({"clients_class_distribution": wandb.Image(fig)}, commit=False)

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
        self.pre() # 初始化 Clients 和 Global Model
        
        # 初始化权重
        if len(self.weights) == 0:
            for client in self.clients:
                self.weights.append(len(client.train_dataset) * 1.0)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=128,
            shuffle=False,
        )

        clients_test_loaders = []
        for client in self.clients:
            clients_test_loaders.append(DataLoader(dataset=self.client_test_datasets[client.client_id],
                                                   batch_size=128,
                                                   shuffle=False))
        # 记录已覆盖的类别数 (用于观察系统何时学会了所有类)
        covered_classes_history = []

        
        
        
        acc = []

        for r in range(self.config.rounds):
            state_dicts = list()
            
            # [FedODP] 用于收集本轮各 Client 贡献的原型
            local_prototypes_collect = [] 

            k = int(len(self.clients) * self.config.ratio)
            selected_clients = random.sample(self.clients, k)
            self.logger.info(f"Round {r} Selected | clients={[client.client_id for client in selected_clients]}")

            vote_model_state_dicts = None
            if self.config.use_vote_pseudo:
                num_vote_models = min(self.config.vote_num_models, len(self.clients))
                if self.config.share_noisy_vote_models:
                    vote_model_state_dicts = [
                        client.get_shared_vote_state_dict()
                        for client in self.clients[:num_vote_models]
                    ]
                else:
                    vote_model_state_dicts = [
                        copy.deepcopy(client.local_model.state_dict())
                        for client in self.clients[:num_vote_models]
                    ]

            selected_weights = [self.weights[self.clients.index(client)] for client in selected_clients]
            global_anchor_margin = self.compute_anchor_margin(self.semantic_anchors) if self.config.fedsa else 0.0

            # 计算全局梯度 (GA Loss 用，保持不变)
            global_grad_vector = self.aggregate_flattened_gradients(selected_clients, num_batches_per_client=1)
            
            # --- Client 训练循环 ---
            for client in selected_clients:
                # [FedODP 关键修改] 
                # 传入 self.global_prototypes 给 Client 进行“按需消歧”
                state_dict = client.train(
                    global_model_state_dict=self.global_model.state_dict(), 
                    roud=r, 
                    global_prototypes=self.global_prototypes, # <--- 传情报
                    semantic_anchors=self.semantic_anchors.detach().clone() if self.semantic_anchors is not None else None,
                    global_anchor_margin=global_anchor_margin,
                    global_grad_vector=global_grad_vector,
                    vote_model_state_dicts=vote_model_state_dicts,
                )
                state_dicts.append(state_dict)
                
                # [FedODP 关键修改]
                # 训练完后，让 Client 贡献它的“情报”（本地原型）
                # 建议：前几轮 (Warm-up) 模型太差，不要收集，以免污染库
                if r >= self.config.warmup: 
                    # threshold=0.8 表示只确信度>0.8的才上传
                    local_protos = client.get_local_prototype_stats(threshold=0.8)
                    if len(local_protos) > 0:
                        local_prototypes_collect.append(local_protos)

            # --- Server 聚合 ---
            
            # 1. 聚合模型参数 (原有逻辑)
            new_state_dict = self.fed_avg_simple(state_dicts=state_dicts, weights=selected_weights) 
            self.global_model.load_state_dict(new_state_dict)
            
            # 2. [FedODP] 聚合原型 (更新全局情报库)
            if len(local_prototypes_collect) > 0:
                self.aggregate_prototypes(local_prototypes_collect)
            
            # 3. 测试与评估
            
            server_acc_matrix = self.get_acc_matrix(test_loader=test_loader)
            server_metrics = common.get_metrics(server_acc_matrix)
            test_acc = server_metrics["accuracy"]
            server_confusion_path = os.path.join(self.confusion_matrix_dir, f"round_{r:03d}_server.csv")
            self._write_confusion_matrix_csv(server_confusion_path, server_acc_matrix)
            logger = self.logger
            wandb.log({
                "server_test/acc": test_acc,
                "server_test/macro_recall": server_metrics["recall_mean"],
                "server_test/macro_f1": server_metrics["f1"],
                "server_test/macro_precision": server_metrics["precision_mean"],
                "server/covered_classes": len(self.global_prototypes),
                "server/global_anchor_margin": global_anchor_margin
            }, step=r)
            acc.append(test_acc)

            selected_client_ids = {client.client_id for client in selected_clients}

            all_client_disamb_metrics = []
            for client in self.clients:
                disamb_metrics = client.get_disambiguation_metrics()
                all_client_disamb_metrics.append(disamb_metrics)
            mean_client_disamb_q_acc = float(np.mean([m["accuracy"] for m in all_client_disamb_metrics]))
            mean_client_disamb_q_macro_recall = float(np.mean([m["recall_mean"] for m in all_client_disamb_metrics]))
            mean_client_disamb_q_macro_precision = float(np.mean([m["precision_mean"] for m in all_client_disamb_metrics]))
            mean_client_disamb_q_macro_f1 = float(np.mean([m["f1"] for m in all_client_disamb_metrics]))
            std_client_disamb_q_macro_recall = float(np.std([m["recall_mean"] for m in all_client_disamb_metrics]))

            #每轮都测试一次泛化能力
            all_client_test_metrics = []
            if r >= 0:
                for client in self.clients:
                    client_test_matrix = client.get_acc_matrix(test_loader=clients_test_loaders[client.client_id])
                    client_test_metrics = common.get_metrics(client_test_matrix)
                    all_client_test_metrics.append(client_test_metrics)
                    client_confusion_path = os.path.join(
                        self.confusion_matrix_dir,
                        f"round_{r:03d}_client_{client.client_id}_personalized_test.csv",
                    )
                    self._write_confusion_matrix_csv(client_confusion_path, client_test_matrix)

            vote_metric_clients = [client.last_train_metrics for client in selected_clients if client.last_train_metrics]
            vote_pseudo_accs = [m["vote_pseudo_acc"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            vote_confidences = [m["vote_confidence"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            vote_confidence_p10s = [m["vote_confidence_p10"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            vote_confidence_p50s = [m["vote_confidence_p50"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            vote_confidence_p90s = [m["vote_confidence_p90"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            vote_high_conf_error_rates = [m["vote_high_conf_error_rate"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            vote_low_conf_correct_rates = [m["vote_low_conf_correct_rate"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            vote_sample_coverages = [m["vote_sample_coverage"] for m in vote_metric_clients if m.get("vote_samples", 0) > 0]
            candidate_size_means = [m["candidate_size_mean"] for m in vote_metric_clients]
            candidate_size_p90s = [m["candidate_size_p90"] for m in vote_metric_clients]
            candidate_ambiguity_rates = [m["candidate_ambiguity_rate"] for m in vote_metric_clients]
            client_personalized_test_accs = [m["accuracy"] for m in all_client_test_metrics]
            mean_client_personalized_test_acc = float(np.mean(client_personalized_test_accs)) if client_personalized_test_accs else 0.0
            std_client_personalized_test_acc = float(np.std(client_personalized_test_accs)) if client_personalized_test_accs else 0.0
            min_client_personalized_test_acc = float(np.min(client_personalized_test_accs)) if client_personalized_test_accs else 0.0
            p10_client_personalized_test_acc = float(np.percentile(client_personalized_test_accs, 10)) if client_personalized_test_accs else 0.0
            mean_selected_client_vote_pseudo_acc = self._safe_mean(vote_pseudo_accs)
            mean_selected_client_vote_confidence = self._safe_mean(vote_confidences)
            mean_selected_client_vote_confidence_p10 = self._safe_mean(vote_confidence_p10s)
            mean_selected_client_vote_confidence_p50 = self._safe_mean(vote_confidence_p50s)
            mean_selected_client_vote_confidence_p90 = self._safe_mean(vote_confidence_p90s)
            mean_selected_client_vote_high_conf_error_rate = self._safe_mean(vote_high_conf_error_rates)
            mean_selected_client_vote_low_conf_correct_rate = self._safe_mean(vote_low_conf_correct_rates)
            mean_selected_client_vote_sample_coverage = self._safe_mean(vote_sample_coverages)
            mean_selected_client_candidate_size_mean = self._safe_mean(candidate_size_means)
            mean_selected_client_candidate_size_p90 = self._safe_mean(candidate_size_p90s)
            mean_selected_client_candidate_ambiguity_rate = self._safe_mean(candidate_ambiguity_rates)

            self._log_round_metric_summary(
                round_id=r,
                server_metrics=server_metrics,
                mean_client_disamb_q_acc=mean_client_disamb_q_acc,
                mean_client_disamb_q_macro_recall=mean_client_disamb_q_macro_recall,
                mean_client_disamb_q_macro_precision=mean_client_disamb_q_macro_precision,
                mean_client_disamb_q_macro_f1=mean_client_disamb_q_macro_f1,
                std_client_disamb_q_macro_recall=std_client_disamb_q_macro_recall,
                mean_client_personalized_test_acc=mean_client_personalized_test_acc,
                std_client_personalized_test_acc=std_client_personalized_test_acc,
                min_client_personalized_test_acc=min_client_personalized_test_acc,
                p10_client_personalized_test_acc=p10_client_personalized_test_acc,
                mean_selected_client_vote_pseudo_acc=mean_selected_client_vote_pseudo_acc,
                mean_selected_client_vote_confidence=mean_selected_client_vote_confidence,
                mean_selected_client_vote_confidence_p10=mean_selected_client_vote_confidence_p10,
                mean_selected_client_vote_confidence_p50=mean_selected_client_vote_confidence_p50,
                mean_selected_client_vote_confidence_p90=mean_selected_client_vote_confidence_p90,
                mean_selected_client_vote_high_conf_error_rate=mean_selected_client_vote_high_conf_error_rate,
                mean_selected_client_vote_low_conf_correct_rate=mean_selected_client_vote_low_conf_correct_rate,
                mean_selected_client_vote_sample_coverage=mean_selected_client_vote_sample_coverage,
                mean_selected_client_candidate_size_mean=mean_selected_client_candidate_size_mean,
                mean_selected_client_candidate_size_p90=mean_selected_client_candidate_size_p90,
                mean_selected_client_candidate_ambiguity_rate=mean_selected_client_candidate_ambiguity_rate,
            )
            self._log_round_interpretation(
                round_id=r,
                server_acc=server_metrics["accuracy"],
                mean_client_personalized_test_acc=mean_client_personalized_test_acc,
                min_client_personalized_test_acc=min_client_personalized_test_acc,
                p10_client_personalized_test_acc=p10_client_personalized_test_acc,
                mean_client_disamb_q_macro_f1=mean_client_disamb_q_macro_f1,
                mean_selected_client_vote_pseudo_acc=mean_selected_client_vote_pseudo_acc,
                mean_selected_client_vote_confidence=mean_selected_client_vote_confidence,
                mean_selected_client_vote_confidence_p90=mean_selected_client_vote_confidence_p90,
                mean_selected_client_vote_high_conf_error_rate=mean_selected_client_vote_high_conf_error_rate,
                mean_selected_client_candidate_ambiguity_rate=mean_selected_client_candidate_ambiguity_rate,
            )
            logger.info(f"Round {r} Client Metrics")
            for line in self._format_client_metric_lines(all_client_disamb_metrics, all_client_test_metrics):
                logger.info(f"Round {r}   {line}")
            logger.info(f"Round {r} Worst Clients")
            for line in self._format_worst_clients(all_client_test_metrics):
                logger.info(f"Round {r}   {line}")
            logger.info(
                f"Round {r} Confusion | server={os.path.basename(server_confusion_path)}, "
                f"clients=round_{r:03d}_client_<id>_personalized_test.csv"
            )
            logger.info(f"Round {r} Server Worst Classes")
            for line in self._format_worst_confusion_classes(server_acc_matrix, server_metrics).split("; "):
                logger.info(f"Round {r}   {line}")

            wandb.log({
                "client_disamb_q/mean_acc": mean_client_disamb_q_acc,
                "client_disamb_q/mean_macro_recall": mean_client_disamb_q_macro_recall,
                "client_disamb_q/mean_macro_precision": mean_client_disamb_q_macro_precision,
                "client_disamb_q/mean_macro_f1": mean_client_disamb_q_macro_f1,
                "client_disamb_q/std_macro_recall": std_client_disamb_q_macro_recall,
                "client_personalized_test/mean_acc": mean_client_personalized_test_acc,
                "client_personalized_test/std_acc": std_client_personalized_test_acc,
                "client_personalized_test/min_acc": min_client_personalized_test_acc,
                "client_personalized_test/p10_acc": p10_client_personalized_test_acc,
                "selected_client_vote/mean_pseudo_acc": mean_selected_client_vote_pseudo_acc,
                "selected_client_vote/mean_confidence": mean_selected_client_vote_confidence,
                "selected_client_vote/mean_confidence_p10": mean_selected_client_vote_confidence_p10,
                "selected_client_vote/mean_confidence_p50": mean_selected_client_vote_confidence_p50,
                "selected_client_vote/mean_confidence_p90": mean_selected_client_vote_confidence_p90,
                "selected_client_vote/mean_high_conf_error_rate": mean_selected_client_vote_high_conf_error_rate,
                "selected_client_vote/mean_low_conf_correct_rate": mean_selected_client_vote_low_conf_correct_rate,
                "selected_client_vote/mean_coverage": mean_selected_client_vote_sample_coverage,
                "selected_client_candidate/mean_size": mean_selected_client_candidate_size_mean,
                "selected_client_candidate/mean_size_p90": mean_selected_client_candidate_size_p90,
                "selected_client_candidate/mean_ambiguity_rate": mean_selected_client_candidate_ambiguity_rate,
            }, step=r)
            per_client_wandb_logs = {}
            for client, disamb_metrics, test_metrics in zip(self.clients, all_client_disamb_metrics, all_client_test_metrics):
                client_id = client.client_id
                train_metrics = client.last_train_metrics or {}
                per_client_wandb_logs.update({
                    f"client_disamb_q/{client_id}/acc": disamb_metrics["accuracy"],
                    f"client_disamb_q/{client_id}/macro_recall": disamb_metrics["recall_mean"],
                    f"client_disamb_q/{client_id}/macro_precision": disamb_metrics["precision_mean"],
                    f"client_disamb_q/{client_id}/macro_f1": disamb_metrics["f1"],
                    f"client_personalized_test/{client_id}/acc": test_metrics["accuracy"],
                    f"client_personalized_test/{client_id}/macro_recall": test_metrics["recall_mean"],
                    f"client_personalized_test/{client_id}/macro_precision": test_metrics["precision_mean"],
                    f"client_personalized_test/{client_id}/macro_f1": test_metrics["f1"],
                    f"client_candidate/{client_id}/size_mean": train_metrics.get("candidate_size_mean", 0.0),
                    f"client_candidate/{client_id}/ambiguity_rate": train_metrics.get("candidate_ambiguity_rate", 0.0),
                    f"client_vote/{client_id}/confidence_p90": train_metrics.get("vote_confidence_p90", 0.0),
                    f"client_vote/{client_id}/high_conf_error_rate": train_metrics.get("vote_high_conf_error_rate", 0.0),
                })
            if per_client_wandb_logs:
                wandb.log(per_client_wandb_logs, step=r)

            with open(self.round_metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    r,
                    test_acc,
                    server_metrics["recall_mean"],
                    server_metrics["f1"],
                    mean_client_disamb_q_acc,
                    mean_client_disamb_q_macro_recall,
                    mean_client_disamb_q_macro_precision,
                    mean_client_disamb_q_macro_f1,
                    std_client_disamb_q_macro_recall,
                    mean_client_personalized_test_acc,
                    std_client_personalized_test_acc,
                    min_client_personalized_test_acc,
                    p10_client_personalized_test_acc,
                    mean_selected_client_vote_pseudo_acc,
                    mean_selected_client_vote_confidence,
                    mean_selected_client_vote_confidence_p10,
                    mean_selected_client_vote_confidence_p50,
                    mean_selected_client_vote_confidence_p90,
                    mean_selected_client_vote_high_conf_error_rate,
                    mean_selected_client_vote_low_conf_correct_rate,
                    mean_selected_client_vote_sample_coverage,
                    mean_selected_client_candidate_size_mean,
                    mean_selected_client_candidate_size_p90,
                    mean_selected_client_candidate_ambiguity_rate,
                    "|".join(str(client.client_id) for client in selected_clients),
                ])
            with open(self.per_client_round_metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                for client, disamb_metrics, test_metrics in zip(self.clients, all_client_disamb_metrics, all_client_test_metrics):
                    train_metrics = client.last_train_metrics or {}
                    writer.writerow([
                        r,
                        client.client_id,
                        int(client.client_id in selected_client_ids),
                        disamb_metrics["accuracy"],
                        disamb_metrics["recall_mean"],
                        disamb_metrics["precision_mean"],
                        disamb_metrics["f1"],
                        test_metrics["accuracy"],
                        test_metrics["recall_mean"],
                        test_metrics["precision_mean"],
                        test_metrics["f1"],
                        train_metrics.get("vote_confidence", 0.0),
                        train_metrics.get("vote_confidence_p10", 0.0),
                        train_metrics.get("vote_confidence_p50", 0.0),
                        train_metrics.get("vote_confidence_p90", 0.0),
                        train_metrics.get("vote_high_conf_error_rate", 0.0),
                        train_metrics.get("vote_low_conf_correct_rate", 0.0),
                        train_metrics.get("vote_sample_coverage", 0.0),
                        train_metrics.get("candidate_size_mean", 0.0),
                        train_metrics.get("candidate_size_std", 0.0),
                        train_metrics.get("candidate_size_p50", 0.0),
                        train_metrics.get("candidate_size_p90", 0.0),
                        train_metrics.get("candidate_ambiguity_rate", 0.0),
                    ])
                
            # 每100轮画一次热力图 (原有逻辑)
            if (r + 1) % 100 == 0:
                client_acc_matrix = [client.get_acc_matrix(test_loader=test_loader) for client in self.clients]
                server_acc_matrix = self.get_acc_matrix(test_loader=test_loader)

                for idx, acc_matrix in enumerate(client_acc_matrix):
                    fig = plot_acc_counts_heatmap_blue_test(counts=acc_matrix, normalize='none', figsize=(20, 10))
                    wandb.log({f"client{idx}/True_Pred": wandb.Image(fig)}, commit=False)

                fig = plot_acc_counts_heatmap_blue_test(counts=server_acc_matrix, normalize='none', figsize=(20, 10))
                wandb.log({f"Server/True_Pred": wandb.Image(fig)}, commit=False)

        
        last_10_acc = acc[-10:] 
        # 2. 使用 numpy 计算均值和标准差
        mean_acc = np.mean(last_10_acc)
        std_acc = np.std(last_10_acc)
        # 3. 格式化打印 (保留 2 位小数，带上 ± 标准差)
        log_msg = f'Result: Last 10 Rounds Avg: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%'
        logger.info(log_msg)
        print(log_msg)

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
    
    # --- [FedODP] 聚合原型 ---
    def aggregate_prototypes(self, clients_prototypes_list):
        """
        聚合来自 Client 的本地原型，按类别样本数加权更新全局原型。
        当启用 FedSA 时，进一步使用 EMA 更新 semantic anchors。
        """
        new_protos_sum = {}
        new_protos_count = {}
        
        for client_protos in clients_prototypes_list:
            for k, item in client_protos.items():
                p = item['prototype'].to(self.device)
                count = item['count']
                
                if k not in new_protos_sum:
                    new_protos_sum[k] = p * count
                    new_protos_count[k] = count
                else:
                    new_protos_sum[k] += p * count
                    new_protos_count[k] += count
        
        for k in new_protos_sum:
            current_round_avg = new_protos_sum[k] / new_protos_count[k]
            current_round_avg = torch.nn.functional.normalize(current_round_avg, p=2, dim=0)

            if self.config.fedsa and self.semantic_anchors is not None:
                self.global_prototypes[k] = current_round_avg
                alpha = self.config.fedsa_anchor_ema
                updated_anchor = alpha * self.semantic_anchors[k] + (1 - alpha) * current_round_avg
                self.semantic_anchors[k] = torch.nn.functional.normalize(updated_anchor, p=2, dim=0)
            else:
                alpha = 0.99
                if k not in self.global_prototypes:
                    self.global_prototypes[k] = current_round_avg
                else:
                    old_proto = self.global_prototypes[k]
                    new_proto = alpha * old_proto + (1 - alpha) * current_round_avg
                    self.global_prototypes[k] = torch.nn.functional.normalize(new_proto, p=2, dim=0)

        return self.global_prototypes
    
    
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

        

    def eval(self, test_loader)->float:
        
        
        self.global_model.eval()
        
        test_acc:float = 0.0
        test_loss:float = 0
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



    
