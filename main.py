# -*- coding: utf-8 -*-
from client import Client
from server import Server
from torchvision import datasets, transforms
import wandb
import numpy as np
import torch
import random
import os
from common import seed_everything

import yaml
from common import ExperimentConfig, setup_logger
from datetime import datetime

import json
import argparse
import csv
from pathlib import Path


def build_experiment_name(config: ExperimentConfig) -> str:
    if config.partition.lower() == "non_iid":
        partition_kind = f"noniid_p{config.p}_alpha{config.alpha_dir}"
    else:
        partition_kind = "iid"

    loss_parts = []
    if config.lc:
        loss_parts.append("lc")
    if config.mix:
        loss_parts.append("mix")
    if config.ga:
        loss_parts.append("ga")
    if config.proto:
        loss_parts.append("proto")
    if config.fedsa:
        loss_parts.append("fedsa")
    loss_kind = "-".join(loss_parts) if loss_parts else "base"

    vote_kind = f"vote{config.vote_num_models}" if config.use_vote_pseudo else "vote0"

    return (
        f"fedpll_{config.dataset.lower()}_{config.model_name.lower()}_"
        f"{vote_kind}_s{config.seed}_r{config.rounds}_le{config.local_epochs}_"
        f"noise{config.noise_level}_{partition_kind}_"
        f"{config.optimizer}_lr{config.lr:g}_{loss_kind}"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="联邦偏标记学习实验管理")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件')
    args = parser.parse_args()

    wandb_mode = os.getenv("WANDB_MODE", "online").lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        print(f"❌ 非法的 WANDB_MODE: {wandb_mode}，仅支持 online/offline/disabled")
        exit(1)

    if wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
        print("❌ 缺少环境变量 WANDB_API_KEY，请先在 shell 中导出后再运行。")
        print("   例如: export WANDB_API_KEY=your_wandb_api_key")
        exit(1)

    try:
        config = ExperimentConfig.from_yaml(args.config)
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        exit(1)
    
    experiment_name = build_experiment_name(config)
    config_path = Path(args.config).resolve()
    config_stem = config_path.stem
    
         
    with wandb.init(
        project='test-fl', 
        entity='whitewall_9-jinan-university', 
        config=config.model_dump(), # 传字典给 wandb
        name=experiment_name,
        mode=wandb_mode,
        group=f"vote_{config.dataset.lower()}_{config.model_name.lower()}_noise{config.noise_level}",
        tags=[
            f"seed:{config.seed}",
            f"vote:{config.vote_num_models if config.use_vote_pseudo else 0}",
            f"rounds:{config.rounds}",
            f"partition:{config.partition.lower()}",
        ],
        allow_val_change=True,
    ) as run:
        
        # 1. [Inject] 回填运行时 ID
        config.exp_id = run.id
        config.exp_name = run.name
        
        # 2. [Sync] 更新 WandB 云端配置
        wandb.config.update({
            "exp_id": run.id,
            "exp_name": run.name
        }, allow_val_change=True)

        # 3. [Logger] 初始化日志
        current_date = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join('./logs', current_date)
        os.makedirs(save_dir, exist_ok=True)
        log_file_name = f"{config_stem}__{config.exp_id}.log"
        log_file_path = os.path.abspath(os.path.join(save_dir, log_file_name))
        logger = setup_logger(save_path=save_dir, log_file_name=log_file_name)

        run_index_path = os.path.join(save_dir, "run_index.csv")
        run_index_exists = os.path.exists(run_index_path)
        with open(run_index_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not run_index_exists:
                writer.writerow([
                    "timestamp",
                    "run_id",
                    "exp_name",
                    "config_stem",
                    "config_path",
                    "log_file",
                    "dataset",
                    "model_name",
                    "noise_level",
                    "vote_num_models",
                ])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                config.exp_id,
                config.exp_name,
                config_stem,
                str(config_path),
                log_file_path,
                config.dataset,
                config.model_name,
                config.noise_level,
                config.vote_num_models if config.use_vote_pseudo else 0,
            ])
        
        # 4. [Print Config] 打印简洁实验摘要；完整配置已在 W&B config 中保存。
        logger.info(f"Experiment | name={config.exp_name}")
        logger.info(f"Experiment | wandb_mode={wandb_mode}")
        logger.info(f"Experiment | config_file={config_path}")
        logger.info(f"Experiment | log_file={log_file_path}")
        logger.info(
            f"Experiment | dataset={config.dataset}, model={config.model_name}, "
            f"seed={config.seed}, rounds={config.rounds}, clients={config.num_clients}"
        )
        logger.info(
            f"Experiment | partition={config.partition}, p={config.p}, "
            f"alpha={config.alpha_dir}, optimizer={config.optimizer}, lr={config.lr}"
        )
        logger.info(
            f"Experiment | vote={config.vote_num_models if config.use_vote_pseudo else 0}, "
            f"losses=lc:{config.lc},mix:{config.mix},ga:{config.ga},"
            f"proto:{config.proto},fedsa:{config.fedsa}"
        )
        
        # 5. [Seed] 固定种子
        seed_everything(config.seed)
        
        # 6. [Dataset] 统一的数据集加载逻辑 (修复了之前的覆盖 Bug)
        dataset_name = config.dataset.lower()
        logger.info(f"Loading dataset: {dataset_name}...")

        if dataset_name == 'svhn':
            # SVHN 统计数据
            mean = [0.4377, 0.4438, 0.4728]
            std  = [0.1980, 0.2010, 0.1970]
            
            # 训练集：RandomCrop + Normalize (禁止水平翻转)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            # 测试集：Normalize
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_dataset = datasets.SVHN(root='./data', split='train', transform=transform_train, download=True)
            test_dataset = datasets.SVHN(root='./data', split='test', transform=transform_test, download=True)

        elif dataset_name == 'cifar10':
            # CIFAR-10 统计数据
            mean = [0.4914, 0.4822, 0.4465]
            std  = [0.2023, 0.1994, 0.2010]

            # 训练集：RandomCrop + HorizontalFlip (CIFAR必须加翻转)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            
            train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
            test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

        elif dataset_name == 'fashionmnist':
            # FashionMNIST (单通道)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
            test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        
        elif dataset_name == 'mnist':
            # MNIST (单通道)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

        # 7. [Start] 启动 Server
        server = Server(config=config, train_dataset=train_dataset, test_dataset=test_dataset, logger=logger)
        server.start()
