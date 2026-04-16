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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="联邦偏标记学习实验管理")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件')
    args = parser.parse_args()

    if not os.getenv("WANDB_API_KEY"):
        print("❌ 缺少环境变量 WANDB_API_KEY，请先在 shell 中导出后再运行。")
        print("   例如: export WANDB_API_KEY=your_wandb_api_key")
        exit(1)

    try:
        config = ExperimentConfig.from_yaml(args.config)
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        exit(1)
    
    data_partition = ""
    if config.partition.lower() == "non_iid":
        data_partition = f'noniid_class_p:{config.p}_alpha_dir:{config.alpha_dir}'
    else:
        data_partition = f'iid'

    loss_kind = ""
    if config.lc == True:
        loss_kind += "lc"
    if config.mix == True:
        loss_kind += "_mix"
    if config.ga == True:
        loss_kind += "_ga"
    if config.proto == True:
        loss_kind += "_proto"
    if config.fedsa == True:
        loss_kind += "_fedsa"

    vote_kind = "novote"
    if config.use_vote_pseudo:
        vote_kind = f"vote{config.vote_num_models}"
    
         
    with wandb.init(
        project='test-fl', 
        entity='whitewall_9-jinan-university', 
        config=config.model_dump(), # 传字典给 wandb
        name=(
            f'fedpll_noise:{config.noise_level}_m:{config.model_name.lower()}_' 
            f'd:{config.dataset}_{data_partition}_lr:{config.lr:.3f}_' 
            f'optim:{config.optimizer}_{loss_kind}_{vote_kind}_findsgd'
        ),
        group='fedpll',
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
        logger = setup_logger(save_path=save_dir, log_file_name=f"{config.exp_id}.log")
        
        # 4. [Print Config] 打印漂亮的配置信息 (你的需求)
        config_dict = config.model_dump()
        config_str = json.dumps(config_dict, indent=4, ensure_ascii=False)
        logger.info(f"\n{'='*20} Experiment Configuration {'='*20}\n{config_str}\n{'='*65}")
        
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
