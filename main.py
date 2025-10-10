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

if __name__ == "__main__":

    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['WANDB_API_KEY'] = "364095e05fdc1c26991c0347c509cfc8a4ef138c" 

    path = './config.yaml'
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # config = {
    #     "seed":42,
    #     "local_epochs": 5,
    #     "batch_size": 128,
    #     "optimizer":"sgd",
    #     "lr": 0.1,
    #     "model_name": "resnet18",
    #     "dataset": "SVHN",
    #     "noise_level": 0.3,
    #     "rounds":250,
    #     "num_classes":10,
    #     "num_clients":10,
    #     "ratio":1,
    #     "partition":"non_iid",
    #     "p": 0.7,
    #     "alpha_dir": 0.5,
    #     "mixup_alpha":0.2,
    #     "lc":True,
    #     "mix":False,
    #     "ga":False
    # }
    data_partition = ""
    if config['partition'].lower() == "non_iid":
        data_partition = f'noniid_class_p:{config["p"]}_alpha_dir:{config["alpha_dir"]}'
    else:
        data_partition = f'iid'

    loss_kind = ""
    if config['lc'] == True:
        loss_kind += "lc"
    if config['mix'] == True:
        loss_kind += "_mix"
    if config['ga'] == True:
        loss_kind += "_ga"
    
         
    with wandb.init(project='test-fl', 
                 entity='whitewall_9-jinan-university', 
                 config=config, 
                 name = (f'fedpll_noise:{config["noise_level"]}_m:{config["model_name"].lower()}_' 
                        f'd:{config["dataset"]}_{data_partition}_lr:{config["lr"]:.3f}_' 
                        f'optim:{config["optimizer"]}_{loss_kind}_' 
                        # f'sim'
                        f'findsgd'
                        ),
                 group='fedpll',
                 allow_val_change=True,
                 ) as run:
        
        seed_everything(run.config.seed)

        # mnist fashionmnist
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307, ), (0.3081,))
        # ])

        
        # train_dataset = datasets.FashionMNIST(
        #     root='./data',
        #     train=True,
        #     transform=transform,
        #     download=True
        #  )

        # test_dataset = datasets.FashionMNIST(
        #     root='./data',
        #     train=False,
        #     transform=transform,
        #     download=True
        # )
        # train_dataset = datasets.MNIST(
        #     root='./data',
        #     train=True,
        #     transform=transform,
        #     download=True
        # )

        # test_dataset = datasets.MNIST(
        #     root='./data',
        #     train=False,
        #     transform=transform,
        #     download=True
        # )
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],  # SVHN �ٷ���ֵ
                                std=[0.1980, 0.2010, 0.1970])    # SVHN �ٷ���׼��
        ])
        train_dataset = datasets.SVHN(
            root='./data',
            split='train',
            transform=transform,
            download=True
        )

        test_dataset = datasets.SVHN(
            root='./data',
            split='test',
            transform=transform,
            download=True
        )
        server = Server(config=run.config, train_dataset=train_dataset, test_dataset=test_dataset)
        server.start()
