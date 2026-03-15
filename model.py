import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x =  F.relu((self.conv1(x)))
        x = self.pool(x)
        x = F.relu((self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ������ 1: 3ͨ�� -> 32ͨ��, kernel=3, padding=1 ���ֳߴ�
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ������ 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # ������ 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # ��ѡ������ 4
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)
        
        # ȫ���Ӳ�
        self.fc1 = nn.Linear(128*4*4, 256)  # ����ػ�������ͼΪ4x4
        self.fc2 = nn.Linear(256, num_classes)
        
        # ���ػ�
        self.pool = nn.MaxPool2d(2,2)
        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool -> BN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv2 -> ReLU -> Pool -> BN
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv3 -> ReLU -> Pool -> BN
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # �����conv4, ���Լ�����
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # ����3ͨ��
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    
    
from torchvision.models import resnet18

def get_resnet18(num_classes=10, pretrained=False):
    # if not pretrained:
    #     model = resnet18(weights=None)
    model = resnet18(pretrained=pretrained)
    # ���� 32x32��С kernel + ȥ����һ�� maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # �����һ����������
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_model(model_name):
    model = None
    if model_name.lower() == "cnn3":
        model = CNN3()
    elif model_name.lower() == "cnn":
        model = CNN()
    elif model_name.lower() == "resnet18":
        model = get_resnet18()
    elif model_name.lower() == "lenet5":
        model = LeNet5()
    elif model_name.lower() == "smallcnn":
        model = SmallCNN()
    return model
