import torch
import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
LeNet architecture.
"""


def hyper_params():
    hparams = {
        "dataset": 'cifar10',
        "output_directory": "lr_lenet5",
        "training": {
            "batch_size": 128,
            "num_epochs": 60,
            "learning_rate": 0.01,
            "milestones": [],
            "gamma": 0.1,
            "weight_decay": 0,
            "mini_batch": 1000000
        },
        "pruning": {
            "iterations": 1,
            "percentage": 0.2
        }
    }
    return hparams


class Real(nn.Module):
    model_name = 'lenet_real'

    def __init__(self):
        super().__init__()
        # Changed (3, 6, 5) to (3, 8, 5)
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @classmethod
    def name(cls):
        return cls.model_name


class Quat(nn.Module):
    """
    LeNet architecture using htorch funtions.
    """
    model_name = 'lenet_quat'

    def __init__(self):
        super().__init__()
        self.conv1 = layers.QConv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = layers.QConv2d(2, 4, 5)
        self.fc1 = layers.QLinear(4 * 5 * 5, 30)
        self.fc2 = layers.QLinear(30, 21)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @classmethod
    def name(cls):
        return cls.model_name
