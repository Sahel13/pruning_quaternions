import torch
import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
LeNet_300_100 architecture.
"""


def hyper_params():
    hparams = {
        "dataset": 'mnist',
        "output_directory": "18-10_1",
        "training": {
            "batch_size": 128,
            "num_epochs": 3,
            "learning_rate": 0.05,
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
    model_name = 'lenet_300_100_real'

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @classmethod
    def name(cls):
        return cls.model_name


class Quat(nn.Module):
    model_name = 'lenet_300_100_quat'

    def __init__(self):
        super().__init__()
        self.fc1 = layers.QLinear(196, 75)
        self.fc2 = layers.QLinear(75, 25)
        self.fc3 = layers.QLinear(25, 10)
        self.abs = layers.QuaternionToReal(10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.abs(self.fc3(x))
        return x

    @classmethod
    def name(cls):
        return cls.model_name
