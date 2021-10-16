
import torch
import torch.nn as nn
from htorch import layers
import core_qnn.quaternion_layers as par
import torch.nn.functional as F

"""
LeNet_300_100 architecture.
"""


class Real(nn.Module):
    model_name = 'lenet_300_100'

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


class Quat_P(nn.Module):
    model_name = 'lenet_300_100_p'

    def __init__(self):
        super().__init__()
        self.fc1 = par.QuaternionLinear(784, 300)
        self.fc2 = par.QuaternionLinear(300, 100)
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


class Quat_H(nn.Module):
    model_name = 'lenet_300_100_h'

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
