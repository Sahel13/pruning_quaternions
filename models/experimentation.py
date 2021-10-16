import torch
import torch.nn as nn
from htorch import layers
import core_qnn.quaternion_layers as par
import torch.nn.functional as F


class Quat_P(nn.Module):
    model_name = 'lenet_p'

    def __init__(self):
        super().__init__()
        self.conv1 = par.QuaternionConv(4, 8, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = par.QuaternionConv(8, 16, 5, 1)
        self.fc1 = par.QuaternionLinear(16 * 5 * 5, 120)
        self.fc2 = par.QuaternionLinear(120, 84)
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


class Exp1(nn.Module):
    model_name = 'trial 1'

    # Learns one extra dimension for the input.
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 4, 3, 1, 1)
        self.conv1 = par.QuaternionConv(4, 8, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = par.QuaternionConv(8, 16, 5, 1)
        self.fc1 = par.QuaternionLinear(16 * 5 * 5, 120)
        self.fc2 = par.QuaternionLinear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
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


class Exp2(nn.Module):
    model_name = 'trial 2'

    # Learns one extra dimension for the input.
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 12, 3, 1, 1)
        self.conv1 = par.QuaternionConv(12, 8, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = par.QuaternionConv(8, 16, 5, 1)
        self.fc1 = par.QuaternionLinear(16 * 5 * 5, 120)
        self.fc2 = par.QuaternionLinear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
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