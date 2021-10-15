import torch
from torch.nn import Module, Linear, MaxPool2d, Conv2d
import torch.nn.functional as F
from htorch.layers import QConv2d, QLinear, QuaternionToReal


class LeNet(Module):
    def __init__(self):
        super().__init__()
        # Changed (3, 6, 5) to (3, 8, 5)
        self.conv1 = Conv2d(3, 8, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(8, 16, 5)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLeNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QConv2d(1, 2, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(2, 4, 5)
        self.fc1 = Linear(4 * 5 * 5, 30)
        self.fc2 = Linear(30, 21)
        self.fc3 = Linear(21, 10)
        self.abs = QuaternionToReal(10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.abs(self.fc3(x))
        return x


class LeNet_300_100(Module):
    """
    LeNet_300_100 neural network architecture.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 300)
        self.fc2 = Linear(300, 100)
        self.fc3 = Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QLeNet_300_100(Module):
    """
    LeNet_300_100 neural network architecture.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = QLinear(196, 75)
        self.fc2 = QLinear(75, 25)
        self.fc3 = QLinear(25, 10)
        self.abs = QuaternionToReal(10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.abs(self.fc3(x))
        return x


class CNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 8, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(8, 16, 5)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QConv2d(1, 2, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = QConv2d(2, 4, 5)
        self.fc1 = QLinear(4 * 5 * 5, 30)
        self.fc2 = QLinear(30, 21)
        self.fc3 = QLinear(21, 10)
        self.abs = QuaternionToReal(10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.abs(self.fc3(x))
        return x
