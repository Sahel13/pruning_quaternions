import torch
from htorch import layers
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture for CIFAR-10 classification
proposed by Gaudet.
"""


def hyper_params():
    hparams = {
        "dataset": 'cifar10',
        "output_directory": "morning_test",
        "training": {
            "batch_size": 128,
            "num_epochs": 120,
            "learning_rate": 0.1,
            "milestones": [],
            "gamma": 0.1,
            "weight_decay": 1e-4,
            "mini_batch": 1000000
        },
        "pruning": {
            "iterations": 10,
            "percentage": 0.96
        }
    }
    return hparams


def lr_scheduler(epochs):
    """
    Custom lr function used by Gaudet.
    """
    if epochs <= 10:
        return 0.1
    elif epochs < 120:
        return 1
    elif epochs >= 120 and epochs < 150:
        return 0.1
    else:
        return 0.01


class Block(nn.Module):
    """
    A ResNet block.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Shortcut connection
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return F.relu(out)


class Real(nn.Module):
    model_name = 'gaudet_1'

    def __init__(self):
        super().__init__()
        architecture = [(32, 2), (64, 1), (128, 1)]

        # Initial convolutional layer.
        current_filters = architecture[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # ResNet blocks
        blocks = []
        # First segment
        blocks.append(Block(32, 32))
        blocks.append(Block(32, 32))
        # Second segment
        blocks.append(Block(32, 64, True))
        # Third segment
        blocks.append(Block(64, 128, True))

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer.
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    @classmethod
    def name(cls):
        return cls.model_name


class Quat_Block(nn.Module):
    """
    A quaternion ResNet block.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1

        self.bn1 = layers.QBatchNorm2d(in_channels)
        self.conv1 = layers.QConv2d(in_channels, out_channels, kernel_size=3,
                                    stride=stride, padding=1, bias=False)

        self.bn2 = layers.QBatchNorm2d(out_channels)
        self.conv2 = layers.QConv2d(out_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=False)

        # Shortcut connection
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layers.QConv2d(in_channels, out_channels, kernel_size=1,
                               stride=2, bias=False),
                layers.QBatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return F.relu(out)


class Quat3(nn.Module):
    model_name = 'gaudet_4'

    def __init__(self):
        super().__init__()
        architecture = [(8, 2), (16, 1), (32, 1)]

        self.i_block = Block(3, 3)
        self.j_block = Block(3, 3)
        self.k_block = Block(3, 3)

        # Initial convolutional layer.
        current_filters = architecture[0][0]
        self.conv = layers.QConv2d(3, current_filters, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.bn = layers.QBatchNorm2d(current_filters)

        # ResNet blocks
        blocks = []
        # First segment
        blocks.append(Quat_Block(8, 8))
        blocks.append(Quat_Block(8, 8))
        # Second segment
        blocks.append(Quat_Block(8, 16, True))
        # Third segment
        blocks.append(Quat_Block(16, 32, True))

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer.
        self.fc = layers.QLinear(architecture[-1][0], 10)
        self.abs = layers.QuaternionToReal(10)

    def forward(self, x):
        # Preparation
        out = x
        out = torch.cat((out, self.i_block(x)), 1)
        out = torch.cat((out, self.j_block(x)), 1)
        out = torch.cat((out, self.k_block(x)), 1)
        # The rest
        out = F.relu(self.bn(self.conv(out)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return self.abs(out)

    @classmethod
    def name(cls):
        return cls.model_name


class Quat1(nn.Module):
    model_name = 'gaudet_2'

    def __init__(self):
        super().__init__()
        architecture = [(8, 2), (16, 1), (32, 1)]

        # Initial convolutional layer.
        current_filters = architecture[0][0]
        self.conv = layers.QConv2d(1, current_filters, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.bn = layers.QBatchNorm2d(current_filters)

        # ResNet blocks
        blocks = []
        # First segment
        blocks.append(Quat_Block(8, 8))
        blocks.append(Quat_Block(8, 8))
        # Second segment
        blocks.append(Quat_Block(8, 16, True))
        # Third segment
        blocks.append(Quat_Block(16, 32, True))

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer.
        self.fc = layers.QLinear(architecture[-1][0], 10)
        self.abs = layers.QuaternionToReal(10)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return self.abs(out)

    @classmethod
    def name(cls):
        return cls.model_name


class Quat2(nn.Module):
    model_name = 'gaudet_3'

    def __init__(self):
        super().__init__()
        architecture = [(8, 2), (16, 1), (32, 1)]

        self.conv0 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(12)

        # Initial convolutional layer.
        current_filters = architecture[0][0]
        self.conv = layers.QConv2d(3, current_filters, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.bn = layers.QBatchNorm2d(current_filters)

        # ResNet blocks
        blocks = []
        # First segment
        blocks.append(Quat_Block(8, 8))
        blocks.append(Quat_Block(8, 8))
        # Second segment
        blocks.append(Quat_Block(8, 16, True))
        # Third segment
        blocks.append(Quat_Block(16, 32, True))

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer.
        self.fc = layers.QLinear(architecture[-1][0], 10)
        self.abs = layers.QuaternionToReal(10)

    def forward(self, x):
        out = self.bn0(self.conv0(x))
        out = F.relu(self.bn(self.conv(out)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return self.abs(out)

    @classmethod
    def name(cls):
        return cls.model_name
