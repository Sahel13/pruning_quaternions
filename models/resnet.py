import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
Resnet architecture.
"""


def hyper_params():
    hparams = {
        "dataset": 'cifar10',
        "output_directory": "17-10_2",
        "training": {
            "batch_size": 128,
            "num_epochs": 160,
            "learning_rate": 0.1,
            "milestones": [80, 120],
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


class Block(nn.Module):
    """
    A ResNet block.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Real(nn.Module):
    model_name = 'resnet_real'

    def __init__(self):
        super().__init__()
        num_segments = 3
        filters_per_segment = [16, 32, 64]
        architecture = [(num_filters, num_segments) for num_filters in
                        filters_per_segment]

        # Initial convolutional layer.
        current_filters = architecture[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # ResNet blocks
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(architecture):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer.
        self.fc = nn.Linear(architecture[-1][0], 10)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
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

        self.conv1 = layers.QConv2d(in_channels, out_channels, kernel_size=3,
                                    stride=stride, padding=1, bias=False)
        self.bn1 = layers.QBatchNorm2d(out_channels)

        self.conv2 = layers.QConv2d(out_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.bn2 = layers.QBatchNorm2d(out_channels)

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Quat(nn.Module):
    model_name = 'resnet_quat'

    def __init__(self):
        super().__init__()
        num_segments = 3
        filters_per_segment = [4, 8, 16]
        architecture = [(num_filters, num_segments) for num_filters in
                        filters_per_segment]

        # Initial convolutional layer.
        current_filters = architecture[0][0]
        self.conv = layers.QConv2d(1, current_filters, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.bn = layers.QBatchNorm2d(current_filters)

        # ResNet blocks
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(architecture):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Quat_Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer.
        self.fc = layers.QLinear(architecture[-1][0], 10)
        self.abs = layers.QuaternionToReal(10)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.abs(out)

    @classmethod
    def name(cls):
        return cls.model_name
