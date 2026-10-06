"""
ResNet for CIFAR-10/CIFAR-100, implemented in PyTorch.
Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'ResBlock', 'ResUnit', 'ResInitBlock',
           'get_resnet_cifar']

import os
import torch.nn as nn
from .common import conv1x1_block, conv3x3_block


class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.
    """
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResUnit(nn.Module):
    """
    ResNet unit with residual connection.
    """
    def __init__(self, in_channels, out_channels, stride):
        super(ResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.body = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class ResInitBlock(nn.Module):
    """
    ResNet CIFAR specific initial block.
    """
    def __init__(self, in_channels, out_channels):
        super(ResInitBlock, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResNet(nn.Module):
    """
    ResNet model for CIFAR, suitable for resnet20.
    """
    def __init__(self, channels, init_block_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=3,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 1
                if (i != 0) and (j == 0):
                    stride = 2
                unit = ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(kernel_size=8))
        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_resnet_cifar(blocks, model_name=None, num_classes=10, **kwargs):
    """
    Create ResNet model for CIFAR with specific parameters.
    """
    if (blocks - 2) % 6 != 0:
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))
    layers = [(blocks - 2) // 6] * 3

    channels_per_layers = [16, 32, 64]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    init_block_channels = 16

    net = ResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        num_classes=num_classes,
        **kwargs)

    return net


def resnet20(num_classes=10, **kwargs):
    """
    ResNet-20 model for CIFAR-10/CIFAR-100.
    """
    return get_resnet_cifar(
        blocks=20,
        model_name="resnet20",
        num_classes=num_classes,
        **kwargs)


def resnet32(num_classes=10, **kwargs):
    return get_resnet_cifar(blocks=32, model_name="resnet32", num_classes=num_classes, **kwargs)


def resnet44(num_classes=10, **kwargs):
    return get_resnet_cifar(blocks=44, model_name="resnet44", num_classes=num_classes, **kwargs)


def resnet56(num_classes=10, **kwargs):
    return get_resnet_cifar(blocks=56, model_name="resnet56", num_classes=num_classes, **kwargs)
