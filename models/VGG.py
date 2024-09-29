import os
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_blocks, num_classes=100):
        super(VGG, self).__init__()      
        self.in_channels = 3
        self.out_channels_list = [64, 128, 256, 512, 512]
        self.features = nn.Sequential()
        for i, num_convs in enumerate(num_blocks):
            block = nn.Sequential()
            for j in range(num_convs):
                block.add_module(
                    "conv{}".format(j + 1),
                    nn.Conv2d(self.in_channels, self.out_channels_list[i], kernel_size=3, padding=1)
                )
                block.add_module("bn{}".format(j + 1), nn.BatchNorm2d(self.out_channels_list[i]))
                block.add_module("relu", nn.ReLU(inplace=True))
                self.in_channels = self.out_channels_list[i]
            block.add_module("MaxPool", nn.MaxPool2d(kernel_size=2, stride=2))
            self.features.add_module("block{}".format(i + 1), block)

        self.classifier = nn.Sequential()
        self.classifier.add_module("classifier_linear_1", nn.Linear(512, 4096))
        self.classifier.add_module("classifier_relu_1", nn.ReLU(True))
        self.classifier.add_module("classifier_linear_2", nn.Linear(4096, 4096))
        self.classifier.add_module("classifier_relu_2", nn.ReLU(True))
        self.classifier.add_module("classifier_linear_3", nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg11(num_classes=10, **kwargs):
    return VGG(num_blocks=[1, 1, 2, 2, 2], num_classes=num_classes, **kwargs)

def vgg19(num_classes=10, **kwargs):
    return VGG(num_blocks=[2, 2, 4, 4, 4], num_classes=num_classes, **kwargs)
