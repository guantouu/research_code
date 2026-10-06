import torch
import torch.nn as nn
from modules.linear import QuantLinear
from modules.pool import QuantAdaptiveAvgPool2d
from modules.conv import QuantBnConv2d


class Q_ResNet20(nn.Module):
    def __init__(self, model=None):
        super().__init__()

        self.num_blocks = [3, 3, 3]

        # Initial convolution layer
        self.conv1 = QuantBnConv2d()
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        for i, num_blocks in enumerate(self.num_blocks):
            for j in range(num_blocks):
                # Convolutional layers within a block
                setattr(self, f"layer{i+1}_block{j+1}_conv1", QuantBnConv2d())
                setattr(self, f"layer{i+1}_block{j+1}_relu1", nn.ReLU(inplace=True))
                setattr(self, f"layer{i+1}_block{j+1}_conv2", QuantBnConv2d())
                setattr(self, f"layer{i+1}_block{j+1}_relu2", nn.ReLU(inplace=True))
                # Downsample layer for shortcut connection if needed
                if j == 0 and i != 0:
                    setattr(self, f"layer{i+1}_block{j+1}_downsample", QuantBnConv2d())
                else:
                    setattr(self, f"layer{i+1}_block{j+1}_downsample", None)

        # Average pooling and fully connected layer
        self.avgpool = QuantAdaptiveAvgPool2d((1, 1))
        self.fc = QuantLinear()

        if model is not None:
            # Copy parameters from the existing model
            self.conv1.set_param(model.conv1, model.bn1)
            for i, num_blocks in enumerate(self.num_blocks):
                for j in range(num_blocks):
                    block = model.layer1[j] if i == 0 else model.layer2[j] if i == 1 else model.layer3[j]
                    conv1 = getattr(self, f"layer{i+1}_block{j+1}_conv1")
                    conv1.set_param(block.conv1, block.bn1)
                    conv2 = getattr(self, f"layer{i+1}_block{j+1}_conv2")
                    conv2.set_param(block.conv2, block.bn2)
                    downsample = getattr(self, f"layer{i+1}_block{j+1}_downsample")
                    if downsample is not None:
                        downsample.set_param(block.downsample[0], block.downsample[1])
            self.fc.set_param(model.fc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        for i, num_blocks in enumerate(self.num_blocks):
            for j in range(num_blocks):
                identity = x
                conv1 = getattr(self, f"layer{i+1}_block{j+1}_conv1")
                relu1 = getattr(self, f"layer{i+1}_block{j+1}_relu1")
                conv2 = getattr(self, f"layer{i+1}_block{j+1}_conv2")
                relu2 = getattr(self, f"layer{i+1}_block{j+1}_relu2")
                downsample = getattr(self, f"layer{i+1}_block{j+1}_downsample")

                out = conv1(x)
                out = relu1(out)
                out = conv2(out)

                if downsample is not None:
                    identity = downsample(x)

                out += identity
                out = relu2(out)
                x = out

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Q_VGG(nn.Module):
    """
    Quantized VGG (models/VGG.py); the number of convs per block follows the given model.
    Dropout is omitted: the quantized model is used for evaluation and short fine-tuning.
    """
    def __init__(self, model):
        super().__init__()

        features = getattr(model, 'features')
        self.num_blocks = [sum(1 for name, _ in block.named_children() if name.startswith('conv'))
                           for block in features.children()]

        for i, num_convs in enumerate(self.num_blocks):
            block = getattr(features, "block{}".format(i + 1))
            for j in range(num_convs):
                quant_conv = QuantBnConv2d()
                quant_conv.set_param(getattr(block, "conv{}".format(j + 1)), getattr(block, "bn{}".format(j + 1)))
                setattr(self, f"block{i + 1}.conv{j + 1}", quant_conv)
            setattr(self, f"block{i + 1}.MaxPool", nn.MaxPool2d(kernel_size=2, stride=2))

        classifier = getattr(model, 'classifier')
        for k in range(1, 4):
            quant_linear = QuantLinear()
            quant_linear.set_param(getattr(classifier, 'classifier_linear_{}'.format(k)))
            setattr(self, 'classifier_linear_{}'.format(k), quant_linear)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for i, num_convs in enumerate(self.num_blocks):
            for j in range(num_convs):
                x = self.relu(getattr(self, f"block{i+1}.conv{j+1}")(x))
            x = getattr(self, f"block{i+1}.MaxPool")(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.classifier_linear_1(x))
        x = self.relu(self.classifier_linear_2(x))
        x = self.classifier_linear_3(x)
        return x

def q_vgg11(model=None):
    return Q_VGG(model)

def q_vgg13(model=None):
    return Q_VGG(model)

def q_vgg16(model=None):
    return Q_VGG(model)

def q_vgg19(model=None):
    return Q_VGG(model)

def q_resnet20(model=None):
    net = Q_ResNet20(model)
    return net
