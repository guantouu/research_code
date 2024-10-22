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

class Q_VGG11(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.num_blocks = [1, 1, 2, 2, 2]

        for i, num_convs in enumerate(self.num_blocks):
            for j in range(num_convs):
                setattr(self, f"block{i + 1}.conv{j + 1}", QuantBnConv2d())
                setattr(self, f"block{i + 1}.relu", nn.ReLU(inplace=True))
            setattr(self, f"block{i + 1}.MaxPool", nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier_linear_1 = QuantLinear()
        self.classifier_relu_1 = nn.ReLU(True)
        self.classifier_linear_2 = QuantLinear()
        self.classifier_relu_2 = nn.ReLU(True)
        self.classifier_linear_3 = QuantLinear()

        if model != None:
            features = getattr(model, 'features')
            for i, num_convs in enumerate(self.num_blocks):
                block = getattr(features, "block{}".format(i +1))
                for j in range(num_convs):
                    conv = getattr(block, "conv{}".format(j + 1))
                    bn = getattr(block, "bn{}".format(j + 1))
                    quant_conv = QuantBnConv2d()
                    quant_conv.set_param(conv, bn)
                    setattr(self, f"block{i + 1}.conv{j + 1}", quant_conv)
            classifier = getattr(model, 'classifier')
            linear = getattr(classifier, 'classifier_linear_1')
            self.classifier_linear_1.set_param(linear)
            linear = getattr(classifier, 'classifier_linear_2')
            self.classifier_linear_2.set_param(linear)
            linear = getattr(classifier, 'classifier_linear_3')
            self.classifier_linear_3.set_param(linear)                        

    def forward(self, x):
        for i, num_convs in enumerate(self.num_blocks):
            for j in range(num_convs):
                con_tmp_func = getattr(self, f"block{i+1}.conv{j+1}")
                relu_tmp_func = getattr(self, f"block{i+1}.relu")
                x = con_tmp_func(x)
                x = relu_tmp_func(x)
            pool_tmp_func = getattr(self, f"block{i+1}.MaxPool")
            x = pool_tmp_func(x)
        x = self.classifier_linear_1(x)
        x = self.classifier_relu_1(x)
        x = self.classifier_linear_2(x)
        x = self.classifier_relu_2(x)
        x = self.classifier_linear_3(x)

        return x

class Q_VGG19(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.num_blocks = [2, 2, 4, 4, 4]

        for i, num_convs in enumerate(self.num_blocks):
            for j in range(num_convs):
                setattr(self, f"block{i + 1}.conv{j + 1}", QuantBnConv2d())
                setattr(self, f"block{i + 1}.relu", nn.ReLU(inplace=True))
            setattr(self, f"block{i + 1}.MaxPool", nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier_linear_1 = QuantLinear()
        self.classifier_relu_1 = nn.ReLU(True)
        self.classifier_linear_2 = QuantLinear()
        self.classifier_relu_2 = nn.ReLU(True)
        self.classifier_linear_3 = QuantLinear()

        if model != None:
            features = getattr(model, 'features')
            for i, num_convs in enumerate(self.num_blocks):
                block = getattr(features, "block{}".format(i +1))
                for j in range(num_convs):
                    conv = getattr(block, "conv{}".format(j + 1))
                    bn = getattr(block, "bn{}".format(j + 1))
                    quant_conv = QuantBnConv2d()
                    quant_conv.set_param(conv, bn)
                    setattr(self, f"block{i + 1}.conv{j + 1}", quant_conv)
            classifier = getattr(model, 'classifier')
            linear = getattr(classifier, 'classifier_linear_1')
            self.classifier_linear_1.set_param(linear)
            linear = getattr(classifier, 'classifier_linear_2')
            self.classifier_linear_2.set_param(linear)
            linear = getattr(classifier, 'classifier_linear_3')
            self.classifier_linear_3.set_param(linear)                        

    def forward(self, x):
        for i, num_convs in enumerate(self.num_blocks):
            for j in range(num_convs):
                con_tmp_func = getattr(self, f"block{i+1}.conv{j+1}")
                relu_tmp_func = getattr(self, f"block{i+1}.relu")
                x = con_tmp_func(x)
                x = relu_tmp_func(x)
            pool_tmp_func = getattr(self, f"block{i+1}.MaxPool")
            x = pool_tmp_func(x)
        x = self.classifier_linear_1(x)
        x = self.classifier_relu_1(x)
        x = self.classifier_linear_2(x)
        x = self.classifier_relu_2(x)
        x = self.classifier_linear_3(x)

        return x

def q_vgg11(model=None):
    net = Q_VGG11(model)
    return net

def q_vgg19(model=None):
    net = Q_VGG19(model)
    return net    

def q_resnet20(model=None):
    net = Q_ResNet20(model)
    return net