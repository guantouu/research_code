import torch
import torch.nn as nn
from modules.linear import QuantLinear
from modules.pool import QuantAdaptiveAvgPool2d
from modules.conv import QuantBnConv2d

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

def q_vgg11(model=None):
    net = Q_VGG11(model)
    return net

def q_vgg19(model=None):
    net = Q_VGG19(model)
    return net    
