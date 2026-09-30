import torch
import torch.nn as nn
from modules.linear import QuantLinear
from modules.pool import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from modules.conv import QuantBnConv2d

class Q_ResNet20(nn.Module):
    """
    Quantized CIFAR ResNet (resnet20/32/44/56); the number of units per stage follows the given model.
    """
    def __init__(self, model=None, units_per_stage=(3, 3, 3)):
        super().__init__()

        # Initial convolution layer
        self.init_block = QuantBnConv2d()
        self.act = nn.ReLU(inplace=True)

        if model is not None:
            units_per_stage = [len(getattr(model.features, "stage{}".format(i + 1))) for i in range(3)]
        self.channel = list(units_per_stage)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                quant_unit = Q_ResBlockBn()
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        # Average pooling and fully connected layer
        self.final_pool = QuantAvgPool2d(kernel_size=8)
        self.quant_output = QuantLinear()

        if model is not None:
            features = getattr(model, 'features')
            origin_init_block = getattr(features, 'init_block')
            self.init_block.set_param(origin_init_block.conv.conv, origin_init_block.conv.bn)

            for stage_num in range(0, 3):
                stage = getattr(features, "stage{}".format(stage_num + 1))
                for unit_num in range(0, self.channel[stage_num]):
                    unit = getattr(stage, "unit{}".format(unit_num + 1))
                    quant_unit = Q_ResBlockBn()
                    quant_unit.set_param(unit)
                    setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

            output = getattr(model, 'output')
            self.quant_output.set_param(output)

    def forward(self, x):
        x = self.init_block(x)

        x = self.act(x)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x
    
class Q_ResBlockBn(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        convbn1 = unit.body.conv1
        self.conv1 = QuantBnConv2d()
        self.conv1.set_param(convbn1.conv, convbn1.bn)

        convbn2 = unit.body.conv2
        self.conv2 = QuantBnConv2d()
        self.conv2.set_param(convbn2.conv, convbn2.bn)

        if self.resize_identity:
            self.identity_conv = QuantBnConv2d()
            self.identity_conv.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x

        x = self.conv1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)

        x = x + identity

        x = nn.ReLU()(x)

        return x

def q_resnet20(model=None):
    net = Q_ResNet20(model)
    return net

def q_resnet32(model=None):
    return Q_ResNet20(model, units_per_stage=(5, 5, 5))

def q_resnet44(model=None):
    return Q_ResNet20(model, units_per_stage=(7, 7, 7))

def q_resnet56(model=None):
    return Q_ResNet20(model, units_per_stage=(9, 9, 9))