import torch
import torch.nn as nn
from modules.linear import QuantLinear
from modules.pool import QuantAdaptiveAvgPool2d
from modules.conv import QuantBnConv2d


class Q_ResNet50(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.init_block = QuantBnConv2d()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                quant_unit = Q_ResUnitBn()
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAdaptiveAvgPool2d((1, 1))
        self.quant_output = QuantLinear()
        
        if model != None:
            features = getattr(model, 'features')
            origin_init_block = getattr(features, 'init_block')
            self.init_block.set_param(origin_init_block.conv.conv, origin_init_block.conv.bn)

            for stage_num in range(0, 4):
                stage = getattr(features, "stage{}".format(stage_num + 1))
                for unit_num in range(0, self.channel[stage_num]):
                    unit = getattr(stage, "unit{}".format(unit_num + 1))
                    quant_unit = Q_ResUnitBn()
                    quant_unit.set_param(unit)
                    setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

            output = getattr(model, 'output')
            self.quant_output.set_param(output)




    def forward(self, x):
        x = self.init_block(x)

        x = self.pool(x)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x

class Q_ResUnitBn(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self):
        super(Q_ResUnitBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        convbn1 = unit.body.conv1
        self.conv1 = QuantBnConv2d()
        self.conv1.set_param(convbn1.conv, convbn1.bn)

        convbn2 = unit.body.conv2
        self.conv2 = QuantBnConv2d()
        self.conv2.set_param(convbn2.conv, convbn2.bn)

        convbn3 = unit.body.conv3
        self.conv3 = QuantBnConv2d()
        self.conv3.set_param(convbn3.conv, convbn3.bn)

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
        x = nn.ReLU()(x)

        x = self.conv3(x)

        x = x + identity

        x = nn.ReLU()(x)

        return x


def q_resnet50(model=None):
    net = Q_ResNet50(model)
    return net
