import torch
import torch.nn as nn
from modules.quantizer import ste_round, transfer_float_averaging_to_int_averaging

class QuantAdaptiveAvgPool2d(nn.Module):
    """
    Quantized Average Pooling Layer

    Parameters:
    ----------
    kernel_size : int, default 7
        Kernel size for average pooling.
    stride : int, default 1
        stride for average pooling.
    padding : int, default 0
        padding for average pooling.
    """

    def __init__(self,
                 output_size=(1, 1)):
        super(QuantAdaptiveAvgPool2d, self).__init__()

        self.output_size = output_size
        self.final_pool = nn.AdaptiveAvgPool2d(output_size)

    def set_param(self, pool):
        self.final_pool = pool

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]

        if x_scaling_factor is None:
            return self.final_pool(x)

        x_scaling_factor = x_scaling_factor.view(-1)
        correct_scaling_factor = x_scaling_factor

        x_int = x / correct_scaling_factor
        x_int = ste_round.apply(x_int)
        x_int = self.final_pool(x_int)

        x_int = transfer_float_averaging_to_int_averaging.apply(x_int)

        return (x_int * correct_scaling_factor, correct_scaling_factor)
