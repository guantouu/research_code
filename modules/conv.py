import torch
import torch.nn as nn
from modules.quantizer import SymmetricQuantFunction, AsymmetricQuantFunction, symmetric_linear_quantization_params
import torch.nn.functional as F

class QuantBnConv2d(nn.Module):
    """
    Class to quantize given convolutional layer weights, with support for both folded BN and separate BN.

    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    fix_BN : bool, default False
        Whether to fix BN statistics during training.
    fix_BN_threshold: int, default None
        When to start training with folded BN.
    """

    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 quant_mode="symmetric",
                 per_strip=False):
        super(QuantBnConv2d, self).__init__()
        self.weight_bit = weight_bit
        self.per_strip = per_strip
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode
        self.counter = 1

    def set_param(self, conv, bn):
        self.out_channels = conv.out_channels
        self.register_buffer('convbn_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(conv.weight.data))
        self.register_buffer('bias_integer', torch.zeros_like(bn.bias))

        self.conv = conv
        self.bn = bn
        self.bn.momentum = 0.99

    def __repr__(self):
        conv_s = super(QuantBnConv2d, self).__repr__()
        s = "({0}, weight_bit={1}, bias_bit={2}, groups={3}, wt-strip-wise={4}, quant_mode={6})".format(
            conv_s, self.weight_bit, self.bias_bit, self.conv.groups, self.per_strip, self.quant_mode)
        return s

    def forward(self, x, pre_act_scaling_factor=None):
        """
        x: the input activation
        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer
        """
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        running_std = torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.conv.weight * scale_factor.reshape([self.conv.out_channels, 1, 1, 1])

        if self.conv.bias is not None:
            scaled_bias = self.conv.bias
        else:
            scaled_bias = torch.zeros_like(self.bn.running_mean)
        scaled_bias = (scaled_bias - self.bn.running_mean.detach()) * scale_factor + self.bn.bias

        if self.per_strip:
            w_permute = scaled_weight.data.permute(1, 0, 2, 3)
            strip_w_transform = w_permute.contiguous().view(self.conv.in_channels, -1).transpose(0, 1)
            w_min = strip_w_transform.min(dim=1).values
            w_max = strip_w_transform.max(dim=1).values

            w_transform = scaled_weight.data.contiguous().view(self.conv.out_channels, -1)
            bias_w_min = w_transform.min(dim=1).values
            bias_w_max = w_transform.max(dim=1).values

        else:
            w_min = scaled_weight.data.min()
            w_max = scaled_weight.data.max()
            bias_w_min = w_min
            bias_w_max = w_max

        if self.quant_mode == 'symmetric':
            self.convbn_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_strip)
            self.convbn_bias_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, bias_w_min, bias_w_max, self.per_strip)
            self.weight_integer = self.weight_function(scaled_weight, self.weight_bit, self.convbn_scaling_factor)            
            if self.quantize_bias:
                bias_scaling_factor = self.convbn_bias_scaling_factor.view(1, -1)
                self.bias_integer = self.weight_function(scaled_bias, self.bias_bit, bias_scaling_factor)
            self.convbn_scaled_bias = scaled_bias
        else:
            raise Exception('For weight, we only support symmetric quantization.')

        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return F.conv2d(x, self.weight_integer, self.bias_integer, self.conv.stride, self.conv.padding,
                         self.conv.dilation, self.conv.groups) * correct_output_scale

