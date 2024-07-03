import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.quantizer import (SymmetricQuantFunction, AsymmetricQuantFunction, 
                                symmetric_linear_quantization_params, ste_round)

class QuantLinear(nn.Module):
    """
    Class to quantize weights of given linear layer

    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 weight_percentile=0,
                 ):
        super(QuantLinear, self).__init__()
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.counter = 0

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quantize_fn={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_integer', torch.zeros_like(linear.bias))
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        w = self.weight
        w_transform = w.data.detach()
        # calculate the quantization range of weights and bias
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
            if self.quantize_bias:
                b_min = self.bias.data
                b_max = self.bias.data
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)
            if self.quantize_bias:
                b_min = self.bias.data.min()
                b_max = self.bias.data.max()

        # perform the quantization
        if self.quant_mode == 'symmetric':
            self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                          self.per_channel)
            self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.fc_scaling_factor)

            bias_scaling_factor = self.fc_scaling_factor.view(1, -1) * prev_act_scaling_factor.view(1, -1)
            self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor)
        else:
            raise Exception('For weight, we only support symmetric quantization.')

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor
        correct_output_scale = bias_scaling_factor[0].view(1, -1)

        return ste_round.apply(
            F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)) * correct_output_scale