import torch
import torch.nn as nn
from modules.quantizer import (SymmetricQuantFunction, AsymmetricQuantFunction, 
                                symmetric_linear_quantization_params, asymmetric_linear_quantization_params)

class QuantAct(nn.Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int, default 4
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    act_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    fixed_point_quantization : bool, default False
        Whether to skip deployment-oriented operations and use fixed-point rather than integer-only quantization.
    """

    def __init__(self,
                 activation_bit=4,
                 act_range_momentum=0.95,
                 running_stat=True,
                 quant_mode="symmetric",
                 act_percentile=0,
                 fixed_point_quantization=False):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.act_percentile = act_percentile
        self.fixed_point_quantization = fixed_point_quantization

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        self.register_buffer('pre_weight_scaling_factor', torch.ones(1))
        self.register_buffer('identity_weight_scaling_factor', torch.ones(1))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode={2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(),
                                          self.x_max.item())

    def forward(self, x, pre_act_scaling_factor=None, pre_weight_scaling_factor=None, identity=None,
                identity_scaling_factor=None, identity_weight_scaling_factor=None):
        """
        x: the activation that we need to quantize
        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer
        pre_weight_scaling_factor: the scaling factor of the previous weight quantization layer
        identity: if True, we need to consider the identity branch
        identity_scaling_factor: the scaling factor of the previous activation quantization of identity
        identity_weight_scaling_factor: the scaling factor of the weight quantization layer in the identity branch

        Note that there are two cases for identity branch:
        (1) identity branch directly connect to the input featuremap
        (2) identity branch contains convolutional layers that operate on the input featuremap
        """
        if type(x) is tuple:
            if len(x) == 3:
                channel_num = x[2]
            pre_act_scaling_factor = x[1]
            x = x[0]

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.act_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        # calculate the quantization range of the activations
        if self.act_percentile == 0:
            x_min = x.data.min()
            x_max = x.data.max()
        elif self.quant_mode == 'symmetric':
            x_min, x_max = get_percentile_min_max(x.detach().view(-1), 100 - self.act_percentile,
                                                  self.act_percentile, output_tensor=True)
        # Note that our asymmetric quantization is implemented using scaled unsigned integers without zero_points,
        # that is to say our asymmetric quantization should always be after ReLU, which makes
        # the minimum value to be always 0. As a result, if we use percentile mode for asymmetric quantization,
        # the lower_percentile will be set to 0 in order to make sure the final x_min is 0.
        elif self.quant_mode == 'asymmetric':
            x_min, x_max = get_percentile_min_max(x.detach().view(-1), 0, self.act_percentile, output_tensor=True)

        # Initialization
        if self.x_min == self.x_max:
            self.x_min += x_min
            self.x_max += x_max

        # use momentum to update the quantization range
        elif self.act_range_momentum == -1:
            self.x_min = min(self.x_min, x_min)
            self.x_max = max(self.x_max, x_max)
        else:
            self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
            self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        # perform the quantization
        if self.quant_mode == 'symmetric':
            self.act_scaling_factor = symmetric_linear_quantization_params(self.activation_bit,
                                                                           self.x_min, self.x_max, False)
        # Note that our asymmetric quantization is implemented using scaled unsigned integers
        # without zero_point shift. As a result, asymmetric quantization should be after ReLU,
        # and the self.act_zero_point should be 0.
        else:
            self.act_scaling_factor, self.act_zero_point = asymmetric_linear_quantization_params(
                self.activation_bit, self.x_min, self.x_max, True)
        if (pre_act_scaling_factor is None) or (self.fixed_point_quantization == True):
            # this is for the case of input quantization,
            # or the case using fixed-point rather than integer-only quantization
            quant_act_int = self.act_function(x, self.activation_bit, self.act_scaling_factor)
        elif type(pre_act_scaling_factor) is list:
            # this is for the case of multi-branch quantization
            branch_num = len(pre_act_scaling_factor)
            quant_act_int = x
            start_channel_index = 0
            for i in range(branch_num):
                quant_act_int[:, start_channel_index: start_channel_index + channel_num[i], :, :] \
                    = fixedpoint_fn.apply(x[:, start_channel_index: start_channel_index + channel_num[i], :, :],
                                          self.activation_bit, self.quant_mode, self.act_scaling_factor, 0,
                                          pre_act_scaling_factor[i],
                                          pre_act_scaling_factor[i] / pre_act_scaling_factor[i])
                start_channel_index += channel_num[i]
        else:
            if identity is None:
                if pre_weight_scaling_factor is None:
                    pre_weight_scaling_factor = self.pre_weight_scaling_factor
                quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                    self.act_scaling_factor, 0, pre_act_scaling_factor,
                                                    pre_weight_scaling_factor)
            else:
                if identity_weight_scaling_factor is None:
                    identity_weight_scaling_factor = self.identity_weight_scaling_factor
                quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                    self.act_scaling_factor, 1, pre_act_scaling_factor,
                                                    pre_weight_scaling_factor,
                                                    identity, identity_scaling_factor,
                                                    identity_weight_scaling_factor)
        correct_output_scale = self.act_scaling_factor.view(-1)
        return (quant_act_int * correct_output_scale, self.act_scaling_factor)