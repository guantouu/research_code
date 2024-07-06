import torch
from torch.autograd import Function, Variable

class ste_round(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize floating point input tensor to integers with the given scaling factor and zeropoint.

    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # reshape scale and zeropoint for convolutional weights and activations
    size = input.size()
    if len(input.shape) == 4:
        input = input.view(-1, input.size(1))
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input
    quantize = (1. / scale * input).view(size)
    return torch.round(quantize + zero_point)
    
def symmetric_linear_quantization_params(num_bits,
                                         saturation_min,
                                         saturation_max,
                                         per_strip=False):
    """
    Compute the scaling factor and zeropoint with the given quantization range for symmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    per_channel: if True, calculate the scaling factor per channel.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        if per_strip:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            if (scale.shape[0] == len(num_bits)):
                n = [2 ** (bit - 1) - 1  for bit in num_bits]
                n = torch.tensor(n).cuda(0)
            else:
                n = 2 ** (8 - 1) - 1 
            scale = torch.clamp(scale, min=1e-8) / n
        else:
            n = 2 ** (num_bits - 1) - 1 
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n

    return scale

def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range for asymmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    integral_zero_point: if True, adjust zero_point accordingly to make sure 0.0 in floating point tensor
                         be exactly mapped to an integer value.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** num_bits - 1
        scale = torch.clamp((saturation_max - saturation_min), min=1e-8) / float(n)

        # For asymmetric quantization, the current hardware support scaled unsigned integers without zero_point.
        # So saturation_min = 0 (we only do asymmetric quantization for activations after ReLU.)
        zero_point = -saturation_min / scale

        if integral_zero_point:
            if isinstance(zero_point, torch.Tensor):
                zero_point = zero_point.round()
            else:
                zero_point = float(round(zero_point))

        return scale, zero_point


class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0.).cuda()

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

        x_size = new_quant_x.size()

        if (isinstance(k, list)): 
            new_quant_x = new_quant_x.view(-1, new_quant_x.size(1))
            strip_quant_x = []
            for i in range (new_quant_x.shape[0]):
                n = 2 ** (k[i] - 1) - 1
                strip_quant_x.append(torch.clamp(new_quant_x[i], -n - 1, n))
            quant_x = torch.stack(strip_quant_x).view(x_size)
        else:
            n = 2 ** (k - 1) - 1
            quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        return quant_x

    @staticmethod
    def backward(ctx, grad_output):
        
        scale = ctx.scale
        
        output_size = grad_output.clone().shape
        output = grad_output.clone()
        if len(grad_output.shape) == 4:
            output = output.view(-1, output.size(1))
            scale = scale.view(-1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)
        result = (output / scale).view(output_size)
        return result, None, None, None


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using asymmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None, specified_zero_point=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of AsymmetricQuantFunction requires pre-calculated scaling factor.
        The current hardware support requires asymmetric quantization to use scaled unsigned integers
        without zero_point, so asymmetric quantization is for activations after ReLU, and zero_point is set to 0.
        specified_scale: pre-calculated scaling factor for the tensor x
        specified_zero_point: pre-calculated zero_point for the tensor x
        """
        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The AsymmetricQuantFunction requires a pre-calculated scaling factor")

        if specified_zero_point is not None:
            zero_point = specified_zero_point
        else:
            zero_point = torch.tensor(0).cuda()

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2 ** k - 1

        new_quant_x = torch.clamp(new_quant_x, 0, n)

        ctx.scale = scale

        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)
        return grad_output.clone() / scale, None, None, None


class transfer_float_averaging_to_int_averaging(Function):
    """
    Straight-through Estimator(STE) for Int Averaging

    The eps is used to avoid pytorh representation error like 2 = 1.99999999
    However, the eps has upper bound,
    take 7x7 integer average pooling as an example, the eps should be chosen to satisfy 48/49 + eps < 1.
    """

    @staticmethod
    def forward(ctx, x, eps=0.01):
        with torch.no_grad():
            x_int = torch.trunc(x + eps)
            return x_int

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None