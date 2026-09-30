"""
Generic conversion of a float model into the quantized model used by the pipeline, for networks that
have no hand-written Q_ class (torchvision ResNet / VGG for ImageNet).
"""
import torch.nn as nn
from modules.conv import QuantBnConv2d
from modules.linear import QuantLinear

def quantize_model(model):
    """
    In place: every Conv2d and the BatchNorm2d registered right after it (the pairing rule of
    get_bn_fold_scale) become one QuantBnConv2d at the conv's name, the BatchNorm2d becomes
    Identity, and every Linear becomes a QuantLinear. The model's own forward is unchanged, so
    quantized layer names equal the float conv names used as saliency keys.
    """
    for parent in list(model.modules()):
        children = list(parent.named_children())
        for k, (name, child) in enumerate(children):
            if isinstance(child, nn.Conv2d):
                bn_name, bn = children[k + 1] if k + 1 < len(children) else (None, None)
                if not (isinstance(bn, nn.BatchNorm2d) and bn.num_features == child.out_channels):
                    raise ValueError("Conv2d {} is not followed by its BatchNorm2d".format(name))
                quant_conv = QuantBnConv2d()
                quant_conv.set_param(child, bn)
                setattr(parent, name, quant_conv)
                setattr(parent, bn_name, nn.Identity())
            elif isinstance(child, nn.Linear):
                quant_linear = QuantLinear()
                quant_linear.set_param(child)
                setattr(parent, name, quant_linear)
    return model
