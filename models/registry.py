"""
Registry of the networks used by the pipeline. Add a network here to make it available everywhere.

CIFAR: net -> (float model module, constructor, quantized model module, constructor); saliency keys
are the simplified float conv names (utils.strip_utils.simplify_attribute_path).
ImageNet: net -> torchvision constructor, pretrained on IMAGENET1K_V1; the quantized model comes from
models.quantize.quantize_model and saliency keys are the full conv names.
"""
import importlib
import torchvision
from models.quantize import quantize_model
from utils.strip_utils import simplify_attribute_path

MODELS = {
    'resnet20': ('models.ResNet20', 'resnet20', 'models.Q_ResNet20', 'q_resnet20'),
    'resnet32': ('models.ResNet20', 'resnet32', 'models.Q_ResNet20', 'q_resnet32'),
    'resnet44': ('models.ResNet20', 'resnet44', 'models.Q_ResNet20', 'q_resnet44'),
    'resnet56': ('models.ResNet20', 'resnet56', 'models.Q_ResNet20', 'q_resnet56'),
    'resnet18': ('models.ResNet', 'resnet18', 'models.Q_ResNet', 'q_resnet18'),
    'resnet50': ('models.ResNet', 'resnet50', 'models.Q_ResNet', 'q_resnet50'),
    'vgg11': ('models.VGG', 'vgg11', 'models.Q_VGG', 'q_vgg11'),
    'vgg13': ('models.VGG', 'vgg13', 'models.Q_VGG', 'q_vgg13'),
    'vgg16': ('models.VGG', 'vgg16', 'models.Q_VGG', 'q_vgg16'),
    'vgg19': ('models.VGG', 'vgg19', 'models.Q_VGG', 'q_vgg19'),
}

IMAGENET_MODELS = {
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',
    'vgg11': 'vgg11_bn',
    'vgg13': 'vgg13_bn',
    'vgg16': 'vgg16_bn',
    'vgg19': 'vgg19_bn',
}

def _lookup(net, dataset):
    table = IMAGENET_MODELS if dataset == 'imagenet' else MODELS
    if net not in table:
        raise ValueError("Unknown model type: {} for {} (available: {})".format(net, dataset, list(table)))
    return table[net]

def build_float_model(net, num_classes, state_dict=None, dataset='cifar10'):
    if dataset == 'imagenet':
        model = torchvision.models.get_model(_lookup(net, dataset), weights=None, num_classes=num_classes)
    else:
        float_module, float_ctor, _, _ = _lookup(net, dataset)
        model = getattr(importlib.import_module(float_module), float_ctor)(num_classes)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model

def build_quant_model(net, float_model, dataset='cifar10'):
    if dataset == 'imagenet':
        _lookup(net, dataset)
        return quantize_model(float_model)
    _, _, quant_module, quant_ctor = _lookup(net, dataset)
    return getattr(importlib.import_module(quant_module), quant_ctor)(float_model)

def saliency_key(conv_name, dataset='cifar10'):
    """
    Name of a float conv in saliency files and bit configs, i.e. its name in the quantized model.
    """
    return conv_name if dataset == 'imagenet' else simplify_attribute_path(conv_name)

def imagenet_pretrained(net):
    """
    (state dict, published top-1 accuracy) of the torchvision IMAGENET1K_V1 weights.
    """
    weights = torchvision.models.get_model_weights(_lookup(net, 'imagenet')).IMAGENET1K_V1
    return weights.get_state_dict(progress=True), weights.meta['_metrics']['ImageNet-1K']['acc@1']
