"""
Registry of the networks used by the pipeline: net -> (float model module, constructor,
quantized model module, constructor). Add a network here to make it available everywhere.
"""
import importlib

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

def _lookup(net):
    if net not in MODELS:
        raise ValueError("Unknown model type: {} (available: {})".format(net, list(MODELS)))
    return MODELS[net]

def build_float_model(net, num_classes, state_dict=None):
    float_module, float_ctor, _, _ = _lookup(net)
    model = getattr(importlib.import_module(float_module), float_ctor)(num_classes)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model

def build_quant_model(net, float_model):
    _, _, quant_module, quant_ctor = _lookup(net)
    return getattr(importlib.import_module(quant_module), quant_ctor)(float_model)
