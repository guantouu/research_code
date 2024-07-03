import torch
import numpy as np
from modules.conv import QConv2d
from modules.linear import QLinear

def get_modules_list(model, known_modules):
    modules = []
    for module in model.modules():
        classname = module.__class__.__name__
        if classname in known_modules:
            modules.append(module)
    return modules

def filter_indices(values, threshold):
    indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            indices.append(idx)
    if len(indices) <= 1:
        indices = [0]
    return indices

def get_params_grad(model):
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def enable_calibrate(model):
    for name, child in model.named_children():
        if isinstance(child, QConv2d) or isinstance(child, QLinear):
            child.inference = False
        else:
            enable_calibrate(child)
    return model

def disable_calibrate(model):
    for name, child in model.named_children():
        if isinstance(child, QConv2d) or isinstance(child, QLinear):
            child.inference = True
        else:
            disable_calibrate(child)
    return model

def enable_hessian_quantizer(model):
    for name, child in model.named_children():
        if isinstance(child, QConv2d):
            child.modifyQuantizer()
        else:
            enable_hessian_quantizer(child)
    return model

def compute_strip_group(model, dataloader, criterion, device):
    importances = {}
    strip_importances_per_layer = {}

    known_modules = {'QConv2d'}
    modules = get_modules_list(model, known_modules)

    for m in model.parameters():
        shape_list = [4]
        if len(m.shape) in shape_list:
            m.requires_grad = True
        else:
            m.requires_grad = False

    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    model.to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward(create_graph = True)

    params, gradsH = get_params_grad(model)

    trace_vhv = []
    for index, p in enumerate(params):
        trace_vhv.append([])
        for c in range(p.size(0) * p.size(2) * p.size(3)):
            trace_vhv[index].append([])

    for i in range(1):
        v = [torch.randint_like(p, high = 2, device = device).float() * 2 - 1 for p in params]

        THv = [torch.zeros(p.size()).to(device) for p in params]
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(create_graph = True)

            params, gradsH = get_params_grad(model)
            Hv = torch.autograd.grad(gradsH, params, grad_outputs = v, only_inputs = True, retain_graph = False)
            THv = [THv1 + Hv1/float(len(dataloader)) + 0. for THv1, Hv1 in zip(THv, Hv)]
        Hv = THv

        Hv = [Hvi.detach().cpu() for Hvi in Hv]
        v = [vi.detach().cpu() for vi in v]

        with torch.no_grad():
            for Hv_i in range(len(Hv)):
                strip_Hv = Hv[Hv_i].view(-1, Hv[Hv_i].size(1))
                strip_v = v[Hv_i].view(-1, Hv[Hv_i].size(1))
                strip_i = 0
                for strip_Hv_i, strip_v_i in zip(strip_Hv, strip_v):
                    trace_vhv[Hv_i][strip_i].append(strip_Hv_i.flatten().dot(strip_v_i.flatten()).item())
                    strip_i += 1
                            
    for m in model.parameters():
        m.requires_grad = True

    strip_trace = []
    for k, layer in enumerate(trace_vhv):
        strip_trace.append(torch.zeros(len(layer)))
        for cnt, strip in enumerate(layer):
            strip_trace[k][cnt] = sum(strip) / len(strip)

    for k, m in enumerate(modules):
        tmp = []
        weight = m.weight.data
        strip_weight = weight.view(-1, weight.size(1))
        strip_importances_per_layer[m] = []
        for cnt, strip_w in enumerate(strip_weight):
            saliency = (strip_trace[k][cnt] * strip_w.detach().norm()**2 / strip_w.numel()).cpu().item()
            tmp.append(saliency)
            strip_importances_per_layer[m].append(saliency)
        importances[m] = (tmp, len(tmp))
    
    return importances, strip_importances_per_layer

def model_strip_group(model, importances, strip_importances_per_layer, bits, ratio=0.9):
    all_importances = []
    known_modules = {'QConv2d'}
    modules = get_modules_list(model, known_modules)
    
    for m in modules:
        imp_m = importances[m]
        imps = imp_m[0]
        all_importances += (imps)
    all_importances = sorted(all_importances)
    idx = int(ratio * len(all_importances)) - 1
    threshold = all_importances[idx]

    idx_recomputed = len(filter_indices(all_importances, threshold))
    print("all importances: {}".format(len(all_importances)))
    print('=> The threshold is: %.5f (%d), computed by function is: %.5f (%d).' %
        (threshold, idx, threshold, idx_recomputed))  
    # do pruning
    print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' %
        (max(all_importances), min(all_importances), threshold))

    for module in model.modules():
        classname = module.__class__.__name__
        if classname not in known_modules:
            continue
        strip_group_per_layer = [0] * len(strip_importances_per_layer[module])
        for k in range(0, len(strip_importances_per_layer[module])):
            if strip_importances_per_layer[module][k] > threshold:
                strip_group_per_layer[k] = bits['highly_sensitive']
            else:
                strip_group_per_layer[k] = bits['insensitive']
        module.weight_quantizer.sensitive = strip_group_per_layer
        print("low bit:{}".format(strip_group_per_layer.count(bits['insensitive'])))
        print("heighly bit:{}".format(strip_group_per_layer.count(bits['highly_sensitive'])))
    
    return model