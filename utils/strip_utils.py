import torch
import numpy as np

def get_modules_list(model, known_modules):
    modules = []
    for module in model.modules():
        classname = module.__class__.__name__
        if classname in known_modules:
            modules.append(module)
    return modules

def get_params_grad(model):
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def filter_indices(values, threshold):
    indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            indices.append(idx)
    if len(indices) <= 1:
        indices = [0]
    return indices

def simplify_attribute_path(attribute_path):
    parts = attribute_path.split('.')
    if(len(parts) == 4):
        simplified_parts = [parts[1]]
    elif(len(parts) == 5):
        simplified_parts = [parts[1], parts[2], parts[3]]
    else:
        simplified_parts = [parts[1], parts[2], parts[4]]
    simplified_path = '.'.join(simplified_parts)
    
    return simplified_path

def compute_strip_importances(model, dataloader, criterion):
    importances = {}
    strip_importances_per_layer = {}
    known_modules = {"Conv2d"}
    
    modules = get_modules_list(model, known_modules)

    for m in model.parameters():
        shape_list = [4]
        if len(m.shape) in shape_list:
            m.requires_grad = True
        else:
            m.requires_grad = False

    image, target = next(iter(dataloader))
    image = image.cuda(0)
    target = target.cuda(0)
    
    output = model(image)
    loss = criterion(output, target)
    loss.backward(create_graph = True)

    params, gradsH = get_params_grad(model)

    trace_vhv = []
    for index, p in enumerate(params):
        trace_vhv.append([])
        for c in range(p.size(0) * p.size(2) * p.size(3)):
            trace_vhv[index].append([])

    for i in range(1):
        v = [torch.randint_like(p, high = 2).float().cuda(0) * 2 - 1 for p in params]

        THv = [torch.zeros(p.size()).cuda(0) for p in params]

        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(0), targets.cuda(0)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(create_graph = True)

            params, gradsH = get_params_grad(model)
            Hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs = True, retain_graph = False)
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
    known_modules = {'Conv2d'}
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

    strip_group = {}
    module_to_name = {}
    for name, module in model.named_modules():
        classname = module.__class__.__name__
        if classname not in known_modules:
            continue
        module_name = simplify_attribute_path(name)
        module_to_name[module] = module_name

    for module in model.modules():
        classname = module.__class__.__name__
        if classname not in known_modules:
            continue
        strip_group_per_layer = [0] * len(strip_importances_per_layer[module])
        module_name = module_to_name.get(module, None)
        strip_group[module_name] = []
        for k in range(0, len(strip_importances_per_layer[module])):
            if strip_importances_per_layer[module][k] > threshold:
                strip_group_per_layer[k] = bits['highly_sensitive']
                strip_group[module_name].append(bits['highly_sensitive'])
            else:
                strip_group_per_layer[k] = bits['insensitive']
                strip_group[module_name].append(bits['insensitive'])
        print("heighly/low bit :{}/{}".format(
            strip_group_per_layer.count(bits['highly_sensitive']),
            strip_group_per_layer.count(bits['insensitive']) 
        ))
    
    return strip_group