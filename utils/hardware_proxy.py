"""
Analytic hardware proxies of a per-strip bit config, cheap enough for every point of a dense ratio grid
(no NeuroSIM run). Validated against NeuroSIM on the 8/2 runs:
    area   follows avg_weight_bits            (bits per weight)
    energy follows column_read_weighted_bits  (bits per strip weighted by its column reads, i.e. the
           output positions of its layer: the ADC reads a column once per output position, whatever
           the strip length). Across all 14 saliency/random pairs, the design with the lower value used
           less NeuroSIM energy.
NeuroSIM latency does not follow either, so there is no latency proxy: measure it with hardware_eval.py.
"""
import torch
from models import registry

def conv_output_positions(model, input_shape, dataset):
    """
    net_structure for estimate_hardware_proxy: {saliency key: output positions (H*W) of the conv}, from one
    forward pass of a zero input of input_shape (C, H, W).
    """
    positions, handles = {}, []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            key = registry.saliency_key(name, dataset)
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, key=key: positions.__setitem__(key, out.shape[2] * out.shape[3])))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)
    for h in handles:
        h.remove()
    return positions

def estimate_hardware_proxy(bit_config, net_structure, saliency):
    """
    Energy / area proxies of bit_config ({layer: [bit per strip]}); net_structure from conv_output_positions,
    saliency gives the strip lengths.
    """
    reads = sum(len(b) * net_structure[n] for n, b in bit_config.items())
    read_bits = sum(sum(b) * net_structure[n] for n, b in bit_config.items())
    weights = sum(len(b) * saliency[n]['strip_len'] for n, b in bit_config.items())
    weight_bits = sum(sum(b) * saliency[n]['strip_len'] for n, b in bit_config.items())
    return {'column_read_weighted_bits': read_bits / reads, 'avg_weight_bits': weight_bits / weights}
