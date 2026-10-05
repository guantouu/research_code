"""
Back-fill an existing 'quant_perturbation' saliency file with the per-strip traces and the HAWQ-v2 layer
score ('trace' / 'layer_saliency') that the hawqv2_layer allocator needs, without a new Hutchinson run:
the traces are recovered from the saliencies (utils.strip_utils.strip_traces_from_saliency).
Uses the FP32 model {logdir}/{net}/{dataset}/best.pth of the Hessian config that wrote the file
(ImageNet: the torchvision weights, as in strip_wise_hessian_trace.py).
Run from the repo root:
    python scripts/add_layer_saliency.py --config configs/exp_for_cifar/bits_8_2/hessian_trace_resnet20.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dataset, registry
from utils.common_utils import process_config
from utils.strip_utils import strip_traces_from_saliency, aggregate_layer_trace, layer_quant_perturbation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='the Hessian config that wrote the saliency file')
    parser.add_argument('--saliency_file', type=str, default=None, help='default: the file the config writes')
    args = parser.parse_args()
    configs = process_config(args.config)
    if configs.get('saliency', 'quant_perturbation') != 'quant_perturbation':
        raise ValueError('Traces can only be recovered from quant_perturbation saliencies')
    suffix = f"_{configs.tag}" if configs.get('tag') else ''
    saliency_file = args.saliency_file or os.path.join(
        'saliency', f'{configs.net}_{configs.dataset}_quant_perturbation{suffix}.json')
    with open(saliency_file, 'r') as f:
        saliency = json.load(f)

    torch.cuda.set_device(0)
    state = torch.load(os.path.join(configs.logdir, configs.net, configs.dataset, 'best.pth'))
    model = registry.build_float_model(configs.net, dataset.NUM_CLASSES[configs.dataset], state, configs.dataset).cuda(0)
    model.eval()
    modules = {registry.saliency_key(name, configs.dataset): m for name, m in model.named_modules()
               if m.__class__.__name__ == 'Conv2d'}
    if set(modules) != set(saliency):
        raise ValueError('Layers of {} do not match the Conv2d layers of {}'.format(saliency_file, configs.net))
    mods = [modules[n] for n in saliency]

    traces, num_unknown = strip_traces_from_saliency(model, configs.bits, mods, {modules[n]: s['saliency'] for n, s in saliency.items()})
    layer_saliency = layer_quant_perturbation(model, configs.bits, mods, aggregate_layer_trace(traces, mods))
    for n, m in zip(saliency, mods):
        saliency[n]['trace'] = traces[m]
        saliency[n]['layer_saliency'] = layer_saliency[m]
    with open(saliency_file, 'w') as f:
        json.dump(saliency, f)
    print('=> {}: traces of {} strips recovered ({} strips with zero quantization gain set to 0), '
          'layer_saliency of {} layers added'.format(saliency_file, sum(len(t) for t in traces.values()),
                                                     num_unknown, len(mods)))

if __name__ == '__main__':
    main()
