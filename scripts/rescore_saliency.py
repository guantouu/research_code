"""
Rescore a saliency file for another bit pair without a new Hutchinson run. The Hessian traces do not depend
on the bitwidths, only the quantization errors do, so the per-strip 'trace' stored in a saliency file (written
by strip_wise_hessian_trace.py, or back-filled by scripts/add_layer_saliency.py) gives the 'quant_perturbation'
saliency and the HAWQ-v2 layer score for any bit pair:
    saliency_s = 1/2 * tr(H_ss)/n_s * (||Q_low(w_s) - w_s||^2 - ||Q_high(w_s) - w_s||^2)
The bit pair and the output tag come from the Hessian config given (e.g. one with bits 8/4 and tag 8_4).
Run from the repo root:
    python scripts/rescore_saliency.py --config configs/imagenet/8_4/hessian_trace_resnet18.json \
        --from_file saliency/resnet18_imagenet_quant_perturbation_8_2.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dataset, registry
from utils.common_utils import process_config
from utils.strip_utils import get_bn_fold_scale, strip_quant_error, aggregate_layer_trace, layer_quant_perturbation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Hessian config with the new bits (and tag)')
    parser.add_argument('--from_file', type=str, required=True, help='saliency file with per-strip traces')
    args = parser.parse_args()
    configs = process_config(args.config)
    with open(args.from_file, 'r') as f:
        source = json.load(f)
    missing = [n for n, s in source.items() if 'trace' not in s]
    if missing:
        raise ValueError('{} has no per-strip traces (e.g. {}): back-fill it with scripts/add_layer_saliency.py'.format(
            args.from_file, missing[0]))

    torch.cuda.set_device(0)
    state = torch.load(os.path.join(configs.logdir, configs.net, configs.dataset, 'best.pth'))
    model = registry.build_float_model(configs.net, dataset.NUM_CLASSES[configs.dataset], state, configs.dataset).cuda(0)
    model.eval()
    modules = {registry.saliency_key(name, configs.dataset): m for name, m in model.named_modules()
               if m.__class__.__name__ == 'Conv2d'}
    if set(modules) != set(source):
        raise ValueError('Layers of {} do not match the Conv2d layers of {}'.format(args.from_file, configs.net))
    mods = [modules[n] for n in source]
    traces = {modules[n]: s['trace'] for n, s in source.items()}
    layer_saliency = layer_quant_perturbation(model, configs.bits, mods, aggregate_layer_trace(traces, mods))

    rescored = {}
    for n, m in zip(source, mods):
        fold_scale = get_bn_fold_scale(model, m)
        gain = strip_quant_error(m, fold_scale, configs.bits['insensitive']) - \
               strip_quant_error(m, fold_scale, configs.bits['highly_sensitive'])
        trace = torch.tensor(source[n]['trace'], dtype=gain.dtype, device=gain.device)
        rescored[n] = {'strip_len': source[n]['strip_len'],
                       'saliency': (0.5 * trace / m.in_channels * gain).cpu().tolist(),
                       'trace': source[n]['trace'], 'layer_saliency': layer_saliency[m]}
    suffix = f"_{configs.tag}" if configs.get('tag') else ''
    os.makedirs('saliency', exist_ok=True)
    out = os.path.join('saliency', f'{configs.net}_{configs.dataset}_quant_perturbation{suffix}.json')
    with open(out, 'w') as f:
        json.dump(rescored, f)
    print('=> {}: saliency of {} strips rescored for bits {} -> {}'.format(
        args.from_file, sum(len(s['saliency']) for s in rescored.values()), dict(configs.bits), out))

if __name__ == '__main__':
    main()
