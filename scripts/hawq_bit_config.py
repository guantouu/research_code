"""
Convert a bit configuration published by HAWQ (github.com/Zhen-Dong/HAWQ, bit_config.py: one bitwidth per
quantized conv, pytorchcv layer names) into this pipeline's per-strip bit config (torchvision layer names,
every strip of a layer at the layer's bitwidth), for hardware_eval.py's "bit_config_file" designs.

This is the original HAWQ allocation: Hessian traces, sensitivities and the ILP solution all come from the
HAWQ release, nothing is recomputed here. Only the weight bitwidths are used (the HAWQ activation bits and
QAT are not): the configs are evaluated with this pipeline's PTQ and NeuroSIM.

Name mapping (pytorchcv ResNet -> torchvision ResNet):
    quant_init_convbn / quant_init_block_convbn  -> conv1
    stage{s}.unit{u}.quant_convbn{k}              -> layer{s}.{u-1}.conv{k}
    stage{s}.unit{u}.quant_identity_convbn        -> layer{s}.{u-1}.downsample.0
Every mapped layer must exist in the saliency file and every layer of the saliency file must be mapped.

Run from the repo root:
    python scripts/hawq_bit_config.py --hawq_bit_config path/to/HAWQ/bit_config.py \
        --name bit_config_resnet18_modelsize_0.5 --saliency_file saliency/resnet18_imagenet_quant_perturbation_8_2.json \
        --out bit_config/hawq_resnet18_modelsize_0.5.json
"""
import argparse
import json
import os
import re

def torchvision_name(key):
    if key in ('quant_init_convbn', 'quant_init_block_convbn'):
        return 'conv1'
    m = re.fullmatch(r'stage(\d+)\.unit(\d+)\.quant_convbn(\d+)', key)
    if m:
        return 'layer{}.{}.conv{}'.format(m.group(1), int(m.group(2)) - 1, m.group(3))
    m = re.fullmatch(r'stage(\d+)\.unit(\d+)\.quant_identity_convbn', key)
    if m:
        return 'layer{}.{}.downsample.0'.format(m.group(1), int(m.group(2)) - 1)
    return None

def convert(hawq_config, saliency):
    """
    {torchvision layer: [bit per strip]} in the saliency file's layer order.
    """
    layer_bits = {}
    for key, bit in hawq_config.items():
        name = torchvision_name(key)
        if name is not None:
            layer_bits[name] = int(bit)
    unknown = [n for n in layer_bits if n not in saliency]
    missing = [n for n in saliency if n not in layer_bits]
    if unknown or missing:
        raise ValueError('HAWQ layers not in the saliency file: {}; saliency layers without a HAWQ bitwidth: {}'.format(
            unknown, missing))
    return {n: [layer_bits[n]] * len(s['saliency']) for n, s in saliency.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawq_bit_config', required=True, help="HAWQ's bit_config.py")
    parser.add_argument('--name', required=True, nargs='+', help='config names in bit_config_dict')
    parser.add_argument('--saliency_file', required=True, help='saliency file of the same network (layer names, strips)')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    namespace = {}
    with open(args.hawq_bit_config, 'r') as f:
        exec(f.read(), namespace)
    with open(args.saliency_file, 'r') as f:
        saliency = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.name:
        bit_config = convert(namespace['bit_config_dict'][name], saliency)
        weights = {n: len(s['saliency']) * s['strip_len'] for n, s in saliency.items()}
        avg = sum(weights[n] * b[0] for n, b in bit_config.items()) / sum(weights.values())
        low = min(min(b) for b in bit_config.values())
        low_fraction = sum(weights[n] for n, b in bit_config.items() if b[0] == low) / sum(weights.values())
        out = os.path.join(args.out_dir, 'hawq_{}.json'.format(name.replace('bit_config_', '')))
        with open(out, 'w') as f:
            json.dump(bit_config, f, indent=4)
        print('{}: {} layers, avg weight bits {:.3f}, {}-bit weights {:.3f} -> {}'.format(
            name, len(bit_config), avg, low, low_fraction, out))

if __name__ == '__main__':
    main()
