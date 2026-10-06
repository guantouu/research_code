"""
Fit the dual-crossbar energy model used by fim_pareto_search.py's 'adc' budget:
    energy (uJ, q + p passes, without the modeled merge) ~ offset + a * crossbars + b * ADC conversions
to the NeuroSIM results of dual_crossbar_eval.py runs (summary.csv + saved bit configs). ADC conversions are
column reads x row tiles (utils.bit_allocation.dual_adc_reads). Simpler models are fitted too for comparison.
Prints the fits and the energy_model entry to put in a fim_search config (a negative crossbar coefficient is
not physical: the crossbar term is then dropped and b refitted).
Run from the repo root:
    python scripts/fit_dual_energy.py --config configs/cifar10/8_2/dual_crossbar_resnet20.json \
        log/resnet20/cifar10/dual_crossbar/*
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dataset, registry
from utils.common_utils import process_config
from utils.bit_allocation import bit_config_crossbars, bit_config_column_reads, bit_config_adc_reads
from utils.hardware_proxy import conv_output_positions
from hardware_eval import read_neurosim_params

def fit(X, y):
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return coef, 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), (np.abs(pred - y) / y).max()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='the dual_crossbar_eval.py config (net, dataset, saliency, geometry)')
    parser.add_argument('result_dirs', nargs='+', help='dual_crossbar_eval.py output directories')
    args = parser.parse_args()
    configs = process_config(args.config)
    with open(configs.saliency_file, 'r') as f:
        saliency = json.load(f)
    params = read_neurosim_params(configs.neurosim_dir)
    geometry = {'rows': configs.subArray, 'cols': int(params['numColSubArray']), 'cell_bit': int(params['cellBit'])}
    shape = (3, 224, 224) if configs.dataset == 'imagenet' else (3, 32, 32)
    net_structure = conv_output_positions(
        registry.build_float_model(configs.net, dataset.NUM_CLASSES[configs.dataset], None, configs.dataset), shape, configs.dataset)

    X, y = [], []
    for d in args.result_dirs:
        for r in csv.DictReader(open(os.path.join(d, 'summary.csv'))):
            with open(os.path.join(d, 'bit_config_{}.json'.format(r['design'])), 'r') as f:
                bc = json.load(f)
            X.append((bit_config_crossbars(bc, saliency, configs.bits, geometry),
                      bit_config_column_reads(bc, net_structure, configs.bits, geometry),
                      bit_config_adc_reads(bc, net_structure, saliency, configs.bits, geometry)))
            y.append(float(r['energy_uJ_q']) + float(r['energy_uJ_p']))
    X, y = np.array(X, float), np.array(y)
    print('{} designs from {} runs'.format(len(y), len(args.result_dirs)))
    for label, cols in [('crossbars', [0]), ('column reads', [1]), ('ADC conversions', [2]),
                        ('crossbars + ADC conversions', [0, 2])]:
        coef, r2, err = fit(X[:, cols], y)
        print('{:30} R2 {:.5f}  max rel error {:5.2f}%  coefficients {}'.format(
            label, r2, 100 * err, ', '.join('{:.4g}'.format(c) for c in coef)))
    coef, _, _ = fit(X[:, [0, 2]], y)
    if coef[0] < 0:
        b, offset = fit(X[:, [2]], y)[0]
        model = {'per_crossbar_uJ': 0.0, 'per_adc_read_uJ': float(b), 'offset_uJ': float(offset)}
    else:
        model = {'per_crossbar_uJ': float(coef[0]), 'per_adc_read_uJ': float(coef[1]), 'offset_uJ': float(coef[2])}
    print('energy_model: {}'.format(json.dumps(model)))

if __name__ == '__main__':
    main()
