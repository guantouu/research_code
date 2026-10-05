"""
Global mixed-precision ratio search on the diagonal empirical Fisher (paper Sec. 4.2).

The strip ranking of each allocator is fixed (it comes from the saliency file); only the global ratio r
(fraction of weights at the low bitwidth) is searched, over a 1-D grid. For every r:
    delta_fim = ||F(quantized model at r) - F(FP32 model)||^2   (Fisher diagonal of the conv weights on the
                Hutchinson calibration subset, utils.fisher_utils; fisher_type 'true' by default, since the
                empirical Fisher vanishes on the CIFAR training images the model fits), also relative to ||F(FP32)||^2
    column_read_weighted_bits / avg_weight_bits: analytic energy / area proxies (utils.hardware_proxy)
The Pareto front of (delta_fim, energy proxy) is kept. With a fixed ranking both usually move monotonically
with r, so most of the grid is on the front; r* is the largest ratio on the front whose relative delta_fim
stays within fim_threshold. Latency has no reliable proxy and is measured by NeuroSIM only.

Outputs in {logdir}/{net}/{dataset}/fim_search/{exp_name}/: results.csv (every grid point), selected.json
(r* per allocator) and hardware_eval.json, a hardware_eval.py config for the selected ratios (taken from
hardware_config, with the designs replaced), to be run separately for real NeuroSIM numbers.
Crossbar-capacity alignment is not part of the ratio search: set "snap_to_crossbar" in that hardware config.

With "crossbar_search": true, the allocation is also optimized directly in crossbars for a dual-crossbar chip
(high-bit and low-bit strips in separate arrays; geometry from hardware_config: rows = subArray, columns and
cellBit from its Param.cpp): for a crossbar budget, crossbar_optimal_allocation picks per layer the number of
high-bit strips (best saliency first) that maximizes the total saliency (exact knapsack DP), so a partly used
crossbar is filled with the next-best strips or given up when they are not worth it; a binary search over the
budget (an even scan, then bisection below the first passing budget) finds the fewest crossbars whose relative
delta FIM is <= fim_threshold (allocator 'crossbar_opt'). Every grid point also reports its dual_crossbars count,
and 'min_crossbars' in selected.json is the passing config with the fewest crossbars over everything evaluated.
"""
import argparse
import csv
import json
import logging
import os
import time

import torch
import torch.nn as nn
from models import dataset, registry
from modules.conv import QuantBnConv2d
from utils.common_utils import process_config
from utils.bit_allocation import allocate_bits, bit_config_cost, bit_config_crossbars, crossbar_options, crossbar_optimal_allocation
from utils.fisher_utils import compute_fisher_diag, fisher_distance, fisher_norm
from utils.hardware_proxy import conv_output_positions, estimate_hardware_proxy
from ratio_sweep import build_quant_model
from fine_tuning import validate
from hardware_eval import read_neurosim_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/cifar10/8_2/fim_search_resnet20.json', required=False)
    args = parser.parse_args()
    configs = process_config(args.config)

    log_path = os.path.join(configs.logdir, configs.net, configs.dataset)
    out_dir = os.path.join(log_path, 'fim_search', configs.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(out_dir, 'log.log'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(configs)

    with open(configs.saliency_file, 'r') as f:
        saliency = json.load(f)
    float_state = torch.load(os.path.join(log_path, 'best.pth'))
    num_classes = dataset.NUM_CLASSES[configs.dataset]
    torch.cuda.set_device(0)
    criterion = nn.CrossEntropyLoss().cuda(0)

    # the calibration subset of the Hutchinson pass (strip_wise_hessian_trace.py), never the test set
    calib_loader = dataset.get_hessian_loader(configs.dataset, configs.batch_size, configs.fisher_samples,
                                              seed=configs.seed)
    val_loader = dataset.get_loaders(configs.dataset, configs.batch_size, train=False)[1] \
        if configs.get('evaluate_accuracy', False) else None

    float_model = registry.build_float_model(configs.net, num_classes, float_state, configs.dataset).cuda(0)
    input_shape = next(iter(calib_loader))[0].shape[1:]
    net_structure = conv_output_positions(float_model, input_shape, configs.dataset)

    geometry = crossbar_geometry(configs.hardware_config) if configs.get('hardware_config') else None

    def bit_config_fn(allocator, r):
        return allocate_bits(allocator, saliency, configs.bits, r, seed=configs.seed)

    def build_fn(bit_config):
        return build_quant_model(configs, num_classes, float_state, bit_config)

    def hardware_proxy_fn(bit_config):
        proxy = estimate_hardware_proxy(bit_config, net_structure, saliency)
        if geometry is not None:
            proxy['dual_crossbars'] = bit_config_crossbars(bit_config, saliency, configs.bits, geometry)
        return proxy

    def fisher_fn(model, params):
        return compute_fisher_diag(model, params, calib_loader, criterion, fisher_type=configs.get('fisher_type', 'true'),
                                   mc_samples=configs.get('fisher_mc_samples', 0), seed=configs.seed)

    def accuracy_fn(model):
        return float(validate(val_loader, model, criterion, configs)) if val_loader is not None else None

    t = time.time()
    keys = list(saliency)
    float_convs = {registry.saliency_key(n, configs.dataset): m for n, m in float_model.named_modules()
                   if isinstance(m, nn.Conv2d)}
    baseline_fisher = fisher_fn(float_model, [float_convs[k].weight for k in keys])
    logging.info('=> FP32 {} Fisher diagonal on {} calibration samples ({:.0f}s)'.format(
        configs.get('fisher_type', 'true'), configs.fisher_samples, time.time() - t))

    records = []
    for allocator in configs.allocators:
        records += search_pareto(allocator, keys, baseline_fisher, fisher_fn, bit_config_fn, build_fn,
                                 configs.r_grid, hardware_proxy_fn, accuracy_fn, saliency, configs.bits, out_dir)

    if configs.get('crossbar_search', False):
        cb_records, cb_best = search_crossbar_budget(saliency, configs.bits, geometry, keys, baseline_fisher, fisher_fn,
                                                     build_fn, hardware_proxy_fn, accuracy_fn, configs.fim_threshold, out_dir,
                                                     configs.get('crossbar_scan_points', 16))
        records += cb_records

    selected = {}
    for allocator in configs.allocators:
        group = [r for r in records if r['allocator'] == allocator]
        for r, flag in zip(group, pareto_front(group)):
            r['pareto'] = flag
        best = select_ratio(group, configs.fim_threshold)
        best['selected'] = True
        selected[allocator] = dict(best)
        logging.info('=> [{}] r* = {:.4g} (relative delta FIM {:.4g}, threshold {}), avg weight bits {:.3f}'.format(
            allocator, selected[allocator]['ratio'], selected[allocator]['rel_delta_fim'], configs.fim_threshold,
            selected[allocator]['avg_weight_bits']))

    fields = ['allocator', 'ratio', 'budget', 'low_weight_fraction', 'avg_weight_bits', 'column_read_weighted_bits',
              'dual_crossbars', 'delta_fim', 'rel_delta_fim', 'acc1', 'pareto', 'selected', 'bit_config_file']
    if configs.get('crossbar_search', False):
        overall = min_crossbars(records, configs.fim_threshold)
        for name, best in [('crossbar_opt', cb_best), ('min_crossbars', overall)]:
            if best is None:
                continue
            best['selected'] = True
            selected[name] = dict(best)
            logging.info('=> [{}] {} crossbars from {} (relative delta FIM {:.4g}, threshold {}), avg weight bits {:.3f}'.format(
                name, best['dual_crossbars'], best['allocator'] if best['ratio'] is None else
                '{} r {:.4g}'.format(best['allocator'], best['ratio']), best['rel_delta_fim'], configs.fim_threshold,
                best['avg_weight_bits']))

    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    with open(os.path.join(out_dir, 'selected.json'), 'w') as f:
        json.dump(selected, f, indent=4)
    if configs.get('hardware_config'):
        write_hardware_config(configs, selected, out_dir)

    logging.info('{:>13} {:>6} {:>8} {:>9} {:>10} {:>10} {:>11} {:>7} {:>7}'.format(
        'allocator', 'ratio', 'lowW', 'avg bits', 'crw bits', 'crossbars', 'rel dFIM', 'acc1', 'pareto'))
    for r in records:
        logging.info('{:>13} {:>6} {:>8.3f} {:>9.3f} {:>10.3f} {:>10} {:>11.4g} {:>7} {:>7}'.format(
            r['allocator'], '' if r.get('ratio') is None else '{:.4g}'.format(r['ratio']), r['low_weight_fraction'],
            r['avg_weight_bits'], r['column_read_weighted_bits'], r.get('dual_crossbars', ''), r['rel_delta_fim'],
            '' if r['acc1'] is None else '{:.2f}'.format(r['acc1']),
            ('*' if r.get('pareto') else '') + ('<' if r.get('selected') else '')))
    logging.info('=> Results saved to {}'.format(out_dir))

def search_pareto(allocator, keys, baseline_fisher, fisher_fn, bit_config_fn, build_fn, r_grid,
                  hardware_proxy_fn, accuracy_fn, saliency, bits, out_dir):
    """
    1. baseline_fisher = fisher_fn(FP32 model)                      (computed once by the caller, r irrelevant)
    2. for r in r_grid:
         bit_config = bit_config_fn(allocator, r)                   allocate_bits, unchanged
         model_c    = build_fn(bit_config)                          ratio_sweep.build_quant_model, unchanged
         fisher_c   = fisher_fn(model_c)                            utils.fisher_utils.compute_fisher_diag
         delta_fim  = fisher_distance(baseline_fisher, fisher_c)
         energy / area proxies of bit_config                        (no NeuroSIM)
    The Pareto front and r* are taken by the caller (pareto_front, select_ratio).
    """
    base_norm = fisher_norm(baseline_fisher)
    records = []
    for r in r_grid:
        t = time.time()
        record = {'allocator': allocator, 'ratio': r,
                  **evaluate_bit_config('{}_{}'.format(allocator, r), bit_config_fn(allocator, r), keys, baseline_fisher,
                                        base_norm, fisher_fn, build_fn, hardware_proxy_fn, accuracy_fn, saliency, bits, out_dir)}
        logging.info('=> [{}] r {:.4g}: relative delta FIM {:.4g}, column-read-weighted bits {:.3f}{} ({:.0f}s)'.format(
            allocator, r, record['rel_delta_fim'], record['column_read_weighted_bits'],
            '' if record['acc1'] is None else ', acc {:.2f}'.format(record['acc1']), time.time() - t))
        records.append(record)
    return records

def evaluate_bit_config(name, bit_config, keys, baseline_fisher, base_norm, fisher_fn, build_fn, hardware_proxy_fn,
                        accuracy_fn, saliency, bits, out_dir):
    """
    Record of one bit config: relative delta FIM, hardware proxies, optional test accuracy; the bit config is saved.
    """
    model_c = build_fn(bit_config)
    quant_convs = {n: m for n, m in model_c.named_modules() if isinstance(m, QuantBnConv2d)}
    delta = fisher_distance(baseline_fisher, fisher_fn(model_c, [quant_convs[k].conv.weight for k in keys]))
    bit_config_file = os.path.join(out_dir, 'bit_config_{}.json'.format(name))
    with open(bit_config_file, 'w') as f:
        json.dump(bit_config, f, indent=4)
    record = {'delta_fim': delta, 'rel_delta_fim': delta / base_norm,
              'low_weight_fraction': bit_config_cost(bit_config, saliency, bits)['low_weight_fraction'],
              **hardware_proxy_fn(bit_config), 'acc1': accuracy_fn(model_c), 'bit_config_file': bit_config_file}
    del model_c
    torch.cuda.empty_cache()
    return record

def search_crossbar_budget(saliency, bits, geometry, keys, baseline_fisher, fisher_fn, build_fn, hardware_proxy_fn,
                           accuracy_fn, threshold, out_dir, scan_points=16):
    """
    Fewest dual crossbars whose optimal allocation (crossbar_optimal_allocation) keeps the relative delta FIM
    <= threshold. delta FIM is noisy near the threshold (not monotone in the budget), so a binary search
    alone can stop on the wrong side: scan_points budgets evenly spaced between the all-low-bit and
    all-high-bit crossbar counts are evaluated first, then the gap below the smallest passing one is
    bisected, and the smallest evaluated budget that passes is selected.
    Returns (records of every evaluated budget, record of the selected budget or None if none passes).
    """
    options = crossbar_options(saliency, bits, geometry)
    lo = sum(o[0][0] for o in options.values())
    hi = sum(o[-1][0] for o in options.values())
    base_norm = fisher_norm(baseline_fisher)
    records = {}

    def evaluate(budget):
        if budget not in records:
            t = time.time()
            bit_config, used = crossbar_optimal_allocation(saliency, bits, budget, geometry, options)
            records[budget] = {'allocator': 'crossbar_opt', 'ratio': None, 'budget': budget,
                               **evaluate_bit_config('crossbar_opt_{}'.format(budget), bit_config, keys, baseline_fisher,
                                                     base_norm, fisher_fn, build_fn, hardware_proxy_fn, accuracy_fn,
                                                     saliency, bits, out_dir)}
            r = records[budget]
            logging.info('=> [crossbar_opt] budget {} ({} used): relative delta FIM {:.4g}, avg weight bits {:.3f}{} ({:.0f}s)'.format(
                budget, used, r['rel_delta_fim'], r['avg_weight_bits'],
                '' if r['acc1'] is None else ', acc {:.2f}'.format(r['acc1']), time.time() - t))
        return records[budget]['rel_delta_fim'] <= threshold

    logging.info('=> [crossbar_opt] dual-crossbar geometry {}, budget {}..{} crossbars'.format(geometry, lo, hi))
    scan = sorted({lo + round(i * (hi - lo) / (scan_points - 1)) for i in range(scan_points)})
    passing = [b for b in scan if evaluate(b)]
    if passing:
        below = [b for b in scan if b < passing[0]]
        fail, ok = (below[-1], passing[0]) if below else (None, passing[0])
        while fail is not None and ok - fail > 1:
            mid = (fail + ok) // 2
            if evaluate(mid):
                ok = mid
            else:
                fail = mid
    ordered = [records[b] for b in sorted(records)]
    best = min_crossbars(ordered, threshold)
    if best is None:
        logging.info('=> [crossbar_opt] no budget reaches relative delta FIM <= {}'.format(threshold))
    return ordered, best

def min_crossbars(records, threshold):
    """
    Record with the fewest dual crossbars among those whose relative delta FIM is <= threshold
    (ties: lower delta FIM); None if there is none.
    """
    ok = [r for r in records if r['rel_delta_fim'] <= threshold]
    return min(ok, key=lambda r: (r['dual_crossbars'], r['rel_delta_fim'])) if ok else None

def crossbar_geometry(hardware_config):
    """
    Dual-crossbar geometry of a hardware_eval.py config: rows = subArray (NeuroSIM numRowSubArray from the
    command line), columns = numColSubArray and cell_bit = cellBit of its NeuroSIM Param.cpp.
    """
    with open(hardware_config, 'r') as f:
        hw = json.load(f)
    params = read_neurosim_params(hw['neurosim_dir'])
    return {'rows': hw['subArray'], 'cols': int(params['numColSubArray']), 'cell_bit': int(params['cellBit'])}

def pareto_front(points, objectives=('delta_fim', 'column_read_weighted_bits'), minimize=True):
    """
    O(K^2) dominance filter over the swept r grid: keep a point iff no other point is at
    least as good on every objective and strictly better on at least one.
    Returns one flag per point.
    """
    sign = 1 if minimize else -1
    vals = [[sign * p[o] for o in objectives] for p in points]
    return [not any(all(b <= a for a, b in zip(v, w)) and any(b < a for a, b in zip(v, w)) for w in vals)
            for v in vals]

def select_ratio(records, threshold):
    """
    r*: the largest ratio on the Pareto front whose relative delta FIM is <= threshold
    (the smallest ratio of the grid if none is).
    """
    ok = [r for r in records if r['pareto'] and r['rel_delta_fim'] <= threshold]
    return max(ok, key=lambda r: r['ratio']) if ok else min(records, key=lambda r: r['ratio'])

def write_hardware_config(configs, selected, out_dir):
    """
    hardware_eval.py config for the selected ratios: hardware_config with exp_name and designs replaced
    (all8 / all-low references plus each allocator at its r*).
    """
    with open(configs.hardware_config, 'r') as f:
        hw = json.load(f)
    high, low = configs.bits['highly_sensitive'], configs.bits['insensitive']
    designs = [{'name': 'all{}'.format(high), 'uniform': high}]
    seen = set()
    for a, r in selected.items():
        if r['bit_config_file'] in seen:   # min_crossbars is one of the other selections
            continue
        seen.add(r['bit_config_file'])
        if r.get('ratio') is None:   # crossbar_opt: no ratio, use the saved bit config
            designs.append({'name': '{}_{}'.format(a, r['budget']), 'bit_config_file': r['bit_config_file']})
        else:
            designs.append({'name': '{}_{}'.format(r['allocator'], r['ratio']), 'allocator': r['allocator'], 'ratio': r['ratio']})
    designs.append({'name': 'all{}'.format(low), 'uniform': low})
    hw.update(exp_name=configs.exp_name, designs=designs, reference='all{}'.format(high))
    path = os.path.join(out_dir, 'hardware_eval.json')
    with open(path, 'w') as f:
        json.dump(hw, f, indent=4)
    logging.info('=> NeuroSIM for the selected ratios: python hardware_eval.py --config {}'.format(path))

if __name__ == '__main__':
    main()
