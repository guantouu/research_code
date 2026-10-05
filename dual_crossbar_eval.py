"""
Dual-crossbar hardware evaluation (paper Sec. 4.3, Z = Zq + expand(Zp)) as a two-pass NeuroSIM approximation.

The chip of the paper has separate high-bit and low-bit crossbar arrays. NeuroSIM simulates one array type
per run, so every design is run twice with the same binary and Param.cpp, differing only in the synapse
precision passed on the command line:
    q pass: only the high-bit strips, synapseBit = high bitwidth (one column group per strip)
    p pass: only the low-bit strips,  synapseBit = low bitwidth
A layer with no strip of a pass is left out of that pass (utee/hook.py strip_bits); a pass with no layer is
not run and counts as zero. The two runs are then combined:
    area_total    = area_q + area_p
    energy_total  = energy_q + energy_p + energy_align
    latency_total = max(latency_q, latency_p) + latency_align
MODELING ASSUMPTIONS (report them as such, they are not NeuroSIM outputs):
  - The two runs are two separate chips: chip-level overhead (global buffer, interconnect) is counted in both,
    so area_total over-estimates a real dual-array chip, and the arrays are assumed to compute in parallel.
  - energy_align / latency_align model the merge Zq + expand(Zp): one shift-and-add per output position of
    every output channel that has strips in both arrays (expand is a shift, i.e. wiring). Its cost per
    operation is taken from NeuroSIM's own accumulation circuits (ShiftAdd / adders) of the q pass:
        energy_align = sum_l AccumEnergy_q,l * O_both,l / G_q,l
        latency_align = max_l AccumLatency_q,l * O_both,l / G_q,l
    with G_q,l the column groups of layer l in the q pass (one per high-bit strip) and O_both,l the output
    channels with strips in both arrays: the merge is one more accumulation per merged channel and position,
    against G_q,l per position already done in the q pass. Layers are pipelined, so the slowest merge counts.
    The merge adders' area is not added.

Run from the repo root with a hardware_eval.py config (every design is evaluated as a dual-crossbar design;
"mode": "dual_crossbar" may be set on the designs and is required for them in hardware_eval.py's list to be
skipped there). Results: {logdir}/{net}/{dataset}/dual_crossbar/{exp_name}/summary.csv.
"""
import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from models import dataset, registry
from utils.common_utils import process_config
from utils.bit_allocation import bit_config_cost, bit_config_crossbars, bit_config_column_reads
from utils.hardware_proxy import conv_output_positions, estimate_hardware_proxy
from fine_tuning import validate
from hardware_eval import read_neurosim_params, build_design, parse_neurosim
from modules.conv import QuantBnConv2d
from utee import hook

LAYER_BLOCK = re.compile(r'Estimation of Layer (\d+) -+\n(.*?)(?=Estimation of Layer|\Z)', re.S)
ACCUM_ENERGY = re.compile(r'Accumulation Circuits .*? readDynamicEnergy is : ([\d.e+-]+)pJ')
ACCUM_LATENCY = re.compile(r'Accumulation Circuits .*? readLatency is : ([\d.e+-]+)ns')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/cifar10/8_2/dual_crossbar_resnet20.json', required=False)
    args = parser.parse_args()
    configs = process_config(args.config)

    log_path = os.path.join(configs.logdir, configs.net, configs.dataset)
    out_dir = os.path.join(log_path, 'dual_crossbar', configs.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(out_dir, 'log.log'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(configs)

    neurosim_params = read_neurosim_params(configs.neurosim_dir)
    logging.info('=> NeuroSIM Param.cpp: {}'.format(neurosim_params))
    if int(neurosim_params['numColSubArray']) != configs.subArray:
        raise ValueError('subArray {} differs from numColSubArray {} in {}/Param.cpp'.format(
            configs.subArray, neurosim_params['numColSubArray'], configs.neurosim_dir))
    if neurosim_params['novelMapping'] != 'false':
        raise ValueError('Set novelMapping = false in {}/Param.cpp and re-run make'.format(configs.neurosim_dir))
    shutil.copy(os.path.join(configs.neurosim_dir, 'Param.cpp'), os.path.join(out_dir, 'Param.cpp'))
    shutil.copy(args.config, os.path.join(out_dir, 'config.json'))
    geometry = {'rows': configs.subArray, 'cols': int(neurosim_params['numColSubArray']),
                'cell_bit': int(neurosim_params['cellBit'])}
    high, low = configs.bits['highly_sensitive'], configs.bits['insensitive']

    _, val_loader, num_classes = dataset.get_loaders(configs.dataset, configs.batch_size, train=False)
    with open(configs.saliency_file, 'r') as f:
        saliency = json.load(f)
    float_state = torch.load(os.path.join(log_path, 'best.pth'))
    torch.cuda.set_device(0)
    criterion = nn.CrossEntropyLoss().cuda(0)
    trace_images, _ = next(iter(val_loader))
    net_structure = conv_output_positions(registry.build_float_model(configs.net, num_classes, float_state, configs.dataset),
                                          trace_images.shape[1:], configs.dataset)

    # quantize, evaluate and export both passes of every design (GPU, sequential)
    records = []
    for design in configs.designs:
        model, bit_config = build_design(design, configs, num_classes, float_state, saliency, log_path, out_dir,
                                         geometry['cell_bit'])
        acc1 = float(validate(val_loader, model, criterion, configs))
        base = '{}_{}_{}_{}'.format(configs.net, configs.dataset, configs.exp_name, design['name'])
        passes = {}
        for tag, bits in [('q', high), ('p', low)]:
            name = '{}_{}'.format(base, tag)
            export_trace(model, trace_images, name, configs, bits)
            with open('layer_record_{}/layers.txt'.format(name), 'r') as f:
                passes[tag] = {'record_name': name, 'layers': f.read().split()}
        # module names of the hook (Conv_k) -> bit config layer names
        conv_names = {m.name: n for n, m in model.named_modules() if isinstance(m, QuantBnConv2d)}
        out_channels = {n: m.out_channels for n, m in model.named_modules() if isinstance(m, QuantBnConv2d)}
        records.append({'design': design['name'], 'acc1': acc1, 'bit_config': bit_config, 'passes': passes,
                        'conv_names': conv_names, 'out_channels': out_channels})
        logging.info('=> [{}] acc {:.2f}, layers in q / p pass: {} / {}'.format(
            design['name'], acc1, len(passes['q']['layers']), len(passes['p']['layers'])))

    # NeuroSIM: two runs per design (CPU, in parallel)
    def simulate(job):
        record, tag = job
        p = record['passes'][tag]
        if not p['layers']:
            return None
        t = time.time()
        output_file = os.path.join(out_dir, 'neurosim_{}_{}.txt'.format(record['design'], tag))
        with open('layer_record_{}/trace_command.sh'.format(p['record_name']), 'r') as f:
            args = f.read().split()
        args[0] = os.path.join(configs.neurosim_dir, 'main')
        with open(output_file, 'w') as f:
            subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)
        logging.info('=> [{} {}] NeuroSIM done ({:.0f}s)'.format(record['design'], tag, time.time() - t))
        return output_file
    jobs = [(r, tag) for r in records for tag in ('q', 'p')]
    with ThreadPoolExecutor(max_workers=configs.get('max_parallel', 4)) as pool:
        outputs = list(pool.map(simulate, jobs))
    for (record, tag), output_file in zip(jobs, outputs):
        record['passes'][tag]['output'] = output_file

    rows = [combine(r, configs, saliency, net_structure, geometry) for r in records]
    write_summary(rows, configs, neurosim_params, out_dir)

def export_trace(model, images, record_name, configs, bits):
    """
    One pass of a dual-crossbar design: only the strips of bitwidth bits, with synapse precision bits.
    """
    model.eval()
    hook_handle_list = hook.hardware_evaluation(model, bits, configs.wl_activate, configs.subArray,
                                                configs.parallelRead, record_name, strip_bits_={bits})
    with torch.no_grad():
        model(images.cuda(0, non_blocking=True))
    hook.remove_hook_list(hook_handle_list)

def per_layer_accumulation(output_file):
    """
    {layer index (1-based): (accumulation energy pJ, accumulation latency ns)} of a NeuroSIM output.
    """
    with open(output_file, 'r') as f:
        text = f.read()
    layers = {}
    for index, block in LAYER_BLOCK.findall(text):
        energy, latency = ACCUM_ENERGY.search(block), ACCUM_LATENCY.search(block)
        if energy and latency:
            layers[int(index)] = (float(energy.group(1)), float(latency.group(1)))
    return layers

def estimate_expand_alignment_cost(record, configs):
    """
    Modeled cost of Z = Zq + expand(Zp) (MODELING ASSUMPTION, see the module docstring): per layer with strips
    in both arrays, one shift-and-add per output position of each output channel that has strips in both,
    priced with the q pass's own NeuroSIM accumulation energy / latency per column group.
    Returns (energy uJ, latency us).
    """
    q = record['passes']['q']
    if q.get('output') is None or record['passes']['p'].get('output') is None:
        return 0.0, 0.0
    accum = per_layer_accumulation(q['output'])
    high = configs.bits['highly_sensitive']
    energy, latency = 0.0, 0.0
    for index, conv in enumerate(q['layers'], start=1):
        name = record['conv_names'][conv]
        bits = record['bit_config'][name]
        channels = record['out_channels'][name]
        per_channel = len(bits) // channels
        groups = sum(b == high for b in bits)
        both = sum(1 for o in range(channels) if len(set(bits[o * per_channel:(o + 1) * per_channel])) > 1)
        if both == 0 or index not in accum:
            continue
        e, l = accum[index]
        energy += e * both / groups
        latency = max(latency, l * both / groups)
    return energy * 1e-6, latency * 1e-3

def combine(record, configs, saliency, net_structure, geometry):
    """
    One summary row: both passes' NeuroSIM metrics, the modeled alignment cost and the combined totals.
    """
    metrics = {}
    for tag in ('q', 'p'):
        out = record['passes'][tag].get('output')
        m = parse_neurosim(out) if out else {'area_mm2': 0.0, 'energy_uJ': 0.0, 'latency_us': 0.0, 'adc_energy_uJ': 0.0,
                                               'accum_energy_uJ': 0.0}
        for k in ('area_mm2', 'energy_uJ', 'latency_us', 'adc_energy_uJ', 'accum_energy_uJ'):
            metrics['{}_{}'.format(k, tag)] = m[k]
    energy_align, latency_align = estimate_expand_alignment_cost(record, configs)
    bit_config = record['bit_config']
    row = {'design': record['design'], 'acc1': record['acc1'],
           **{k: v for k, v in bit_config_cost(bit_config, saliency, configs.bits).items()},
           'column_read_weighted_bits': estimate_hardware_proxy(bit_config, net_structure, saliency)['column_read_weighted_bits'],
           'dual_crossbars': bit_config_crossbars(bit_config, saliency, configs.bits, geometry),
           'column_reads': bit_config_column_reads(bit_config, net_structure, configs.bits, geometry),
           **metrics,
           'energy_align_uJ': energy_align, 'latency_align_us': latency_align,
           'area_mm2': metrics['area_mm2_q'] + metrics['area_mm2_p'],
           'energy_uJ': metrics['energy_uJ_q'] + metrics['energy_uJ_p'] + energy_align,
           'latency_us': max(metrics['latency_us_q'], metrics['latency_us_p']) + latency_align}
    return row

def write_summary(rows, configs, neurosim_params, out_dir):
    """
    summary.csv with one row per design: *_q / *_p are NeuroSIM outputs of each pass, *_align are modeled,
    area_mm2 / energy_uJ / latency_us are the combined totals; *_vs_ref relative to the reference design.
    """
    ref = next(r for r in rows if r['design'] == configs.get('reference', rows[0]['design']))
    for r in rows:
        for k in ['area_mm2', 'energy_uJ', 'latency_us']:
            r[k + '_vs_ref'] = r[k] / ref[k] - 1
        r['acc1_vs_ref'] = r['acc1'] - ref['acc1']
        r.update({'wl_activate': configs.wl_activate, 'subArray': configs.subArray,
                  **{'param_' + k: v for k, v in neurosim_params.items()}})
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logging.info('=> Reference: {} (q / p: NeuroSIM runs of each array; align: modeled merge cost)'.format(ref['design']))
    logging.info('{:>18} {:>6} {:>6} {:>17} {:>24} {:>24} {:>8} {:>8}'.format(
        'design', 'acc1', 'bits', 'area q+p', 'energy q+p+align', 'latency max(q,p)+align', 'd energy', 'd area'))
    for r in rows:
        logging.info('{:>18} {:6.2f} {:6.2f} {:7.2f}+{:<6.2f}={:<6.2f} {:6.2f}+{:<5.2f}+{:<5.3f}={:<6.2f} {:6.1f},{:<6.1f}+{:<5.2f}={:<6.1f} {:+8.1%} {:+8.1%}'.format(
            r['design'], r['acc1'], r['avg_weight_bits'], r['area_mm2_q'], r['area_mm2_p'], r['area_mm2'],
            r['energy_uJ_q'], r['energy_uJ_p'], r['energy_align_uJ'], r['energy_uJ'],
            r['latency_us_q'], r['latency_us_p'], r['latency_align_us'], r['latency_us'],
            r['energy_uJ_vs_ref'], r['area_mm2_vs_ref']))
    logging.info('=> Results saved to {}'.format(out_dir))

if __name__ == '__main__':
    main()
