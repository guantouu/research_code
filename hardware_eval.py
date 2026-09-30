"""
ReRAM hardware evaluation of strip-wise bit configs with NeuroSIM.

For each design in the config: build the quantized model (PTQ from the FP32 best.pth, or a saved
QAT model), measure its test accuracy, export the NeuroSIM trace of one test image with utee/hook.py,
run NeuroSIM, and collect area / energy / latency per image into one summary.csv.

A design is one of
    {"name": ..., "uniform": 8}                                   every strip at 8 bits
    {"name": ..., "allocator": "saliency", "ratio": 0.74}         allocate_bits on the saliency file
    {"name": ..., "bit_config_file": "bit_config/xxx.json"}       an existing bit config
    {"name": ..., "model_file": "saliency_0.74.pth"}              a saved model (e.g. QAT from fine_tuning.py)

Run from the repo root (the hook writes ./layer_record_* and calls ./NeuroSIM/main).
NeuroSIM hardware parameters live in NeuroSIM/Param.cpp; a copy is saved with the results.
"""
import os
import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from models import dataset
from utils.common_utils import process_config
from utils.bit_allocation import allocate_bits
from fine_tuning import validate
from ratio_sweep import build_quant_model
from utee import hook
from modules.conv import QuantBnConv2d

# NeuroSIM summary line -> (column, unit scale)
METRICS = {
    'area_mm2': (r'ChipArea : ([\d.e+-]+)um\^2', 1e-6),
    'cim_area_mm2': (r'Chip total CIM array : ([\d.e+-]+)um\^2', 1e-6),
    'adc_area_mm2': (r'Total ADC \(or S/As and precharger for SRAM\) Area on chip : ([\d.e+-]+)um\^2', 1e-6),
    'latency_us': (r'Chip pipeline-system-clock-cycle \(per image\) is: ([\d.e+-]+)ns', 1e-3),
    'dyn_energy_uJ': (r'Chip pipeline-system readDynamicEnergy \(per image\) is: ([\d.e+-]+)pJ', 1e-6),
    'leak_energy_uJ': (r'Chip pipeline-system leakage Energy \(per image\) is: ([\d.e+-]+)pJ', 1e-6),
    'adc_energy_uJ': (r'ADC \(or S/As and precharger for SRAM\) readDynamicEnergy is : ([\d.e+-]+)pJ', 1e-6),
    'accum_energy_uJ': (r'Accumulation Circuits .* readDynamicEnergy is : ([\d.e+-]+)pJ', 1e-6),
    'periph_energy_uJ': (r'Other Peripheries .* readDynamicEnergy is : ([\d.e+-]+)pJ', 1e-6),
    'fps': (r'Throughput FPS \(Pipelined Process\): ([\d.e+-]+)', 1),
}
# Param.cpp options recorded with every run
PARAMS = ['memcelltype', 'technode', 'cellBit', 'numColSubArray', 'levelOutput', 'numColMuxed',
          'operationmode', 'accesstype', 'novelMapping', 'resistanceOn', 'resistanceOff']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/exp_for_cifar/hardware_eval.json', required=False)
    args = parser.parse_args()
    configs = process_config(args.config)

    log_path = os.path.join(configs.logdir, configs.net, configs.dataset)
    out_dir = os.path.join(log_path, 'hardware', configs.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(out_dir, 'log.log'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(configs)

    neurosim_params = read_neurosim_params(configs.neurosim_dir)
    logging.info('=> NeuroSIM Param.cpp: {}'.format(neurosim_params))
    if int(neurosim_params['numColSubArray']) != configs.subArray:
        raise ValueError('subArray {} differs from numColSubArray {} in {}/Param.cpp: edit one and re-run make'.format(
            configs.subArray, neurosim_params['numColSubArray'], configs.neurosim_dir))
    if neurosim_params['novelMapping'] != 'false':
        # every layer is declared 1x1, so novel mapping puts all of them on its path, whose tile split
        # reads past the weight matrix when the column count is not divisible (garbage ADC energy)
        raise ValueError('Set novelMapping = false in {}/Param.cpp and re-run make'.format(configs.neurosim_dir))
    shutil.copy(os.path.join(configs.neurosim_dir, 'Param.cpp'), os.path.join(out_dir, 'Param.cpp'))
    shutil.copy(args.config, os.path.join(out_dir, 'config.json'))

    #--------------------------------------------------------------------------------------------------
    _, val_loader, num_classes = dataset.get_loaders(configs.dataset, configs.batch_size, train=False)
    #--------------------------------------------------------------------------------------------------

    with open(configs.saliency_file, 'r') as f:
        saliency = json.load(f)
    float_state = torch.load(os.path.join(log_path, 'best.pth'))
    torch.cuda.set_device(0)
    criterion = nn.CrossEntropyLoss().cuda(0)
    trace_images, _ = next(iter(val_loader))

    # quantize, evaluate and export every design (GPU, sequential)
    records = []
    for design in configs.designs:
        model, bit_config = build_design(design, configs, num_classes, float_state, saliency, log_path, out_dir)
        acc1 = float(validate(val_loader, model, criterion, configs))
        record_name = '{}_{}_{}_{}'.format(configs.net, configs.dataset, configs.exp_name, design['name'])
        export_trace(model, trace_images, record_name, configs)
        records.append({'design': design['name'], 'acc1': acc1, 'avg_weight_bits': avg_weight_bits(bit_config, saliency),
                        'high_strip_fraction': high_strip_fraction(bit_config), 'record_name': record_name})
        logging.info('=> [{}] acc {:.2f}, avg weight bits {:.3f}, trace in layer_record_{}'.format(
            design['name'], acc1, records[-1]['avg_weight_bits'], record_name))

    # NeuroSIM (CPU, in parallel)
    def simulate(record):
        t = time.time()
        output_file = os.path.join(out_dir, 'neurosim_{}.txt'.format(record['design']))
        with open('layer_record_{}/trace_command.sh'.format(record['record_name']), 'r') as f:
            args = f.read().split()
        args[0] = os.path.join(configs.neurosim_dir, 'main')   # the binary whose Param.cpp was checked
        with open(output_file, 'w') as f:
            subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)
        logging.info('=> [{}] NeuroSIM done ({:.0f}s)'.format(record['design'], time.time() - t))
        return parse_neurosim(output_file)
    with ThreadPoolExecutor(max_workers=configs.get('max_parallel', 4)) as pool:
        for record, metrics in zip(records, pool.map(simulate, records)):
            record.update(metrics)

    write_summary(records, configs, neurosim_params, out_dir)

def build_design(design, configs, num_classes, float_state, saliency, log_path, out_dir):
    """
    Quantized model and per-strip bit config of one design; the bit config is saved to out_dir.
    """
    if 'model_file' in design:
        model = torch.load(os.path.join(log_path, design['model_file']), weights_only=False).cuda(0)
        bit_config = {name: [int(b) for b in m.weight_bit] for name, m in model.named_modules()
                      if isinstance(m, QuantBnConv2d)}
    else:
        if 'uniform' in design:
            bit_config = {name: [design['uniform']] * len(s['saliency']) for name, s in saliency.items()}
        elif 'allocator' in design:
            bit_config = allocate_bits(design['allocator'], saliency, configs.bits, design['ratio'], seed=configs.seed)
        elif 'bit_config_file' in design:
            with open(design['bit_config_file'], 'r') as f:
                bit_config = json.load(f)
        else:
            raise ValueError("Design {} needs one of uniform / allocator / bit_config_file / model_file".format(design['name']))
        model = build_quant_model(configs, num_classes, float_state, bit_config)
    with open(os.path.join(out_dir, 'bit_config_{}.json'.format(design['name'])), 'w') as f:
        json.dump(bit_config, f, indent=4)
    return model, bit_config

def export_trace(model, images, record_name, configs):
    """
    Forward one batch with the hooks attached; the hook exports the first image of the batch.
    """
    model.eval()
    hook_handle_list = hook.hardware_evaluation(model, configs.wl_weight, configs.wl_activate,
                                                configs.subArray, configs.parallelRead, record_name)
    with torch.no_grad():
        model(images.cuda(0, non_blocking=True))
    hook.remove_hook_list(hook_handle_list)

def avg_weight_bits(bit_config, saliency):
    """
    Average bits per weight (strips weighted by their length).
    """
    bits = sum(sum(b) * saliency[name]['strip_len'] for name, b in bit_config.items())
    weights = sum(len(b) * saliency[name]['strip_len'] for name, b in bit_config.items())
    return bits / weights

def high_strip_fraction(bit_config):
    all_bits = [b for bits in bit_config.values() for b in bits]
    return sum(b > min(all_bits) for b in all_bits) / len(all_bits)

def read_neurosim_params(neurosim_dir):
    with open(os.path.join(neurosim_dir, 'Param.cpp'), 'r') as f:
        text = f.read()
    params = {}
    for name in PARAMS:
        match = re.search(r'^\s*{}\s*=\s*([^;]+);'.format(name), text, re.MULTILINE)
        params[name] = match.group(1).strip() if match else None
    return params

def parse_neurosim(output_file):
    with open(output_file, 'r') as f:
        text = f.read()
    metrics = {}
    for name, (pattern, scale) in METRICS.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError('"{}" not found in {}: check the NeuroSIM output'.format(name, output_file))
        metrics[name] = float(matches[-1]) * scale   # the last match is the chip-level value
    per_layer = [float(e) for e in re.findall(r"layer\d+'s readDynamicEnergy is: ([\d.e+-]+)pJ", text)]
    if not all(0 <= e < 1e12 for e in per_layer + [metrics['dyn_energy_uJ'] * 1e6]):
        raise RuntimeError('Implausible energy in {}: NeuroSIM probably read uninitialized memory'.format(output_file))
    metrics['energy_uJ'] = metrics['dyn_energy_uJ'] + metrics['leak_energy_uJ']
    return metrics

def write_summary(records, configs, neurosim_params, out_dir):
    """
    summary.csv with one row per design; *_vs_ref columns are relative to the reference design.
    NeuroSIM's TOPS/W is not reported: it counts one op per column group, so it is not comparable
    between designs with different bitwidths. Compare the energy per image instead.
    """
    ref = next(r for r in records if r['design'] == configs.get('reference', records[0]['design']))
    for r in records:
        for k in ['area_mm2', 'energy_uJ', 'latency_us']:
            r[k + '_vs_ref'] = r[k] / ref[k] - 1
        r['acc1_vs_ref'] = r['acc1'] - ref['acc1']
        r.update({'wl_weight': configs.wl_weight, 'wl_activate': configs.wl_activate, 'subArray': configs.subArray,
                  **{'param_' + k: v for k, v in neurosim_params.items()}})
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    logging.info('=> Reference: {}'.format(ref['design']))
    logging.info('{:>16} {:>6} {:>9} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8}'.format(
        'design', 'acc1', 'avg bits', 'area', 'energy', 'latency', 'd acc', 'd area', 'd energy'))
    logging.info('{:>16} {:>6} {:>9} {:>8} {:>10} {:>8}'.format('', '%', '', 'mm2', 'uJ/img', 'us/img'))
    for r in records:
        logging.info('{:>16} {:6.2f} {:9.3f} {:8.2f} {:10.2f} {:8.1f} {:+8.2f} {:+8.1%} {:+8.1%}'.format(
            r['design'], r['acc1'], r['avg_weight_bits'], r['area_mm2'], r['energy_uJ'], r['latency_us'],
            r['acc1_vs_ref'], r['area_mm2_vs_ref'], r['energy_uJ_vs_ref']))
    logging.info('=> Results saved to {}'.format(out_dir))

if __name__ == '__main__':
    main()
