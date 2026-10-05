import os
import argparse
import csv
import json
import logging
import time

import torch
import torch.nn as nn
from models import dataset, registry
from utils.common_utils import process_config
from utils.bit_allocation import allocate_bits, bit_config_cost, pareto_front
import fine_tuning
from fine_tuning import train, validate
from utils import misc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/exp_for_cifar/ratio_sweep.json', required=False)
    args = parser.parse_args()

    print('Using config!')
    configs = process_config(args.config)

    log_path = os.path.join(configs.logdir, configs.net, configs.dataset)
    sweep_dir = os.path.join(log_path, 'sweep', configs.exp_name)
    os.makedirs(sweep_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(sweep_dir, 'log.log'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(configs)

    #--------------------------------------------------------------------------------------------------
    train_loader, val_loader, num_classes = dataset.get_loaders(configs.dataset, configs.batch_size)
    if train_loader is None and (configs.finetune_epochs > 0 or configs.qat_after_search):
        raise ValueError("No training set for {}: set finetune_epochs 0 and qat_after_search false".format(configs.dataset))
    #--------------------------------------------------------------------------------------------------

    with open(configs.saliency_file, 'r') as f:
        saliency = json.load(f)
    float_state = torch.load(os.path.join(log_path, 'best.pth'))

    torch.cuda.set_device(0)
    criterion = nn.CrossEntropyLoss().cuda(0)

    def evaluate(model, loader):
        return float(validate(loader, model.cuda(0), criterion, configs))

    def run_point(allocator, ratio, loader):
        """
        Allocate bits for one ratio, (optionally) fine-tune, and evaluate on loader.
        """
        t = time.time()
        bit_config = allocate_bits(allocator, saliency, configs.bits, ratio, seed=configs.seed)
        model = build_quant_model(configs, num_classes, float_state, bit_config)
        if configs.finetune_epochs > 0:
            optimizer = torch.optim.SGD(model.parameters(), configs.lr,
                                        momentum=configs.momentum, weight_decay=configs.weight_decay)
            for epoch in range(configs.finetune_epochs):
                train(train_loader, model, criterion, optimizer, epoch, configs)
        acc1 = evaluate(model, loader)

        tag = f'{allocator}_{ratio}'
        bit_config_file = os.path.join(sweep_dir, f'bit_config_{tag}.json')
        with open(bit_config_file, 'w') as f:
            json.dump(bit_config, f, indent=4)
        if configs.save_models:
            misc.model_save(model, os.path.join(sweep_dir, f'{tag}.pth'))

        record = {'allocator': allocator, 'ratio': ratio, 'acc1': acc1, 'bit_config_file': bit_config_file,
                  **bit_config_cost(bit_config, saliency, configs.bits)}
        logging.info('=> [{}] ratio {:.4g}: acc {:.2f}, avg weight bits {:.3f}, high-bit strips {:.1%}, low-bit weights {:.1%} ({:.0f}s)'.format(
            allocator, ratio, acc1, record['avg_weight_bits'], record['high_strip_fraction'],
            record['low_weight_fraction'], time.time() - t))
        return record, model

    float_model = registry.build_float_model(configs.net, num_classes, float_state, configs.dataset)
    fp_acc = evaluate(float_model, val_loader)
    logging.info(f'=> FP32 accuracy: {fp_acc:.2f}')

    if configs.search == 'target':
        # the ratio is selected on a held-out training subset, not on the test set
        select_loader = dataset.get_hessian_loader(configs.dataset, configs.batch_size, configs.search_samples,
                                                   seed=configs.seed + 1)
        select_fp_acc = evaluate(float_model, select_loader)
        logging.info(f'=> FP32 accuracy on {configs.search_samples} held-out training images: {select_fp_acc:.2f}')

    records = []
    for allocator in configs.allocators:
        if configs.search == 'grid':
            for ratio in configs.ratios:
                records.append(run_point(allocator, ratio, val_loader)[0])
        elif configs.search == 'target':
            record, model = search_ratio(allocator, run_point, select_loader, select_fp_acc - configs.max_acc_drop, configs)
            record['select_acc1'], record['acc1'] = record['acc1'], evaluate(model, val_loader)
            logging.info('=> [{}] selected ratio {:.4g}: test acc {:.2f}'.format(allocator, record['ratio'], record['acc1']))
            if configs.qat_after_search and allocator in configs.qat_allocators:
                record['qat_acc1'], record['qat_model'] = run_qat(record, configs)
            records.append(record)
        else:
            raise ValueError("Unknown search: {}".format(configs.search))

    # Pareto front of each allocator's own accuracy / cost curve
    for allocator in configs.allocators:
        group = [r for r in records if r['allocator'] == allocator]
        for r, flag in zip(group, pareto_front(group)):
            r['pareto'] = flag
    fields = ['allocator', 'ratio', 'acc1', 'avg_weight_bits', 'high_strip_fraction', 'low_weight_fraction', 'pareto'] + \
             (['select_acc1', 'qat_acc1', 'qat_model'] if configs.search == 'target' else []) + ['bit_config_file']
    with open(os.path.join(sweep_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    if configs.search == 'target':
        # the selected ratio per allocator, for fine_tuning.py / inference.py
        with open(os.path.join(sweep_dir, 'selected.json'), 'w') as f:
            json.dump({r['allocator']: r for r in records}, f, indent=4)

    logging.info(f'=> FP32 accuracy: {fp_acc:.2f}')
    logging.info('{:>10} {:>7} {:>7} {:>9} {:>10} {:>7}'.format('allocator', 'ratio', 'acc1', 'avg bits', 'high-bit', 'pareto'))
    for r in records:
        logging.info('{:>10} {:>7.4g} {:>7.2f} {:>9.3f} {:>10.1%} {:>7}'.format(
            r['allocator'], r['ratio'], r['acc1'], r['avg_weight_bits'], r['high_strip_fraction'], '*' if r['pareto'] else ''))
    logging.info(f'=> Results saved to {sweep_dir}')

def search_ratio(allocator, run_point, select_loader, target, configs):
    """
    Binary search for the largest ratio (most low-bit strips) whose accuracy on select_loader stays >= target.
    Assumes accuracy decreases as the ratio grows. Returns the record and model of the selected ratio.
    """
    logging.info(f'=> [{allocator}] searching the largest ratio with accuracy >= {target:.2f}')
    lo, hi = 0.0, 1.0
    best = run_point(allocator, hi, select_loader)
    if best[0]['acc1'] >= target:
        return best
    best = None
    while hi - lo > configs.ratio_tol:
        mid = (lo + hi) / 2
        point = run_point(allocator, mid, select_loader)
        if point[0]['acc1'] >= target:
            lo, best = mid, point
        else:
            hi = mid
    if best is None:
        # even the most high-bit ratio tried failed: fall back to all strips at the high bitwidth
        best = run_point(allocator, lo, select_loader)
    return best

def run_qat(record, configs):
    """
    Hand the selected ratio over to fine_tuning.py: QAT with the sweep's bit config.
    The model is saved as {logdir}/{net}/{dataset}/{allocator}_{ratio}.pth (saliency_{ratio}.pth for the
    saliency allocator, the name inference.py loads by default).
    """
    ft_configs = process_config(configs.finetune_config)
    ft_configs.update(net=configs.net, dataset=configs.dataset, logdir=configs.logdir, bias_bit=configs.bias_bit,
                      strip_wise=True, ratio=record['ratio'], bit_config_file=record['bit_config_file'],
                      save_name=f"{record['allocator']}_{record['ratio']}.pth")
    logging.info('=> [{}] QAT of the selected ratio {} for {} epochs'.format(record['allocator'], record['ratio'], ft_configs.epochs))
    best_acc1, save_file = fine_tuning.run(ft_configs)
    best_acc1 = float(best_acc1)
    logging.info('=> [{}] QAT done: best test acc {:.2f}, model saved to {}'.format(record['allocator'], best_acc1, save_file))
    return best_acc1, save_file

def build_quant_model(configs, num_classes, float_state, bit_config):
    """
    Fresh quantized model from the FP32 weights with the given per-strip bit config (as in fine_tuning.py).
    """
    pre_trained_model = registry.build_float_model(configs.net, num_classes, float_state, configs.dataset)
    model = registry.build_quant_model(configs.net, pre_trained_model, configs.dataset)

    name_counter = 0
    for name, m in model.named_modules():
        if name in bit_config.keys():
            name_counter += 1
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', configs.bias_bit)
            setattr(m, 'quantize_bias', (configs.bias_bit != 0))
            setattr(m, 'per_strip', True)
            setattr(m, 'name', f"Conv_{name_counter}")
            setattr(m, 'weight_bit', bit_config[name])
    if name_counter != len(bit_config):
        raise ValueError("Only {} of {} layers in the bit config were found in the quantized model".format(
            name_counter, len(bit_config)))
    return model.cuda(0)

if __name__ == '__main__':
    main()
