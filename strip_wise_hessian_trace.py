import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.common_utils import process_config
from datetime import datetime
from utils.strip_utils import compute_strip_importances
from utils.bit_allocation import allocate_bits
import json
from models import dataset, registry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/exp_for_cifar/hessian_trace.json', required=False)
    args = parser.parse_args()

    print('Using config!')
    configs = process_config(args.config)

    log_path = os.path.join(configs.logdir, configs.net, configs.dataset, 'log.log')
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=log_path)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(configs)

    inference_log_dir = os.path.join(configs.logdir, configs.net, configs.dataset, 'best.pth')
    net = configs.net

    #--------------------------------------------------------------------------------------------------
    _, val_loader, num_classes = dataset.get_loaders(configs.dataset, configs.batch_size, train=False)
    #--------------------------------------------------------------------------------------------------
    model = registry.build_float_model(net, num_classes, torch.load(inference_log_dir), configs.dataset)

    t_begin = time.time()

    torch.cuda.set_device(0)
    model = model.cuda(0)
    criterion = nn.CrossEntropyLoss().cuda(0)

    # estimate the Hessian on a training subset, not on the test set used for validation
    hessian_loader = dataset.get_hessian_loader(configs.dataset, configs.batch_size, configs.hessian_samples,
                                                seed=configs.seed)
    saliency = hessian_trace(model, hessian_loader, criterion, configs)
    logging.info('Hessian trace time: {:.1f}s'.format(time.time() - t_begin))

    # saliency is computed once; ratio_sweep.py reuses it for any ratio / allocator.
    # It depends on the bit pair through the quantization error, so other pairs than 8/4 set a tag (e.g. "8_2")
    suffix = f"_{configs.tag}" if configs.get('tag') else ''
    os.makedirs('saliency', exist_ok=True)
    saliency_file = os.path.join('saliency', f'{configs.net}_{configs.dataset}_{configs.saliency}{suffix}.json')
    with open(saliency_file, 'w') as json_file:
        json.dump(saliency, json_file)
    logging.info(f'Saliency saved to {saliency_file}')

    strip_group = allocate_bits('saliency', saliency, configs.bits, configs.ratio, log=True)
    strip_bit_config = f'{configs.net}_{configs.dataset}_saliency_{configs.ratio}{suffix}.json'
    os.makedirs('bit_config', exist_ok=True)
    strip_bit_config = os.path.join('bit_config', strip_bit_config)
    with open(strip_bit_config, 'w') as json_file:
        json.dump(strip_group, json_file, indent=4)

    validate(val_loader, model, criterion, configs)

def hessian_trace(model, dataloader, criterion, configs):
    importances, strip_importances_per_layer = compute_strip_importances(
        model, dataloader, criterion, configs.bits, saliency=configs.saliency,
        max_iters=configs.hutchinson_max_iters, min_iters=configs.hutchinson_min_iters,
        tol=configs.hutchinson_tol, seed=configs.seed)
    saliency = {}
    for name, m in model.named_modules():
        if m in strip_importances_per_layer:
            saliency[registry.saliency_key(name, configs.dataset)] = {'strip_len': m.in_channels,
                                                       'saliency': strip_importances_per_layer[m]}
    return saliency

def validate(val_loader, model, criterion, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.print_freq == 0:
                progress.display(i)

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()