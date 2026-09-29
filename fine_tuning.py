import os
import argparse
import time
import logging
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models import dataset, registry
from utils.bit_config import bit_config_dict
from utils.common_utils import process_config
from utils import misc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/exp_for_cifar/fine_tuning.json', required=False)
    args = parser.parse_args()

    print('Using config!')
    configs = process_config(args.config)

    log_path = os.path.join(configs.logdir, configs.net, configs.dataset)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=log_path + '/log.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    run(configs)

def run(configs):
    """
    QAT fine-tuning of one bit config; returns the best test accuracy and the saved model path.
    Optional configs: bit_config_file (default bit_config/{net}_{dataset}_saliency_{ratio}.json)
    and save_name (default saliency_{ratio}.pth), used when called from ratio_sweep.py.
    """
    best_acc1 = 0
    net = configs.net
    inference_log_dir = os.path.join(configs.logdir, configs.net, configs.dataset, 'best.pth')
    log_path = os.path.join(configs.logdir, configs.net, configs.dataset)

    logging.info(configs)

    #--------------------------------------------------------------------------------------------------
    if configs.dataset == 'cifar10':
        train_loader, val_loader = dataset.get_cifar10(batch_size=configs.batch_size)
        num_classes=10
    elif configs.dataset == 'cifar100':
        train_loader, val_loader = dataset.get_cifar100(batch_size=configs.batch_size)
        num_classes=100
    else:
        raise ValueError("Unknown dataset type")
    #--------------------------------------------------------------------------------------------------
    pre_trained_model = registry.build_float_model(net, num_classes, torch.load(inference_log_dir))
    model = registry.build_quant_model(net, pre_trained_model)
    #--------------------------------------------------------------------------------------------------

    # print(pre_trained_model)
    # exit()
    #--------------------------------------------------------------------------------------------------

    if configs.get('bit_config_file'):
        bit_config_path = configs.bit_config_file
        with open(bit_config_path, 'r') as bit_config_file:
            bit_config = json.load(bit_config_file)
        print(bit_config_path)
    elif configs.strip_wise == True:
        bit_config_path = f'{configs.net}_{configs.dataset}_saliency_{configs.ratio}.json'
        bit_config_path = os.path.join('bit_config', bit_config_path)
        with open(bit_config_path, 'r') as bit_config_file:
            bit_config = json.load(bit_config_file)
        print(bit_config_path)
    else:
        bit_config = bit_config_dict["bit_config_" + net + "_" + configs.quant_scheme]

    #--------------------------------------------------------------------------------------------------
    name_counter = 0
    for name, m in model.named_modules():
        if name in bit_config.keys():
            name_counter += 1
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', configs.bias_bit)
            setattr(m, 'quantize_bias', (configs.bias_bit != 0))
            setattr(m, 'per_strip', configs.strip_wise)

            setattr(m, 'name', f"Conv_{name_counter}")

            bitwidth = bit_config[name]
            setattr(m, 'weight_bit', bitwidth)

    #--------------------------------------------------------------------------------------------------

    torch.cuda.set_device(0)
    model = model.cuda(0)

    criterion = nn.CrossEntropyLoss().cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), configs.lr,
                                momentum=configs.momentum,
                                weight_decay=configs.weight_decay)


    save_file = os.path.join(log_path, configs.get('save_name', f'saliency_{configs.ratio}.pth'))
    for epoch in range(configs.epochs):
        # adjust_learning_rate(optimizer, epoch, configs)

        train(train_loader, model, criterion, optimizer, epoch, configs)

        acc1 = validate(val_loader, model, criterion, configs)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
        if is_best:
            misc.model_save(model, save_file)

    return best_acc1, save_file

def train(train_loader, model, criterion, optimizer, epoch, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % configs.print_freq == 0:
            progress.display(i)


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

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()