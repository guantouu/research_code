import os
import sys
import argparse
import time
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pytorchcv.model_provider import get_model as ptcv_get_model
from utils.bit_config import bit_config_dict
from utils.common_utils import process_config

best_acc1 = 0

def main():
    global best_acc1
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/exp_for_cifar/training.json', required=False)
    args = parser.parse_args()

    print('Using config!')
    configs = process_config(args.config)
    net = configs.net
    inference_log_dir = os.path.join(configs.logdir, configs.net, configs.dataset, 'best.pth')
 
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=configs.save_path + 'log.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(configs)

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    if net == 'resnet50':
        from models.ResNet import resnet50
        pre_trained_model = resnet50()
        pre_trained_model.load_state_dict(torch.load(inference_log_dir))

        from models.Q_ResNet import q_resnet50
        model = q_resnet50(pre_trained_model)
    else:
        raise ValueError("Unknown model type")
                
    #--------------------------------------------------------------------------------------------------

    bit_config = bit_config_dict["bit_config_" + net + "_" + configs.quant_scheme]
    name_counter = 0

    for name, m in model.named_modules():
        if name in bit_config.keys():
            name_counter += 1
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', configs.bias_bit)
            setattr(m, 'quantize_bias', (configs.bias_bit != 0))
            setattr(m, 'per_strip', configs.strip_wise)
            setattr(m, 'act_percentile', configs.act_percentile)
            setattr(m, 'act_range_momentum', configs.act_range_momentum)
            setattr(m, 'checkpoint_iter_threshold', configs.checkpoint_iter)
            setattr(m, 'save_path', configs.save_path)
            setattr(m, 'fixed_point_quantization', configs.fixed_point_quantization)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
                if bit_config[name][1] == 'hook':
                    m.register_forward_hook(hook_fn_forward)
                    global hook_keys
                    hook_keys.append(name)
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                if bitwidth == 4:
                    setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)

    #--------------------------------------------------------------------------------------------------
    torch.cuda.set_device(0)
    model = model.cuda(0)

    criterion = nn.CrossEntropyLoss().cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), configs.lr,
                                momentum=configs.momentum,
                                weight_decay=configs.weight_decay)

    #--------------------------------------------------------------------------------------------------
    data_root = os.path.expanduser(os.path.join(configs.data, 'cifar100-data'))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ])),
        batch_size=configs.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root=data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ])),
        batch_size=configs.batch_size, shuffle=True)
    #--------------------------------------------------------------------------------------------------

    best_epoch = 0
    for epoch in range(configs.epochs):
        # adjust_learning_rate(optimizer, epoch, configs)

        train(train_loader, model, criterion, optimizer, epoch, configs)

        acc1 = validate(val_loader, model, criterion, configs)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
        if is_best:
            best_epoch = epoch

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