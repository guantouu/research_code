import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.common_utils import process_config
from datetime import datetime
from utils.strip_utils import compute_strip_importances, model_strip_group
import json


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

    if net == 'resnet50':
        from models.ResNet import resnet50
        model = resnet50()
        model.load_state_dict(torch.load(inference_log_dir))
    else:
        raise ValueError("Unknown model type")

    #--------------------------------------------------------------------------------------------------
    data_root = os.path.expanduser(os.path.join(configs.data, 'cifar100-data'))
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

    t_begin = time.time()

    torch.cuda.set_device(0)
    model = model.cuda(0)
    criterion = nn.CrossEntropyLoss().cuda(0)

    strip_group = hessian_trace(model, val_loader, criterion, configs)
    strip_bit_config = f'{configs.net}_{configs.dataset}_saliency_{configs.ratio}.json'
    strip_bit_config = os.path.join('bit_config', strip_bit_config)
    with open(strip_bit_config, 'w') as json_file:
        json.dump(strip_group, json_file, indent=4)

def hessian_trace(model, val_loader, criterion, configs):
    importances, strip_importances_per_layer = compute_strip_importances(model, val_loader, criterion)
    strip_group = model_strip_group(model, importances, strip_importances_per_layer, configs.bits, configs.ratio)
    return strip_group

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