"""
Replace FP32 training with pretrained CIFAR ResNet20 weights from chenyaofo/pytorch-cifar-models:
download (or read) the weights, rename them to models/ResNet20.py, check the test accuracy and
save them as {logdir}/resnet20/{dataset}/best.pth for the rest of the pipeline.
"""
import os
import re
import argparse

import torch
from models import dataset
from models.ResNet20 import resnet20

PRETRAINED = {
    'cifar10': ('https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt', 10, 92.60),
    'cifar100': ('https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt', 100, 68.83),
}

def convert_key(key):
    """
    chenyaofo CifarResNet parameter name -> models/ResNet20.py parameter name.
    """
    m = re.match(r'(conv1|bn1)\.(.+)', key)
    if m:
        return 'features.init_block.conv.{}.{}'.format('conv' if m.group(1) == 'conv1' else 'bn', m.group(2))
    m = re.match(r'layer(\d+)\.(\d+)\.(.+)', key)
    if m:
        stage, unit, rest = m.group(1), int(m.group(2)) + 1, m.group(3)
        rest = re.sub(r'^conv(\d)\.', r'body.conv\1.conv.', rest)
        rest = re.sub(r'^bn(\d)\.', r'body.conv\1.bn.', rest)
        rest = re.sub(r'^downsample\.0\.', 'identity_conv.conv.', rest)
        rest = re.sub(r'^downsample\.1\.', 'identity_conv.bn.', rest)
        return 'features.stage{}.unit{}.{}'.format(stage, unit, rest)
    m = re.match(r'fc\.(.+)', key)
    if m:
        return 'output.' + m.group(1)
    raise KeyError('Unexpected pretrained parameter: {}'.format(key))

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, target in loader:
            output = model(images.cuda(0))
            correct += (output.argmax(dim=1).cpu() == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=list(PRETRAINED))
    parser.add_argument('--weights', type=str, default=None, help='local weight file; downloaded if not given')
    parser.add_argument('--logdir', type=str, default='/app/log')
    parser.add_argument('--force', action='store_true', help='overwrite an existing best.pth')
    args = parser.parse_args()

    url, num_classes, published_acc = PRETRAINED[args.dataset]
    if args.weights:
        state_dict = torch.load(args.weights, map_location='cpu')
    else:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=True)

    model = resnet20(num_classes)
    model.load_state_dict({convert_key(k): v for k, v in state_dict.items()}, strict=True)
    model = model.cuda(0)

    _, val_loader = dataset.get_cifar10(batch_size=256) if args.dataset == 'cifar10' else \
        dataset.get_cifar100(batch_size=256)
    acc = evaluate(model, val_loader)
    print('=> Test accuracy: {:.2f} (published: {:.2f})'.format(acc, published_acc))
    if abs(acc - published_acc) > 0.5:
        raise RuntimeError('Accuracy differs from the published value: check the conversion and the normalization in models/dataset.py')

    save_dir = os.path.join(args.logdir, 'resnet20', args.dataset)
    save_path = os.path.join(save_dir, 'best.pth')
    if os.path.exists(save_path) and not args.force:
        raise FileExistsError('{} exists; use --force to overwrite'.format(save_path))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print('=> Saved to {}'.format(save_path))

if __name__ == '__main__':
    main()
