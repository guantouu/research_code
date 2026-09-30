"""
Replace FP32 training with pretrained weights: chenyaofo/pytorch-cifar-models for CIFAR, torchvision
IMAGENET1K_V1 for ImageNet. Download (or read) the weights, map them onto the model in
models/registry.py, check the test accuracy against the published value and save them as
{logdir}/{net}/{dataset}/best.pth for the rest of the pipeline.
"""
import os
import argparse

import torch
from models import dataset, registry

BASE_URL = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/'
# (net, dataset) -> (release file, published top-1 accuracy)
PRETRAINED = {
    ('resnet20', 'cifar10'): ('resnet/cifar10_resnet20-4118986f.pt', 92.60),
    ('resnet32', 'cifar10'): ('resnet/cifar10_resnet32-ef93fc4d.pt', 93.53),
    ('resnet44', 'cifar10'): ('resnet/cifar10_resnet44-2a3cabcb.pt', 94.01),
    ('resnet56', 'cifar10'): ('resnet/cifar10_resnet56-187c023a.pt', 94.37),
    ('vgg11', 'cifar10'): ('vgg/cifar10_vgg11_bn-eaeebf42.pt', 92.79),
    ('vgg13', 'cifar10'): ('vgg/cifar10_vgg13_bn-c01e4a43.pt', 94.00),
    ('vgg16', 'cifar10'): ('vgg/cifar10_vgg16_bn-6ee7ea24.pt', 94.16),
    ('vgg19', 'cifar10'): ('vgg/cifar10_vgg19_bn-57191229.pt', 93.91),
    ('resnet20', 'cifar100'): ('resnet/cifar100_resnet20-23dac2f1.pt', 68.83),
    ('resnet32', 'cifar100'): ('resnet/cifar100_resnet32-84213ce6.pt', 70.16),
    ('resnet44', 'cifar100'): ('resnet/cifar100_resnet44-ffe32858.pt', 71.63),
    ('resnet56', 'cifar100'): ('resnet/cifar100_resnet56-f2eff4c8.pt', 72.63),
    ('vgg11', 'cifar100'): ('vgg/cifar100_vgg11_bn-57d0759e.pt', 70.78),
    ('vgg13', 'cifar100'): ('vgg/cifar100_vgg13_bn-5ebe5778.pt', 74.63),
    ('vgg16', 'cifar100'): ('vgg/cifar100_vgg16_bn-7d8c4031.pt', 74.00),
    ('vgg19', 'cifar100'): ('vgg/cifar100_vgg19_bn-b98f7bd7.pt', 73.87),
}

def convert_state_dict(pretrained, model):
    """
    Map the chenyaofo parameters onto the model by position: both register the same layers in the
    same order (conv/bn/downsample/fc, features/classifier), so the i-th tensors must agree in
    shape and in the last name component (weight, bias, running_mean, ...).
    """
    target = model.state_dict()
    if len(pretrained) != len(target):
        raise ValueError('{} pretrained tensors vs {} in the model'.format(len(pretrained), len(target)))
    converted = {}
    for (src, value), (dst, ref) in zip(pretrained.items(), target.items()):
        if value.shape != ref.shape or src.split('.')[-1] != dst.split('.')[-1]:
            raise ValueError('Pretrained {} {} does not match {} {}'.format(src, tuple(value.shape), dst, tuple(ref.shape)))
        converted[dst] = value
    return converted

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
    parser.add_argument('--net', type=str, default='resnet20',
                        choices=sorted({net for net, _ in PRETRAINED} | set(registry.IMAGENET_MODELS)))
    parser.add_argument('--dataset', type=str, default='cifar10', choices=list(dataset.NUM_CLASSES))
    parser.add_argument('--weights', type=str, default=None, help='local CIFAR weight file; downloaded if not given')
    parser.add_argument('--logdir', type=str, default='/app/log')
    parser.add_argument('--force', action='store_true', help='overwrite an existing best.pth')
    args = parser.parse_args()

    model = registry.build_float_model(args.net, dataset.NUM_CLASSES[args.dataset], dataset=args.dataset)
    if args.dataset == 'imagenet':
        # torchvision weights, already named for the torchvision model
        state_dict, published_acc = registry.imagenet_pretrained(args.net)
        model.load_state_dict(state_dict, strict=True)
    else:
        if (args.net, args.dataset) not in PRETRAINED:
            raise ValueError('No pretrained {} for {}'.format(args.net, args.dataset))
        release_file, published_acc = PRETRAINED[(args.net, args.dataset)]
        if args.weights:
            state_dict = torch.load(args.weights, map_location='cpu')
        else:
            state_dict = torch.hub.load_state_dict_from_url(BASE_URL + release_file, map_location='cpu', progress=True)
        model.load_state_dict(convert_state_dict(state_dict, model), strict=True)
    model = model.cuda(0)

    # the whole test set (for ImageNet the whole validation set), as for the published accuracy
    _, val_loader, _ = dataset.get_loaders(args.dataset, 256, train=False, full_val=True)
    acc = evaluate(model, val_loader)
    print('=> Test accuracy: {:.2f} (published: {:.2f})'.format(acc, published_acc))
    if abs(acc - published_acc) > 0.5:
        raise RuntimeError('Accuracy differs from the published value: check the conversion and the normalization in models/dataset.py')

    save_dir = os.path.join(args.logdir, args.net, args.dataset)
    save_path = os.path.join(save_dir, 'best.pth')
    if os.path.exists(save_path) and not args.force:
        raise FileExistsError('{} exists; use --force to overwrite'.format(save_path))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print('=> Saved to {}'.format(save_path))

if __name__ == '__main__':
    main()
