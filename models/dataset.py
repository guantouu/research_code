import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os

# Normalization of the pretrained weights (chenyaofo/pytorch-cifar-models); models trained here use the same values
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
CIFAR100_MEAN, CIFAR100_STD = (0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)

def get_cifar10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_cifar100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

# ImageNet (PTQ only): the validation set is split once into a calibration part, used for the Hessian
# and ratio selection, and a test part for the reported accuracy, so no bit allocation is chosen on
# the images it is evaluated on. The root holds ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz;
# torchvision extracts them into val/<wnid>/ on first use.
IMAGENET_ROOT = os.environ.get('IMAGENET_ROOT', '/workspace/imagenet')
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMAGENET_CALIB_SIZE = 10000
IMAGENET_SPLIT_SEED = 0
IMAGENET_WORKERS = min(16, os.cpu_count() or 1)

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}

def imagenet_val(data_root=IMAGENET_ROOT):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return datasets.ImageNet(data_root, split='val', transform=transform)

def imagenet_split(num_images):
    """
    Indices (calibration, test) of the fixed validation split.
    """
    perm = torch.randperm(num_images, generator=torch.Generator().manual_seed(IMAGENET_SPLIT_SEED)).tolist()
    return perm[:IMAGENET_CALIB_SIZE], sorted(perm[IMAGENET_CALIB_SIZE:])

def get_loaders(dataset, batch_size, train=True, full_val=False):
    """
    (train loader or None, test loader, number of classes) of a dataset.
    ImageNet has no train loader (PTQ only); its test loader is the test part of the validation
    split, or the whole validation set with full_val=True (to compare with published accuracies).
    """
    if dataset == 'cifar10':
        train_loader, val_loader = get_cifar10(batch_size=batch_size)
    elif dataset == 'cifar100':
        train_loader, val_loader = get_cifar100(batch_size=batch_size)
    elif dataset == 'imagenet':
        ds = imagenet_val()
        if not full_val:
            ds = torch.utils.data.Subset(ds, imagenet_split(len(ds))[1])
        train_loader = None
        val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=IMAGENET_WORKERS, pin_memory=True)
    else:
        raise ValueError("Unknown dataset type: {}".format(dataset))
    return (train_loader if train else None), val_loader, NUM_CLASSES[dataset]

def get_hessian_loader(dataset, batch_size, num_samples, data_root='/tmp/public_dataset/pytorch', seed=0, **kwargs):
    """
    Fixed random subset of the training set (no augmentation) for Hessian estimation,
    so the bit allocation is not selected on the test set. For ImageNet the subset is drawn from
    the calibration part of the validation split instead.
    """
    num_workers = kwargs.setdefault('num_workers', 1)
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        ds = datasets.CIFAR10(root=os.path.join(data_root, 'cifar10-data'), train=True, download=True, transform=transform)
        pool = list(range(len(ds)))
    elif dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
        ds = datasets.CIFAR100(root=os.path.join(data_root, 'cifar100-data'), train=True, download=True, transform=transform)
        pool = list(range(len(ds)))
    elif dataset == 'imagenet':
        ds = imagenet_val()
        pool = imagenet_split(len(ds))[0]
        num_workers = IMAGENET_WORKERS
    else:
        raise ValueError("Unknown dataset type")
    if num_samples > len(pool):
        raise ValueError("{} samples requested, {} available".format(num_samples, len(pool)))
    g = torch.Generator().manual_seed(seed)
    indices = [pool[i] for i in torch.randperm(len(pool), generator=g)[:num_samples].tolist()]
    return DataLoader(torch.utils.data.Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
