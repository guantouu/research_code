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
            batch_size=batch_size, shuffle=True, **kwargs)
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
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_imagenet(batch_size, data_root='/data/imagenet', train=True, val=True, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    train_path = os.path.join(data_root, 'train')
    print("Building ImageNet data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                train_path,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ])),
            batch_size=batch_size, shuffle=True, num_workers=16, prefetch_factor=4, pin_memory=True)
        ds.append(train_loader)
    if val:
        val_path = os.path.join(data_root, 'val')
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                val_path,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                ])),
            batch_size=batch_size, shuffle=True, num_workers=16)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
def get_hessian_loader(dataset, batch_size, num_samples, data_root='/tmp/public_dataset/pytorch', seed=0, **kwargs):
    """
    Fixed random subset of the training set (no augmentation) for Hessian estimation,
    so the bit allocation is not selected on the test set.
    """
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        ds = datasets.CIFAR10(root=os.path.join(data_root, 'cifar10-data'), train=True, download=True, transform=transform)
    elif dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
        ds = datasets.CIFAR100(root=os.path.join(data_root, 'cifar100-data'), train=True, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset type")
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:num_samples].tolist()
    num_workers = kwargs.setdefault('num_workers', 1)
    return DataLoader(torch.utils.data.Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
