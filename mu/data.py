"""Deterministic data splits for machine unlearning.

Splits the CIFAR train set into:
  * val   : 10% per class held out (matches SalUn's validation convention)
  * pool  : the remaining 45000 training images
and then partitions the pool into:
  * forget : the samples to be unlearned
  * retain : everything else in the pool

Two forget protocols:
  * "random" : a seeded random `forget_ratio` fraction of the pool
  * "class"  : every pool sample whose label == `forget_class`

The model carries its own normalization layer (see mu/resnet.py), so the
loaders only apply ToTensor (+ augmentation for the training loaders), exactly
like the SalUn pipeline this reproduces.
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

_AUG = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
_PLAIN = transforms.Compose([transforms.ToTensor()])

_DSET = {"cifar10": (CIFAR10, 10), "cifar100": (CIFAR100, 100)}


@dataclass
class Splits:
    """Index sets (into the full 50k train set) and ready-made loaders."""
    val_idx: np.ndarray
    forget_idx: np.ndarray
    retain_idx: np.ndarray
    num_classes: int
    # training loaders (augmented)
    retain_loader: DataLoader
    forget_loader: DataLoader
    full_loader: DataLoader
    # evaluation loaders (no augmentation, not shuffled)
    retain_eval: DataLoader
    forget_eval: DataLoader
    full_eval: DataLoader
    val_eval: DataLoader
    test_eval: DataLoader


def _targets(ds):
    return np.array(ds.targets)


def get_splits(
    dataset="cifar10",
    data_dir="data",
    protocol="random",
    forget_ratio=0.1,
    forget_class=0,
    batch_size=256,
    seed=1,
    num_workers=4,
):
    cls, num_classes = _DSET[dataset]
    train_aug = cls(data_dir, train=True, transform=_AUG, download=False)
    train_plain = cls(data_dir, train=True, transform=_PLAIN, download=False)
    test_plain = cls(data_dir, train=False, transform=_PLAIN, download=False)

    y = _targets(train_aug)
    rng = np.random.RandomState(seed)

    # ---- val: 10% per class held out (SalUn convention) ----
    val_idx = []
    for c in range(num_classes):
        c_idx = np.where(y == c)[0]
        val_idx.append(rng.choice(c_idx, int(0.1 * len(c_idx)), replace=False))
    val_idx = np.hstack(val_idx)
    pool_idx = np.array(sorted(set(range(len(y))) - set(val_idx.tolist())))

    # ---- forget / retain partition of the pool ----
    if protocol == "random":
        n_forget = int(round(forget_ratio * len(pool_idx)))
        forget_idx = rng.choice(pool_idx, size=n_forget, replace=False)
    elif protocol == "class":
        forget_idx = pool_idx[y[pool_idx] == forget_class]
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    forget_idx = np.array(sorted(forget_idx.tolist()))
    retain_idx = np.array(sorted(set(pool_idx.tolist()) - set(forget_idx.tolist())))

    def _loader(base, idx, train):
        return DataLoader(
            Subset(base, idx.tolist()),
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True,
        )

    return Splits(
        val_idx=val_idx,
        forget_idx=forget_idx,
        retain_idx=retain_idx,
        num_classes=num_classes,
        retain_loader=_loader(train_aug, retain_idx, True),
        forget_loader=_loader(train_aug, forget_idx, True),
        full_loader=_loader(train_aug, np.concatenate([retain_idx, forget_idx]), True),
        retain_eval=_loader(train_plain, retain_idx, False),
        forget_eval=_loader(train_plain, forget_idx, False),
        full_eval=_loader(train_plain, np.concatenate([retain_idx, forget_idx]), False),
        val_eval=_loader(train_plain, val_idx, False),
        test_eval=DataLoader(test_plain, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True),
    )
