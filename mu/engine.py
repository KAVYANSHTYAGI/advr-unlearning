"""Training / evaluation helpers shared across the harness."""
import random

import numpy as np
import torch
import torch.nn as nn

from mu.resnet import resnet18, resnet34, resnet50

_ARCH = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(arch="resnet18", num_classes=10, device="cuda"):
    model = _ARCH[arch](num_classes=num_classes)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    """Top-1 accuracy (%) over a loader."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def train_epochs(model, loader, epochs, lr, device="cuda",
                 momentum=0.9, weight_decay=5e-4, milestones=None, log_prefix=""):
    """Standard SGD + MultiStepLR training loop. Returns the trained model."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    sched = None
    if milestones:
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running = correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        if sched:
            sched.step()
        print(f"{log_prefix}epoch {epoch+1}/{epochs} "
              f"loss {running/total:.3f} acc {100*correct/total:.2f} "
              f"lr {opt.param_groups[0]['lr']:.4f}", flush=True)
    return model
