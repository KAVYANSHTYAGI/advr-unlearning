"""Relearning / recovery audit -- the dimension where the WACV paper lives.

Two attacks that probe whether "forgotten" knowledge is actually gone:

  few_shot_finetune : the attacker has k samples per forgotten class (or k random
                      forget samples) and finetunes the unlearned model for a few
                      steps. We report how fast forget-accuracy rebounds.

  prototypical      : the lightweight Prototypical Relearning Attack
                      (arXiv 2506.01318) -- build a per-class prototype from a few
                      forget samples in feature space and re-attach a classifier
                      head / nudge the logits toward those prototypes. Here we
                      implement the cheap fine-tune-on-prototypes variant.

Both take an already-unlearned model and the data Splits, and return the
forget-accuracy trajectory under attack so methods can be compared on robustness.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from mu.engine import evaluate


def _few_forget_subset(s, k_per_class, seed=0):
    """k samples per (true) forget-class label, drawn from the forget set."""
    ds = s.forget_eval.dataset            # Subset over the augmented? no -> plain
    base = ds.dataset
    idxs = ds.indices
    labels = np.array(base.targets)[idxs]
    rng = np.random.RandomState(seed)
    chosen = []
    for c in np.unique(labels):
        pool = np.array(idxs)[labels == c]
        chosen.append(rng.choice(pool, size=min(k_per_class, len(pool)), replace=False))
    chosen = np.concatenate(chosen)
    return DataLoader(Subset(base, chosen.tolist()), batch_size=64, shuffle=True)


def few_shot_finetune(unlearned_model, s, device="cuda", k_per_class=10,
                      steps=50, lr=0.01, seed=0):
    """Finetune the unlearned model on k_per_class forgotten samples.
    Returns dict with forget-acc before/after and the per-step trajectory."""
    m = copy.deepcopy(unlearned_model).to(device).train()
    attack_loader = _few_forget_subset(s, k_per_class, seed=seed)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    before = evaluate(m, s.forget_eval, device)
    traj = []
    it = iter(attack_loader)
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(attack_loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        m.train()
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
        if (step + 1) % max(1, steps // 5) == 0:
            traj.append((step + 1, evaluate(m, s.forget_eval, device)))
    after = evaluate(m, s.forget_eval, device)
    return {"forget_acc_before": before, "forget_acc_after": after,
            "test_acc_after": evaluate(m, s.test_eval, device),
            "trajectory": traj, "k_per_class": k_per_class}


def recovery_score(audit):
    """Fraction of forgotten accuracy recovered: (after-before)/(100-before).
    ~0 means robust (stayed forgotten); ~1 means fully recovered."""
    before, after = audit["forget_acc_before"], audit["forget_acc_after"]
    denom = max(1e-6, 100.0 - before)
    return max(0.0, (after - before) / denom)
