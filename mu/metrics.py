"""Evaluation metrics for unlearning.

The headline numbers, following the SalUn / unlearning-benchmark convention:
  UA  (unlearning acc)   = 100 - forget_acc      (higher = more forgotten)
  RA  (retain acc)       = accuracy on retain set (higher = better)
  TA  (test acc)         = accuracy on test set   (higher = better)
  MIA = SVC membership-inference success on the forget set (confidence attack);
        closer to the retrain oracle is better.
  gap_to_retrain        = mean |metric - retrain_metric| over {UA,RA,TA,MIA}
                          (the "Tug-of-War" / avg-gap summary; lower = better)
"""
import numpy as np
import torch

from mu.engine import evaluate
from mu.svc_mia import SVC_MIA


@torch.no_grad()
def _subset_loader(loader, n, seed=0):
    """Deterministically take the first ~n samples' loader for MIA balancing."""
    from torch.utils.data import DataLoader, Subset
    ds = loader.dataset
    idx = np.arange(min(n, len(ds)))
    return DataLoader(Subset(ds, idx.tolist()), batch_size=loader.batch_size,
                      shuffle=False, num_workers=2)


def mia_forget_efficacy(model, s, device="cuda"):
    """SVC-MIA on the forget set: shadow = retain vs test, target = forget.
    Returns the confidence-attack success in [0,1] (lower = better forgetting)."""
    n = len(s.forget_eval.dataset)
    shadow_train = _subset_loader(s.retain_eval, n)
    shadow_test = _subset_loader(s.test_eval, n)
    res = SVC_MIA(
        shadow_train=shadow_train,
        shadow_test=shadow_test,
        target_train=s.forget_eval,
        target_test=None,
        model=model,
    )
    return res["confidence"]


def full_metrics(model, s, device="cuda"):
    fa = evaluate(model, s.forget_eval, device)
    ra = evaluate(model, s.retain_eval, device)
    ta = evaluate(model, s.test_eval, device)
    mia = 100.0 * mia_forget_efficacy(model, s, device)
    return {"UA": 100 - fa, "RA": ra, "TA": ta, "MIA": mia,
            "forget_acc": fa}


def gap_to_retrain(row, retrain_row):
    keys = ["UA", "RA", "TA", "MIA"]
    return float(np.mean([abs(row[k] - retrain_row[k]) for k in keys]))
