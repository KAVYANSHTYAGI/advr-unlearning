"""Privacy/leakage metrics and the relearning audit for person-unlearning.

Forgetting quality in segmentation is NOT just "is the person class gone". Two
failure modes a good unlearner must avoid:

  residual_person : the model still predicts `person` in the person region
                    (direct leak). Want ~0.

  blend_fail      : in the person region, the prediction disagrees with the
                    dominant-neighbor target -- i.e. the region does not blend
                    into its surroundings and a person-shaped anomaly remains
                    (the "segmentation by subtraction" leak). Want ~0.

  silhouette_IoU  : how well a naive attacker recovers the person mask from the
                    prediction alone, by flagging person-region pixels whose label
                    differs from the surrounding ring. IoU vs the true person mask;
                    lower = more private.

Plus a relearning audit (few-shot finetune on person pixels) mirroring the
classification track: does person segmentation come back?
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from seg.data import IGNORE, PERSON
from seg.engine import seg_metrics
from seg.model import seg_forward
from seg.utils import masked_ce as _masked_ce


@torch.no_grad()
def leakage_metrics(model, loader, device="cuda"):
    """loader yields (img, orig, merged). Aggregates over person images."""
    model.eval()
    res_person = res_total = 0
    blend_fail = blend_total = 0
    sil_inter = sil_union = 0
    for img, orig, merged in loader:
        img, orig, merged = img.to(device), orig.to(device), merged.to(device)
        pred = seg_forward(model, img).argmax(1)
        M = orig == PERSON
        if M.sum() == 0:
            continue
        # direct residual: still predicted person inside the region
        res_person += (pred[M] == PERSON).sum().item()
        res_total += M.sum().item()
        # blend failure: prediction != dominant-neighbor target inside the region
        blend_fail += (pred[M] != merged[M]).sum().item()
        blend_total += M.sum().item()
        # silhouette attacker: region pixels whose pred differs from the merged
        # target form the "anomaly"; IoU against the true mask
        anomaly = (pred != merged) & (orig != IGNORE)
        sil_inter += (anomaly & M).sum().item()
        sil_union += (anomaly | M).sum().item()
    return {
        "residual_person": 100 * res_person / max(res_total, 1),
        "blend_fail": 100 * blend_fail / max(blend_total, 1),
        "silhouette_IoU": 100 * sil_inter / max(sil_union, 1),
    }


def relearn_audit(unlearned_model, loader, device="cuda", steps=50, lr=1e-3,
                  val_loader=None):
    """Finetune the unlearned model on person pixels (orig labels) for a few steps;
    report how much person segmentation is recovered."""
    m = copy.deepcopy(unlearned_model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    before = seg_metrics(m, val_loader, device)["person_IoU"] if val_loader else None
    it = iter(loader); done = 0
    while done < steps:
        try:
            img, orig, _merged = next(it)
        except StopIteration:
            it = iter(loader); continue
        img, orig = img.to(device), orig.to(device)
        forget = orig == PERSON
        m.train()
        loss = _masked_ce(seg_forward(m, img), orig, forget)
        opt.zero_grad(); loss.backward(); opt.step(); done += 1
    after = seg_metrics(m, val_loader, device)["person_IoU"] if val_loader else None
    out = {"person_IoU_before": before, "person_IoU_after": after}
    if before is not None:
        out["recover_pts"] = round(after - before, 1)
    return out
