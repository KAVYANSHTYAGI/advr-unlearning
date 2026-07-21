"""T1.4 -- Transfer attack (answers reviewer W1: "why not transfer attackers").

Train the attacker on model A's field, evaluate it on model B's field, for every
(A,B) pair among {original, oracle, ADVR, ...}. A defense that merely *obscures*
(shifts the field to something a co-trained D can't read) would still leak when
attacked by a D trained on a different model's field. If the ADVR column stays at
the floor even for attackers trained on the original/oracle, the protection is real,
not co-adaptation artifact.

Output: an AUC transfer matrix per surface (rows=train model, cols=eval model).

Usage: PYTHONPATH=. python scripts/experiments/rebuttal/exp_transfer_attack.py
Env: TRANSFER_MODELS (default "original,oracle,ADVR"), N_TRIALS, ARCH, OUT_JSON
"""
import json
import os

import numpy as np
import torch

from seg.attack import eval_attacker, train_attacker
from seg.exp_utils import attack_loaders, load_model

N_TRIALS = int(os.environ.get("N_TRIALS", "3"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "1600"))
ATK = dict(kind="cnn", width=128, depth=6, dilation=2)


def _resolve(names, arch, cls):
    """Map friendly model names to checkpoint paths."""
    out = []
    for n in names:
        if n == "original":
            p = f"checkpoints/seg_original_{arch}_cls{cls}.pth"
        elif n == "oracle":
            p = f"checkpoints/seg_oracle_{arch}_cls{cls}.pth"
        else:
            p = f"checkpoints/unlearned_{n}.pth"
        if os.path.exists(p):
            out.append((n, p))
        else:
            print(f"[skip] {n}: {p} missing")
    return out


def main():
    arch = os.environ.get("ARCH", "deeplabv3r50")
    cls = os.environ.get("FORGET_CLS", "12")
    device = "cuda"
    surfaces = [("out", False)] + ([("feat", True)] if arch == "deeplabv3r50" else [])
    names = [s for s in os.environ.get("TRANSFER_MODELS", "original,oracle,ADVR").split(",") if s]
    models = _resolve(names, arch, cls)
    tl, vl = attack_loaders()

    # preload all models once
    loaded = {n: load_model(p, device) for n, p in models}
    result = {}
    for surface, feat in surfaces:
        mat = {}
        for atrain, _ in models:
            # train N_TRIALS attackers on atrain, keep the best by its own val AUC
            best_D, best_auc = None, -1.0
            for _t in range(N_TRIALS):
                D = train_attacker(loaded[atrain], tl, device, feat=feat,
                                   epochs=10 ** 6, max_iters=MAX_ITERS, **ATK)
                a = eval_attacker(D, loaded[atrain], vl, device, feat=feat)["attacker_AUC"]
                if np.isfinite(a) and a > best_auc:
                    import copy; best_D, best_auc = copy.deepcopy(D), a
            row = {}
            for aeval, _ in models:
                r = eval_attacker(best_D, loaded[aeval], vl, device, feat=feat)
                row[aeval] = round(r["attacker_AUC"], 1)
            mat[atrain] = row
            print(f"  [{surface}] train={atrain:10s} -> " +
                  "  ".join(f"{k}:{v:.1f}" for k, v in row.items()), flush=True)
        result[surface] = mat

    out = os.environ.get("OUT_JSON", "results/seg/exp_transfer_attack.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("read the ADVR *column*: highest AUC any attacker (incl. cross-trained) "
          "achieves against ADVR. Compare to the oracle column (floor).")


if __name__ == "__main__":
    main()
