"""T2.1 -- Oracle-construction ablation (answers reviewer W6: "floor depends on
arbitrary preprocessing").

Attacks each oracle variant (person pixels relabelled by nn / mode / rand / ignore)
with the standard N_TRIALS silhouette attacker and reports the leakage FLOOR (AUC/
IoU, both surfaces) for each. If the floor is stable across schemes, the oracle-floor
protocol is robust to the relabelling choice; the spread across schemes is the
honest uncertainty on the floor.

Assumes the variant oracles were trained by scripts/exp_oracle_ablation.sh into
checkpoints/seg_oracle_<arch>_cls<cls>[_<scheme>].pth (nn = no suffix).

Usage: PYTHONPATH=. python scripts/experiments/rebuttal/exp_oracle_ablation.py
Env: N_TRIALS, SCHEMES (default "nn,mode,rand,ignore"), ARCH, OUT_JSON
"""
import json
import os

import numpy as np
import torch

from seg.attack import eval_attacker, train_attacker
from seg.exp_utils import attack_loaders, load_model

N_TRIALS = int(os.environ.get("N_TRIALS", "3"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "1400"))
SCHEMES = [s for s in os.environ.get("SCHEMES", "nn,mode,rand,ignore").split(",") if s]
ATK = dict(kind="cnn", width=128, depth=6, dilation=2)


def _ckpt(scheme, arch, cls):
    suf = "" if scheme == "nn" else f"_{scheme}"
    return f"checkpoints/seg_oracle_{arch}_cls{cls}{suf}.pth"


def main():
    arch = os.environ.get("ARCH", "deeplabv3r50")
    cls = os.environ.get("FORGET_CLS", "12")
    device = "cuda"
    surfaces = [("out", False)] + ([("feat", True)] if arch == "deeplabv3r50" else [])
    tl, vl = attack_loaders()

    rows = {}
    for scheme in SCHEMES:
        path = _ckpt(scheme, arch, cls)
        if not os.path.exists(path):
            print(f"[skip] {scheme}: {path} missing (run exp_oracle_ablation.sh)")
            continue
        model = load_model(path, device)
        rows[scheme] = {}
        for surface, feat in surfaces:
            aucs, ious = [], []
            for _t in range(N_TRIALS):
                D = train_attacker(model, tl, device, feat=feat, epochs=10 ** 6,
                                   max_iters=MAX_ITERS, **ATK)
                r = eval_attacker(D, model, vl, device, feat=feat)
                aucs.append(r["attacker_AUC"]); ious.append(r["attacker_IoU"])
            rows[scheme][surface] = {
                "AUC_mean": round(float(np.nanmean(aucs)), 1),
                "AUC_std": round(float(np.nanstd(aucs)), 1),
                "IoU_mean": round(float(np.nanmean(ious)), 1),
            }
            print(f"  {scheme:7s} {surface}: floor AUC "
                  f"{rows[scheme][surface]['AUC_mean']:.1f}"
                  f"+/-{rows[scheme][surface]['AUC_std']:.1f}", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.environ.get("OUT_JSON", "results/seg/exp_oracle_ablation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)

    print("\n=== floor stability across relabel schemes (out surface) ===")
    outs = [rows[s]["out"]["AUC_mean"] for s in rows]
    if outs:
        print(f"  floor AUC range: {min(outs):.1f} - {max(outs):.1f} "
              f"(spread {max(outs)-min(outs):.1f}) across {list(rows)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
