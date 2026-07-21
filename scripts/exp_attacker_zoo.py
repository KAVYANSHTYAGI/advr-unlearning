"""T1.2 -- Attacker architecture zoo (answers reviewer W1: "why not U-Net / ViT").

Runs three qualitatively different attacker architectures against each model:
  cnn  -- the dilated conv silhouette net (baseline attacker)
  unet -- encoder/decoder with skips (strong boundary + layout prior, ~8-17M)
  vit  -- patch transformer, global attention from layer 1 (~14M+)
Both surfaces (output field + penultimate features), N_TRIALS each, mean+/-std.

If ADVR stays <= floor+2sigma under ALL THREE architectures, the claim "no
learned function recovers the silhouette" is defensible against architecture choice.

Usage: PYTHONPATH=. python scripts/exp_attacker_zoo.py
Env: N_TRIALS, KINDS (default "cnn,unet,vit"), UNLEARNED, ARCH, OUT_JSON, MAX_ITERS
"""
import json
import os

import numpy as np
import torch

from seg.attack import (build_attacker, eval_attacker, n_params, train_attacker,
                        FEAT_DIM)
from seg.data import NUM_CLASSES
from seg.exp_utils import attack_loaders, load_model, model_paths

N_TRIALS = int(os.environ.get("N_TRIALS", "3"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "2000"))
KINDS = [k for k in os.environ.get("KINDS", "cnn,unet,vit").split(",") if k]
# per-kind build kwargs (sized for a genuinely strong attacker)
KW = {
    "cnn":  dict(width=128, depth=6, dilation=2),
    "unet": dict(base=96),
    "vit":  dict(dim=384, vdepth=8, heads=6),
}


def _pcount(kind, feat):
    c_in = FEAT_DIM if feat else NUM_CLASSES
    return n_params(build_attacker(kind, c_in=c_in, out_size=128, **KW[kind]))


def main():
    arch = os.environ.get("ARCH", "deeplabv3r50")
    device = "cuda"
    surfaces = [("out", False)] + ([("feat", True)] if arch == "deeplabv3r50" else [])
    tl, vl = attack_loaders()
    models = model_paths()

    print("attacker param counts (out surface): " +
          ", ".join(f"{k}={_pcount(k, False)/1e6:.1f}M" for k in KINDS))

    rows = {}
    for name, path, transform in models:
        model = load_model(path, device)
        rows[name] = {}
        for surface, feat in surfaces:
            for kind in KINDS:
                ious, aucs = [], []
                for t in range(N_TRIALS):
                    D = train_attacker(model, tl, device, transform=transform,
                                       feat=feat, kind=kind, epochs=10 ** 6,
                                       max_iters=MAX_ITERS, **KW[kind])
                    r = eval_attacker(D, model, vl, device, transform=transform, feat=feat)
                    ious.append(r["attacker_IoU"]); aucs.append(r["attacker_AUC"])
                cell = {
                    "params": _pcount(kind, feat),
                    "AUC_mean": round(float(np.nanmean(aucs)), 1),
                    "AUC_std": round(float(np.nanstd(aucs)), 1),
                    "IoU_mean": round(float(np.nanmean(ious)), 1),
                    "IoU_std": round(float(np.nanstd(ious)), 1),
                }
                rows[name].setdefault(surface, {})[kind] = cell
                print(f"  {name:16s} {surface} {kind:4s}: AUC {cell['AUC_mean']:.1f}"
                      f"+/-{cell['AUC_std']:.1f}  IoU {cell['IoU_mean']:.1f}", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.environ.get("OUT_JSON", "results/seg/exp_attacker_zoo.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)

    print("\n=== worst-case leak per model (max excess AUC over floor, any arch) ===")
    if "oracle_FLOOR" in rows:
        for surface, _ in surfaces:
            fo = {k: rows["oracle_FLOOR"][surface][k] for k in KINDS}
            for name in rows:
                if name == "oracle_FLOOR":
                    continue
                worst = max(
                    rows[name][surface][k]["AUC_mean"] - fo[k]["AUC_mean"]
                    for k in KINDS)
                wk = max(KINDS, key=lambda k: rows[name][surface][k]["AUC_mean"]
                         - fo[k]["AUC_mean"])
                thr = 2 * fo[wk]["AUC_std"]
                flag = "LEAK" if worst > thr else "safe"
                print(f"  {name:16s} {surface}: worst +{worst:+.1f} via {wk} ({flag})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
