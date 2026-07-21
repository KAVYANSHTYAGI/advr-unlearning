"""T1.1 -- Attacker capacity sweep (answers reviewer W1: "weak attacker").

For a fixed set of models (floor + DELETE + unlearned), sweep the CNN attacker
capacity (width/depth/dilation) from small to large and record the leakage AUC/IoU
on both surfaces. The headline plot is `excess AUC over floor` vs `attacker params`:
if ADVR's excess *saturates* (stays <= floor+2sigma) as the attacker grows, the
privacy claim is not an artifact of a weak adversary.

Each cell runs N_TRIALS attacker trainings; we report mean+/-std.

Usage: PYTHONPATH=. python scripts/exp_attacker_capacity.py
Env: N_TRIALS, UNLEARNED, ARCH, FORGET_CLS, OUT_JSON, MAX_ITERS
"""
import json
import os

import numpy as np
import torch

from seg.attack import eval_attacker, train_attacker, build_attacker, n_params, FEAT_DIM
from seg.data import NUM_CLASSES
from seg.exp_utils import attack_loaders, load_model, model_paths

# (label, width, depth, dilation) -- small -> large
CONFIGS = [
    ("tiny",  32, 2, 1),
    ("small", 64, 3, 1),
    ("mid",  128, 4, 2),
    ("large",192, 6, 2),
    ("xl",   256, 8, 2),
]
N_TRIALS = int(os.environ.get("N_TRIALS", "3"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "1600"))
EPOCHS = 10 ** 6  # iter-bounded


def _params_for(width, depth, dilation, feat):
    c_in = FEAT_DIM if feat else NUM_CLASSES
    return n_params(build_attacker("cnn", c_in=c_in, width=width, depth=depth,
                                   dilation=dilation))


def main():
    arch = os.environ.get("ARCH", "deeplabv3r50")
    device = "cuda"
    surfaces = [("out", False)] + ([("feat", True)] if arch == "deeplabv3r50" else [])
    tl, vl = attack_loaders()
    models = model_paths()

    rows = {}
    for name, path, transform in models:
        model = load_model(path, device)
        rows[name] = {}
        for surface, feat in surfaces:
            for label, w, d, dil in CONFIGS:
                ious, aucs = [], []
                for t in range(N_TRIALS):
                    D = train_attacker(model, tl, device, transform=transform,
                                       feat=feat, kind="cnn", width=w, depth=d,
                                       dilation=dil, epochs=EPOCHS, max_iters=MAX_ITERS)
                    r = eval_attacker(D, model, vl, device, transform=transform, feat=feat)
                    ious.append(r["attacker_IoU"]); aucs.append(r["attacker_AUC"])
                cell = {
                    "params": _params_for(w, d, dil, feat),
                    "AUC_mean": round(float(np.nanmean(aucs)), 1),
                    "AUC_std": round(float(np.nanstd(aucs)), 1),
                    "IoU_mean": round(float(np.nanmean(ious)), 1),
                    "IoU_std": round(float(np.nanstd(ious)), 1),
                }
                rows[name].setdefault(surface, {})[label] = cell
                print(f"  {name:16s} {surface} {label:5s} "
                      f"({cell['params']/1e3:.0f}k): AUC {cell['AUC_mean']:.1f}"
                      f"+/-{cell['AUC_std']:.1f}", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.environ.get("OUT_JSON", "results/seg/exp_attacker_capacity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)

    # excess-over-floor summary at the LARGEST attacker
    print("\n=== excess AUC over floor at each capacity (out surface) ===")
    if "oracle_FLOOR" in rows:
        for label, *_ in CONFIGS:
            fo = rows["oracle_FLOOR"]["out"][label]
            thr = fo["AUC_mean"] + 2 * fo["AUC_std"]
            line = f"{label:5s} floor {fo['AUC_mean']:.1f}+/-{fo['AUC_std']:.1f} | "
            for name in rows:
                if name == "oracle_FLOOR":
                    continue
                a = rows[name]["out"][label]["AUC_mean"]
                flag = "LEAK" if a > thr else "safe"
                line += f"{name} {a-fo['AUC_mean']:+.1f}({flag})  "
            print(line)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
