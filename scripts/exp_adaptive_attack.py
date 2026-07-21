"""T1.3 -- Adaptive / white-box attacker (answers reviewer W5, the biggest
experimental omission: "does not establish robustness against adaptive adversaries").

Against a FROZEN ADVR (and the oracle floor), we give the adversary far more than
the discriminator ADV trained against: the strongest architecture, multiple random
restarts, and training to convergence with val-AUC early stopping. If ADVR's best-
of-all-restarts AUC still does not exceed floor+2sigma, the privacy guarantee holds
against an adaptive white-box adversary -- the sentence that otherwise sinks the paper.

For each model we report the single best (over restarts x rounds) val AUC/IoU.

Usage: PYTHONPATH=. python scripts/exp_adaptive_attack.py
Env: KIND (default vit), RESTARTS, ROUNDS, ITERS, UNLEARNED, ARCH, OUT_JSON
"""
import json
import os

import torch

from seg.attack import train_attacker_adaptive
from seg.exp_utils import attack_loaders, load_model, model_paths

KIND = os.environ.get("KIND", "vit")
RESTARTS = int(os.environ.get("RESTARTS", "3"))
ROUNDS = int(os.environ.get("ROUNDS", "8"))
ITERS = int(os.environ.get("ITERS", "1500"))
KW = {"cnn": dict(width=128, depth=6, dilation=2), "unet": dict(base=96),
      "vit": dict(dim=384, vdepth=8, heads=6)}[KIND]


def main():
    arch = os.environ.get("ARCH", "deeplabv3r50")
    device = "cuda"
    surfaces = [("out", False)] + ([("feat", True)] if arch == "deeplabv3r50" else [])
    tl, vl = attack_loaders()
    models = model_paths()
    print(f"[adaptive] kind={KIND} restarts={RESTARTS} rounds={ROUNDS} "
          f"iters/round={ITERS}  (max {RESTARTS*ROUNDS*ITERS} iters/cell)")

    rows = {}
    for name, path, transform in models:
        model = load_model(path, device)
        rows[name] = {}
        for surface, feat in surfaces:
            _, best = train_attacker_adaptive(
                model, tl, vl, device, transform=transform, feat=feat, kind=KIND,
                restarts=RESTARTS, rounds=ROUNDS, iters_per_round=ITERS,
                verbose=True, **KW)
            rows[name][surface] = {
                "AUC": round(best["attacker_AUC"], 1),
                "IoU": round(best["attacker_IoU"], 1),
            }
            print(f"  {name:16s} {surface} BEST: AUC {rows[name][surface]['AUC']:.1f}"
                  f"  IoU {rows[name][surface]['IoU']:.1f}", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.environ.get("OUT_JSON", "results/seg/exp_adaptive_attack.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)

    print("\n=== adaptive white-box excess over floor (best-of-all-restarts) ===")
    if "oracle_FLOOR" in rows:
        for surface, _ in surfaces:
            fo = rows["oracle_FLOOR"][surface]["AUC"]
            for name in rows:
                if name == "oracle_FLOOR":
                    continue
                a = rows[name][surface]["AUC"]
                # single best draw -> compare directly to floor (no sigma band here)
                print(f"  {name:16s} {surface}: {a:.1f} vs floor {fo:.1f} "
                      f"({a-fo:+.1f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
