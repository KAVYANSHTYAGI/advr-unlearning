"""Validate HYP: can a silhouette attacker reconstruct the person mask from a model's
output field? Compares original / oracle (floor) / DELETE-style logit-mask, plus any
saved unlearned checkpoints (checkpoints/unlearned_*.pth).

Usage:
  PYTHONPATH=. python scripts/run_attack.py
"""
import glob
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from seg.attack import eval_attacker, train_attacker
from seg.data import ADE20KUnlearn
from seg.model import get_seg_model


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strong", action="store_true",
                    help="stronger/deeper attacker for an honest transfer-attack eval")
    args = ap.parse_args()
    atk = dict(width=128, depth=6, dilation=2, epochs=3, max_iters=900) if args.strong \
        else dict(width=64, depth=3, dilation=1, epochs=2, max_iters=400)
    device = "cuda"
    tr = ADE20KUnlearn("train", size=512, crop=False)
    tr = Subset(tr, list(range(min(1600, len(tr)))))     # cap for ~1 GPU-h budget
    va = ADE20KUnlearn("val", size=512, crop=False)
    tl = DataLoader(tr, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)
    vl = DataLoader(va, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    original = load("checkpoints/seg_original_deeplabv3r50.pth", device)
    oracle = load("checkpoints/seg_oracle_deeplabv3r50.pth", device)

    variants = [
        ("original", original, None),
        ("oracle_FLOOR", oracle, None),
        ("DELETE_logitmask", original, "mask"),
    ]
    for ck in sorted(glob.glob("checkpoints/unlearned_*.pth")):
        name = os.path.basename(ck)[len("unlearned_"):-4]
        variants.append((name, load(ck, device), None))

    rows = {}
    for name, model, transform in variants:
        D = train_attacker(model, tl, device, transform=transform,
                           epochs=atk["epochs"], max_iters=atk["max_iters"],
                           width=atk["width"], depth=atk["depth"], dilation=atk["dilation"])
        r = eval_attacker(D, model, vl, device, transform=transform)
        rows[name] = r
        print(f"{name:20s} attacker_IoU {r['attacker_IoU']:.1f}  attacker_AUC {r['attacker_AUC']:.1f}", flush=True)

    os.makedirs("results/seg", exist_ok=True)
    json.dump(rows, open("results/seg/attack.json", "w"), indent=2)
    floor = rows["oracle_FLOOR"]["attacker_IoU"]
    print("\n--- leakage above oracle context-prior floor (IoU - floor) ---")
    for name, r in rows.items():
        print(f"{name:20s} +{r['attacker_IoU'] - floor:+.1f}")
    print(f"\nfloor (oracle) = {floor:.1f} IoU. A method LEAKS if it beats the floor.")


if __name__ == "__main__":
    main()
