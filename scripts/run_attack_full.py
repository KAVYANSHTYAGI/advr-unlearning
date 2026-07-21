"""Red-team: attack every model on BOTH surfaces -- output field (softmax) AND
penultimate features -- with a strong attacker. The feature attack is the one that
could break ADV (it only defends the output). Bigger batch + many workers to use the
whole machine.

Usage: PYTHONPATH=. python scripts/run_attack_full.py
"""
import glob
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from seg.attack import eval_attacker, train_attacker
from seg.data import ADE20KUnlearn
from seg.model import get_seg_model

ATK = dict(width=128, depth=6, dilation=2, epochs=4, max_iters=1400)
BS, NW = 12, 16


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    device = "cuda"
    tr = ADE20KUnlearn("train", size=512, crop=False)
    tr = Subset(tr, list(range(min(2400, len(tr)))))
    va = ADE20KUnlearn("val", size=512, crop=False)
    tl = DataLoader(tr, batch_size=BS, shuffle=True, num_workers=NW, pin_memory=True)
    vl = DataLoader(va, batch_size=BS, shuffle=False, num_workers=NW, pin_memory=True)

    original = load("checkpoints/seg_original_deeplabv3r50.pth", device)
    oracle = load("checkpoints/seg_oracle_deeplabv3r50.pth", device)
    variants = [("original", original, None), ("oracle_FLOOR", oracle, None),
                ("DELETE_logitmask", original, "mask")]
    for ck in sorted(glob.glob("checkpoints/unlearned_*.pth")):
        variants.append((os.path.basename(ck)[len("unlearned_"):-4], load(ck, device), None))

    rows = {}
    for name, model, transform in variants:
        rec = {}
        for surface, feat in [("out", False), ("feat", True)]:
            D = train_attacker(model, tl, device, transform=transform, feat=feat,
                               epochs=ATK["epochs"], max_iters=ATK["max_iters"],
                               width=ATK["width"], depth=ATK["depth"], dilation=ATK["dilation"])
            r = eval_attacker(D, model, vl, device, transform=transform, feat=feat)
            rec[surface + "_IoU"] = round(r["attacker_IoU"], 1)
            rec[surface + "_AUC"] = round(r["attacker_AUC"], 1)
        rows[name] = rec
        print(f"{name:18s} out[IoU {rec['out_IoU']:5.1f} AUC {rec['out_AUC']:5.1f}]  "
              f"feat[IoU {rec['feat_IoU']:5.1f} AUC {rec['feat_AUC']:5.1f}]", flush=True)

    json.dump(rows, open("results/seg/attack_full.json", "w"), indent=2)
    fo, ff = rows["oracle_FLOOR"]["out_AUC"], rows["oracle_FLOOR"]["feat_AUC"]
    print(f"\nfloors: out_AUC {fo}  feat_AUC {ff}")
    print("--- excess AUC over floor (leak) ---")
    for n, r in rows.items():
        print(f"{n:18s} out +{r['out_AUC']-fo:+.1f}   feat +{r['feat_AUC']-ff:+.1f}")


if __name__ == "__main__":
    main()
