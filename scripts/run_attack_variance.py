"""Multi-trial attacker protocol: train the attacker N times per model (both
surfaces) and report mean+/-std. Needed because a single attacker-training run has
real variance (oracle floor measured at IoU 0.0/10.9/2.2 across 3 trials) -- a
single-draw leakage number is not trustworthy on its own.

Usage: PYTHONPATH=. python scripts/run_attack_variance.py
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seg.attack import eval_attacker, train_attacker
from seg.data import UnlearnDS
from seg.model import get_seg_model

ATK = dict(width=128, depth=6, dilation=2, epochs=4, max_iters=1400)
BS, NW = 12, 24
N_TRIALS = int(os.environ.get("N_TRIALS", "3"))  # reviewer W2: bump to 10 for CIs


class _RAMSet(torch.utils.data.Dataset):
    """Images + labels preloaded into RAM as tensors -> zero decode on reuse.
    Yields (img, orig, orig) to match the UnlearnDS 3-tuple the attacker expects."""
    def __init__(self, imgs, labs):
        self.imgs, self.labs = imgs, labs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.labs[i], self.labs[i]


def _preload(ds, workers=8):
    """Decode the whole dataset once (parallel), stack into RAM tensors. The 36
    attacker trainings then reuse these without ever touching disk again."""
    from torch.utils.data import DataLoader as _DL
    ld = _DL(ds, batch_size=16, num_workers=workers, shuffle=False, pin_memory=False)
    imgs, labs = [], []
    for img, orig, *_ in ld:
        imgs.append(img); labs.append(orig)
    ram = _RAMSet(torch.cat(imgs), torch.cat(labs))
    print(f"  [preload] {len(ram)} samples cached in RAM "
          f"({torch.cat(imgs).element_size()*torch.cat(imgs).nelement()/1e9:.1f} GB)", flush=True)
    return ram

import os as _os
_ARCH = _os.environ.get("ARCH", "deeplabv3r50")
_CLS = _os.environ.get("FORGET_CLS", "12")
# UNLEARNED = comma-separated basenames of checkpoints/unlearned_<name>.pth to attack.
# Defaults to the person deeplab suite for backward compat.
_UNL = _os.environ.get("UNLEARNED", "ADV,ADVR,ADVR_s1,B1")
# floor + prior-art are always the oracle/original for THIS arch+class.
MODELS = [
    ("oracle_FLOOR",     f"checkpoints/seg_oracle_{_ARCH}_cls{_CLS}.pth", None),
    ("DELETE_logitmask", f"checkpoints/seg_original_{_ARCH}_cls{_CLS}.pth", "mask"),
]
for _name in [s for s in _UNL.split(",") if s]:
    MODELS.append((_name, f"checkpoints/unlearned_{_name}.pth", None))


def load(path, device, arch=None):
    if arch is None:
        import os; arch = os.environ.get("ARCH", "deeplabv3r50")
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=arch); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    import os
    arch = os.environ.get("ARCH", "deeplabv3r50")
    # SegFormer has no ASPP penultimate layer accessible via classifier[4], so
    # the feature-surface attack is only valid for DeepLabV3-style models.
    surfaces = [("out", False)]
    if arch == "deeplabv3r50":
        surfaces.append(("feat", True))
    device = "cuda"
    _full = UnlearnDS("train", size=512, crop=False)
    tr = _preload(Subset(_full, list(range(min(2400, len(_full))))))
    va = _preload(UnlearnDS("val", size=512, crop=False))
    # in-RAM -> num_workers=0 (data already decoded tensors, nothing to parallelize)
    tl = DataLoader(tr, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(va, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

    rows = {}
    for name, path, transform in MODELS:
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing"); continue
        model = load(path, device)
        for surface, feat in surfaces:
            ious, aucs = [], []
            for t in range(N_TRIALS):
                D = train_attacker(model, tl, device, transform=transform, feat=feat, **ATK)
                r = eval_attacker(D, model, vl, device, transform=transform, feat=feat)
                ious.append(r["attacker_IoU"]); aucs.append(r["attacker_AUC"])
                print(f"  {name} {surface} trial{t}: IoU {ious[-1]:.1f} AUC {aucs[-1]:.1f}", flush=True)
            rows.setdefault(name, {})[surface] = {
                "IoU_mean": round(float(np.mean(ious)), 1), "IoU_std": round(float(np.std(ious)), 1),
                "AUC_mean": round(float(np.mean(aucs)), 1), "AUC_std": round(float(np.std(aucs)), 1),
            }
        del model; torch.cuda.empty_cache()

    out_path = os.environ.get("OUT_JSON", "results/seg/attack_variance.json")
    json.dump(rows, open(out_path, "w"), indent=2)
    has_feat = all("feat" in surf for surf in rows.values())
    print(f"\n=== summary (mean +/- std over {N_TRIALS} attacker trials) ===")
    for name, surf in rows.items():
        o = surf["out"]
        line = f"{name:20s} out AUC {o['AUC_mean']:5.1f}+/-{o['AUC_std']:.1f}"
        if has_feat:
            f = surf["feat"]
            line += f"  feat AUC {f['AUC_mean']:5.1f}+/-{f['AUC_std']:.1f}"
        print(line)

    if "oracle_FLOOR" in rows:
        fo = rows["oracle_FLOOR"]["out"]
        ff = rows["oracle_FLOOR"].get("feat")
        feat_str = f"  feat_AUC {ff['AUC_mean']}+/-{ff['AUC_std']}" if ff else ""
        print(f"\nfloors: out_AUC {fo['AUC_mean']}+/-{fo['AUC_std']}{feat_str}")
        print("a method LEAKS only if its AUC exceeds floor_mean + 2*floor_std")
        for name, surf in rows.items():
            if name == "oracle_FLOOR":
                continue
            o = surf["out"]
            leak_o = "LEAK" if o["AUC_mean"] > fo["AUC_mean"] + 2 * fo["AUC_std"] else "safe"
            line = f"{name:20s} out: {leak_o} ({o['AUC_mean']-fo['AUC_mean']:+.1f})"
            if ff and "feat" in surf:
                f = surf["feat"]
                leak_f = "LEAK" if f["AUC_mean"] > ff["AUC_mean"] + 2 * ff["AUC_std"] else "safe"
                line += f"   feat: {leak_f} ({f['AUC_mean']-ff['AUC_mean']:+.1f})"
            print(line)


if __name__ == "__main__":
    main()
