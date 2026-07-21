"""Shared helpers for the reviewer-response experiments (attacker zoo, capacity
sweep, adaptive/white-box, transfer). Mirrors the RAM-preload trick in
scripts/run_attack_variance.py so the many attacker trainings never re-decode disk.
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from seg.data import UnlearnDS
from seg.model import get_seg_model


class RAMSet(Dataset):
    """Images + labels preloaded into RAM. Yields (img, orig, orig) to match the
    UnlearnDS 3-tuple the attacker expects."""
    def __init__(self, imgs, labs):
        self.imgs, self.labs = imgs, labs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.labs[i], self.labs[i]


def preload(ds, workers=8, tag=""):
    ld = DataLoader(ds, batch_size=16, num_workers=workers, shuffle=False,
                    pin_memory=False)
    imgs, labs = [], []
    for img, orig, *_ in ld:
        imgs.append(img); labs.append(orig)
    imgs, labs = torch.cat(imgs), torch.cat(labs)
    print(f"  [preload{tag}] {len(imgs)} samples cached "
          f"({imgs.element_size()*imgs.nelement()/1e9:.1f} GB)", flush=True)
    return RAMSet(imgs, labs)


def load_model(path, device="cuda", arch=None):
    if arch is None:
        arch = os.environ.get("ARCH", "deeplabv3r50")
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=arch)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def attack_loaders(bs=12, n_train=2400, workers=24):
    """Preloaded person-image train/val loaders for attacker training/eval.
    Env EXP_N_TRAIN / EXP_N_VAL / EXP_BS override sizes (used by SMOKE mode)."""
    bs = int(os.environ.get("EXP_BS", bs))
    n_train = int(os.environ.get("EXP_N_TRAIN", n_train))
    n_val = int(os.environ.get("EXP_N_VAL", "0"))
    full = UnlearnDS("train", size=512, crop=False)
    tr = preload(Subset(full, list(range(min(n_train, len(full))))), workers, " train")
    vfull = UnlearnDS("val", size=512, crop=False)
    if n_val > 0:
        vfull = Subset(vfull, list(range(min(n_val, len(vfull)))))
    va = preload(vfull, workers, " val")
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    return tl, vl


def model_paths(unlearned=None, arch=None, cls=None):
    """Return an ordered list of (name, ckpt_path, transform) covering the floor,
    the DELETE prior-art, and the requested unlearned checkpoints. Mirrors the
    MODELS block in run_attack_variance.py so results line up."""
    arch = arch or os.environ.get("ARCH", "deeplabv3r50")
    cls = cls or os.environ.get("FORGET_CLS", "12")
    if unlearned is None:
        unlearned = os.environ.get("UNLEARNED", "ADV,ADVR,ADVR_s1,B1")
    models = [
        ("oracle_FLOOR",     f"checkpoints/seg_oracle_{arch}_cls{cls}.pth", None),
        ("DELETE_logitmask", f"checkpoints/seg_original_{arch}_cls{cls}.pth", "mask"),
    ]
    for name in [s for s in unlearned.split(",") if s]:
        models.append((name, f"checkpoints/unlearned_{name}.pth", None))
    return [(n, p, t) for n, p, t in models if os.path.exists(p) or _warn(n, p)]


def _warn(name, path):
    print(f"[skip] {name}: {path} missing", flush=True)
    return False


def leak_flag(auc, floor_mean, floor_std):
    return "LEAK" if auc > floor_mean + 2 * floor_std else "safe"
