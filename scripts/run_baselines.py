"""Run a faithful baseline with its OWN published hyper-parameters, save + eval.

Each method uses the defaults from its source paper/repo (not tuned for us) — the
honest transfer test. Trains from the reference original, saves as
unlearned_bl_<name>_s<seed>.pth.

Usage:
  PYTHONPATH=. python scripts/run_baselines.py --name SSD --seed 1
  PYTHONPATH=. python scripts/run_baselines.py --name FT  --long   # 3x longer epochs
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from seg.baselines import run_baseline
from seg.data import PERSON, SegDS, UnlearnDS
from seg.engine import setup_seed
from seg.model import get_seg_model

# faithful defaults from each source (see seg/baselines/<mod>.py docstrings)
FAITHFUL_HP = {
    "FT":       dict(epochs=10, lr=0.01),
    "NegGrad":  dict(epochs=5,  lr=1e-4),
    "NegGrad+": dict(epochs=5,  lr=0.01, beta=0.999),
    "RL":       dict(epochs=10, lr=0.013),
    "SalUn":    dict(epochs=10, lr=0.013, threshold=0.5),
    "SSD":      dict(selection_weighting=10.0, dampening_constant=1.0),
    "SCRUB":    dict(epochs=5,  lr=5e-4, kd_T=4.0, alpha=0.5, gamma=1.0, msteps=2),
    "DELETE":   dict(epochs=5,  lr=0.01),
}

# Extended epochs for segmentation convergence (same algorithm, just more compute).
# NegGrad/SSD/SCRUB are capped — they destroy the model regardless of epoch count.
LONG_HP = {
    "FT":       dict(epochs=30, lr=0.01),
    "NegGrad":  dict(epochs=10, lr=1e-4),          # capped: longer = more destruction
    "NegGrad+": dict(epochs=20, lr=0.01, beta=0.999),
    "RL":       dict(epochs=30, lr=0.013),
    "SalUn":    dict(epochs=30, lr=0.013, threshold=0.5),
    "SSD":      dict(selection_weighting=10.0, dampening_constant=1.0),  # one-shot
    "SCRUB":    dict(epochs=20, lr=5e-4, kd_T=4.0, alpha=0.5, gamma=1.0, msteps=5),
    "DELETE":   dict(epochs=20, lr=0.01),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--arch", default="deeplabv3r50")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--prefix", default="", help="checkpoint name prefix, e.g. 'city_'")
    p.add_argument("--long", action="store_true", help="use extended-epoch HPs for segmentation convergence")
    args = p.parse_args()
    device = "cuda"
    setup_seed(args.seed)

    ck = torch.load(f"checkpoints/seg_original_{args.arch}_cls{PERSON}.pth",
                    map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=args.arch); m.load_state_dict(ck["state_dict"])

    un = UnlearnDS("train", size=512, crop=True)
    ul = DataLoader(un, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    hp = (LONG_HP if args.long else FAITHFUL_HP)[args.name]
    tag = "long" if args.long else "faithful"
    print(f"[{args.name}] {tag} HP {hp}", flush=True)
    um = run_baseline(args.name, m, ul, device=device, **hp)

    save = f"{args.prefix}bl_{args.name}_s{args.seed}"
    torch.save({"state_dict": um.state_dict()}, f"checkpoints/unlearned_{save}.pth")
    print(f"[{args.name}] saved checkpoints/unlearned_{save}.pth", flush=True)


if __name__ == "__main__":
    main()
