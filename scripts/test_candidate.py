"""Test one candidate method end-to-end at full scale and append to the tracker.

Usage:
  PYTHONPATH=. python scripts/test_candidate.py --name A2_ft_anchored --lam_r 1.0 --epochs 3
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader as _DL

from seg.audit import leakage_metrics, relearn_audit
from seg.candidates import run_candidate
from seg.data import SegDS, UnlearnDS
from seg.engine import seg_metrics
from seg.model import get_seg_model

RESULTS = "results/seg/candidates.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam_r", type=float, default=1.0)
    p.add_argument("--lam_c", type=float, default=None,
                   help="default: the candidate's own default (ADVR=0.0, ADVF/"
                        "ADV_full/B1=1.0) unless explicitly set here")
    p.add_argument("--lam_f", type=float, default=0.0)
    p.add_argument("--lam_adv", type=float, default=1.0)
    p.add_argument("--freeze_bn", action="store_true")
    p.add_argument("--blend", action="store_true")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--tag", default="")
    p.add_argument("--save_as", default="")
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--arch", default="deeplabv3r50")
    args = p.parse_args()
    device = "cuda"
    from seg.engine import setup_seed
    setup_seed(args.seed)

    from seg.data import PERSON as FORGET_CLS
    orig_ck = f"checkpoints/seg_original_{args.arch}_cls{FORGET_CLS}.pth"
    ck = torch.load(orig_ck, map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=args.arch); m.load_state_dict(ck["state_dict"])

    un = UnlearnDS("train", size=args.size, crop=True)
    ul = DataLoader(un, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)
    uv = UnlearnDS("val", size=args.size, crop=False)
    uvl = DataLoader(uv, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)
    va = SegDS("val", "orig", size=args.size, crop=False)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    kw = dict(epochs=args.epochs, lr=args.lr, lam_r=args.lam_r,
              lam_f=args.lam_f, lam_adv=args.lam_adv, freeze_bn=args.freeze_bn,
              blend=args.blend)
    if args.lam_c is not None:                   # else: the candidate's own default
        kw["lam_c"] = args.lam_c
    if args.name in ("ADV_full", "ADVR", "ADVF"):  # retain replay on the full distribution
        rep = SegDS("train", "orig", size=args.size, crop=True)
        kw["replay_loader"] = _DL(rep, batch_size=args.bs, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"[{args.name}{args.tag}] running with {kw}", flush=True)
    t0 = time.time()
    um = run_candidate(args.name, m, ul, device=device, **kw)
    dt = time.time() - t0
    if args.save_as:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"state_dict": um.state_dict()}, f"checkpoints/unlearned_{args.save_as}.pth")
        print(f"  saved checkpoints/unlearned_{args.save_as}.pth")

    if args.no_eval:
        print(f"[{args.name}{args.tag}] trained+saved in {dt:.0f}s (eval skipped)")
        return

    row = seg_metrics(um, vl, device)
    row.update(leakage_metrics(um, uvl, device))
    audit = relearn_audit(um, ul, device=device, steps=50, val_loader=vl)
    row["recover_pts"] = audit.get("recover_pts")
    row["person_IoU_post_attack"] = audit.get("person_IoU_after")
    row["runtime_s"] = round(dt, 1)
    row["config"] = {**kw, "name": args.name, "tag": args.tag}

    os.makedirs("results/seg", exist_ok=True)
    allrows = []
    if os.path.exists(RESULTS):
        allrows = json.load(open(RESULTS))
    allrows.append({args.name + args.tag: row})
    json.dump(allrows, open(RESULTS, "w"), indent=2)

    print(f"[{args.name}{args.tag}] DONE {dt:.0f}s")
    print(f"  person_IoU {row['person_IoU']:.1f} | retain_mIoU {row['retain_mIoU']:.1f} "
          f"| blend_fail {row['blend_fail']:.1f} | silhouette_IoU {row['silhouette_IoU']:.1f} "
          f"| recover_pts {row['recover_pts']} | {dt:.0f}s")
    print("  TARGET oracle: person 0 / retain 34.5 / blend 44 / silhouette 11 / recover ~0")


if __name__ == "__main__":
    main()
