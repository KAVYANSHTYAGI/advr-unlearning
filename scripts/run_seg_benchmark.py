"""First segmentation-unlearning experiment: do the classification methods port?

Loads the original DeepLabV3 + oracle, runs each ported method, and reports:
  person_IoU (want -> low)        retain_mIoU (want -> preserved, ~ original)
  residual_person / blend_fail / silhouette_IoU (leakage; want -> low)
  recover_pts (relearn audit; want -> low)        runtime

Compares everything to the oracle (gold standard). Writes results/seg/*.{json,md}.

Usage:
  PYTHONPATH=. python scripts/run_seg_benchmark.py --methods ft ga neggrad+ rl salun ssd scrub
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from seg.audit import leakage_metrics, relearn_audit
from seg.data import ADE20KSeg, ADE20KUnlearn
from seg.engine import seg_metrics
from seg.methods import run_method
from seg.model import get_seg_model


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device)
    m.load_state_dict(ckpt["state_dict"])
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["ft", "ga", "neggrad+", "rl", "salun", "ssd", "scrub"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out", default="results/seg")
    args = p.parse_args()
    device = "cuda"

    # loaders
    val = ADE20KSeg("val", "orig", size=args.size, crop=False)
    vl = DataLoader(val, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    un_train = ADE20KUnlearn("train", size=args.size, crop=True)
    ul = DataLoader(un_train, batch_size=args.bs, shuffle=True, num_workers=6,
                    pin_memory=True, drop_last=True)
    un_val = ADE20KUnlearn("val", size=args.size, crop=False)
    uvl = DataLoader(un_val, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    original = load(os.path.join(args.ckpt_dir, "seg_original_deeplabv3r50.pth"), device)
    rows = {}

    def evaluate_model(m):
        r = seg_metrics(m, vl, device)
        r.update(leakage_metrics(m, uvl, device))
        return r

    rows["original"] = evaluate_model(original)
    oracle_path = os.path.join(args.ckpt_dir, "seg_oracle_deeplabv3r50.pth")
    if os.path.exists(oracle_path):
        rows["oracle"] = evaluate_model(load(oracle_path, device))

    for name in args.methods:
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        m = run_method(name, original, ul, device=device)
        dt = time.time() - t0
        r = evaluate_model(m)
        r["runtime_s"] = round(dt, 1)
        audit = relearn_audit(m, ul, device=device, steps=50, val_loader=vl)
        r["recover_pts"] = audit.get("recover_pts")
        r["person_IoU_post_attack"] = audit.get("person_IoU_after")
        rows[name] = r
        print(f"  {name}: " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "benchmark.json"), "w") as f:
        json.dump(rows, f, indent=2)
    cols = ["person_IoU", "retain_mIoU", "residual_person", "blend_fail",
            "silhouette_IoU", "recover_pts", "person_IoU_post_attack", "runtime_s"]
    lines = ["| method | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, r in rows.items():
        lines.append("| " + name + " | " +
                     " | ".join(f"{r.get(c):.2f}" if isinstance(r.get(c), float)
                               else str(r.get(c, "")) for c in cols) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "benchmark.md"), "w") as f:
        f.write("# Segmentation person-unlearning (ADE20K, DeepLabV3-R50)\n\n" + table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
