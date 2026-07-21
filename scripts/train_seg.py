"""Train the reference segmentation models for person-unlearning.

  --mode original : DeepLabV3-R50 on ADE20K with original labels.
  --mode oracle   : same, but person pixels dominant-neighbor relabeled
                    (target="merged"). This is the gold standard a perfect
                    person-unlearner should match.

Usage:
  PYTHONPATH=. python scripts/train_seg.py --mode original --epochs 24 --bs 4
  PYTHONPATH=. python scripts/train_seg.py --mode oracle   --epochs 24 --bs 4
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from seg.data import SegDS
from seg.engine import seg_metrics, train_seg
from seg.model import get_seg_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["original", "oracle"], required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--arch", default="deeplabv3r50")
    p.add_argument("--suffix", default="",
                   help="appended to the checkpoint tag, e.g. '_mode' for the "
                        "oracle relabel-scheme ablation (set RELABEL env to match)")
    args = p.parse_args()

    target = "orig" if args.mode == "original" else "merged"
    tr = SegDS("train", target=target, size=args.size)
    va = SegDS("val", target="orig", size=args.size, crop=False)  # eval vs ORIGINAL gt
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=24,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    from seg.data import PERSON as FORGET_CLS
    model = get_seg_model(device="cuda", arch=args.arch)
    tag = f"seg_{args.mode}_{args.arch}_cls{FORGET_CLS}{args.suffix}"
    print(f"[{tag}] {len(tr)} train imgs, {args.epochs} epochs, bs{args.bs} res{args.size}")
    t0 = time.time()
    train_seg(model, tl, args.epochs, args.lr, device="cuda",
              log_prefix=f"[{tag}] ", val_loader=vl, log_every=2)
    hrs = (time.time() - t0) / 3600

    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, tag + ".pth")
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, path)
    m = seg_metrics(model, vl, "cuda")
    print(f"[{tag}] DONE {hrs:.1f}h -> {path}")
    print(f"[{tag}] val person_IoU {m['person_IoU']:.1f} retain_mIoU {m['retain_mIoU']:.1f} mIoU {m['mIoU']:.1f}")


if __name__ == "__main__":
    main()
