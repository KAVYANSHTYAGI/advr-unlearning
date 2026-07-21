"""Train the reference models for the unlearning benchmark.

  --mode original : train on the full pool (retain + forget = 45k). One original
                    model serves every forget protocol, so it is trained once.
  --mode retrain  : train on the retain set only. This is the gold-standard
                    "oracle" every unlearning method is measured against, and it
                    depends on the forget protocol (random vs class).

Recipe matches SalUn (SGD, lr 0.1, 182 epochs, decay @ 91/136) so our numbers
are directly comparable to the published table.

Usage:
  PYTHONPATH=. python scripts/train.py --mode original --dataset cifar10
  PYTHONPATH=. python scripts/train.py --mode retrain  --protocol random
  PYTHONPATH=. python scripts/train.py --mode retrain  --protocol class --forget_class 0
"""
import argparse
import os
import time

import torch

from mu.data import get_splits
from mu.engine import evaluate, get_model, setup_seed, train_epochs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["original", "retrain"], required=True)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--protocol", default="random", choices=["random", "class"])
    p.add_argument("--forget_ratio", type=float, default=0.1)
    p.add_argument("--forget_class", type=int, default=0)
    p.add_argument("--epochs", type=int, default=182)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ckpt_dir", default="checkpoints")
    args = p.parse_args()

    setup_seed(args.seed)
    device = "cuda"
    s = get_splits(args.dataset, args.data_dir, protocol=args.protocol,
                   forget_ratio=args.forget_ratio, forget_class=args.forget_class,
                   batch_size=args.batch_size, seed=args.seed)

    if args.mode == "original":
        loader, tag = s.full_loader, f"original_{args.dataset}_{args.arch}"
    else:
        loader = s.retain_loader
        suffix = f"r{args.forget_ratio}" if args.protocol == "random" else f"c{args.forget_class}"
        tag = f"retrain_{args.dataset}_{args.arch}_{args.protocol}_{suffix}"

    model = get_model(args.arch, s.num_classes, device)
    print(f"[{tag}] training on {len(loader.dataset)} images for {args.epochs} epochs")
    t0 = time.time()
    train_epochs(model, loader, args.epochs, args.lr, device=device,
                 milestones=[91, 136], log_prefix=f"[{tag}] ")
    mins = (time.time() - t0) / 60

    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, tag + ".pth")
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, path)

    acc = {
        "forget": evaluate(model, s.forget_eval, device),
        "retain": evaluate(model, s.retain_eval, device),
        "val": evaluate(model, s.val_eval, device),
        "test": evaluate(model, s.test_eval, device),
    }
    print(f"[{tag}] DONE in {mins:.1f} min -> {path}")
    print(f"[{tag}] acc forget {acc['forget']:.2f} retain {acc['retain']:.2f} "
          f"val {acc['val']:.2f} test {acc['test']:.2f}")


if __name__ == "__main__":
    main()
