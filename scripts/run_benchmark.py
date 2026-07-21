"""Run the full unlearning benchmark for one (dataset, protocol) setting.

Loads the original model and the retrain oracle, runs every unlearning method,
and reports the metric table (UA/RA/TA/MIA/gap-to-retrain/runtime) plus the
relearning-recovery audit. Results are written as JSON and a markdown table.

Usage:
  PYTHONPATH=. python scripts/run_benchmark.py --protocol random \
      --methods ft ga neggrad+ rl salun ssd scrub
"""
import argparse
import json
import os
import time

import torch

from mu.data import get_splits
from mu.engine import get_model, setup_seed
from mu.methods import run_method
from mu.metrics import full_metrics, gap_to_retrain
from mu.relearn import few_shot_finetune, recovery_score


def load_model(path, arch, num_classes, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = get_model(arch, num_classes, device)
    m.load_state_dict(ckpt["state_dict"])
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--protocol", default="random", choices=["random", "class"])
    p.add_argument("--forget_ratio", type=float, default=0.1)
    p.add_argument("--forget_class", type=int, default=0)
    p.add_argument("--methods", nargs="+",
                   default=["ft", "ga", "neggrad+", "rl", "salun", "ssd", "scrub"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out", default="results")
    p.add_argument("--no_attack", action="store_true")
    args = p.parse_args()

    setup_seed(args.seed)
    device = "cuda"
    s = get_splits(args.dataset, args.data_dir, protocol=args.protocol,
                   forget_ratio=args.forget_ratio, forget_class=args.forget_class,
                   seed=args.seed)

    suffix = f"r{args.forget_ratio}" if args.protocol == "random" else f"c{args.forget_class}"
    orig_path = os.path.join(args.ckpt_dir, f"original_{args.dataset}_{args.arch}.pth")
    retr_path = os.path.join(args.ckpt_dir,
                             f"retrain_{args.dataset}_{args.arch}_{args.protocol}_{suffix}.pth")

    original = load_model(orig_path, args.arch, s.num_classes, device)
    rows = {}

    # reference rows
    rows["original"] = full_metrics(original, s, device)
    rows["original"]["runtime_s"] = 0.0
    retrain_row = None
    if os.path.exists(retr_path):
        retrain = load_model(retr_path, args.arch, s.num_classes, device)
        retrain_row = full_metrics(retrain, s, device)
        retrain_row["runtime_s"] = 0.0
        rows["retrain"] = retrain_row
    else:
        print(f"[warn] retrain oracle not found at {retr_path}; gap_to_retrain skipped")

    # unlearning methods
    for name in args.methods:
        print(f"=== running {name} ===", flush=True)
        t0 = time.time()
        m = run_method(name, original, s, device=device)
        dt = time.time() - t0
        row = full_metrics(m, s, device)
        row["runtime_s"] = round(dt, 1)
        if retrain_row is not None:
            row["gap"] = round(gap_to_retrain(row, retrain_row), 2)
        if not args.no_attack:
            audit = few_shot_finetune(m, s, device=device, k_per_class=10, steps=50)
            row["recover"] = round(100 * recovery_score(audit), 1)
            row["forget_acc_post_attack"] = round(audit["forget_acc_after"], 1)
        rows[name] = row
        print(f"    {name}: " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    os.makedirs(args.out, exist_ok=True)
    stem = f"{args.dataset}_{args.arch}_{args.protocol}_{suffix}"
    with open(os.path.join(args.out, stem + ".json"), "w") as f:
        json.dump(rows, f, indent=2)

    # markdown table
    cols = ["UA", "RA", "TA", "MIA", "gap", "recover", "forget_acc_post_attack", "runtime_s"]
    lines = ["| method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, r in rows.items():
        lines.append("| " + name + " | " +
                     " | ".join(f"{r.get(c, ''):.2f}" if isinstance(r.get(c), float)
                               else str(r.get(c, "")) for c in cols) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, stem + ".md"), "w") as f:
        f.write(f"# {stem}\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
