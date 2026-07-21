# project1 — Applied Vision Machine Unlearning (WACV-first)

A reproducible benchmark harness for **approximate machine unlearning** on image
classifiers, plus a **relearning/recovery audit** — the axis where our paper lives.

**Thesis.** Vision unlearning methods (SalUn, SCRUB, SSD, NegGrad+) report
"forgetting" via forget-accuracy + MIA, but forgotten knowledge is recoverable
with a few samples (arXiv 2506.01318; ICLR 2025 *Machine Unlearning Fails to
Remove…*). Almost all relearning-robustness work is LLM-centric — **the vision
side is open.** We build the trusted baseline first, then quantify which methods
survive recovery, then propose a robust fix.

## Layout
```
mu/            harness package
  data.py        deterministic forget/retain/val/test splits (random & class)
  engine.py      model factory + train/eval loops
  resnet.py      CIFAR ResNet-18/34/50 (vendored from SalUn; built-in normalize)
  methods.py     ft, ga, neggrad+, rl, salun, ssd, scrub
  metrics.py     UA/RA/TA, SVC-MIA, gap-to-retrain
  svc_mia.py     SVC membership-inference (vendored from SalUn)
  relearn.py     few-shot finetune + prototypical recovery audit
scripts/
  train.py           train original model / retrain oracle
  run_benchmark.py   run all methods + metrics + attack -> results/*.{json,md}
vendor/          upstream repos (SalUn, SSD) for reference
data/            CIFAR-10 / CIFAR-100
checkpoints/     trained models
results/         metric tables + logs
```

## Reproduce
```bash
conda activate myenv
cd machine_unlearning/project1

# 1. reference models
PYTHONPATH=. python scripts/train.py --mode original --dataset cifar10
PYTHONPATH=. python scripts/train.py --mode retrain  --protocol random

# 2. benchmark (random-10% forgetting) + recovery audit
PYTHONPATH=. python scripts/run_benchmark.py --protocol random
```

## Metrics
- **UA** = 100 − forget-acc (higher = more forgotten)
- **RA / TA** = retain / test accuracy (higher = better)
- **MIA** = SVC confidence-attack success on the forget set
- **gap** = mean |metric − retrain-oracle| over {UA,RA,TA,MIA} (lower = better)
- **recover** = % of forgotten accuracy restored by a 10-shot/class finetune
  attack (lower = more robust) — the audit dimension.

## Compute
RTX 5080 local for CIFAR (~18 min/original). RTX 5090 over SSH for Tiny-ImageNet
/ ViT scale-up.
