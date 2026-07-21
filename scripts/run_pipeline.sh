#!/usr/bin/env bash
# Classification (mu/) comparator pipeline: train original -> train retrain
# oracle -> run benchmark. Segmentation (the paper headline) uses
# scripts/run_axis_pipeline.sh instead — see docs/REPRODUCING.md.
set -e
cd "$(dirname "$0")/.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
export PYTHONPATH=.

ORIG=checkpoints/original_cifar10_resnet18.pth
if [ ! -f "$ORIG" ]; then
  echo "[pipeline] training original"
  python scripts/train.py --mode original --dataset cifar10 \
      2>&1 | grep -v "normalize layer" | tee results/train_original.log
fi
echo "[pipeline] training retrain oracle (random-10%)"

python scripts/train.py --mode retrain --protocol random --epochs 182 \
    2>&1 | grep -v "normalize layer" | tee results/train_retrain_random.log

echo "[pipeline] running benchmark (random-10%)"
python scripts/run_benchmark.py --protocol random \
    2>&1 | grep -v "normalize layer" | tee results/benchmark_random.log
echo "[pipeline] DONE"
