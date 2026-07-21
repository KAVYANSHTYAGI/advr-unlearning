#!/usr/bin/env bash
# Drives the rest of the baseline once the original model finishes training:
#   wait for original checkpoint -> train retrain oracle -> run benchmark.
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=.

ORIG=checkpoints/original_cifar10_resnet18.pth
echo "[pipeline] waiting for original checkpoint: $ORIG"
until grep -q "DONE in" results/train_original.log 2>/dev/null; do sleep 15; done
echo "[pipeline] original done; training retrain oracle (random-10%)"

python scripts/train.py --mode retrain --protocol random --epochs 182 \
    2>&1 | grep -v "normalize layer" | tee results/train_retrain_random.log

echo "[pipeline] running benchmark (random-10%)"
python scripts/run_benchmark.py --protocol random \
    2>&1 | grep -v "normalize layer" | tee results/benchmark_random.log
echo "[pipeline] DONE"
