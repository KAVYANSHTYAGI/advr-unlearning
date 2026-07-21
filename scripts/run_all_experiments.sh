#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[queue] waiting for lambda sweep to finish..."
until grep -q "=== SWEEP DONE ===" results/seg/lambda_sweep.txt 2>/dev/null; do sleep 60; done
echo "[queue] sweep done; launching car experiment (B)"
bash scripts/run_car_experiment.sh

echo "[queue] car done; launching SegFormer experiment (C)"
bash scripts/run_segformer_experiment.sh

echo "[queue] ALL EXPERIMENTS DONE"
