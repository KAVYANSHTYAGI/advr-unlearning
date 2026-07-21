#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[competent] original 60ep, scale-crop aug, bs6"
python scripts/train_seg.py --mode original --epochs 60 --bs 6 --size 512 > results/seg/train_original_v2.log 2>&1 \
  || python scripts/train_seg.py --mode original --epochs 60 --bs 4 --size 512 > results/seg/train_original_v2.log 2>&1
echo "[competent] oracle 60ep"
python scripts/train_seg.py --mode oracle --epochs 60 --bs 6 --size 512 > results/seg/train_oracle_v2.log 2>&1 \
  || python scripts/train_seg.py --mode oracle --epochs 60 --bs 4 --size 512 > results/seg/train_oracle_v2.log 2>&1
echo "[competent] DONE"
