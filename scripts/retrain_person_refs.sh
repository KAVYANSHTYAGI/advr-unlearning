#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True   # FORGET_CLS=12 default (person)
echo "[refs] retrain person deeplab original+oracle (cls12)"
python scripts/train_seg.py --mode original --epochs 60 --bs 6 --size 512 \
  > results/seg/person_train_original.log 2>&1
python scripts/train_seg.py --mode oracle --epochs 60 --bs 6 --size 512 \
  > results/seg/person_train_oracle.log 2>&1
echo "[refs] DONE"
