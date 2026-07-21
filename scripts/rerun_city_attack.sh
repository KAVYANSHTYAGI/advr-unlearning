#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
export DATASET=cityscapes NUM_CLASSES=19 FORGET_CLS=11
echo "[city-attack] rerun on cityscapes models"
UNLEARNED=city_A2_s1,city_ADV_s1,city_ADVR_s1,city_B1_s1 \
  python scripts/run_attack_variance.py 2>&1 | tee results/seg/city_attack_rerun.log
cp results/seg/attack_variance.json results/seg/city_attack_variance.json
echo "[city-attack] DONE"
