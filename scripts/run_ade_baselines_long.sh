#!/usr/bin/env bash
# ADE20K baselines re-run with extended epochs for segmentation convergence.
# Overwrites the short-run checkpoints (unlearned_bl_*_s1.pth).
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH=deeplabv3r50
CLS=12     # ADE20K person
NC=150
BS=4
SEED=1

export DATASET=ade20k FORGET_CLS=$CLS NUM_CLASSES=$NC ARCH=$ARCH
LOG=results/seg/ade_baselines_long.log

echo "=== ADE20K baselines long-epoch re-run ===" | tee $LOG
date | tee -a $LOG

for NAME in FT NegGrad "NegGrad+" RL SalUn SSD SCRUB DELETE; do
    echo "[long] $NAME" | tee -a $LOG
    PYTHONPATH=. python scripts/run_baselines.py \
        --name "$NAME" --seed $SEED --arch $ARCH \
        --bs $BS --workers 24 --long \
        2>&1 | tee -a $LOG
done

echo "[long] baselines done, running leakage attack (N=3)" | tee -a $LOG

UNLEARNED="ours_A2_s1,ours_ADV_s1,ours_ADVR_s1,ours_B1_s1,\
bl_FT_s1,bl_NegGrad_s1,bl_NegGrad+_s1,\
bl_RL_s1,bl_SalUn_s1,bl_SSD_s1,\
bl_SCRUB_s1,bl_DELETE_s1"

UNLEARNED="$UNLEARNED" \
OUT_JSON="results/seg/ade_leakage_long.json" \
PYTHONPATH=. python scripts/run_attack_variance.py \
    2>&1 | tee -a $LOG

echo "=== ALL DONE ===" | tee -a $LOG
