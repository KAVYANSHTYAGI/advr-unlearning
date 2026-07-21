#!/usr/bin/env bash
# SegFormer-B2 backbone — 3-seed ADVR + DELETE (long) on ADE20K person, then leakage attack.
# Reference checkpoints (original + oracle segformer_b2 cls12) already exist.
# Seed 1 of ADVR already exists; this trains s2/s3 + all 3 seeds of DELETE.
set -eo pipefail
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv
set -u

export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
export DATASET=ade20k FORGET_CLS=12 NUM_CLASSES=150 ARCH=segformer_b2
LOG=results/seg/sf_cvpr.log; mkdir -p results/seg

echo "=== SegFormer-B2 CVPR run (ADE20K person, FORGET_CLS=12) ===" | tee $LOG
date | tee -a $LOG

# ---- ADVR seeds 2 + 3 (seed 1 already in checkpoints/unlearned_sf_ADVR_s1.pth) ----
for SEED in 2 3; do
    echo "[sf] ADVR seed $SEED" | tee -a $LOG
    python scripts/test_candidate.py --name ADV_full --arch segformer_b2 \
        --bs 4 --workers 24 --size 512 \
        --epochs 20 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0 \
        --save_as sf_ADVR_s${SEED} --no_eval --seed $SEED \
        2>&1 | tee -a $LOG
done

# ---- DELETE trained seeds 1 + 2 + 3 (long: 20 epochs, segformer backbone) ----
for SEED in 1 2 3; do
    echo "[sf] DELETE seed $SEED" | tee -a $LOG
    python scripts/run_baselines.py \
        --name DELETE --seed $SEED --arch segformer_b2 \
        --bs 4 --workers 24 --prefix sf_ --long \
        2>&1 | tee -a $LOG
done

# ---- leakage attack over sf_ADVR s1/2/3 + sf_DELETE trained s1/2/3 ----
echo "[sf] leakage attack (N=3 trials)" | tee -a $LOG
UNLEARNED="sf_ADVR_s1,sf_ADVR_s2,sf_ADVR_s3,sf_bl_DELETE_s1,sf_bl_DELETE_s2,sf_bl_DELETE_s3" \
OUT_JSON="results/seg/sf_leakage.json" \
python scripts/run_attack_variance.py 2>&1 | tee -a $LOG

echo "=== ALL DONE ===" | tee -a $LOG
