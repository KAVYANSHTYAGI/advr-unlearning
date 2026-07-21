#!/usr/bin/env bash
# Car-class unlearning on ADE20K — 3-seed ADVR + DELETE (long), then leakage attack.
# Reference checkpoints (original + oracle cls20) already exist.
# Seed 1 of our methods already exists; this trains s2/s3 + all 3 seeds of DELETE.
set -eo pipefail
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv
set -u

export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
export DATASET=ade20k FORGET_CLS=20 NUM_CLASSES=150 ARCH=deeplabv3r50
LOG=results/seg/car_cvpr.log; mkdir -p results/seg

echo "=== Car class CVPR run (ADE20K, FORGET_CLS=20) ===" | tee $LOG
date | tee -a $LOG

# ---- ADVR seeds 2 + 3 (seed 1 already in checkpoints/unlearned_car_ADVR_s1.pth) ----
for SEED in 2 3; do
    echo "[car] ADVR seed $SEED" | tee -a $LOG
    python scripts/test_candidate.py --name ADV_full --arch deeplabv3r50 \
        --bs 4 --workers 24 --size 512 \
        --epochs 20 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0 \
        --save_as car_ADVR_s${SEED} --no_eval --seed $SEED \
        2>&1 | tee -a $LOG
done

# ---- DELETE trained seeds 1 + 2 + 3 (long: 20 epochs) ----
for SEED in 1 2 3; do
    echo "[car] DELETE seed $SEED" | tee -a $LOG
    python scripts/run_baselines.py \
        --name DELETE --seed $SEED --arch deeplabv3r50 \
        --bs 4 --workers 24 --prefix car_ --long \
        2>&1 | tee -a $LOG
done

# ---- leakage attack: oracle + DELETE_logitmask (prior art) + our ADVR s1/2/3 + DELETE trained s1/2/3 ----
echo "[car] leakage attack (N=3 trials)" | tee -a $LOG
UNLEARNED="car_ADVR_s1,car_ADVR_s2,car_ADVR_s3,car_bl_DELETE_s1,car_bl_DELETE_s2,car_bl_DELETE_s3" \
OUT_JSON="results/seg/car_leakage.json" \
python scripts/run_attack_variance.py 2>&1 | tee -a $LOG

echo "=== ALL DONE ===" | tee -a $LOG
