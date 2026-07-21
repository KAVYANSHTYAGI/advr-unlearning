#!/usr/bin/env bash
# Eval retain_mIoU / person_IoU / recover_pts for all Cityscapes long-epoch checkpoints.
set -eo pipefail
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv
set -u

export DATASET=cityscapes FORGET_CLS=11 NUM_CLASSES=19
ARCH=deeplabv3r50
LOG=results/seg/city_eval.log

echo "=== Cityscapes quality eval ===" | tee $LOG
date | tee -a $LOG

NAMES=(
    city_A2_s1
    city_ADV_s1
    city_ADVR_s1
    city_B1_s1
    city_bl_FT_s1
    city_bl_NegGrad_s1
    "city_bl_NegGrad+_s1"
    city_bl_RL_s1
    city_bl_SalUn_s1
    city_bl_SSD_s1
    city_bl_SCRUB_s1
    city_bl_DELETE_s1
)

for NAME in "${NAMES[@]}"; do
    echo "[eval] $NAME" | tee -a $LOG
    PYTHONPATH=. python scripts/eval_ckpt.py --name "$NAME" --arch $ARCH \
        2>&1 | tee -a $LOG
done

echo "=== DONE ===" | tee -a $LOG
