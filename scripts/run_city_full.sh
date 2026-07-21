#!/usr/bin/env bash
# Full Cityscapes baseline comparison pipeline.
# Runs 8 faithful baselines + N=3 leakage attack over all city_* methods.
# Env: DATASET=cityscapes  FORGET_CLS=11  NUM_CLASSES=19  ARCH=deeplabv3r50
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH=deeplabv3r50
CLS=11          # Cityscapes trainId for person
NC=19
BS=4
SEED=1

export DATASET=cityscapes FORGET_CLS=$CLS NUM_CLASSES=$NC ARCH=$ARCH
LOG=results/seg/city_baselines.log

echo "=== Cityscapes full baseline run ===" | tee $LOG
date | tee -a $LOG

# ---- 8 baselines (sequential, faithful HPs) ----------------------------
for NAME in FT NegGrad "NegGrad+" RL SalUn SSD SCRUB DELETE; do
    echo "[cmp] baseline $NAME" | tee -a $LOG
    PYTHONPATH=. python scripts/run_baselines.py \
        --name "$NAME" --seed $SEED --arch $ARCH \
        --bs $BS --workers 24 --prefix city_ --long \
        2>&1 | tee -a $LOG
done

echo "[cmp] baselines done, running leakage attack (N=3)" | tee -a $LOG

# ---- leakage attack over all city_* methods ----------------------------
# our 4 methods (already trained) + 8 new baselines
UNLEARNED="city_A2_s1,city_ADV_s1,city_ADVR_s1,city_B1_s1,\
city_bl_FT_s1,city_bl_NegGrad_s1,city_bl_NegGrad+_s1,\
city_bl_RL_s1,city_bl_SalUn_s1,city_bl_SSD_s1,\
city_bl_SCRUB_s1,city_bl_DELETE_s1"

UNLEARNED="$UNLEARNED" \
OUT_JSON="results/seg/city_leakage.json" \
PYTHONPATH=. python scripts/run_attack_variance.py \
    2>&1 | tee -a $LOG

echo "=== ALL DONE ===" | tee -a $LOG
