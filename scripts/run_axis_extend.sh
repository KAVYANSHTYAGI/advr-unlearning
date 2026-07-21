#!/usr/bin/env bash
# Stage-2 extension for one axis: adds extra ADVR seeds and/or the full faithful-
# baseline suite (long HPs) on top of an axis already run through
# run_axis_pipeline.sh, then re-runs the leakage attack over the combined set.
# Requires reference + our-methods-seed-1 checkpoints to already exist (run
# run_axis_pipeline.sh first) — errors clearly instead of polling for them.
#
# Replaces: run_car_cvpr.sh, run_sf_cvpr.sh, run_city_full.sh,
#           run_baseline_comparison.sh, run_ade_baselines_long.sh
#
# Usage:
#   # DELETE baseline across 3 seeds + ADVR seeds 2-3 (matches old *_cvpr.sh scripts)
#   bash scripts/run_axis_extend.sh --axis car_ade20k \
#       --baselines DELETE --baseline-seeds 1,2,3 --extra-adv-seeds 2,3
#
#   # full 8-baseline faithful comparison suite, seed 1 (matches run_*_full.sh /
#   # run_baseline_comparison.sh)
#   bash scripts/run_axis_extend.sh --axis person_cityscapes \
#       --baselines FT,NegGrad,NegGrad+,RL,SalUn,SSD,SCRUB,DELETE
set -eo pipefail
cd "$(dirname "$0")/.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
source configs/axes.sh

AXIS="" BASELINES="" BASELINE_SEEDS="1" EXTRA_ADV_SEEDS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --axis) AXIS="$2"; shift 2 ;;
    --baselines) BASELINES="$2"; shift 2 ;;
    --baseline-seeds) BASELINE_SEEDS="$2"; shift 2 ;;
    --extra-adv-seeds) EXTRA_ADV_SEEDS="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done
[ -n "$AXIS" ] || { echo "usage: $0 --axis <name> [--baselines FT,...] [--baseline-seeds 1,2,3] [--extra-adv-seeds 2,3]" >&2; exit 1; }
axis_lookup "$AXIS"

export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p results/seg
LOG="results/seg/${PREFIX}extend.log"
echo "[axis=$AXIS/extend] DATASET=$DATASET FORGET_CLS=$FORGET_CLS ARCH=$ARCH PREFIX='$PREFIX'" | tee "$LOG"

ORIG_CK="checkpoints/seg_original_${ARCH}_cls${FORGET_CLS}.pth"
ORACLE_CK="checkpoints/seg_oracle_${ARCH}_cls${FORGET_CLS}.pth"
STAGE1_CK="checkpoints/unlearned_${PREFIX}ADVR_s1.pth"
for ck in "$ORIG_CK" "$ORACLE_CK" "$STAGE1_CK"; do
  [ -f "$ck" ] || { echo "missing $ck — run: bash scripts/run_axis_pipeline.sh --axis $AXIS first" >&2; exit 1; }
done

UNLEARNED="${PREFIX}A2_s1,${PREFIX}ADV_s1,${PREFIX}ADVR_s1,${PREFIX}B1_s1"

# ---- extra ADVR seeds (20ep, matches the *_cvpr.sh recipe) ----------------
if [ -n "$EXTRA_ADV_SEEDS" ]; then
  IFS=',' read -ra SEEDS <<< "$EXTRA_ADV_SEEDS"
  for SEED in "${SEEDS[@]}"; do
    SAVE="${PREFIX}ADVR_s${SEED}"
    if [ -f "checkpoints/unlearned_${SAVE}.pth" ]; then
      echo "[extend] $SAVE exists, skip" | tee -a "$LOG"
    else
      echo "[extend] ADVR seed $SEED -> $SAVE" | tee -a "$LOG"
      python scripts/test_candidate.py --name ADVR --arch "$ARCH" --bs 4 --workers 24 \
          --size 512 --epochs 20 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0 \
          --save_as "$SAVE" --no_eval --seed "$SEED" 2>&1 | tee -a "$LOG"
    fi
    UNLEARNED="${UNLEARNED},${SAVE}"
  done
fi

# ---- faithful baselines (long HPs), one or more seeds ----------------------
if [ -n "$BASELINES" ]; then
  IFS=',' read -ra BL <<< "$BASELINES"
  IFS=',' read -ra BSEEDS <<< "$BASELINE_SEEDS"
  for NAME in "${BL[@]}"; do
    for SEED in "${BSEEDS[@]}"; do
      SAVE="${PREFIX}bl_${NAME}_s${SEED}"
      if [ -f "checkpoints/unlearned_${SAVE}.pth" ]; then
        echo "[extend] $SAVE exists, skip" | tee -a "$LOG"
      else
        echo "[extend] baseline $NAME seed $SEED -> $SAVE" | tee -a "$LOG"
        python scripts/run_baselines.py --name "$NAME" --seed "$SEED" --arch "$ARCH" \
            --bs 4 --workers 24 --prefix "$PREFIX" --long 2>&1 | tee -a "$LOG"
      fi
      UNLEARNED="${UNLEARNED},${SAVE}"
    done
  done
fi

echo "[extend] leakage attack over: $UNLEARNED" | tee -a "$LOG"
OUT_JSON="results/seg/${PREFIX}leakage_full.json"
UNLEARNED="$UNLEARNED" OUT_JSON="$OUT_JSON" \
  python scripts/run_attack_variance.py 2>&1 | tee -a "$LOG"
echo "[extend] DONE -> $OUT_JSON" | tee -a "$LOG"
