#!/usr/bin/env bash
# Stage-1 pipeline for one experiment axis: train reference models (if missing),
# train+eval our 4 methods (A2, ADV, ADVR, B1) at seed 1 (if missing), run the
# leakage attack. Idempotent — safe to re-run, skips anything already on disk.
#
# Replaces: run_car_experiment.sh, run_cityscapes_experiment.sh,
#           run_segformer_experiment.sh, run_car_methods.sh
#
# Usage: bash scripts/run_axis_pipeline.sh --axis person_ade20k
#        bash scripts/run_axis_pipeline.sh --axis car_ade20k
#        bash scripts/run_axis_pipeline.sh --axis person_cityscapes
#        bash scripts/run_axis_pipeline.sh --axis person_segformer
# See configs/axes.sh for the axis table (dataset/class/backbone/prefix per axis).
set -eo pipefail
cd "$(dirname "$0")/.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
source configs/axes.sh

AXIS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --axis) AXIS="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done
[ -n "$AXIS" ] || { echo "usage: $0 --axis <name> (see configs/axes.sh)" >&2; exit 1; }
axis_lookup "$AXIS"

export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p results/seg
LOG="results/seg/${PREFIX}exp.log"
echo "[axis=$AXIS] DATASET=$DATASET FORGET_CLS=$FORGET_CLS NUM_CLASSES=$NUM_CLASSES ARCH=$ARCH PREFIX='$PREFIX'" | tee "$LOG"

ORIG_CK="checkpoints/seg_original_${ARCH}_cls${FORGET_CLS}.pth"
ORACLE_CK="checkpoints/seg_oracle_${ARCH}_cls${FORGET_CLS}.pth"

train_ref () {  # mode -> checkpoint path already resolved by train_seg.py's naming
  local MODE=$1
  echo "[axis=$AXIS] train $MODE (60ep, bs6->bs4 fallback)" | tee -a "$LOG"
  python scripts/train_seg.py --mode "$MODE" --arch "$ARCH" --epochs 60 --bs 6 --size 512 \
      > "results/seg/${PREFIX}train_${MODE}.log" 2>&1 || \
  python scripts/train_seg.py --mode "$MODE" --arch "$ARCH" --epochs 60 --bs 4 --size 512 \
      > "results/seg/${PREFIX}train_${MODE}.log" 2>&1
  grep -E "DONE|retain_mIoU|person_IoU" "results/seg/${PREFIX}train_${MODE}.log" | tail -3 | tee -a "$LOG"
}

[ -f "$ORIG_CK" ] && echo "[axis=$AXIS] original exists, skip" | tee -a "$LOG" || train_ref original
[ -f "$ORACLE_CK" ] && echo "[axis=$AXIS] oracle exists, skip" | tee -a "$LOG" || train_ref oracle

run_method () {  # candidate_name save_suffix epochs lr extra_args...
  local NAME=$1 SUF=$2; shift 2
  local SAVE="${PREFIX}${SUF}"
  if [ -f "checkpoints/unlearned_${SAVE}.pth" ]; then
    echo "[axis=$AXIS] $SAVE exists, skip training" | tee -a "$LOG"
  else
    echo "[axis=$AXIS] train $NAME -> $SAVE" | tee -a "$LOG"
    python scripts/test_candidate.py --name "$NAME" --arch "$ARCH" --bs 4 --workers 24 \
        --save_as "$SAVE" --no_eval --seed 1 "$@" 2>&1 | tee -a "$LOG"
  fi
  python scripts/eval_ckpt.py --name "$SAVE" --arch "$ARCH" | tee -a "$LOG"
}

run_method A2_ft_anchored  A2_s1   --epochs 3 --lr 1e-3 --lam_r 1.0
run_method ADV_silhouette  ADV_s1  --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0
run_method ADVR            ADVR_s1 --epochs 8 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0
run_method B1_proto_collapse B1_s1 --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0

echo "[axis=$AXIS] two-surface leakage attack (3 trials)" | tee -a "$LOG"
OUT_JSON="results/seg/${PREFIX}leakage.json"
UNLEARNED="${PREFIX}A2_s1,${PREFIX}ADV_s1,${PREFIX}ADVR_s1,${PREFIX}B1_s1" \
  OUT_JSON="$OUT_JSON" \
  python scripts/run_attack_variance.py 2>&1 | tee -a "$LOG"
echo "[axis=$AXIS] DONE -> $OUT_JSON" | tee -a "$LOG"
