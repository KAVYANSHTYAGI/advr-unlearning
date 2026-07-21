#!/usr/bin/env bash
# Produces the 3-seed headline table (A2/ADV/ADVR retain_mIoU + recover_pts,
# person_ade20k axis) reported in docs/EXPERIMENT_LOG.md 2026-06-30.
set -e
cd "$(dirname "$0")/.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
EVLOG=results/seg/multiseed_eval.txt
[ -f "$EVLOG" ] || : > $EVLOG

run_method () {  # name save_tag extra_args...
  local NAME=$1; shift; local SAVE=$1; shift
  if [ -f "checkpoints/unlearned_${SAVE}.pth" ] && grep -q "^${SAVE}:" "$EVLOG" 2>/dev/null; then
    echo "[skip] ${SAVE} already done"; return 0
  fi
  python scripts/test_candidate.py --name "$NAME" --bs 4 --workers 16 --save_as "$SAVE" --no_eval --seed "$SEED" "$@"
  python scripts/eval_ckpt.py --name "$SAVE" | tee -a $EVLOG
}

for SEED in 1 2 3; do
  echo "############ SEED $SEED ############" | tee -a $EVLOG
  run_method A2_ft_anchored      "A2_s${SEED}"   --epochs 3 --lr 1e-3 --lam_r 1.0
  run_method ADV_silhouette      "ADV_s${SEED}"  --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0
  run_method ADV_full            "ADVR_s${SEED}" --epochs 8 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0
  if [ "$SEED" = "1" ]; then
    run_method B1_proto_collapse "B1_s1"  --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0
  fi
done

echo "==== two-surface learned attack over all checkpoints ====" | tee -a $EVLOG
UNLEARNED="A2_s1,A2_s2,A2_s3,ADV_s1,ADV_s2,ADV_s3,ADVR_s1,ADVR_s2,ADVR_s3,B1_s1" \
  python scripts/run_attack_variance.py | tee -a $EVLOG
echo "ALL DONE" | tee -a $EVLOG
