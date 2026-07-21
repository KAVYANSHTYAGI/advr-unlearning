#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
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
python scripts/run_attack_full.py | tee -a $EVLOG
echo "ALL DONE" | tee -a $EVLOG
