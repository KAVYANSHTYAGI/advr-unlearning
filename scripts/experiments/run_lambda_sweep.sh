#!/usr/bin/env bash
# lambda-sweep ablation for ADVR (lam_r x lam_adv) -- supplementary to the
# headline table, not required to reproduce it. See docs/REPRODUCING.md.
set -e
cd "$(dirname "$0")/../.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
EVLOG=results/seg/lambda_sweep.txt; : > $EVLOG

for LAM_R in 0.5 1.0 2.0 4.0; do
for LAM_A in 0.5 1.0; do
  TAG="r${LAM_R}_a${LAM_A}"
  SAVE="ADVR_${TAG}"
  if [ -f "checkpoints/unlearned_${SAVE}.pth" ] && grep -q "^${SAVE}:" "$EVLOG" 2>/dev/null; then
    echo "[skip] $SAVE"; continue
  fi
  echo "[sweep] lam_r=$LAM_R lam_adv=$LAM_A -> $SAVE"
  python scripts/test_candidate.py --name ADV_full --bs 4 --workers 12 \
    --epochs 8 --lr 1e-3 --lam_r "$LAM_R" --lam_adv "$LAM_A" --lam_c 0.0 \
    --save_as "$SAVE" --no_eval --seed 1 --tag "_${TAG}"
  python scripts/eval_ckpt.py --name "$SAVE" | tee -a $EVLOG
done; done

echo "=== SWEEP DONE ===" | tee -a $EVLOG
echo "=== Pareto check (retain vs leakage) ===" | tee -a $EVLOG
cat $EVLOG
