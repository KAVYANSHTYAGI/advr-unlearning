#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[advreplay] ADV + retain-replay (lam_c=0), longer+bigger, OOM-safe batch"
for BS in 8 6 4; do
  echo "[advreplay] trying batch $BS"
  if python scripts/test_candidate.py --name ADV_full --bs $BS --epochs 8 --workers 16 \
       --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0 --save_as ADVR --no_eval --tag "_replay_bs${BS}"; then
    echo "[advreplay] trained at batch $BS"; break
  fi
done
echo "[advreplay] clean eval"
python scripts/eval_ckpt.py --name ADVR
echo "[advreplay] both-surface leakage attack"
python scripts/run_attack_full.py
echo "[advreplay] DONE"
