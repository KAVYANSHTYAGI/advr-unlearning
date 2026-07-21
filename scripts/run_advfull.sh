#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[advfull] waiting for red-team to free GPU..."
until ! pgrep -f run_attack_full.py >/dev/null; do sleep 30; done
echo "[advfull] GPU free; training ADV_full (longer, bigger batch)"
for BS in 8 6 4; do
  echo "[advfull] trying batch $BS"
  if python scripts/test_candidate.py --name ADV_full --bs $BS --epochs 8 --workers 16 \
       --lam_r 1.0 --lam_adv 1.0 --lam_c 1.0 --save_as ADVF --tag "_full_bs${BS}"; then
    echo "[advfull] succeeded at batch $BS"; break
  fi
  echo "[advfull] batch $BS failed (likely OOM); retrying smaller"
done
echo "[advfull] re-running full red-team (out+feat) incl ADVF"
python scripts/run_attack_full.py
echo "[advfull] DONE"
