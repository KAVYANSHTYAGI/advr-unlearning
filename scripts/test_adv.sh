#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== train ADV_silhouette (full, save) ==="
python scripts/test_candidate.py --name ADV_silhouette --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --save_as ADV --tag _adv
echo "=== leakage attack on all variants (incl A2,B1,ADV) ==="
python scripts/run_attack.py
echo "=== DONE ==="
