#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== re-run A2 (save) ==="
python scripts/test_candidate.py --name A2_ft_anchored --epochs 3 --lr 1e-3 --lam_r 1.0 --save_as A2 --tag _foratk >/dev/null 2>&1 || true
echo "=== re-run B1 (save) ==="
python scripts/test_candidate.py --name B1_proto_collapse --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0 --save_as B1 --tag _foratk >/dev/null 2>&1 || true
echo "=== run silhouette attacker on all variants ==="
python scripts/run_attack.py
echo "=== DONE ==="
