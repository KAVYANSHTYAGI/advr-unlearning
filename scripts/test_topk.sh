#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== A2 ==="; python scripts/test_candidate.py --name A2_ft_anchored --epochs 3 --lr 1e-3 --lam_r 1.0
echo "=== D1 (blend) ==="; python scripts/test_candidate.py --name D1_ga_orth --epochs 2 --lr 1e-3 --blend
echo "=== B1 ==="; python scripts/test_candidate.py --name B1_proto_collapse --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0
echo "=== ALL DONE ==="
