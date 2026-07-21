#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== A2 + freeze_bn ==="
python scripts/test_candidate.py --name A2_ft_anchored --epochs 3 --lr 1e-3 --lam_r 1.0 --freeze_bn --tag "_bn"
echo "=== A2 + freeze_bn + lam_r 3 ==="
python scripts/test_candidate.py --name A2_ft_anchored --epochs 3 --lr 1e-3 --lam_r 3.0 --freeze_bn --tag "_bn_lr3"
echo "=== F1 = proto-collapse + feat-anchor + freeze_bn ==="
python scripts/test_candidate.py --name B1_proto_collapse --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0 --lam_f 1.0 --freeze_bn --tag "_F1"
echo "=== ALL DONE ==="
