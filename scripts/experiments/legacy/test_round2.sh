#!/usr/bin/env bash
# HISTORICAL — round-2 screening (BN-freeze variants). Refuted per
# docs/EXPERIMENT_LOG.md (2026-06-27): freeze_bn did not improve over round-1.
# Kept for the record; not part of the main reproduce path.
set -e
cd "$(dirname "$0")/../../.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== A2 + freeze_bn ==="
python scripts/test_candidate.py --name A2_ft_anchored --epochs 3 --lr 1e-3 --lam_r 1.0 --freeze_bn --tag "_bn"
echo "=== A2 + freeze_bn + lam_r 3 ==="
python scripts/test_candidate.py --name A2_ft_anchored --epochs 3 --lr 1e-3 --lam_r 3.0 --freeze_bn --tag "_bn_lr3"
echo "=== F1 = proto-collapse + feat-anchor + freeze_bn ==="
python scripts/test_candidate.py --name B1_proto_collapse --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0 --lam_f 1.0 --freeze_bn --tag "_F1"
echo "=== ALL DONE ==="
