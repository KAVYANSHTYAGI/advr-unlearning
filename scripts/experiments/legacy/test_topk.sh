#!/usr/bin/env bash
# HISTORICAL — round-1 method screening (A2/D1/B1), kept for the record referenced
# in docs/EXPERIMENT_LOG.md. D1 (gradient-ascent orthogonal to retain) was dropped
# after this run; not part of the main reproduce path (see docs/REPRODUCING.md).
set -e
cd "$(dirname "$0")/../../.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== A2 ==="; python scripts/test_candidate.py --name A2_ft_anchored --epochs 3 --lr 1e-3 --lam_r 1.0
echo "=== D1 (blend) ==="; python scripts/test_candidate.py --name D1_ga_orth --epochs 2 --lr 1e-3 --blend
echo "=== B1 ==="; python scripts/test_candidate.py --name B1_proto_collapse --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0
echo "=== ALL DONE ==="
