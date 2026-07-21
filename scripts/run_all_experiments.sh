#!/usr/bin/env bash
# Runs the lambda sweep, then the car and segformer axes. Updated to call the
# consolidated run_axis_pipeline.sh (see scripts/run_all_axes.sh for all 4 axes).
set -eo pipefail
cd "$(dirname "$0")/.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[queue] running lambda sweep"
bash scripts/experiments/run_lambda_sweep.sh

echo "[queue] lambda sweep done; launching car axis"
bash scripts/run_axis_pipeline.sh --axis car_ade20k

echo "[queue] car done; launching segformer axis"
bash scripts/run_axis_pipeline.sh --axis person_segformer

echo "[queue] ALL EXPERIMENTS DONE"
