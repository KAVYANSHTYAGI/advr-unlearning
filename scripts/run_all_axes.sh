#!/usr/bin/env bash
# Runs the stage-1 pipeline (train refs + our 4 methods + leakage attack) across
# all 4 generality axes reported in the paper (EXPERIMENT_LOG.md 2026-07-04).
# Renamed from run_cvpr_queue.sh; now calls the consolidated run_axis_pipeline.sh.
set -eo pipefail
cd "$(dirname "$0")/.."
Q=results/seg/all_axes_queue.log
mkdir -p results/seg
echo "[queue] START $(date)" > "$Q"
for AXIS in person_ade20k car_ade20k person_cityscapes person_segformer; do
  echo "[queue] === $AXIS ===" | tee -a "$Q"
  bash scripts/run_axis_pipeline.sh --axis "$AXIS" >> "$Q" 2>&1 || echo "[queue] $AXIS FAILED" | tee -a "$Q"
done
echo "[queue] ALL DONE $(date)" | tee -a "$Q"
