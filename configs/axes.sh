#!/usr/bin/env bash
# One documented table of the 4 experiment axes reported in the paper (see
# EXPERIMENT_LOG.md 2026-07-04 "generality" sweep). Sourced by scripts/run_axis_pipeline.sh
# and scripts/run_axis_extend.sh via `--axis <name>`.
#
# Fields per axis: DATASET FORGET_CLS NUM_CLASSES ARCH PREFIX
#   DATASET      ade20k | cityscapes
#   FORGET_CLS   class index to unlearn (see seg/data.py)
#   NUM_CLASSES  150 (ADE20K) or 19 (Cityscapes)
#   ARCH         deeplabv3r50 | segformer_b2
#   PREFIX       checkpoint-name prefix, e.g. unlearned_<PREFIX>A2_s1.pth
#
# Add a new axis by adding one line here — no script changes needed.

declare -A AXES=(
  [person_ade20k]="ade20k     12 150 deeplabv3r50 "
  [car_ade20k]="ade20k        20 150 deeplabv3r50 car_"
  [person_cityscapes]="cityscapes 11 19 deeplabv3r50 city_"
  [person_segformer]="ade20k    12 150 segformer_b2 sf_"
)

# axis_lookup <axis-name> -> sets DATASET, FORGET_CLS, NUM_CLASSES, ARCH, PREFIX
axis_lookup() {
  local axis="$1"
  local row="${AXES[$axis]:-}"
  if [ -z "$row" ]; then
    echo "unknown axis '$axis'; have: ${!AXES[*]}" >&2
    exit 1
  fi
  read -r DATASET FORGET_CLS NUM_CLASSES ARCH PREFIX <<< "$row"
  export DATASET FORGET_CLS NUM_CLASSES ARCH
}
