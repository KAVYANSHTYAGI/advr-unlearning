#!/usr/bin/env bash
# T2.1 -- retrain the oracle under alternative person-relabelling schemes, then
# attack each to show the leakage floor is stable across the preprocessing choice.
# The default "nn" oracle already exists (checkpoints/seg_oracle_..._cls12.pth), so
# we only retrain the mode/rand/ignore variants.
set -e
cd "$(dirname "$0")/../../.."
if command -v conda >/dev/null 2>&1 && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
fi
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
EPOCHS=${EPOCHS:-24}; BS=${BS:-6}
mkdir -p results/seg

for SCHEME in mode rand ignore; do
  CK="checkpoints/seg_oracle_deeplabv3r50_cls12_${SCHEME}.pth"
  if [ -f "$CK" ]; then echo "[oracle-abl] $SCHEME exists, skip"; continue; fi
  echo "[oracle-abl] retraining oracle scheme=$SCHEME ($EPOCHS ep)"
  RELABEL=$SCHEME python scripts/train_seg.py --mode oracle --epochs $EPOCHS \
    --bs $BS --size 512 --suffix "_${SCHEME}" \
    > results/seg/oracle_ablation_${SCHEME}.log 2>&1
done

echo "[oracle-abl] attacking all oracle variants"
python scripts/experiments/rebuttal/exp_oracle_ablation.py
echo "[oracle-abl] DONE"
