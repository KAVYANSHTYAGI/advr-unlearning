#!/usr/bin/env bash
# Trains the oracle once the original segmentation model finishes.
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[seg-pipeline] waiting for original to finish..."
until grep -q "DONE" results/seg/train_original.log 2>/dev/null; do sleep 60; done
echo "[seg-pipeline] original done; training oracle (merged labels)"
python scripts/train_seg.py --mode oracle --epochs 24 --bs 4 --size 512 \
    > results/seg/train_oracle.log 2>&1
echo "[seg-pipeline] DONE (original + oracle trained)"
