#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
export ARCH=segformer_b2 FORGET_CLS=12   # ADE20K person, 150 classes (defaults)
LOG=results/seg/segformer_exp.log; mkdir -p results/seg
echo "[sf] ARCH=segformer_b2 person=12" | tee $LOG

python scripts/train_seg.py --mode original --arch $ARCH --epochs 60 --bs 4 --size 512 \
  > results/seg/sf_train_original.log 2>&1
grep -E "DONE|retain_mIoU" results/seg/sf_train_original.log | tail -2 | tee -a $LOG
python scripts/train_seg.py --mode oracle --arch $ARCH --epochs 60 --bs 4 --size 512 \
  > results/seg/sf_train_oracle.log 2>&1
grep -E "DONE|retain_mIoU" results/seg/sf_train_oracle.log | tail -2 | tee -a $LOG

echo "[sf] A2 + ADVR" | tee -a $LOG
python scripts/test_candidate.py --name A2_ft_anchored --arch $ARCH --bs 4 --workers 24 \
  --save_as sf_A2_s1 --no_eval --seed 1 --epochs 3 --lr 1e-3 --lam_r 1.0
python scripts/eval_ckpt.py --name sf_A2_s1 --arch $ARCH | tee -a $LOG
python scripts/test_candidate.py --name ADV_full --arch $ARCH --bs 4 --workers 24 \
  --save_as sf_ADVR_s1 --no_eval --seed 1 --epochs 8 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0
python scripts/eval_ckpt.py --name sf_ADVR_s1 --arch $ARCH | tee -a $LOG

echo "[sf] leakage attack (output surface, 3 trials)" | tee -a $LOG
UNLEARNED=sf_A2_s1,sf_ADVR_s1 \
  python scripts/run_attack_variance.py 2>&1 | tee -a $LOG
cp results/seg/attack_variance.json results/seg/sf_attack_variance.json
echo "[sf] DONE" | tee -a $LOG
