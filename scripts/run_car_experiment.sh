#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True FORGET_CLS=20
LOG=results/seg/car_exp.log; mkdir -p results/seg
echo "[car] using FORGET_CLS=20 (car, idx 20)" | tee $LOG

echo "[car] train original (car visible)" | tee -a $LOG
python scripts/train_seg.py --mode original --epochs 60 --bs 6 --size 512 \
    > results/seg/car_train_original.log 2>&1 || \
python scripts/train_seg.py --mode original --epochs 60 --bs 4 --size 512 \
    > results/seg/car_train_original.log 2>&1
grep -E "DONE|retain_mIoU|person_IoU" results/seg/car_train_original.log | tail -3 | tee -a $LOG

echo "[car] train oracle (car -> dominant-neighbor)" | tee -a $LOG
python scripts/train_seg.py --mode oracle --epochs 60 --bs 6 --size 512 \
    > results/seg/car_train_oracle.log 2>&1 || \
python scripts/train_seg.py --mode oracle --epochs 60 --bs 4 --size 512 \
    > results/seg/car_train_oracle.log 2>&1
grep -E "DONE|retain_mIoU|person_IoU" results/seg/car_train_oracle.log | tail -3 | tee -a $LOG

echo "[car] ADVR unlearning (seed 1)" | tee -a $LOG
python scripts/test_candidate.py --name ADV_full --bs 4 --workers 12 \
    --epochs 8 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0 \
    --save_as car_ADVR_s1 --no_eval --seed 1 2>&1 | tee -a $LOG
python scripts/eval_ckpt.py --name car_ADVR_s1 | tee -a $LOG

echo "[car] two-surface leakage attack (3 trials)" | tee -a $LOG
python scripts/run_attack_variance.py 2>&1 | tee -a $LOG
echo "[car] DONE" | tee -a $LOG
