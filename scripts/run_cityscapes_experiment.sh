#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
export DATASET=cityscapes NUM_CLASSES=19 FORGET_CLS=11
LOG=results/seg/city_exp.log; mkdir -p results/seg
echo "[city] DATASET=cityscapes person=trainId 11, 19 classes" | tee $LOG

echo "[city] train original" | tee -a $LOG
python scripts/train_seg.py --mode original --epochs 60 --bs 6 --size 512 \
  > results/seg/city_train_original.log 2>&1 || \
python scripts/train_seg.py --mode original --epochs 60 --bs 4 --size 512 \
  > results/seg/city_train_original.log 2>&1
grep -E "DONE|retain_mIoU" results/seg/city_train_original.log | tail -2 | tee -a $LOG

echo "[city] train oracle" | tee -a $LOG
python scripts/train_seg.py --mode oracle --epochs 60 --bs 6 --size 512 \
  > results/seg/city_train_oracle.log 2>&1 || \
python scripts/train_seg.py --mode oracle --epochs 60 --bs 4 --size 512 \
  > results/seg/city_train_oracle.log 2>&1
grep -E "DONE|retain_mIoU" results/seg/city_train_oracle.log | tail -2 | tee -a $LOG

# method suite (seed 1): baseline A2, ours ADV & ADVR, ablation B1
run () { # candidate_name save extra...
  python scripts/test_candidate.py --name "$1" --bs 4 --workers 24 --save_as "$2" \
    --no_eval --seed 1 "${@:3}"
  python scripts/eval_ckpt.py --name "$2" | tee -a $LOG
}
echo "[city] A2" | tee -a $LOG;   run A2_ft_anchored  city_A2_s1   --epochs 3 --lr 1e-3 --lam_r 1.0
echo "[city] ADV" | tee -a $LOG;  run ADV_silhouette  city_ADV_s1  --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0
echo "[city] ADVR" | tee -a $LOG; run ADV_full        city_ADVR_s1 --epochs 8 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0
echo "[city] B1" | tee -a $LOG;   run B1_proto_collapse city_B1_s1 --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0

echo "[city] two-surface leakage attack (3 trials)" | tee -a $LOG
UNLEARNED=city_A2_s1,city_ADV_s1,city_ADVR_s1,city_B1_s1 \
  python scripts/run_attack_variance.py 2>&1 | tee -a $LOG
cp results/seg/attack_variance.json results/seg/city_attack_variance.json
echo "[city] DONE" | tee -a $LOG
