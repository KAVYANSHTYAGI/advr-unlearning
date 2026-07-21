#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
export FORGET_CLS=20   # ADE20K car; NUM_CLASSES=150, DATASET=ade20k (defaults)
LOG=results/seg/car_methods.log; mkdir -p results/seg
echo "[car] FORGET_CLS=20 (car), reusing seg_*_deeplabv3r50_cls20 reference models" | tee $LOG

run () { python scripts/test_candidate.py --name "$1" --bs 4 --workers 24 --save_as "$2" --no_eval --seed 1 "${@:3}"; python scripts/eval_ckpt.py --name "$2" | tee -a $LOG; }
echo "[car] A2" | tee -a $LOG;   run A2_ft_anchored  car_A2_s1   --epochs 3 --lr 1e-3 --lam_r 1.0
echo "[car] ADV" | tee -a $LOG;  run ADV_silhouette  car_ADV_s1  --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0
echo "[car] ADVR (reuse existing car_ADVR_s1 if present)" | tee -a $LOG
[ -f checkpoints/unlearned_car_ADVR_s1.pth ] && python scripts/eval_ckpt.py --name car_ADVR_s1 | tee -a $LOG || \
  run ADV_full car_ADVR_s1 --epochs 8 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0
echo "[car] B1" | tee -a $LOG;   run B1_proto_collapse car_B1_s1 --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0

echo "[car] leakage attack (3 trials)" | tee -a $LOG
UNLEARNED=car_A2_s1,car_ADV_s1,car_ADVR_s1,car_B1_s1 \
  python scripts/run_attack_variance.py 2>&1 | tee -a $LOG
cp results/seg/attack_variance.json results/seg/car_attack_variance.json
echo "[car] DONE" | tee -a $LOG
