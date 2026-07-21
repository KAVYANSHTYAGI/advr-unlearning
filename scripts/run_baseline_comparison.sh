#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True   # person cls12, ade20k defaults
LOG=results/seg/baseline_comparison.log; mkdir -p results/seg
: > $LOG

echo "[cmp] waiting for person reference models..." | tee -a $LOG
until [ -f checkpoints/seg_original_deeplabv3r50_cls12.pth ] && \
      [ -f checkpoints/seg_oracle_deeplabv3r50_cls12.pth ] && \
      grep -q "\[refs\] DONE" results/seg/person_refs_driver.log 2>/dev/null; do sleep 120; done
echo "[cmp] refs ready" | tee -a $LOG

# --- faithful baselines (their own hyper-params) ---
for B in FT NegGrad NegGrad+ RL SalUn SSD SCRUB DELETE; do
  echo "[cmp] baseline $B" | tee -a $LOG
  python scripts/run_baselines.py --name "$B" --seed 1
  python scripts/eval_ckpt.py --name "bl_${B}_s1" | tee -a $LOG
done

# --- our methods from the SAME reference (apples-to-apples) ---
run_ours () { python scripts/test_candidate.py --name "$1" --bs 4 --workers 24 --save_as "$2" --no_eval --seed 1 "${@:3}"; python scripts/eval_ckpt.py --name "$2" | tee -a $LOG; }
echo "[cmp] ours: A2" | tee -a $LOG;   run_ours A2_ft_anchored   ours_A2_s1   --epochs 3 --lr 1e-3 --lam_r 1.0
echo "[cmp] ours: ADV" | tee -a $LOG;  run_ours ADV_silhouette   ours_ADV_s1  --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0
echo "[cmp] ours: ADVR" | tee -a $LOG; run_ours ADV_full         ours_ADVR_s1 --epochs 8 --lr 1e-3 --lam_r 1.0 --lam_adv 1.0 --lam_c 0.0
echo "[cmp] ours: B1" | tee -a $LOG;   run_ours B1_proto_collapse ours_B1_s1  --epochs 3 --lr 1e-3 --lam_r 1.0 --lam_c 1.0

# --- N=3 leakage attack over ALL methods ---
echo "[cmp] leakage attack (N=3) over all methods" | tee -a $LOG
UNLEARNED=bl_FT_s1,bl_NegGrad_s1,bl_NegGrad+_s1,bl_RL_s1,bl_SalUn_s1,bl_SSD_s1,bl_SCRUB_s1,bl_DELETE_s1,ours_A2_s1,ours_ADV_s1,ours_ADVR_s1,ours_B1_s1 \
  python scripts/run_attack_variance.py 2>&1 | tee -a $LOG
cp results/seg/attack_variance.json results/seg/baseline_comparison_attack.json
echo "[cmp] ALL DONE" | tee -a $LOG
