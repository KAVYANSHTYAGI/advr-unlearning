#!/usr/bin/env bash
# =============================================================================
# Reviewer-response experiment queue (AAAI rebuttal).
# Runs on the 5090:  ssh 5090  ->  copy project1  ->  bash scripts/exp_reviewer_response.sh
#
# Tier 1 (cheap, attacker-side only, NO retraining -- reuses existing checkpoints):
#   T1.5  attacker variance at 10 seeds        run_attack_variance.py
#   T1.1  attacker capacity sweep              exp_attacker_capacity.py
#   T1.2  attacker architecture zoo (cnn/unet/vit)  exp_attacker_zoo.py
#   T1.4  transfer attack matrix               exp_transfer_attack.py
#   T1.3  adaptive / white-box attacker        exp_adaptive_attack.py
#   T2.2  metric-sensitivity sweep             exp_metric_sensitivity.py
# Tier 2 (retrains 3 oracle variants ~overnight):
#   T2.1  oracle-construction ablation         exp_oracle_ablation.sh
#
# Env knobs:
#   SMOKE=1     tiny iters/trials to validate the pipeline end-to-end (~minutes)
#   SKIP_T2=1   skip the oracle retrains (Tier 1 only)
#   N_TRIALS    attacker trials per cell (default 10 for T1.5, 3 elsewhere)
#   UNLEARNED   comma list of unlearned_<name>.pth to attack
# =============================================================================
set -e
cd "$(dirname "$0")/.."
source /home/kavyansh/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p results/seg results/seg/rebuttal
LOG=results/seg/rebuttal/queue.log
: > "$LOG"

# models to attack across the rebuttal: floor + prior-art + our method (+seeds) +
# the two most-cited leaky baselines (to show they DO leak under strong attack).
export UNLEARNED="${UNLEARNED:-ADVR,ADVR_s1,ADVR_s2,ADVR_s3,bl_SSD_s1,bl_SalUn_s1,bl_DELETE_s1}"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "### SMOKE MODE: tiny settings, validates the pipeline only ###" | tee -a "$LOG"
  export N_TRIALS=1 MAX_ITERS=60 RESTARTS=1 ROUNDS=2 ITERS=60 STEPS="10,25"
  export EXP_N_TRAIN=120 EXP_N_VAL=80 EXP_BS=8
  export UNLEARNED="ADVR"
fi

run () {  # tag  outfile  cmd...
  local TAG=$1 OUTF=$2; shift 2
  if [ -f "$OUTF" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $TAG ($OUTF exists; FORCE=1 to rerun)" | tee -a "$LOG"; return 0
  fi
  echo "=========== $TAG  $(date '+%F %T') ===========" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "[done] $TAG -> $OUTF" | tee -a "$LOG"
}

R=results/seg/rebuttal
NT=${N_TRIALS:-3}          # trials for the sweeps (each cell is expensive)
NTV=${N_TRIALS:-10}        # trials for the headline variance table

# ---- Tier 1: attacker-side, no retraining ----
run "T1.5 variance(N=$NTV)" $R/attack_variance_n10.json \
  env N_TRIALS=$NTV OUT_JSON=$R/attack_variance_n10.json python scripts/run_attack_variance.py

run "T1.1 capacity-sweep" $R/attacker_capacity.json \
  env N_TRIALS=$NT OUT_JSON=$R/attacker_capacity.json python scripts/exp_attacker_capacity.py

run "T1.2 attacker-zoo" $R/attacker_zoo.json \
  env N_TRIALS=$NT OUT_JSON=$R/attacker_zoo.json python scripts/exp_attacker_zoo.py

run "T1.4 transfer-attack" $R/transfer_attack.json \
  env N_TRIALS=$NT OUT_JSON=$R/transfer_attack.json python scripts/exp_transfer_attack.py

run "T1.3 adaptive-attack" $R/adaptive_attack.json \
  env OUT_JSON=$R/adaptive_attack.json python scripts/exp_adaptive_attack.py

run "T2.2 metric-sensitivity" $R/metric_sensitivity.json \
  env OUT_JSON=$R/metric_sensitivity.json python scripts/exp_metric_sensitivity.py

# ---- Tier 2: retrains oracle variants (skippable) ----
if [ "${SKIP_T2:-0}" != "1" ] && [ "${SMOKE:-0}" != "1" ]; then
  run "T2.1 oracle-ablation" $R/oracle_ablation.json \
    env OUT_JSON=$R/oracle_ablation.json bash scripts/exp_oracle_ablation.sh
fi

echo "############ REBUTTAL QUEUE DONE  $(date '+%F %T') ############" | tee -a "$LOG"
echo "results in results/seg/rebuttal/*.json ; full log: $LOG"
