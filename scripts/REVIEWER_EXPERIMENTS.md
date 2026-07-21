# Reviewer-response experiments (AAAI rebuttal)

Extra experiments that answer the AAAI reviewer's major concerns, mapped to the
review. **Tier 1 needs no retraining** (reuses existing `checkpoints/`), so it is
cheap; **Tier 2** retrains 3 oracle variants (~overnight).

## Reviewer concern → experiment

| Reviewer weakness | Experiment | Script | Output |
|---|---|---|---|
| W1 "weak attacker / why not bigger" | T1.1 attacker **capacity sweep** | `exp_attacker_capacity.py` | `rebuttal/attacker_capacity.json` |
| W1 "why not U-Net / ViT" | T1.2 attacker **architecture zoo** | `exp_attacker_zoo.py` | `rebuttal/attacker_zoo.json` |
| W1 "transfer attackers" | T1.4 **transfer attack** matrix | `exp_transfer_attack.py` | `rebuttal/transfer_attack.json` |
| W5 "no adaptive adversary" (biggest) | T1.3 **adaptive / white-box** attacker | `exp_adaptive_attack.py` | `rebuttal/adaptive_attack.json` |
| W2 "N=3 is fragile" | T1.5 **10-seed** variance table | `run_attack_variance.py` (`N_TRIALS=10`) | `rebuttal/attack_variance_n10.json` |
| W "recover metric hand-designed", "define blend_fail" | T2.2 **metric sensitivity** | `exp_metric_sensitivity.py` | `rebuttal/metric_sensitivity.json` |
| W6 "floor depends on preprocessing" | T2.1 **oracle-construction ablation** | `exp_oracle_ablation.sh` | `rebuttal/oracle_ablation.json` |

Not fixable with experiments (framing, do in the writing): ADVR novelty is
incremental → lead with **SSL + the oracle floor** as the contribution; soften
absolute language ("universal", "no learned function", "first") → "across all
evaluated methods/settings/attackers".

## Run everything (on the 5090)

```bash
ssh 5090
# copy the whole project1/ over, then:
cd project1
bash scripts/exp_reviewer_response.sh          # Tier 1 + Tier 2, skip-if-done
# knobs:
SMOKE=1 bash scripts/exp_reviewer_response.sh   # tiny end-to-end validation (~mins)
SKIP_T2=1 bash scripts/exp_reviewer_response.sh # Tier 1 only (no retraining)
FORCE=1 bash scripts/exp_reviewer_response.sh   # ignore existing outputs, rerun
N_TRIALS=10 UNLEARNED=ADVR,bl_SSD_s1 bash scripts/exp_reviewer_response.sh
```

Results land in `results/seg/rebuttal/*.json`; full log at
`results/seg/rebuttal/queue.log`. Each step **skips if its JSON already exists**, so
the queue is resumable after interruption.

## What each experiment shows

- **T1.1 capacity sweep** — CNN attacker from 32k→~1M params. Plot *excess AUC over
  floor* vs params; the claim holds iff ADVR's excess **saturates** at ≤ floor+2σ.
- **T1.2 zoo** — same protocol under `cnn` (~1M), `unet` (~17M), `vit` (~18M).
  Reports the **worst-case** leak over all three architectures per model.
- **T1.3 adaptive** — frozen ADVR, strongest arch, `RESTARTS` random inits each
  trained to convergence with val-AUC early stopping; reports **best-of-all** AUC.
  This is the honest adaptive white-box adversary. Env: `KIND, RESTARTS, ROUNDS, ITERS`.
- **T1.4 transfer** — AUC matrix (train-on-A × eval-on-B). Read the **ADVR column**:
  highest AUC any attacker, including ones trained on original/oracle fields,
  reaches against ADVR. If it's at the floor, the defense isn't co-adaptation.
- **T1.5 variance** — the headline leakage table at **N=10** seeds (μ±2σ). Env `N_TRIALS`.
- **T2.2 metric sensitivity** — recovery vs relearning budget (10→200 steps) plus the
  threshold-free `residual_person / blend_fail / silhouette_IoU`. Shows recover_pts
  isn't an artifact of the 50-step choice, and gives one reproducible source for
  `blend_fail` (% person pixels where argmax ≠ dominant-neighbour target).
- **T2.1 oracle ablation** — retrains the oracle with person pixels relabelled by
  `nn` (default) / `mode` / `rand` / `ignore` and re-measures the floor. The spread
  across schemes is the honest uncertainty on the floor. Env `RELABEL`, `SCHEMES`.

## Compute notes

- Tier 1 is attacker-only (attacker ≈1–20M params, ~1–2k iters/cell) — minutes per
  cell. The data is RAM-preloaded once (~9 GB) so trainings never re-decode disk.
- Tier 2 retrains 3 DeepLabV3-R50 oracles (~6 h each at 24 ep) — overnight.
- All scripts honour `ARCH`, `FORGET_CLS`, `NUM_CLASSES`, `DATASET` — so the same
  suite runs for the Cityscapes / car / SegFormer settings by setting those envs.
