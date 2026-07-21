# Reproducing the results

End-to-end walkthrough from a clean checkout to the paper's headline tables. See
[`docs/CONFIGURATION.md`](CONFIGURATION.md) for what every environment variable
below does, and [`configs/axes.sh`](../configs/axes.sh) for the 4 experiment axes.

## 0. Install

```bash
pip install -e .                 # core (DeepLabV3-R50 axes)
pip install -e '.[segformer]'    # + SegFormer-B2 backbone axis
```

## 1. Get the data

- **ADE20K** (`ADEChallengeData2016`, mmseg-style extraction): download from the
  [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/), then either
  `ln -s /path/to/ADEChallengeData2016 data/ADE20K` or `export ADE20K_ROOT=/path/to/it`.
- **Cityscapes**: download `leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`
  from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/), extract into
  `data/cityscapes/` (or `export CITYSCAPES_ROOT=...`).

## 2. Train reference models (original + oracle)

The "original" model is trained on true labels; the "oracle" is trained with person
pixels already dominant-neighbor-relabeled (`seg.data.merge_person_label`) — the
gold-standard target a perfect unlearner should match, without ever training on the
forget class.

```bash
PYTHONPATH=. python scripts/train_seg.py --mode original --epochs 60 --bs 6 --size 512
PYTHONPATH=. python scripts/train_seg.py --mode oracle   --epochs 60 --bs 6 --size 512
```
Each takes ~16-18h on an RTX 5080-class GPU (60 epochs, competent-baseline recipe —
see `docs/EXPERIMENT_LOG.md` 2026-06-28). Writes
`checkpoints/seg_{original,oracle}_deeplabv3r50_cls12.pth`. Drop `--bs 6` to `--bs 4`
if you OOM.

For a different axis (car, Cityscapes, SegFormer), export the axis's env vars first
(see `configs/axes.sh`) — or just use `run_axis_pipeline.sh` below, which does this
for you and skips training if the checkpoint already exists.

## 3. Unlearn with ADVR

```bash
PYTHONPATH=. python scripts/test_candidate.py --name ADVR --bs 4 --epochs 8 \
    --lam_r 1.0 --lam_adv 1.0 --save_as ADVR --no_eval
```
Trains ADVR (output-adversarial silhouette removal + retain-anchor KL + retain
replay — `seg/candidates.py::adv_full`, `lam_c=0`) against the just-trained
original, saves `checkpoints/unlearned_ADVR.pth`. Other candidates: `A2_ft_anchored`,
`ADV_silhouette` (ADVR without retain-replay), `B1_proto_collapse` (feature-collapse
ablation), `ADVF` (ADVR + feature-collapse term, `--lam_c 1.0`).

## 4. Evaluate quality + leakage

```bash
PYTHONPATH=. python scripts/eval_ckpt.py --name ADVR
# person_IoU (want ~0) | retain_mIoU (want close to oracle) | blend_fail |
# silhouette_IoU | recover_pts (few-shot relearning audit, want ~0)

PYTHONPATH=. python scripts/run_attack_variance.py
# trains the silhouette-reconstruction attacker N_TRIALS times per model/surface,
# reports mean+/-std AUC vs the oracle floor -> results/seg/attack_variance.json
```
A method "leaks" iff its attacker AUC exceeds `floor_mean + 2*floor_std` (the
statistically-robust protocol from `docs/EXPERIMENT_LOG.md` 2026-06-30, after the
first single-draw attacker-variance bug was found and fixed).

## 5. One axis end-to-end (recommended path)

Steps 2-4 for one axis, idempotent (skips anything already on disk):

```bash
bash scripts/run_axis_pipeline.sh --axis person_ade20k     # the paper's headline axis
bash scripts/run_axis_pipeline.sh --axis car_ade20k
bash scripts/run_axis_pipeline.sh --axis person_cityscapes
bash scripts/run_axis_pipeline.sh --axis person_segformer
# or all 4 in sequence:
bash scripts/run_all_axes.sh
```
Each produces `results/seg/<prefix>leakage.json` (A2/ADV/ADVR/B1 vs the oracle floor
and DELETE-logitmask prior-art) — this is the 4-axis generality table
(`docs/EXPERIMENT_LOG.md` 2026-07-04).

## 6. Full baseline comparison (8 faithful baselines + extra seeds)

Requires step 5 already run for that axis.

```bash
# full 8-baseline suite (FT/NegGrad/NegGrad+/RL/SalUn/SSD/SCRUB/DELETE), long HPs
bash scripts/run_axis_extend.sh --axis person_ade20k \
    --baselines FT,NegGrad,NegGrad+,RL,SalUn,SSD,SCRUB,DELETE

# extra ADVR seeds + DELETE-only comparison (matches the car/segformer paper runs)
bash scripts/run_axis_extend.sh --axis car_ade20k \
    --baselines DELETE --baseline-seeds 1,2,3 --extra-adv-seeds 2,3
```
Writes `results/seg/<prefix>leakage_full.json` — this is the "12-method" comparison
table (our 4 methods + 8 faithful baselines).

## 7. The 3-seed headline table

```bash
bash scripts/run_multiseed.sh
```
Trains A2/ADV/ADVR at seeds 1-3 (+ B1 at seed 1) on the default axis, evaluates
each, then attacks the full set — the retain_mIoU/recover_pts±std table in
`docs/EXPERIMENT_LOG.md` 2026-06-30.

## 8. Supplementary / rebuttal experiments

- `bash scripts/experiments/run_lambda_sweep.sh` — ADVR's `lam_r × lam_adv` ablation.
- `bash scripts/experiments/rebuttal/exp_reviewer_response.sh` — the AAAI
  reviewer-response queue (attacker capacity/architecture/transfer/adaptive sweeps,
  oracle-construction ablation). `SMOKE=1` dry-runs it in minutes. See
  [`docs/REVIEWER_EXPERIMENTS.md`](REVIEWER_EXPERIMENTS.md) for the full
  concern → script → output map.
- `scripts/experiments/legacy/` — historical method-screening runs (round-1/round-2
  in `docs/EXPERIMENT_LOG.md`) kept for the record, not part of the main path.

## 9. Classification comparator (`mu/`, CIFAR-10)

The vision-unlearning literature's usual classification benchmark, kept as a
comparator (see `docs/EXPERIMENT_LOG.md` 2026-06-26 for why segmentation became the
headline instead):

```bash
PYTHONPATH=. python scripts/train.py --mode original --dataset cifar10
bash scripts/run_pipeline.sh    # retrain oracle + full method benchmark
```

## Figures

```bash
python scripts/figures/gen_pareto_scatter.py   # retain vs leakage Pareto plot
python scripts/figures/gen_roc_curves.py
python scripts/figures/gen_qual_panels.py      # qualitative segmentation panels
```
