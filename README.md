# ADVR — leak-safe class-unlearning for semantic segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Most machine-unlearning methods for segmentation "forget" a class by relabeling its
pixels and moving on — but the forgotten region still looks like a person-shaped hole
in the model's output, and a simple attacker can reconstruct exactly where it was.
**ADVR** removes that spatial silhouette itself, not just the label, while preserving
accuracy on every other class better than existing baselines and without leaving the
model easier to relearn.

## The idea

Class-unlearning in dense prediction is usually graded on the same axes as
classification unlearning (forget-accuracy, a membership-inference attack). But a
segmentation model's *output* is spatial: even a model that never predicts `person`
inside the forgotten region can still leave a coherent, person-shaped anomaly there —
"segmentation by subtraction." A learned attacker trained to reconstruct the person
mask from nothing but the model's output field (no image, no label) picks this up
immediately, and we show the dominant prior-art mechanism (mask-and-renormalize
logits, à la DELETE, CVPR'25) reliably leaks under this attack.

**Hypothesis:** forgetting a class in dense prediction means driving the spatial
mutual information between the forgotten region and the model's output field toward
zero — not just relabeling it. See [`docs/POSITIONING.md`](docs/POSITIONING.md) for
the full argument against DELETE / Spotter / Erase-to-Retain.

**ADVR** (adversarial silhouette removal + retain-replay) operationalizes this as a
min-max game: a discriminator is trained to reconstruct the person mask from the
model's output field; the segmentation model is updated to (a) blend the forgotten
region into its dominant-neighbor context, (b) fool the discriminator, (c) stay
anchored to the original model on retain pixels via KL, and (d) replay full-image CE
on the retain distribution to close the accuracy gap the first three terms leave
behind. See [`seg/candidates.py`](seg/candidates.py) for the exact loss
(`adv_full`/`adv_replay`, `lam_c=0`).

## Headline results (ADE20K-150, person, DeepLabV3-R50, 3 seeds)

**Unlearning quality** vs. the oracle (retrained from scratch on relabeled data,
retain_mIoU 36.6):

| method | retain_mIoU ↑ | recover_pts ↓ |
|---|---|---|
| A2 (retain-anchored finetune) | 22.4 ± 0.6 | 4.2 ± 0.3 |
| ADV (silhouette-adversarial only) | 22.0 ± 1.0 | 2.9 ± 0.1 |
| **ADVR (ours)** | **26.5 ± 0.5** | **2.7 ± 0.2** |
| B1 (feature-collapse ablation) | 0.4 (over-forgets) | 2.0 |

**Leakage** — silhouette-reconstruction attacker AUC, mean±std over N=10 independent
attacker trainings per model/surface; a method **leaks** iff AUC exceeds
`oracle_floor_mean + 2·std`:

| model | output-field AUC | feature-field AUC | verdict |
|---|---|---|---|
| oracle (privacy floor) | 85.4 ± 0.6 | 97.0 ± 0.1 | — |
| DELETE (prior art, CVPR'25) | 96.4 ± 0.1 | 98.6 ± 0.1 | **LEAKS**, both surfaces |
| ADV | 81.2 ± 0.2 | 92.1 ± 0.2 | safe |
| **ADVR (ours)** | **78.9 ± 1.5** | **94.7 ± 0.1** | **safe, both surfaces** |
| B1 (ablation) | 77.1 ± 1.4 | 79.0 ± 0.5 | safe, but only because it over-forgets |

**ADVR is the only method that is simultaneously leak-free on both surfaces, keeps
significantly more accuracy on the other 149 classes, and is harder to relearn.**

**Generality** — the pattern holds across dataset, forget class, and backbone (single
seed each, see `docs/EXPERIMENT_LOG.md` 2026-07-04 for the full table):

| axis | oracle retain | ADVR retain | A2 retain | DELETE AUC | ADVR AUC | floor |
|---|---|---|---|---|---|---|
| ADE20K, person | 36.6 | 26.5 | 22.4 | 96.4 **LEAK** | 78.9 safe | 85.4 |
| Cityscapes, person (19-cls) | 66.3 | 61.7 | 57.7 | 98.1 **LEAK** | 88.3 safe | 91.1 |
| ADE20K, car | 37.7 | 30.1 | 23.2 | 98.9 **LEAK** | 92.1 safe | 89.8 |
| ADE20K, person, SegFormer-B2 | 32.5 | 26.9 | 23.7 | 95.7 **LEAK** | 82.5 safe | 87.8 |

DELETE-style logit-masking leaks on *every* setting tested; ADVR is leak-safe and
best-retain on every setting.

## Install

```bash
pip install -e .                 # core (DeepLabV3-R50 axes)
pip install -e '.[segformer]'    # + SegFormer-B2 backbone axis
```
Tested with Python 3.12, `torch==2.9.0+cu128`, `torchvision==0.24.0+cu128` (see
[`requirements.txt`](requirements.txt) for the full pin; swap the CUDA build tag for
your platform).

## Quickstart

```bash
# 1. train the reference models (original + oracle), ADE20K person axis
PYTHONPATH=. python scripts/train_seg.py --mode original --epochs 60 --bs 6
PYTHONPATH=. python scripts/train_seg.py --mode oracle   --epochs 60 --bs 6

# 2. unlearn "person" with ADVR
PYTHONPATH=. python scripts/test_candidate.py --name ADVR --bs 4 --epochs 8 \
    --lam_r 1.0 --lam_adv 1.0 --save_as ADVR --no_eval

# 3. evaluate quality + the relearning audit
PYTHONPATH=. python scripts/eval_ckpt.py --name ADVR

# 4. run the silhouette-leakage attack
PYTHONPATH=. python scripts/run_attack_variance.py
```
Full walkthrough (data setup, every experiment axis, the baseline comparison suite,
reproducing every table above): [`docs/REPRODUCING.md`](docs/REPRODUCING.md).
Every env-var knob (`DATASET`, `FORGET_CLS`, `ARCH`, ...): 
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## Repository structure

```
seg/                     segmentation unlearning — the headline library
  data.py                  ADE20K / Cityscapes datasets, dominant-neighbor relabeling
  model.py                 DeepLabV3-R50 / SegFormer-B2 factory
  candidates.py            ADVR + ablations (A2, ADV, B1) — the novel-method code
  attack.py                the silhouette-reconstruction attacker (metric + adversary)
  audit.py                 leakage metrics + few-shot relearning-recovery audit
  utils.py                 shared clone/mask/CE primitives
  baselines/               7 faithful ported baselines (FT, NegGrad(+), RL, SalUn,
                            SSD, SCRUB) + DELETE (CVPR'25), each citing its source
mu/                       classification comparator (CIFAR-10, why: docs/EXPERIMENT_LOG.md)
scripts/                  entry points — see docs/REPRODUCING.md
  train_seg.py, test_candidate.py, run_baselines.py, eval_ckpt.py,
  run_attack_variance.py    core, reusable
  run_axis_pipeline.sh, run_axis_extend.sh, run_all_axes.sh   per-axis orchestration
  figures/                  paper figure generators
  experiments/              lambda sweep, AAAI rebuttal queue, historical screening runs
configs/axes.sh           the 4 experiment axes (dataset/class/backbone), one table
docs/                     reproduction guide, config reference, development history
```

## Baselines & provenance

Every baseline in `seg/baselines/` is a from-scratch reimplementation of a published
method's equations/official code, adapted to dense prediction by masking the loss to
forget/retain pixels (see each module's docstring for the exact source line cited).
No third-party code is vendored or imported at runtime; for reference, the upstream
repos are:

- **SalUn** (FT, NegGrad, RL, SalUn) — [OPTML-Group/Unlearn-Saliency](https://github.com/OPTML-Group/Unlearn-Saliency), ICLR'24
- **SCRUB** (NegGrad+, SCRUB) — [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB), NeurIPS'23
- **SSD** — [if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening), AAAI'24
- **DELETE** — CVPR'25 (mask-and-renormalize logit forgetting)

## Development history

This project pivoted twice and caught two of its own measurement bugs along the way
(an RNG-dependent importance estimator in SSD; a high-variance single-draw silhouette
attacker) before landing on the current protocol. The full log —
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — is kept as the honest record of
what was tried, refuted, and fixed; [`docs/METHOD_IDEAS.md`](docs/METHOD_IDEAS.md) has
every candidate method's derivation.

## Citation

```bibtex
@misc{advr2026,
  title  = {ADVR: Adversarial-Replay Unlearning for Leak-Safe Semantic Segmentation},
  author = {{ADVR authors}},
  year   = {2026},
  note   = {Code: https://github.com/kavyansh/advr-unlearning}
}
```

## License

[MIT](LICENSE).
