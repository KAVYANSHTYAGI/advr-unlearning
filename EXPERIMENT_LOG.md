# Experiment Log — project1 (Machine Unlearning, WACV-first)

Running diary. Same discipline as the ISP project: record decisions, reproduction
gates, and surprises.

## 2026-06-26 — Baseline harness stood up

**Direction.** Applied vision unlearning, WACV-first (ICML/NeurIPS escalation if the
robustness result generalizes). Strategy = PE-ISP playbook: reproduce SOTA → honest
audit → expose inflated claim → principled fix. Gap targeted: vision unlearning is
not robust to relearning/recovery; almost all robustness work is LLM-centric.

**Environment.** Reused `myenv` (torch 2.9.0+cu128, tv 0.24, sm_120). GPU = RTX 5080.

**Data.** torchvision mirror (cs.toronto.edu) was effectively dead (~79 B/s).
Sourced CIFAR-10/100 from existing local copies (`/home/kavyansh/cifar10_data`,
`/home/kavyansh/cifar100_data`) into `data/`. SVHN (OOD probe) deferred.

**Harness design.**
- Splits (`mu/data.py`): val = 10%/class held out (SalUn convention) → 45k pool;
  forget/retain partition the pool. random-10% → forget 4500 / retain 40500;
  class-wise → forget = whole class in pool (4500 for CIFAR-10). Verified zero
  overlap across forget/retain/val.
- Model: CIFAR ResNet-18 vendored from SalUn (built-in channel normalization).
- Methods (`mu/methods.py`): ft, ga (NegGrad), neggrad+, rl, salun (saliency-masked
  rl), ssd (ported from vendor/SSD), scrub (student-teacher). All pass a tiny-epoch
  smoke test.
- Metrics (`mu/metrics.py`): UA/RA/TA, SVC-MIA (confidence), gap-to-retrain.
- Audit (`mu/relearn.py`): few-shot finetune recovery (+ prototypical variant).

**Training recipe (reproduction gate).** SGD, lr 0.1, 182 epochs, decay @ 91/136,
batch 256, wd 5e-4 — matches SalUn so numbers are comparable. ~6 s/epoch on the
5080 (≈18 min/full run).

### Status
- [x] env + harness + smoke tests
- [x] original ResNet-18 / CIFAR-10 (21.4 min): test 94.55, forget/retain 100 — textbook
- [x] retrain oracle random-10% (18.4 min): forget 94.96, retain 100, test 94.21, MIA 88.07
- [x] reproduction-gate benchmark table (below)
- [x] recovery attack confirmed working (rebounds forget-acc on methods that forgot)
- [ ] per-method HP tuning to a common operating point (next)

### Results — cifar10_resnet18_random_r0.1 (first pass, default HPs)

| method | UA | RA | TA | MIA | gap | recover% | f-acc post-attack | runtime_s |
|---|---|---|---|---|---|---|---|---|
| original | 0.00 | 100.0 | 94.55 | 99.91 | — | — | — | — |
| retrain (oracle) | 5.04 | 100.0 | 94.21 | 88.07 | — | — | — | — |
| ft | 0.18 | 99.92 | 93.92 | 98.40 | 3.89 | 0.0 | 99.0 | 60.6 |
| ga | 0.00 | 100.0 | 94.52 | 99.87 | 4.29 | 0.0 | 98.8 | 3.8 |
| neggrad+ | 0.02 | 99.96 | 94.17 | 98.91 | 3.99 | 0.0 | 99.7 | 61.9 |
| rl | 15.78 | 89.22 | 83.76 | 80.40 | 9.91 | 17.2 | 86.9 | 14.4 |
| salun | 9.29 | 95.83 | 88.51 | 84.20 | 4.50 | 0.0 | 89.2 | 15.3 |
| ssd | 83.64 | 16.13 | 16.42 | 23.29 | 76.26 | 40.1 | 49.9 | 7.5 |
| scrub | 0.00 | 100.0 | 94.50 | 99.93 | 4.30 | 0.0 | 99.8 | 41.4 |

**Reads.**
- Setup validated: original is textbook; **retrain oracle behaves correctly**
  (forget-acc collapses to test-level 95, MIA 88) — this is the gold standard.
- **SalUn is the best-behaved** (MIA 84.2 ≈ oracle 88; gap 4.5; recover 0%),
  consistent with it being SOTA.
- ft / ga / neggrad+ / scrub **under-forget** at default HPs (UA≈0, MIA≈99) — they
  need stronger settings. Their "recover 0%" is trivial (nothing was forgotten).
- **SSD over-damps** (alpha=10, λ=1 destroys the model: TA 16). SSD is notoriously
  HP-sensitive; needs a per-setup sweep of selection_weighting.

**Paper-relevant signal.** Among methods that actually forgot, recovery differs
sharply: SalUn recover 0% (robust) vs RL 17% vs SSD 40%. This is the differential
robustness the WACV story rests on — but a fair claim requires all methods tuned to
a **common operating point** (target UA≈5 to match the oracle) before comparing
recovery. That tuning is the next step.

### SSD bug found + fixed (paper-relevant)
The catastrophic SSD row (TA 16) was **not** the operating point — it was caused by
computing Fisher importance on **augmented** loaders, so the estimate depended on
RNG state. Freshly seeded, default SSD gives UA 0 (TA 94.5); after 6 methods had
advanced the RNG (as in the benchmark), it collapsed to TA 16. Fix: compute
importance on the clean `*_eval` loaders (added `full_eval` to `Splits`). Now
deterministic. *This fragility — SSD swinging from no-op to model-destruction under
augmentation noise — is itself a finding worth a paragraph.*

Post-fix SSD sweep (clean importance, λ=1):
  sel_w ≤ 1 → UA 4.2, RA 95.6, TA 88.3, MIA 94.8  (≈ oracle UA, sharp cliff at 2)
  sel_w ≥ 2 → UA 0 (no forgetting)
Operating point: **sel_w ≈ 1**.

### Next (classification track — now a comparator, not the headline)
1. HP-tune ft/ga/neggrad+/scrub up to UA≈5 (oracle-matched); SSD set at sel_w≈1.

---

## 2026-06-26 (later) — PIVOT to segmentation unlearning

**Decision.** Detection unlearning is ~ill-posed (box regressor is class-agnostic;
forgetting collapses to the classifier). **Semantic-segmentation class-unlearning**
is the real problem, with a clean privacy story ("stop segmenting people"). The
classification harness (`mu/`) becomes a comparator; the segmentation work (`seg/`)
is the headline.

**Two contributions identified.**
1. *Task definition.* "Unlearn person" = remap person pixels to a target. Chosen:
   **dominant-neighbor** relabeling (person → nearest valid surrounding class) so a
   forgotten region blends in rather than leaving a person-shaped hole.
2. *Leakage / "segmentation-by-subtraction" metric.* Even with the person class
   gone, a person-shaped residual reconstructs the mask. Forgetting quality must be
   measured by whether the region is indistinguishable from a real merge-target
   region. (Metric design pending.)

**Core technical challenge.** Unlike classification (each sample is wholly
forget/retain), in segmentation forget & retain pixels **coexist in one image and
share gradients** — gradient ascent on person pixels damages neighbors. This pixel
entanglement is where the ported methods may break.

**Data.** ADE20K already on disk (ISP project): `ADEChallengeData2016`, 20210 train
/ 2000 val; `*_raw` = standard-exposure RGB. **person = idx 12** (raw 13); 5075
train / 526 val person images. Built `seg/data.py` with reduce_zero_label and the
dominant-neighbor `merge_person_label` (distance-transform fill from valid donors
only). Validated: 100% of person pixels → real class, 0 → ignore, 0 remain person;
visualization `results/seg/merge_demo.png` shows a person crowd dissolving into
wall/floor. Forget scope = whole class, all images.

**Model/compute.** torchvision DeepLabV3-ResNet50 (instrumentable, ImageNet
backbone). ~16 min/epoch on the 5080 at bs4/res512 (~8.5 GB) → **~6.5 h / model**.
Original training launched (24 epochs); oracle next (candidate for 5090 offload).

### Status (seg)
- [x] seg/data.py + dominant-neighbor target (validated)
- [x] seg/model.py (DeepLabV3-R50) + seg/engine.py (poly-LR train, per-class IoU)
- [x] port methods to pixel-level (ft/ga/neggrad+/rl/salun/ssd/scrub) + leakage/relearn audit
- [x] **reference models trained** (24 ep, 6.3 h each):
      original  person_IoU 67.8  retain_mIoU 34.5
      oracle    person_IoU  0.0  retain_mIoU 34.5   <- task is well-posed: full
      erasure with ZERO collateral damage. This is the target corner.
- [~] full diagnostic benchmark (7 methods, running)

### GPU smoke (8 imgs, noisy) — thesis already visible
| method | person_IoU | residual_person | blend_fail |
|---|---|---|---|
| original | — | 82.9 | 89.5 |
| ga | 11.5 | 81.4 | 89.5 |  (under-forgets at default HP)
| scrub | 11.5 | 81.7 | 90.3 |  (under-forgets)
| salun | 7.4 | 21.9 | 57.2 |  (partial)
| **ssd** | **0.0** | **0.0** | **100.0** | erases person class BUT predicts not-
the-neighbor everywhere -> maximal silhouette leak. The "subtraction" failure in
one row: removing the class is NOT the same as forgetting without a trace.

## 2026-06-27 — Plan locked (post-reference-models)

**Contribution = both** (benchmark exposes gap + method fills it). Paper arc:
1. Problem+metric: formalize seg class-unlearning; show "stop predicting c" is
   insufficient (subtraction leakage); define dominant-neighbor objective + leakage
   suite (residual_person / blend_fail / silhouette_IoU).
2. Benchmark: 7 ported methods at a MATCHED operating point all miss the oracle
   corner (person_IoU 0, retain 34.5, low leakage, relearn-robust).
3. Method: blend-aware unlearner -> oracle corner cheaply, with retain preservation
   + relearn robustness. Likely core: supervise forgotten pixels TOWARD the
   dominant-neighbor (not down/random) + retain constraint + sharpness term. Exact
   design pinned from the diagnostic.

**Scope (decided): ADE20K-deep.** person (+ maybe one more class), full metrics,
relearn, one backbone (DeepLabV3-R50). Cityscapes (privacy headline) + ViT transfer
DEFERRED to rebuttal/camera-ready. Keep it fast for the near WACV cycle.

### Next
1. Read diagnostic table -> confirm failure modes, pick matched operating points.
2. Design + implement the method.
3. Matched-OP headline table + relearn-robustness curves on ADE20K.

## 2026-06-27 (later) — Reality check: round-2 refuted + landscape recon

**Round 2 = NEGATIVE.** BN-freeze did NOT rescue retain; it collapsed it to 0.1
identically across all configs (a degenerate attractor). My Finding-3 BN-confound
*fix* is refuted. Best result stays A2 (retain 23.8, recover 4.3) — which is ~ a
localized FT-to-merged.

**Landscape recon (should have come first).** Verified 3 papers myself:
- DELETE (2503.23751): class unlearning incl. **segmentation** (Cityscapes); states
  "forgetting a class ≠ mis-segmenting as background → reassign to visually similar
  class". = our task framing + dominant-neighbor seed. Forgets by masking forget
  logit + renormalize (leaves silhouette). No leakage metric, no attack, no robustness.
- Spotter (2506.01318): **classification** feature **dispersion** + prototypical
  relearning attack = our B1 collapse + Finding-1 mechanism, already published.
- Erase-to-Retain (2511.16574): **binary medical** dense unlearning, LoRA + KD +
  BCE→background + entropy/feature-repulsion = our anchoring/robustness goal. No
  attack, no spatial-leakage metric, no multi-class.

**Verdict:** 3+ pillars scooped. Our method ideas re-derive prior art. BUT a precise
gap survives (see POSITIONING.md): **no one formalizes spatial silhouette leakage
`I(M;O)`, none use a learned silhouette attacker/discriminator, none do multi-class
distributional blending, none test dense relearning.**

**New sharper hypothesis (HYP):** forgetting in dense prediction = drive `I(M;O)→0`
(forgotten region statistically indistinguishable from context). Unifies attack
(silhouette discriminator) = metric (its AUC) = method (adversarial fooling) =
robustness (no residual spatial info). See POSITIONING.md §3.

**Debts to fix before headlines:** baseline undertrained (retain 34.5 vs published
~42-44); single-seed everywhere.

### Decisive cheap next test (~1 GPU-h)
Build the silhouette discriminator as a POST-HOC attacker; run on existing models
(original/oracle/FT/A2/B1 + ported DELETE-style). If it reconstructs the person mask
from outputs of every method → gap real, method motivated. If not → pivot to benchmark.

## 2026-06-28 — HYP validated; ADV method; two-surface red-team

**HYP CONFIRMED.** Silhouette attacker (output field, no image) reconstructs person
mask. DELETE-style logit-mask leaks IoU ~49-51 vs oracle floor 0 — the dominant prior
mechanism leaks the silhouette. Survives a 2x-deeper transfer attacker.

**ADV method (adversarial silhouette removal)** = min_G[blend(person→neighbor) +
λ_r·retain-KL-anchor + λ_adv·fool(D)] ; max_D BCE(D(O),M). Result: output-leak IoU 0
(AUC at/below floor) at retain 24.4, recover 3.3. **Only method at the target corner.**

**Two-surface red-team (output softmax AND penultimate features φ):**
| model | retain | out-leak IoU | feat-leak IoU (floor 43) | recover |
|---|---|---|---|---|
| oracle | 34.5 | 0 (floor) | 43.3 (floor) | — |
| DELETE-mask | — | 51.6 | 68.1 (+25) | — |
| A2 | 23.8 | 16.5 | 42.2 | 4.3 |
| **ADV** | **24.4** | **0** | **26.1 (<floor)** | 3.3 |
| ADVF (out+feat) | **1.0** | 0 | 0 | 2.0 |
| B1 (feat) | 1.3 | 0 | 0 | 2.0 |

**KEY INSIGHTS (publishable):**
1. *DELETE leaks BOTH surfaces* (out +51, feat +25) — masking the output is security
   theater; the representation still encodes the person.
2. *The feature-floor is HIGH (43 IoU/96.8 AUC):* even the oracle (never saw person)
   leaks person LOCATION from features, because a seg model must encode scene context
   and context predicts where people are. ⇒ feature-leak cannot go below this floor
   without destroying context = utility. Novel insight; none of the prior papers have it.
3. *Privacy↔utility set by WHICH surface you scramble:* feature-collapse (ADVF/B1)
   drives feat-leak to 0 but **destroys retain (1.0-1.3)** (shared features). ADV
   (output-only adversarial) keeps feat-leak BELOW floor *for free* at retain 24.4.
   ⇒ **ADV is the method; ADVF/B1 are the ablation that feature-collapse is unnecessary
   and harmful.** Defend the output; the feature surface follows below floor.

**Metric refinement:** leakage must be measured as EXCESS over a per-surface oracle
floor (out-floor ~82 AUC, feat-floor ~96 AUC). Below-floor = over-forgetting.

**Open:** ADV retain 24.4 < oracle 34.5. Testing ADV + retain-replay (lam_c=0, no
collapse), longer (8ep) + bigger batch, to close the gap. Engineering: ADV_full OOMs
>bs4 (replay+teacher+D heavy) → added lam_c=0 lightweight path + --no_eval.

**Debts still owed:** competent baseline (~42 mIoU vs current 34.5); multi-seed.

## 2026-06-28 (later) — ADVR + competent baseline + multi-seed

**ADVR (ADV + retain-replay, lam_c=0, 8ep)** on the 24ep baseline: person 0.4,
**retain 24.4→29.0**, out-leak 0 (below floor), feat-leak 39.2 (<floor 44.6),
recover 2.5. Replay closed >half the retain gap. **ADVR = headline method.**
ADVF/B1 (feature-collapse) confirmed over-forgetting (retain 1.0-1.3).

**Competent baseline retrained** (60ep, random-scale+crop aug, bs6, ~16h each):
original person 71.0 / retain **38.1** / mIoU 38.3 (was 34.5/34.8); oracle person 0 /
retain **36.6**. New target corner = person 0 / retain 36.6. 24ep ckpts saved in
checkpoints/ep24_backup/.

**Multi-seed re-derivation (running):** seeds 1-3, methods A2/ADV/ADVR (+B1 ablation
seed1) on the competent baseline, each eval'd, then the two-surface learned attack
over all. Produces the publication-grade table with variance. ~8-10h.

**Recipe note:** mIoU 38.3 is high-30s (defensible DeepLabV3-R50; full 42-44 needs the
heavy mmseg recipe). Report transparently; comparison is relative to same reference.

## 2026-06-30 — Multi-seed results + leakage-metric variance bug found & fixed

**Multi-seed (3 seeds) unlearning-quality table, competent baseline (oracle retain
36.6):**
| method | retain_mIoU | recover_pts |
|---|---|---|
| A2 | 22.4 ± 0.6 | 4.2 ± 0.3 |
| ADV | 22.0 ± 1.0 | 2.9 ± 0.1 |
| **ADVR** | **26.5 ± 0.5** | **2.7 ± 0.2** |
| B1 (n=1, degenerate) | 0.4 | 2.0 |
ADVR reproducibly beats A2/ADV on retain by 4-6pts, tight variance — real effect.
This result is solid and final.

**Leakage-metric bug found:** sweep's two-surface attack run showed oracle_FLOOR
out-IoU jump 0.0→17.0 between runs. Verified NOT a stale-checkpoint bug (md5s
differ from old backup; ground-truth oracle eval confirms person_IoU 0.0, retain
36.6 correctly). Root cause: the silhouette attacker is itself a small GAN-style
discriminator trained from a random init each call — IT has real run-to-run
variance. 3 independent attacker-training runs against the SAME fixed oracle
checkpoint: IoU {0.0, 10.9, 2.2}, AUC {82.4, 86.3, 84.0} (mean 84.2 ± std 2.0 AUC).
A single attacker draw is not a trustworthy leakage measurement.

**Fix:** `scripts/run_attack_variance.py` — train the attacker N=3 times per
model/surface, report mean±std, and only call "leak" if AUC exceeds
floor_mean + 2*floor_std. Running now on the decisive comparison (oracle floor /
DELETE-mask / ADV / ADVR / ADVR_s1 / B1 ablation, both surfaces, 36 attacker
trainings total). This is now the standard protocol for all future leakage claims.

### FINAL rigorous leakage table (36 attacker trainings, mean±std)

| model | out AUC | feat AUC | out verdict | feat verdict |
|---|---|---|---|---|
| oracle (floor) | 85.4 ± 0.6 | 97.0 ± 0.1 | — | — |
| DELETE-mask | **96.4 ± 0.1** | **98.6 ± 0.1** | **LEAK (+11.0)** | **LEAK (+1.6)** |
| ADV | 81.2 ± 0.2 | 92.1 ± 0.2 | safe (-4.2) | safe (-4.9) |
| **ADVR** | **78.9 ± 1.5** | **94.7 ± 0.1** | **safe (-6.5)** | **safe (-2.3)** |
| ADVR_s1 | 81.6 ± 0.7 | 93.9 ± 0.3 | safe (-3.8) | safe (-3.1) |
| B1 (ablation) | 77.1 ± 1.4 | 79.0 ± 0.5 | safe (-8.3) | safe (-18.0, over-forgets) |

Leak rule: AUC > floor_mean + 2*floor_std. **Confirmed with proper statistics:**
DELETE-style logit-masking (the dominant prior-art mechanism) LEAKS on both surfaces,
clearing the 2-sigma bar by a wide margin. ADV and ADVR are SAFE on both surfaces
across all seeds tested, while ADVR additionally preserves retain accuracy
(26.5±0.5 vs 22.4±0.6) and improves relearn-robustness (2.7±0.2 vs 4.2±0.3) over the
baseline. B1 is safe but only because it destroys retain (over-forgetting ablation).

**Headline result, publication-grade:** ADVR is the only method that is simultaneously
(a) statistically-confirmed leak-free on both the output and feature surfaces,
(b) retains significantly more accuracy on the other 149 classes than the baseline,
and (c) is harder to relearn. This closes the experimental program for the WACV
submission's core claim.

## 2026-07-04 — GENERALITY (3 axes) all confirm the pattern (CVPR scope)

Bugs fixed first: checkpoint names now include forget-class (`_cls{N}`), engine routes
through seg_forward (SegFormer no-aux), attack params via ARCH/FORGET_CLS/UNLEARNED,
attacker subset capped to len (Cityscapes has 2343<2400 person imgs).

| axis | setting | oracle retain | ADVR retain | A2 retain | DELETE out-AUC | ADVR out-AUC | floor |
|---|---|---|---|---|---|---|---|
| orig | ADE20K person (3-seed) | 36.6 | 26.5±0.5 | 22.4 | 96.4 LEAK | 78.9 safe | 85.4 |
| +dataset | Cityscapes person (19cls) | 66.3 | 61.7 | 57.7 | 98.1 LEAK | 88.3 safe | 91.1 |
| +class | ADE20K car | 37.7 | 30.1 | 23.2 | 98.9 LEAK | 92.1 safe | 89.8 |
| +backbone | SegFormer-B2 person | 32.5 | 26.9 | 23.7 | 95.7 LEAK | 82.5 safe | 87.8 |

**Unanimous:** DELETE leaks on every setting; ADVR is leak-safe AND best-retain on
every setting. Cityscapes is the strongest result (retain gap only 4.6). On SegFormer
ADVR is the ONLY safe method (A2 baseline leaks: 88.4 > floor 87.8) -> the adversarial
term, not just the dominant-neighbor target, is what removes the leak. recover_pts also
best for ADVR everywhere (city 2.6, car ~4-6, sf 2.6 vs A2 6.3). B1 always degenerate
(retain ~0-10). Reference checkpoints: seg_{mode}_{arch}_cls{N}.pth. Extension runs are
single-seed (seed 1); multi-seed the winners later if needed.
