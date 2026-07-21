# Positioning & the Sharper Hypothesis (post-landscape)

Written after reading DELETE (2503.23751), Spotter (2506.01318), and
Erase-to-Retain (2511.16574) at the method level. Goal: state exactly what is
claimed, what is left, and a single sharp hypothesis that unifies the
metric/attack/method/robustness into one object.

## 1. What the prior art actually does (verified at equation level)

| | DELETE | Spotter | Erase-to-Retain | **us (gap)** |
|---|---|---|---|---|
| task | class-centric incl. **seg** (Cityscapes, qual.) | **classification only** | **binary medical** seg | multi-class scene parsing (ADE20K-150) |
| forget mechanism | mask forget logit + renormalize ("dark knowledge") | intra-class cosine **dispersion** of forget features | LoRA + **BCE→background** + entropy/feature-repulsion | — |
| where pixels go | argmax over remaining → "visually similar class" (qual. only) | n/a | **background** (binary; no neighbor exists) | distributional context blend |
| spatial leakage metric | **none** (Dice/IoU + Fig.) | **none** | **none** (Dice/IoU) | **silhouette MI — open** |
| adversarial / learned attacker | **none** | prototypical attack (feature, classification) | **none** | **discriminator on output field — open** |
| relearning robustness | **not tested** | yes (feature, classification) | not a held-out attack | **dense relearn — open** |
| multi-class context blend | **no** (qual. only) | no | no (binary) | **open** |

**Key technical reading.** DELETE forgets by *zeroing the forget logit and
renormalizing* — exactly the "remove the label, leave the hole" move our diagnostic
flagged (SSD-like: residual→0 but silhouette intact). Erase-to-Retain forgets by
*BCE→background* — literal hole-dumping, and it is binary so "blend into a neighbor"
is undefined. **Neither makes the forgotten region statistically indistinguishable
from genuine context.** Spotter scatters *features* (classification) but never touches
the *spatial output field*. So all three would **fail a spatial silhouette test**, and
none of them define one.

## 2. The one thing none of them have

In dense prediction the privacy leak is **not the label** (everyone kills it) and
**not the feature prototype** (Spotter, classification). It is the **spatial mutual
information between the forgotten region and the model's output there**:
$$\mathcal{L}_{\text{leak}} \;=\; I\big(\,M\;;\;O_\theta(x)\,\big),\qquad
O_\theta(x)=\{z_i,\phi_i\}_{i\in \text{nbhd}},\; M=\mathbf 1[y=\text{person}].$$
A region leaks iff an observer can recover the mask `M` from the output field `O`
without seeing the image. Point-estimate blending (DELETE's argmax-neighbor,
E2R's background) leaves `I(M;O) > 0` — a coherent, detectable patch. **This object
is unclaimed.**

## 3. The sharper hypothesis (HYP)

> **Forgetting a class in dense prediction = driving `I(M;O)→0`: making the
> forgotten region's output statistically indistinguishable from genuine context,
> not merely re-labeled. This is necessary for privacy (no silhouette) and
> sufficient for relearn-robustness (no spatial person-information remains to
> re-amplify).**

This single statement *is* the attack, the metric, the method, and the robustness
claim — "all related toward one goal":

- **Attack / metric** = a **silhouette discriminator** `D_ψ` trained to predict `M`
  from `O_θ(x)` on held-out images. Its reconstruction IoU / AUC is the **principled
  leakage metric** (replacing our heuristic `silhouette_IoU`) and the **first learned
  silhouette-reconstruction attack for segmentation unlearning**.
- **Method** = **adversarial silhouette removal**. Min–max:
$$\min_\theta\;\Big[\underbrace{\mathcal L_{\text{blend}}}_{\text{predict context}}
\;+\;\lambda_{\text{adv}}\,\underbrace{\text{BCE}\big(D_\psi(O_\theta(x)),\,0\big)}_{\text{fool }D\text{: region looks like context}}
\;+\;\lambda_r\,\underbrace{\mathrm{KL}(q_R\Vert q_R^0)}_{\text{retain anchor}}\Big],
\qquad
\max_\psi\; \text{BCE}\big(D_\psi(O_\theta(x)),\,M\big).$$
- **Distributional blend** (the multi-class-scene contribution): instead of a point
  target `ỹ`, the adversary forces the region's *predicted-label distribution* to
  match the context-conditional `P(y\,|\,\text{surroundings})` — so a person on a
  beach dissolves into a sand+sea *mixture*, not a uniform blob. **This is why the
  problem needs ADE20K-150 clutter** — it is exactly where DELETE/E2R's point/background
  blending visibly fails and distributional blending wins.
- **Robustness** = a consequence: if a strong `D_ψ` cannot localize the region, there
  is no residual person-information for a few-shot relearn to exploit. Measured by
  recover_pts **and** post-defense attacker AUC.

## 4. Why this is defensible vs each paper
- vs **DELETE**: they renormalize logits (leaves silhouette); we minimize `I(M;O)`
  adversarially and prove it with an attacker they never build.
- vs **Spotter**: feature dispersion in classification; we act on the **spatial output
  field** of dense prediction with a spatial attacker — their prototype attack doesn't
  even type-check on segmentation.
- vs **Erase-to-Retain**: binary background-dumping with no attacker; we do multi-class
  distributional blending with a held-out reconstruction attack and a leakage metric.
- vs **post-hoc inpainting / anonymization GANs** (the obvious reviewer question):
  those edit *pixels*; we remove the capability from the model's *learned representation*
  and **verify** removal with an attacker on the output field — a model property, not a
  pixel post-process.

## 5. Honest risks & debts (must fix before headline claims)
1. **Adversarial training is finicky** — start with `D` as a *frozen post-hoc attacker*
   (metric only), confirm the gap, then turn on the min–max.
2. **Baseline is undertrained** (retain mIoU 34.5 vs published DeepLabV3-R50 ADE20K
   ~42–44). Retrain the reference longer before any headline number.
3. **Single-seed.** Every reported delta needs ≥3 seeds + variance.

## 6. Cheapest decisive next experiment (validate HYP before building the defense)
Build `D_ψ` as a **post-hoc attacker** and run it on the models we ALREADY have
(original, oracle, FT, A2, B1, and a ported DELETE-style mask-renormalize). Prediction
from HYP: the attacker reconstructs the person silhouette (high IoU) from *every*
existing method's output **except** where the region is truly context-distributed.
- If true → the gap is real, the metric is principled, the method is motivated. Proceed
  to the adversarial defense.
- If the attacker *can't* reconstruct even from DELETE/B1 → HYP is wrong, leakage isn't
  spatially recoverable, and we pivot to the benchmark framing instead.
This is ~1 GPU-hour and tells us whether the whole direction holds.
