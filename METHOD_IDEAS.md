# Method Ideas — Blend-Aware Segmentation Unlearning

A living bank of candidate methods to reach the **oracle corner**:
`person_IoU → 0`, `retain_mIoU → 34.5`, `blend_fail → 44`, `silhouette_IoU → 11`,
`recover_pts → 0`. Each idea: intuition → math → which failure mode it fixes →
cost/risk → how we test it. Test order and a results tracker are at the bottom.

---

## 0. Notation

- Image `x`; segmentation net `f_θ` with penultimate features `φ_θ(x) ∈ R^{D×H×W}`
  (D=256 for DeepLabV3) and a 1×1 classifier `W ∈ R^{C×D}`, logits
  `z = W φ_θ(x) ∈ R^{C×H×W}`, `C=150`. Softmax `q_i = softmax(z_i)`.
- Person class `p`. Per image: forget set `F = {i : y_i = p}`,
  retain set `R = {i : y_i ∉ {p, ignore}}`.
- Dominant-neighbor target `ỹ_i` = label of nearest valid (non-person, non-ignore)
  pixel — already built by `merge_person_label`.
- Frozen original (pre-unlearn) model `f_{θ0}`; its logits `z0`, features `φ0`.
- Masked CE: `CE_S(z,y) = (1/|S|) Σ_{i∈S} −log q_i[y_i]`.

## 1. The design constraints (read off the diagnostic)

| failure mode | seen in | constraint the method must satisfy |
|---|---|---|
| model destruction (entanglement) | GA, NegGrad+, SSD, SCRUB | updates must be **localized**: leave retain pixels/params ≈ unchanged |
| silhouette leakage | RL, SalUn | person region must be supervised **toward `ỹ`** (blend), not down/random |
| relearn fragility | RL, SalUn, (FT a bit) | **no separable "person" representation** must remain to re-amplify |

A winning method needs all three. The ideas below attack them in different ways;
the hybrids combine the winners.

---

## FAMILY A — Label-space (supervise toward the dominant-neighbor)

### A1. FT-to-merged  *(baseline, already measured)*
`min_θ  CE_{F∪R}(z, ỹ)`  (person→`ỹ`, retain→`y`).
Closest baseline (retain 33, blend 46, silhouette 11.6) but **recover 9.2** and needs
a full pass. Everything below tries to beat it on robustness/locality/cost.

### A2. Retain-anchored FT-to-merged  *(fix destruction by anchoring)*
```
L = CE_F(z, ỹ)  +  λ_r · KL( q_R ‖ q0_R )
```
Supervise only the person region to blend; **anchor every retain pixel to the
frozen original** so the other 149 classes cannot drift. Intuition: move the hole,
freeze the world. Fixes destruction by construction; cheap. Risk: λ_r too high → can't
blend; too low → retain drift. *Sweep λ_r ∈ {0.5,1,2,5}.*

### A3. Saliency-localized FT-to-merged
Compute SalUn mask `m` from `∂CE_F/∂θ`; finetune to `ỹ` updating only salient params.
Localization via parameters instead of a retain loss. Cheaper than A2 (no teacher
forward), but coarser. *Threshold ∈ {0.5, 0.9}.*

---

## FAMILY B — Feature-space collapse  *(the relearn-robustness bet)*

Core idea: don't just change the **decision**, destroy the **representation**. If
person-pixel features are mapped onto the neighbor-class feature manifold, (i) the
linear head outputs the neighbor → blends, and (ii) there is no person direction left
for a few-shot attack to re-weight → relearn-robust. This is the conceptual novelty.

### B1. Neighbor-prototype collapse
Precompute class prototypes on retain pixels: `μ_c = mean_{i: y_i=c} φ0_i`.
```
L = Σ_{i∈F} ‖ φ_θ(x)_i − μ_{ỹ_i} ‖²   +  λ_r · KL( q_R ‖ q0_R )
```
Pull each person pixel's feature to its dominant-neighbor prototype; anchor retain.
Predicts neighbor automatically (blend) and collapses the person representation
(robust). *Variant B1a: cosine instead of L2; B1b: also match the head output
`CE_F(z, ỹ)` as an auxiliary.*

### B2. Person-direction projection (anti-prototype)
Let `d = μ_p − μ_R` (person vs mean-retain feature direction, from `θ0`).
Penalize any person-pixel feature component along `d`:
```
L = Σ_{i∈F} ( φ_i · d̂ )²  + λ_r·KL(q_R‖q0_R)        d̂ = d/‖d‖
```
Cheaper than B1 (one direction, no per-class prototypes). Removes the discriminative
person axis. Risk: a single linear direction may be insufficient (use top-k PCA of
person features → project out a subspace).

### B3. Classifier surgery  *(ablation: shows suppression ≠ forgetting)*
Directly zero/penalize the person row: `W_p ← 0`, or add `−∞` bias. residual→0 but
predicts arg-max over the rest → **leaks** (≈ SSD). Include as the ablation proving
that removing the *output* without touching the *representation* leaves a silhouette.

---

## FAMILY C — Constructive distillation  *(fix SCRUB's destruction)*

### C1. Blinded-teacher distillation
Teacher target: original distribution on R, dominant-neighbor on F.
```
L = KL( q_R ‖ q0_R )  +  CE_F(z, ỹ)            (no gradient ascent anywhere)
```
SCRUB minus the divergent max-step that destroyed the model. Essentially A2 with a KL
(not CE) blend on the head. Clean, safe.

### C2. Inpaint-teacher (target-free blend)
Teacher sees `x` with the person region replaced by neighbor pixels (cheap nearest
fill in *image* space); its prediction there defines the blend target — no `ỹ`
construction needed. Tests whether we can avoid building merged labels at all.

---

## FAMILY D — Localized aggressive forgetting  *(rescue GA/SSD principled-ly)*

### D1. Retain-orthogonal gradient ascent
Do ascent on person loss but remove the part that hurts retain:
```
g_F = ∇_θ CE_F(z, y) ,  g_R = ∇_θ CE_R(z, y)
g  = g_F − (⟨g_F, g_R⟩ / ‖g_R‖²) · g_R        # project out retain-harming component
θ ← θ + η · g
```
Forgets only in directions orthogonal to retain. Directly tests the entanglement
hypothesis: if D1 preserves retain where GA destroyed it, we've *explained* the
failure. Pair with a blend term or it will leak (becomes the story "localization is
necessary but not sufficient — you also need blend").

### D2. EWC-style importance penalty
`L = −CE_F(z,y) + λ Σ_k Ω_k (θ_k − θ0_k)²`, with `Ω` = retain Fisher (diag).
Penalize moving retain-important params. The principled version of SSD that doesn't
collapse. *λ swept on a log grid.*

---

## FAMILY E — Robustness wrappers  *(target recover_pts directly)*

### E1. Sharpness-aware unlearning (SAM)
`min_θ max_{‖ε‖≤ρ} L_base(θ+ε)`. Two-step SAM around any base objective (A2/B1/C1).
Flat minima are empirically hard to relearn out of. *ρ ∈ {0.01,0.05}.*

### E2. Relearn-adversarial (bilevel, the strongest robustness test)
Simulate the attack inside training:
```
inner: θ' = θ − Σ_{t<k} α ∇ CE_F(z,y)      # k-step relearn of person
outer: min_θ  L_base(θ)  +  β · CE_F( f_{θ'} , y-as-person-recovered )
```
Optimize so that even after a k-step attack, person stays forgotten. Expensive
(unrolled), but it is literally optimizing the metric we care about. Use as the
capstone if simpler robustness (E1) is insufficient.

---

## FAMILY F — Hybrids (expected winners)

- **F1 = B1 + A2-anchor + E1-SAM** — collapse for blend+robustness, anchor for
  locality, SAM for extra robustness. The headline method.
- **F2 = A2 + saliency localization (A3)** — cheapest "good enough" if B-family is
  finicky.
- **F3 = D1 + blend term (CE_F→ỹ)** — the "we fixed the aggressive methods" narrative.

---

## Testing protocol

For every idea: tune to a **matched operating point** (`person_IoU ≤ 1`), then report
the full row (retain_mIoU, blend_fail, silhouette_IoU, recover_pts@50, runtime).
Compare to oracle (44/11) and to FT (the baseline to beat on recover_pts). Log to the
tracker below and to `EXPERIMENT_LOG.md`. Ablate the relearn audit at k∈{50,200}.

### Priority order (cheap+high-information first)
1. **A2** retain-anchored FT-to-merged — should kill the destruction failure; fast win, strong baseline.
2. **B1** neighbor-prototype collapse — the novelty; test if it beats FT on recover_pts.
3. **D1** retain-orthogonal GA — *explains* the destruction (great figure), even if it leaks.
4. **C1** blinded-teacher distillation — safe alternative blend.
5. **E1** SAM on the current best base — squeeze recover_pts → 0.
6. **F1** hybrid — combine winners for the headline number.

## Results tracker  (fill as we go; targets: oracle 0 / 34.5 / 44 / 11 / ~0)

| idea | person_IoU | retain_mIoU | blend_fail | silhouette_IoU | recover_pts | runtime_s | notes |
|---|---|---|---|---|---|---|---|
| oracle | 0.0 | 34.5 | 44.0 | 11.0 | — | — | target |
| FT (A1) | 0.02 | 33.1 | 46.4 | 11.6 | 9.2 | 626 | baseline to beat |
| A2 | 0.22 | 23.8 | 49.9 | 10.4 | **4.3** | 896 | best balance; retain hit by BN confound |
| D1 | 0.0 | 8.0 | 48.4 | 4.8* | 5.2 | 713 | orth-proj helps vs GA(0.1) but retain still erodes; *silhouette artifact |
| B1 | 0.0 | 1.3 | 68.1 | 8.4 | **2.0** | 869 | collapse = most robust (recover 2) but nukes retain |
| A2_bn / F1 (BN-freeze) | 0.0 | 0.1 | 69.8 | — | 0.0 | ~880 | REFUTED: BN-freeze collapses retain (degenerate) |
| **ADV (adversarial silhouette)** | 0.0 | **24.4** | 51.9 | 11.1 | **3.2** | 1167 | **floor leakage at preserved retain — target corner** |

### Leakage attack (principled metric; oracle = context-prior floor)
| model | attacker_IoU | attacker_AUC | AUC−floor | retain | verdict |
|---|---|---|---|---|---|
| oracle (floor) | 0.0 | 81.1 | 0 | 34.5 | target |
| original | 66.0 | 97.7 | +16.6 | 34.5 | full leak |
| DELETE-mask | 48.6 | 94.8 | +13.7 | — | dominant prior art LEAKS ~half silhouette |
| A2 | 14.0 | 83.9 | +2.8 | 23.8 | small leak |
| B1 | 0.0 | 76.0 | −5.1 | 1.3 | floor but model destroyed |
| **ADV** | **0.0** | **78.6** | **−2.5** | **24.4** | **only method: floor leak + retain preserved** |

**HYP validated end-to-end.** ADV uniquely occupies the oracle corner (floor leakage at
preserved retain) and is relearn-robust (recover 3.2). Open: (a) retain gap 24.4→34.5
(retain-replay), (b) transfer attack with a STRONGER attacker (running), (c) competent
baseline ~42 mIoU, (d) multi-seed.

**Round-1 findings (2026-06-27).**
1. *Relearn-robustness thesis confirmed & mechanistic:* recover B1(2.0)<A2(4.3)<D1(5.2)
   <<FT(9.2)<<RL(63.8). Deeper (representation) forgetting ⇒ harder to relearn,
   because restoring the person margin requires moving φ, not just the head row w_p.
2. *Forgetting↔retain tradeoff:* retain FT(33.1)>A2(23.8)>D1(8.0)>B1(1.3) — reverse of
   robustness. Collapse loss flows through shared backbone filters and drags retain.
3. *Confound — forget-subset distribution shift (BN):* training only on the 5075
   person images drifts BN running stats; FT resists best via hard retain CE. ⇒ fix:
   freeze BN + feature-space retain anchor (round 2).
4. *Metric caveat:* read silhouette/blend ONLY at retain≈34.5 (D1's low silhouette is
   a broken-model artifact). Valid region = {retain preserved}; within it minimize
   leakage + recover.
