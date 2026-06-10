# Augmentation in Joint-Embedding SSL & the Curriculum-Augmentation Idea

*Research-agent report, 2026-06-04. Web-verified against primary sources. Verdicts labeled **[established]**, **[folklore]**, or **[open]**. Motivates the augmentation-space program (goal: make the 1k split keep improving like 100k).*

## Q1 — Mechanisms: how augmentation shapes what is learned

- **Augmentation defines the invariances.** SimCLR (Chen et al. 2020, arXiv:2002.05709): augmentation composition is the single most important design choice; crop + color-jitter only work *together* (random-crop views otherwise share trivial color histograms). **[established]**
- **BYOL is far more robust to augmentation removal** (Grill et al. 2020, arXiv:2006.07733): removing color costs BYOL −9.1 pts vs SimCLR's −22.2; crop-only gives BYOL 59.4% vs SimCLR 40.3%. Barlow Twins/VICReg sit closer to BYOL (redundancy-reduction, not contrastive), so color invariance is less load-bearing than for SimCLR. **[established]**
- **InfoMin sweet spot** (Tian et al. 2020, arXiv:2005.10243): downstream accuracy is a reverse-U in mutual information between views — too much shared info keeps nuisances; too little discards task-relevant signal ("missing-information regime"). InfoMin Aug 73.0% vs SimCLR 69.3% (RN50, 200ep). **[established]**
- **Augmentation as artificial data enlargement** (Moutakanni et al. 2024, arXiv:2406.09294): invariance is "useful but not necessary"; the dominant effect is artificially increasing sample count/coverage. Matters **more at small scale** (crop-only gap −1.3% at IN-1k vs −0.4% at 142M). They state explicitly: **"training longer on a smaller-scale dataset is harmful when you don't use the dataset-size-increase property of augmentations"** — our 1k-stagnation diagnosis as a published finding. **[established]**
- **Theory.** HaoChen et al. 2021 (spectral contrastive loss, arXiv:2106.04156): the "augmentation graph" connects views; classes separate well only if augmentations create enough intra-class connectivity — more/stronger aug → denser graph → better provable downstream error. Wen & Li 2021 (arXiv:2105.15134): augmentations cause "feature decoupling" onto sparse semantic features. **[established as theory; the few-sample extrapolation is open]**

## Q2 — Size/diversity of the augmented sample space

- **Multi-crop** (SwAV, Caron et al. 2020, arXiv:2006.09882): low-res local crops (2×160 + 4×96) improve *all* tested methods by **+2–4 pts** ImageNet top-1 — cleanest evidence that more views-per-image (denser orbit sampling) helps. Low-res crops are cheap → fits a 64px budget. **[established at large scale; small-data transfer open]**
- No paper cleanly parameterizes accuracy vs a scalar "augmentation-space size"; a unified diversity-axis study is **[open]**.

## Q3 — Curriculum / adaptive augmentation (Yan's idea: mild→strong on stagnation + LR re-warm)

Closest prior art:
- **CUDA** (Ahn et al., ICLR'23, arXiv:2302.05499): adaptive per-class augmentation strength ramped via a difficulty score — supervised long-tailed, but the "raise strength when the model copes" mechanism matches our trigger logic. **[established, supervised]**
- **DYNACL** (Luo et al., ICLR'23, arXiv:2303.01289): scheduled strength annealing in adversarial contrastive SSL — but **strong→weak**, the *opposite* direction. **[established, opposite direction — caution]**
- **CLSA** (arXiv:2104.07713): adds a stronger-aug branch (parallel, not scheduled). **[established]**
- **LR re-warming**: Ibrahim et al. 2024 (arXiv:2403.08763) validates re-warm + re-decay for continual pre-training (LLMs). **[established by analogy]**

**Verdict: (c) open and plausible.** No paper implements the exact recipe — *SSL, mild→strong, triggered by validation stagnation, with LR re-warm*. Theory (augmentation-graph connectivity; Moutakanni's data-enlargement) predicts it should help precisely in the 1k regime.

**Predicted failure modes:**
1. **InfoMin overshoot** — a sudden strength jump crosses the sweet spot; *ramp, don't step*.
2. **Statistic-shift instability** — abrupt aug change shifts feature statistics; BT's cross-correlation normalization can destabilize (ViT's LayerNorm is gentler than BN — mild advantage). *Re-warm LR moderately, not to peak.*
3. **Regime dependence** (DYNACL's opposite result) — the gain hinges on the mild-aug manifold being genuinely exhausted first (our 1k traces suggest it is, ~2.2k steps).

## Q4 — Augmentation families with small-data / low-res evidence

- **i-Mix** (Lee et al., ICLR'21, arXiv:2010.08887): MixUp-in-embedding for contrastive SSL; +6.5% CIFAR-100, +2.6% CIFAR-10 on MoCo-v2, **explicitly helps more as the dataset shrinks** (IN-1k only +0.4%). Strongest small-data evidence found. **[established]**
- **Multi-crop at low res** (SwAV/DINO): +2–4 pts, cheap. **[established large-scale]**
- **Stronger color jitter / RandAugment-in-SSL** (InfoMin Aug, CLSA): pushes toward the sweet spot; gains at ImageNet scale. **[established; small-data transfer folklore]**
- Random erasing/cutout, generative augmentation in SSL views: thin evidence at this scale. **[folklore/open]**

## Top 3 evidence-backed interventions for ViT-Tiny/8 @ 64px, 1k images

1. **i-Mix / MixUp-style view mixing** — best documented small-data gain (arXiv:2010.08887).
2. **Multi-crop (extra low-res local crops)** — more samples per orbit, cheap at 64px (arXiv:2006.09882).
3. **Gradual strength ramp toward the InfoMin sweet spot**, escalated only after stagnation, with moderate LR re-warm — the curriculum idea, instantiated cautiously per CUDA + Ibrahim.

Sources: arXiv 2002.05709, 2006.07733, 2005.10243, 2406.09294, 2106.04156, 2105.15134, 2006.09882, 2302.05499, 2303.01289, 2104.07713, 2403.08763, 2010.08887.
