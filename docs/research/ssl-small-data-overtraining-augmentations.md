# SSL Degradation on Small Data & Augmentation×Scale

*Research-agent report, 2026-06-04. Web-verified against primary sources (some figures read via ar5iv HTML mirrors — treat exact decimals as approximate pending PDF confirmation). Motivated by the observed peak-then-decline of the kNN probe at split=1k vs continued slow improvement at 100k.*

## Q1. Mechanisms for small-data SSL degradation — evidence vs folklore

**Augmentation/data overfitting (memorization of a depleted view distribution) — well evidenced.** El-Nouby et al. 2021 (*Are Large-scale Datasets Necessary for Self-Supervised Pre-training?*, arXiv:2112.10740) hold *total iterations* fixed across dataset sizes, so a small set is seen for vastly more epochs; downstream performance degrades because the network exhausts the finite image+augmentation distribution and memorizes rather than generalizes.

**Dimensional/feature collapse over long training — mechanistically established, but its causal link to small-data peak-then-decline is inferential, not directly measured.** Jing, Vincent, LeCun & Tian 2022 (*Understanding Dimensional Collapse in Contrastive SSL*, arXiv:2110.09348, ICLR 2022) identify two causes: (1) **strong augmentation** — when augmentation variance exceeds data variance along a feature direction, that direction's singular value collapses; (2) **implicit regularization** — over-parameterized nets drive low-rank solutions. The paper does **not** study dataset size, batch size, or training-duration severity; linking collapse to our kNN decline is a plausible hypothesis, not their finding.

**Representation drift while SSL loss still decreases — folklore/under-evidenced**; no primary source isolates it as the driver.

**Instance memorization helping vs hurting — nuanced.** Wang et al. 2024 (*Memorization in SSL Improves Downstream Generalization*, arXiv:2401.12233) show substantial per-sample memorization across contrastive *and* non-contrastive SSL and argue it can *aid* downstream generalization — memorization per se is not automatically the failure mode; the failure is overfitting the *augmentation manifold* on tiny data.

## Q2. Is peak-then-decline documented? (method families)

**Yes, directly** (arXiv:2112.10740 full text). On Stanford Cars (8,144 images): *"pre-training longer than 5k epochs leads to a severe drop in fine-tuning performance"*; on 10% ImageNet, strong results to ~3k epochs then *"slight overfitting."* Attributed mechanism: **overfitting from repeated exposure to a small image set**.

**Method-family difference is the paper's headline result.** Masked/denoising autoencoders (BEiT; their SplitMask) are markedly **more robust** to small data than joint-embedding methods. DINO degrades substantially on small subsets (e.g., iNaturalist ~78.4→70.1 on 1% subsets) while BEiT/SplitMask stay near-flat. **Masked methods resist peak-then-decline; joint-embedding methods (BT/SimCLR/BYOL/DINO) are the vulnerable family.**

## Q3. Barlow Twins specifics

From Zbontar et al. 2021 (arXiv:2103.03230, ICML), verified ablations: accuracy **"almost unaffected for a batch as small as 256"** (tested 256–4096; nothing below); accuracy **keeps improving up to 8192 projector dims** (unlike SimCLR/BYOL which saturate); removing **batch-dimension normalization substantially hurts** — the cross-correlation estimate's stability matters. But there is **no published analysis of cross-correlation rank deficiency** when (small batch × few unique images) is small relative to projector dim D, and no sub-ImageNet small-data experiments. The rank-deficient-C mechanism at 1k images / batch 256 / D=1024 is **plausible but folklore**. (A recent info-geometric BT analysis exists, arXiv:2510.10980 — small-data claims unverified.)

## Q4. Augmentation strength × dataset size

- **Moutakanni et al. 2024** (*You Don't Need Domain-Specific Data Augmentations When Scaling SSL*, arXiv:2406.09294, NeurIPS): with DINOv2, **augmentation's benefit shrinks as data grows** — original-vs-minimal-aug gap −1.3% (IN-1k) → −1.0% (IN-22k) → −0.4% (LVD-142M). Augmentation acts primarily as **artificial dataset enlargement** → matters most at small data.
- **SimCLR** (Chen et al. 2020, arXiv:2002.05709): crop+color is the critical pair; SSL benefits from **stronger** augmentation than supervised learning.
- **Inverted-U caution**: Tian et al. 2020 InfoMin (arXiv:2005.10243): too little *and* too much augmentation both hurt; sweet spots ≈ color-jitter strength 1.0, blur σ≈1.0, patch offset ≈128px.
- **Cookbook** (Balestriero et al. 2023, arXiv:2304.12210): augmentation *defines* learned invariances; larger policies push invariance into the projector rather than the encoder.

## Q5. Small-image (32–64px) constraints

No clean 64px law. InfoMin's crop sweet spot is **resolution-scaled** (less spatial room at 64px); CIFAR-scale SSL is viable but augmentation-sensitive. Our min_scale 0.08 collapse at 64px is consistent with Jing mechanism (1).

## Established vs Folklore

| Claim | Status |
|---|---|
| Peak-then-decline on small data exists | **Established** (El-Nouby 2021) |
| Masked > joint-embedding robustness on small data | **Established** (El-Nouby 2021) |
| Strong augmentation can cause dimensional collapse | **Established mechanism** (Jing 2022) |
| Dimensional collapse *causes* the small-data decline | Plausible, not directly shown |
| Augmentation benefit ↓ as data ↑ | **Established** (Moutakanni 2024) |
| Inverted-U augmentation sweet spot | **Established** (Tian 2020) |
| BT robust at batch 256, gains with large D | **Established** (Zbontar 2021) |
| BT cross-correlation rank-deficiency at tiny data | Folklore / unverified |
| Representation drift as a distinct driver | Folklore / under-evidenced |

## Hypotheses for an augmentation-strength × dataset-size ablation

1. **Inverted-U shifts with data** (Tian 2020 + Moutakanni 2024): sweep min_scale ∈ {0.08, 0.25, 0.5} × size {1k, 10k, 100k}.
2. **Stronger augmentation changes the small-N decline** — accelerates it (collapse view) or delays it (augs-as-data view); the *sign* of the interaction is the open question. Note: the two established results predict opposite signs — apparently untested; a publishable gap.
3. **Decline coincides with rising dimensional collapse** (Jing 2022): track the embedding covariance spectrum. *(Instrumented in every run since tag with `knn/effective_rank`.)*
4. **Reducing projector D mitigates small-N decline** (conditioning of the C estimate; unverified BT regime).
5. **Masked-style pretext is robust at 1k where BT isn't** (El-Nouby 2021). *(Checkable for free against Dimo's MAE curves in the group CSV.)*

**Key sources (arXiv):** 2112.10740 (El-Nouby), 2110.09348 (Jing), 2103.03230 (Zbontar/BT), 2002.05709 (SimCLR), 2005.10243 (InfoMin), 2406.09294 (Moutakanni), 2304.12210 (Cookbook), 2401.12233 (memorization).
