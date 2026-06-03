# Poster Brief — Barlow Twins Data Efficiency (CSE3000 Midterm)

This document contains all the project context and current results needed to design the midterm poster. The headline asset is `reducing-data-in-visual-ai/evaluation/learning_curve.png` — the data-efficiency curve.

## Metadata

- **Author**: Yan Olerinskiy
- **Course**: CSE3000 Research Project, TU Delft, Q4 2025/26
- **Responsible Professor**: Jan van Gemert
- **Supervisors**: Petter Reijalt, Alex Manolache
- **Group**: 5 students, each implementing one what isSSL method:
  - **Barlow Twins** — Yan Olerinskiy *(this poster)*
  - DINO — Leonid Margulis
  - MAE — Dimo Terziev
  - MoCo — Makar Kuleshov
  - I-JEPA — Maksim Plotnikov

## Headline numbers (use prominently)

- **Random-init baseline**: 40.29 %
- **Barlow Twins @ 1k pretraining images**: 53.91 % (+13.6pp over random)
- **Barlow Twins @ 100k pretraining images**: 69.80 % (+29.5pp over random)
- **Total improvement from 1k → 100k**: 15.9pp (100× more data)
- **Saturation point**: roughly 16k images; marginal gain < 1pp per doubling beyond that
- **Total compute cost**: ~$2 of rented GPU time for the full sweep

## 1. Background and motivation

Modern visual AI is powered by enormous datasets, but access to them is uneven: most large-scale image data sits with a handful of large companies, while universities and independent researchers have far less. Large datasets are also hard to curate, which complicates control over fairness, copyright, and bias.

This project investigates how to train visual AI **more data-efficiently**. The standard recipe is a Vision Transformer (ViT) pre-trained self-supervised on a generic dataset, then adapted to a downstream task. Self-supervised methods are mostly evaluated at large scale; they have not been carefully compared in the small-data regime that universities actually live in.

Following the *"scaling down deep learning"* idea, we deliberately reduce both data and compute, and produce **learning curves** (downstream accuracy vs pre-training dataset size) for five popular SSL methods under matched conditions.

### Research question

> *How does the data efficiency of **Barlow Twins** (Zbontar et al., 2021) compare to other self-supervised methods when used to pre-train a ViT-Tiny/8 backbone on Tiny ImageNet subsets and adapted to CIFAR-10 with linear probing?*

The group-level question is *"how do the five methods compare on data efficiency"*; the individual question (this poster) is the Barlow Twins-specific version.

### Why Barlow Twins

- Non-contrastive: doesn't need a queue of negatives (MoCo) or large batches (SimCLR).
- Loss is the cross-correlation matrix of two augmented views' projections, pushed toward the identity — combines an invariance term (diagonal → 1) and a redundancy-reduction term (off-diagonal → 0).
- Known to work well at small batch sizes — a hint it may also be sample-efficient.

## 2. Setup

### Shared experimental pipeline (used by all 5 group members)

- **Backbone**: ViT-Tiny/8 — 5.7M parameters, 192-dim embeddings, 12 transformer layers, 3 heads, 8×8 patches on 64×64 inputs, MLP hidden 768. Randomly initialised.
- **Pre-training dataset**: Tiny ImageNet (100k images, 64×64, 200 classes; from HuggingFace `Maysee/tiny-imagenet`).
- **Pre-training subset sizes (x-axis)**: 1k, 2k, 4k, 8k, 16k, 32k, 64k, 100k images. Subsets are **nested** (smaller is a strict subset of larger) and **stratified by class** (any prefix divisible by 200 is exactly class-balanced).
- **Downstream task**: CIFAR-10, upsampled bilinear 32 → 64 to match pre-training input size, ImageNet-stat normalised.
- **Downstream protocol**: **linear probing** — encoder frozen, single `nn.Linear(192, 10)` trained from scratch on top. SGD + momentum 0.9, cosine LR schedule, base LR = 0.1·(batch/256), 100 epochs, best test top-1 reported. (Standard DINO App. B.2 / MoCo v3 protocol.)

### Barlow Twins-specific configuration

- **Projection head**: 3-layer MLP 192 → 1024 → 1024 (paper uses 8192; scaled down for our 192-d encoder).
- **Loss**: cross-correlation → identity, λ = 1e-2 (scaled to compensate for smaller projector; paper's λ·D ≈ 41 vs ours ≈ 10).
- **Optimiser**: AdamW, base LR = 3e-4, weight-decay = 1e-4, 10-epoch linear warmup → cosine to 0, gradient clipping = 1.0 (ViT-SSL stability).
- **Augmentations**: lightly's BYOL view-1 / view-2 transforms (asymmetric solarize on view-2). GaussianBlur disabled (inputs are 64×64). `RandomResizedCrop` min_scale = 0.25 (default 0.08 too aggressive at 64×64).
- **Training budget per split**: ~10k optimiser steps per split (equal compute), with a 200-epoch floor for the largest splits. 100 epochs for 64k and 100k. Single seed (42).
- **Library choice**: `lightly` for loss + projection head + augmentations; HuggingFace `transformers` for the ViT backbone. (Considered: `solo-learn` — too invasive; Facebook reference — hardcoded for ResNet/ImageNet/LARS.)
- **Compute**: single rented RTX 5060 Ti via vast.ai. Full sweep cost ~$2.

### Evaluation protocol (group-standardised, Leo's protocol document)

Necessary to ensure cross-method comparability:

- Linear classifier `Linear(192, 10)`, freshly initialised.
- Backbone frozen, `eval()` mode, no gradient flow.
- SGD with momentum 0.9, weight-decay 0, cosine LR schedule, base LR scales with batch (0.1·B/256).
- 100 epochs, train with `RandomHorizontalFlip(p=0.5)`, evaluate on full CIFAR-10 test split each epoch, report best test top-1.
- **Random-init baseline** (no pre-training loaded) is run with identical protocol per method, using each method's own backbone library/init — gives a method-specific "no SSL" floor.

## 3. Current results

The headline plot (`reducing-data-in-visual-ai/evaluation/learning_curve.png`) is the data-efficiency learning curve. Suggested visual layout for the poster: this plot at large size, surrounded by the key takeaways below.

### Raw numbers

| Condition | Pre-training images | CIFAR-10 linear-probe accuracy | Lift over random |
| --- | ---: | ---: | ---: |
| Random-init baseline | 0 | **40.29 %** | — |
| Barlow Twins | 1 000 | 53.91 % | +13.6pp |
| Barlow Twins | 2 000 | 56.28 % | +16.0pp |
| Barlow Twins | 4 000 | 60.77 % | +20.5pp |
| Barlow Twins | 8 000 | 64.90 % | +24.6pp |
| Barlow Twins | 16 000 | 66.51 % | +26.2pp |
| Barlow Twins | 32 000 | 67.45 % | +27.2pp |
| Barlow Twins | 64 000 | 68.96 % | +28.7pp |
| Barlow Twins | 100 000 | **69.80 %** | +29.5pp |

### Three takeaways for the poster (suggest as bullet boxes)

**1. Sample-efficient even at the smallest scale.**
At just 1 000 pre-training images (1% of Tiny-IN), Barlow Twins delivers a +13.6pp lift over a randomly-initialised encoder. For comparison, the same protocol applied to DINO at 1 024 images yields 40.3% — basically *at* random. **Barlow Twins extracts useful representations at data scales where DINO doesn't yet.**

**2. Diminishing returns above ~16k images.**
Going from 16k → 32k gains +0.94pp; 32k → 64k gains +1.51pp; 64k → 100k gains +0.84pp. The curve clearly bends. Either ViT-Tiny is hitting its representation ceiling at this scale, or CIFAR-10 linear probing is hitting its ceiling, or both. Distinguishing these requires PEFT fine-tuning or a larger backbone — out of scope for the midterm.

**3. Methodological finding: equal-compute schedules over-train small splits.**
At split=1k, the in-training kNN diagnostic peaked at ~25% of the training budget and degraded afterwards — the encoder learned representations early, then drifted under continued optimisation of the SSL objective. The `final`-checkpoint number (53.9%) is below the peak (~57%). **Direct empirical evidence for switching to convergence-based stopping in the next round** — a group-level methodology decision already in motion.

### Validation: random-init baseline aligns with the group

The random-init baseline at 40.29 % matches DINO's random-init baseline at 40.49 % within seed/initialisation noise. Both use the HuggingFace ViTModel default init (`trunc_normal_`, std=0.02). This **confirms that the evaluation pipeline is comparable across methods** — a precondition for the group-aggregated learning curve scheduled for Week 6.

(Note: MoCo's random-init baseline reads ~45% because it uses a different backbone library — timm with the MoCo-v3 init scheme — illustrating that the random-init floor is **architecture- and init-dependent**, not method-dependent.)

## 4. Possible future steps

In priority order for the remaining 5-6 weeks of the project:

### Week 6 (MVP milestone)

- **Group-aggregated learning curve**: overlay all five methods (BT, DINO, MoCo, MAE, JEPA) on one plot using the shared `evaluation/results.csv`. Identify the data regimes where each method dominates. This is the headline figure for the final paper.

### Weeks 7-9 (extensions and rigour)

- **Convergence-based training**: replace the equal-compute budget with early stopping on a held-out signal (kNN patience on a non-CIFAR-10 dataset, e.g. CIFAR-100, to avoid leakage). Re-run the curve with proper convergence detection — likely raises the small-split numbers.
- **Multi-seed runs**: 3 seeds × 8 split sizes = 24 runs. Report mean ± std per cell with shaded error bands on the learning curve.
- **PEFT fine-tuning**: in addition to linear probing, evaluate the encoders via Parameter-Efficient Fine-Tuning. PEFT typically lifts numbers by 5-15pp and exposes a different kind of data-efficiency comparison.
- **Augmentation ablation specific to BT**: the Barlow Twins paper notes openness to richer augmentations. Test whether augmentation choices materially affect the small-data end of the curve. Natural follow-up to the "min_scale=0.25" decision made for this run.

### Speculative / time-permitting

- **Random-init baseline harmonisation across the group**: the +5pp gap between Yan's/Leo's HF-init baseline and Makar's timm-init baseline is currently a discrepancy. A short additional experiment running each method with both backbone libraries would isolate "method effect" vs "init effect" and strengthen the group plot's interpretation.
- **VTAB / domain-shifted downstream tasks**: CIFAR-10 is a natural-image benchmark close to Tiny-IN. The literature suggests SSL methods diverge more on out-of-domain tasks. Testing a small VTAB subset (specialised tasks, but VTAB-1k size) would broaden the data-efficiency claim.

## 5. Design notes for the poster

- **Headline figure**: `reducing-data-in-visual-ai/evaluation/learning_curve.png` — the data-efficiency curve. Place large and central. The dashed red baseline + shaded SSL-benefit region makes the story self-explanatory.
- **Colour palette suggestion**: a clean academic look — Barlow Twins curve in TU Delft blue (`#00A6D6`) if the design tool can match, otherwise default matplotlib blue. Random-init baseline in muted red (`#d62728`).
- **Three bullet boxes** (next to / below the plot): the three takeaways from §3. Punchy, ~1-2 sentences each. The +13.6pp lift at 1k is the strongest single talking point.
- **Visual hierarchy**: research question prominently up top; plot dominates the centre; takeaways adjacent; setup + future-steps as smaller sidebars. Avoid wall-of-text.
- **No equations** unless space demands it. The Barlow Twins loss formula `Σ(C_ii − 1)² + λ·Σ_{i≠j} C_ij²` could fit in a small "method" box if needed, but the audience is mixed and verbal explanation suffices.
- **Acknowledgements**: include responsible professor + supervisor names plus the group teammates working on the other four methods. CSE3000 Research Project, TU Delft Q4 2025/26.

## 6. Anticipated audience questions (and short answers — useful for poster Q&A)

- *"Why does your random-init baseline differ from teammates'?"* — Same architecture, different init scheme (HF ViTModel vs timm MoCo-v3). Each method probes against its own architecturally-matched baseline.
- *"Why did you over-train at split=1k?"* — Equal-compute schedule, set before observing the kNN drift. The next round uses convergence-based stopping.
- *"Is Barlow Twins genuinely better than DINO or is it your protocol?"* — Random-init agreement with DINO's baseline confirms protocol comparability. With identical eval, BT delivers a +13.6pp lift at 1k where DINO delivers ~0pp.
- *"Why saturation above 16k?"* — Most likely ViT-Tiny capacity + CIFAR-10 linear probe ceiling. Larger backbone or full fine-tuning planned as a follow-up.
- *"Multi-seed?"* — Single seed for the midterm; multi-seed planned for the final paper.
