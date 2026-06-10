# Deep read — El-Nouby et al. 2021, arXiv:2112.10740 (agent-verified 2026-06-10)

*"Are Large-scale Datasets Necessary for Self-Supervised Pre-training?"
Full text read via ar5iv (extracted copy: `elnouby-2112.10740-fulltext.txt`).
This is the primary source for our "joint-embedding family reported less
data-robust" positioning — cite it ONLY within the bounds below.*

## What they compare

- **Denoising autoencoders (DAE):** BEiT, and their own **SplitMask** (BEiT-style
  masked prediction on two disjoint patch sets + InfoNCE match loss; Sec. 4).
- **Joint-embedding:** **DINO is the sole representative** in the robustness
  analysis (Sec. 3). MoCo v3 appears only in an unrelated ImageNet table
  (Table 7). SimCLR never tested. **Barlow Twins never tested** (Related-Work
  mention only).

## The robustness protocol (Sec. 3.1, Table 1)

- Pre-train on ImageNet 1% / 10% / 100% and COCO (~118k, non-object-centric);
  **iterations held constant** by scaling epochs: "we adapt the number of
  epochs to keep the number of iterations constant … 3k and 30k epochs on
  ImageNet 10% and 1%" (300 epochs at 100%).
- **Evaluated by FULL FINE-TUNING** (iNaturalist-2019 top-1), not linear probe.

## Key numbers (Table 1, iNat-2019 fine-tune top-1)

| Method | IN 1% | IN 10% | IN 100% | COCO |
|---|---|---|---|---|
| Supervised | 71.6 | 75.0 | 75.8 | — |
| DINO (joint-emb.) | 70.1 | 73.1 | **78.4** | 71.9 |
| BEiT (DAE) | 74.1 | 74.5 | 75.2 | 74.4 |
| SplitMask (DAE) | 74.8 | 75.4 | 75.4 | 76.3 |

DINO: −8.3 full→COCO, −5.3/−8.3 full→10%/1%; DAEs ≈ flat (≤1.1 swing).
Detection (Table 4, COCO Mask R-CNN ViT-B): COCO-pretrained DINO 43.1 APbox
vs BEiT 46.7 / SplitMask 46.8.

## CRITICAL caveats for our thesis

1. **The ranking REVERSES under linear probing** (their Sec. 5.4): "denoising
   autoencoding methods typically fall behind in terms of linear probing
   compared to instance discrimination methods like DINO" (Table 8: BEiT
   linear probe 41.0 vs fine-tune 82.8 on ImageNet). Their robustness claim is
   established **only in the fine-tuning regime**; robustness-to-size under
   linear probing is simply **untested** there. Our eval is a linear probe —
   say so when citing.
2. DINO's "drop" is partly from a **higher full-data peak** (78.4 > DAE ~75.4):
   "less robust" = steeper slope, NOT uniformly worse. At 1% it is also
   absolutely worse (70.1 vs 74.8).
3. Backbones ViT-S/B at 224px on ImageNet-style data; smallest joint-embedding
   point is 1% ≈ 12.8k images (the 0.1%/1k point exists only for SplitMask,
   Fig. 2). Nothing at ViT-Tiny / 64px / 1k-image scale.
4. **Their stated mechanism is about data TYPE/curation** (hand-designed,
   object-centric-biased augmentations: Sec. 1 & 3.2 hypothesis), not a
   mechanism for the size axis.

## Bonus — published small-data over-training precedent (App. C, Fig. 6)

On Stanford Cars (8,144 images), DAE pre-training: "pre-training longer than
5k epochs leads to a severe drop in finetuning performance" (their cap:
5k epochs for small sets, Table 3; "slight overfitting" already at IN-10%
with very long schedules, Fig. 3). Useful precedent for our §4.2 over-training
question — note it is for the DAE family (even the "robust" family
over-trains on tiny sets).

## Approved citation phrasings for the paper

- "Denoising autoencoders have been reported more robust to the size and type
  of pre-training data than joint-embedding methods (represented by DINO),
  under fine-tuning evaluation [elnouby2021largescale]."
- NOT: "least data-robust", NOT "Barlow Twins is less data-robust", NOT
  unqualified claims in a linear-probe context.
