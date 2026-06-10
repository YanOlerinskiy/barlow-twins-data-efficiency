# Training Budget Reasoning in Self-Supervised Pre-training (Steps vs Epochs)

*Research-agent report, 2026-06-04. Web-verified against primary sources. Motivated the step-denominated control machinery (warmup, probe cadence, stagnation window) with epoch-denominated bounds.*

## Q1 — How do papers justify their epoch budgets?

- **MAE** (He et al. 2021, arXiv:2111.06377): explicit schedule ablation up to **1600 epochs** — *"accuracy improves steadily with longer training… we have not observed saturation even at 1600 epochs."* Contrasts: *"MoCo v3 saturates at 300 epochs for ViT-L."* Notes "epochs" aren't comparable across methods (75% masking → encoder sees 25% of patches/epoch vs 200%+ for two-crop methods).
- **MoCo v3** (arXiv:2104.02057): 300 epochs default; ablation shows saturation: ViT-S 72.5→73.4% (300→600 ep), ViT-B 76.5→76.7%. (Partly via secondary sources, corroborated by MAE's citation.)
- **DINO** (arXiv:2104.14294): 100/300/800-epoch configs; budget appears compute-driven; no explicit overtraining argument verified (*uncertain*).
- **solo-learn** (da Costa et al. 2022, arXiv:2108.01775): benchmark default 400 epochs on ImageNet-100; not a budget-justification study.
- **Small-dataset SSL** — **El-Nouby et al. 2021** (arXiv:2112.10740): for tiny sets (~8k images) they **cap pre-training and warn longer is counterproductive** — *"pre-training longer than 5k epochs leads to a severe drop in finetuning performance."*

## Q2 — Overtraining / data repetition

- **Muennighoff et al. 2023**, *Scaling Data-Constrained Language Models* (arXiv:2305.16264) — verified numbers: up to **~4 epochs** of repeated data ≈ fresh data (4-epoch run only **0.5%** worse val loss than unique-data); meaningful gains persist to **~16 epochs** (fitted repetition half-life R\*_D ≈ 15); beyond that returns *"diminish extremely fast."*
- Vision-SSL analogue: arXiv:2112.10740 (degradation with longer training on small sets; augmentations act as effective fresh data). **No quantitative repetition law for vision SSL found.**

## Q3 — Steps vs epochs

- **Direct precedent for fixing STEPS across dataset sizes**: arXiv:2112.10740 — *"To decouple the effect of using smaller datasets and the effect of doing less training updates, we adapt the number of epochs to keep the number of iterations constant"* (10% ImageNet → 3000 epochs, 1% → 30000, **capped on the smallest sets**). Explicitly argues epochs are the wrong unit when dataset size varies.
- LLM literature reasons in tokens/steps universally; vision-SSL libraries configure in epochs.
- *Not found*: precedent for **step-based probe-cadence / patience windows** — our approach is slightly novel, defensibly so.

## Q4 — Heuristics tying steps to dataset size

**None found** (no steps ∝ √N or ∝ N rule). The only operational heuristic: constant total iterations across N plus a hard cap at the smallest N (arXiv:2112.10740).

## How this shaped our design

- Equal-compute (constant steps) is the defensible budget comparison across split sizes; v1's hand-tuned epoch schedule was implicitly approximating this.
- Control machinery (warmup, probe cadence, stagnation window) runs in **steps** (the optimizer's clock); bounds stay in **epochs** (human-facing); epochs/repetitions reported as derived quantities.
- Subsequent empirical lesson (not from this survey): a purely step-uniform stagnation window truncates large splits — the final criterion became *window = max(4 probe-intervals, 20 epochs)*, with cadence ≈ every 8 epochs clamped to [50, 500] steps. See `run_all_splits.py::SPLIT_CONFIGS` for the evaluated table and the 100k ablation numbers (5/10/20-epoch windows → 65.10/67.60/70.25 linear-probe).
