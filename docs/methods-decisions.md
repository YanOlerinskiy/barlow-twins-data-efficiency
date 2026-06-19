# Methods & Design Decisions — Barlow Twins data-efficiency study

Defense-oriented reference: the full pipeline + every non-obvious decision with its
rationale, the alternatives rejected, and the citation. Deeper backing for most points is
in `docs/research/*.md` (cited inline as [report: …]). Exact hyperparameters live in the
code; this explains the *why*.

---

## 0. Research framing

*This document records WHAT we do and the engineering rationale for each choice. It does NOT
claim novelty: the training recipe and in-domain validation are standard, sensibly-assembled
components, not contributions. This is a careful characterization study — every empirical
statement must be backed by a measurement; interpretations are labelled as such.*

- **Question (individual):** how does Barlow Twins' (BT) downstream accuracy scale with
  pre-training set size for a small ViT in a small-compute regime, and how does it behave at
  the smallest scales?
- **Data efficiency, not compute efficiency.** We study accuracy vs **#images (N)** — N is the
  x-axis / controlled variable. We deliberately do **not** hold compute/steps equal across
  splits (that would answer a compute-efficiency question). Instead each split trains a
  **generous fixed epoch budget**, and we report the accuracy of the **best in-domain-validation
  checkpoint** within that budget (after a short cooldown). **We do NOT claim "convergence"** —
  the budget may not reach it; any split whose validation is still rising at its budget end is
  disclosed (and can be extended via resume). Epochs are the training-length knob, held
  generous; not a controlled variable.
- **Group context:** 5 students, shared pipeline, one aggregated 5-method data-efficiency
  curve. Shared protocol (backbone, splits, CIFAR-10 linear probe) is fixed; my part is the BT
  method and characterizing its small-data behaviour.

---

## 1. Backbone & method (shared / inherited)

- **Backbone:** ViT-Tiny/8 at 64×64 — HF `ViTModel`, patch 8 → 64 patch tokens + CLS, 192-dim,
  12 layers, 3 heads, MLP 768, **5,388,480 params** (counted from the code 2026-06-10; the
  textbook "~5.7M ViT-Tiny" figure is for patch-16@224 — our patch-8@64 embedding layers are
  smaller; projector adds 2,298,880 → 7,687,360 total). Embedding = **CLS token after final
  LayerNorm** (`model.embed`). Config in `shared_config.py:VIT_TINY_8_KWARGS`.
- **Why ViT-Tiny/8 @ 64px:** group-fixed; matches Tiny-ImageNet's native 64px (no upscaling
  for pre-training); small enough for a single consumer GPU.
- **Barlow Twins loss:** cross-correlation matrix of two views' projector outputs pushed to
  identity (invariance diagonal→1 + redundancy off-diagonal→0), weighted by λ. We use
  **lightly's reference `BarlowTwinsLoss`** rather than hand-rolling it — a correctness
  argument for the methods section (the loss has classic silent-bug traps: wrong-axis
  normalization, off-diagonal scaling). Zbontar et al. 2021 (arXiv:2103.03230).
- **Projector:** 192 → 1024 → 1024 (`BarlowTwinsProjectionHead`). Scaled down from the
  paper's 8192 for ViT-Tiny/small data.
- **λ = 1e-2** (`--lambda_param`). Note: lightly's default is 5e-3; we use 1e-2. **Corrected
  2026-06-10:** the BT paper does NOT publish a λ ablation range (it only says "We ran a search
  … found the best results for λ = 5·10⁻³", at projector D=8192) — so "in the paper's ablation
  range" is unsupported. Honest phrasing: chosen a priori when scaling the projector
  8192→1024 (rough λ·D heuristic), **not ablated** — disclose as such.
- **Why `lightly`, not timm/solo-learn:** lightly gives the *reference loss* + the published
  *two-view transform*; timm is a backbone zoo (no SSL losses); solo-learn/VISSL impose their
  own training loop, but our whole contribution lives in a custom loop. We keep the HF ViT
  backbone (group consistency) and only borrow lightly's loss + transforms.

---

## 2. Data & splits (shared)

- **Pre-training:** Tiny-ImageNet (`Maysee/tiny-imagenet`, 100k train images, 200 classes,
  64px native).
- **Splits (x-axis):** {1k, 2k, 4k, 8k, 16k, 32k, 64k, 100k}, **nested & class-stratified**:
  a single master shuffled index list (`tiny_imagenet_shuffled_indices.json`, seed 42) built
  by interleaving one image per class, so any prefix divisible by 200 is exactly
  class-balanced and smaller splits are subsets of larger ones. Lets us attribute curve
  differences to N alone.

---

## 3. Pre-training schedule (the training recipe — standard components, not a novelty claim)

**Shape:** linear warmup (steps) → **constant LR** (stable phase, the full epoch budget) →
pick the **best-validation checkpoint** → short **cooldown branched from that checkpoint**
((1−√t) decay to ~0). No early-stopping trigger; the run always completes its budget.
This is **WSD** (warmup-stable-decay) — we assemble and justify standard components
(Hägele/MiniCPM); we do not claim the schedule as a contribution.

Decisions, each defensible:

1. **Constant LR, not cosine.** *Why:* (a) under cosine, the best model only appears at the
   *scheduled* horizon; an early/mid-cosine checkpoint is under-annealed — Chinchilla shows a
   cosine read before its horizon is strictly worse than one horizoned to that point
   (arXiv:2203.15556 App. B; SGDR arXiv:1608.03983). (b) Constant LR makes **every checkpoint
   schedule-comparable**, so best-val selection isn't confounded by where it sits on a decay
   curve. (c) It commits to **no training horizon** → runs are resumable/extendable.
   *Grounding:* WSD — Hägele 2024 (arXiv:2405.18392, constant+cooldown matches cosine),
   MiniCPM (Hu 2024, arXiv:2404.06395); vision/MAE evidence Singh & Janson 2025
   (arXiv:2503.02844). [report: epoch-budgets-and-lr-scheduling, early-stopping-on-cosine]

2. **No early-stopping trigger (we dropped an earlier kNN-stagnation trigger).** *Why:* the
   trigger machinery (patience/min_delta) was complex and hard to justify; classic early
   stopping is an *overfitting* remedy for supervised training that doesn't transfer to
   label-free SSL; and stopping *on a cosine* gives the under-annealed checkpoint above.
   Replaced by "train a generous budget → select best-val → cooldown." [report:
   how-researchers-pick-epochs, early-stopping-on-cosine]

3. **Best-validation checkpoint selection + cooldown *from the best* (rewind), not from the
   last step.** *Why:* small splits *over-train past their peak* (we measured a 1k run
   declining ~7pp from peak even with annealing), so cooling the last checkpoint would anneal
   a degraded model. We rewind to the peak. *Anneal matters:* annealing before delivering was
   worth **~+1.5pp** in our 16k measurement. The rewind-to-best variant is an adaptation of
   MiniCPM/Hägele's branch-from-checkpoint (they branch *forward*; we rewind to the best-val
   point — disclose as non-standard). [report: epoch-budgets-and-lr-scheduling §Q2c]

4. **Cooldown shape = (1−√t).** *Why:* it's the empirical winner of Hägele 2024's cooldown-
   shape ablation (beats linear, matches/edges cosine: fast initial drop, long low-LR tail).
   Configurable via `--cooldown_shape`; the margin over cosine is small and untested for BT —
   disclose.

5. **Cooldown length = 0.2 × best_step (≥500 steps floor).** *Why:* Hägele 2024: cooldown
   benefit plateaus at ~20% of steps, surpasses cosine at 10–20% (Fig. 5); MiniCPM §4.3 (the
   correct source of the quote — verified 2026-06-10, it is NOT in Hägele): "a decay of 10% of
   the total tokens is sufficient … while a decay of 2.5% of total tokens falls short."
   We size it to **best_step** (the training that
   produced the peak), *not* the full budget — because budgets overshoot the peak, so
   "20% of budget" would give an over-long cooldown when the peak is early. The 500-step floor
   guarantees a *minimum useful* cooldown (a too-short anneal has no low-LR tail); it rarely
   binds at production budgets.

6. **Warmup = 500 optimizer steps (not "10 epochs").** *Why:* warmup is about #optimizer
   updates to stabilize, not data passes. "10 epochs" (BT/DINO convention) assumes one large
   fixed dataset; on our splits it gives 40 steps for 1k (too short, risky at 5e-4) to 4000 for
   100k. 500 steps guarantees a safe ramp for every split. It's a heuristic round number
   (low-sensitivity with grad-clip + moderate LR); disclose, don't over-claim.

---

## 4. Validation probe (in-domain Tiny-ImageNet kNN)

- **What:** every `--knn_every_epochs` (default 4), run a kNN probe and track the
  best-*smoothed* value to select `_best`. Gallery + query = **stratified-disjoint partitions
  of the labelled TI `valid` split** (db 5000 / query 5000 = 25 imgs/class each, k=20).
- **Why in-domain TI, not the downstream CIFAR-10:** selecting the shipped checkpoint by a
  *downstream* probe is "task-aware" — peeking at the eval domain. Published practice probes
  **in-domain** (DINO `eval_knn.py`: db=ImageNet-train, query=ImageNet-val, k=20). Switching to
  TI removes the bias and moves the loop toward convention. We measured TI peaks ~700 steps
  *later* than CIFAR — confirming the downstream proxy would stop the in-domain representation
  early. [report: ssl-pretraining-validation-monitoring]
- **Why the stratified helper (not a prefix slice):** the TI `valid` split is **grouped by
  class** (rows 0–49 = class 0, …). A naive `[:5000]`/`[5000:]` slice would put classes 0–99
  entirely in gallery and 100–199 in query → **zero class overlap → ~0% kNN**. The helper
  splits *each* class 25/25 so both partitions span all 200 classes. (Verified empirically.)
- **k=20:** DINO's default. Borderline-sparse here (25 gallery/class vs DINO's ~1300), so the
  signal is noisier — handled by smoothing.
- **Smoothing (window 3):** the probe is *deterministic* (fixed gallery/query), so probe-to-
  probe jitter is **real model wobble** (constant LR keeps the model moving) **amplified by a
  brittle 200-way/25-gallery metric** — not measurement noise. Best-val on a smoothed curve
  avoids selecting a fluke spike. Also mitigates model-selection (winner's-curse) inflation,
  which grows with #checkpoints (Cawley & Talbot 2010). [report: validation-eval-cadence]
- **Cadence in epochs (`--knn_every_epochs=4`), converted to steps from loader length.** *Why:*
  one interpretable number; batch 250 makes the step conversion exact; small splits (which run
  more epochs) automatically get more probes — no opaque per-split step table. A *constant
  step* cadence would starve small splits (1k runs only ~4k steps total). [report:
  validation-eval-cadence]
- **CIFAR-10 probe retained as an opt-in logged-only diagnostic** (`--diagnostic_cifar_probe`):
  never affects selection; gives the dense kNN(TI)↔kNN(CIFAR) correlation figure (a validity
  check) and was used in the seed-42 saturation pass.

---

## 5. Per-split epoch budgets & runtime

- **Budgets (`run_all_splits.py:SPLIT_CONFIGS`):** {1k–8k:1000, 16k:600, 32k:400, 64k:300,
  100k:250}, capped at 1000. **Decrease with N** because small splits peak *later* in epochs
  (1k peaks ~600–830 ep) while big splits peak earlier (tens of epochs) — peak-in-epochs falls
  as N grows because steps/epoch grows.
- **Time-targeted (~9h, single GPU).** Calibrated from a real run: 8k @ 500 ep (16k steps) =
  **1440 s ≈ 13 steps/s** (Blackwell + TF32). Per-epoch ≈ (split/250)×0.078 + probe overhead.
- **Budgets are non-critical & resumable — CORRECTED 2026-06-10:** best-val + cooldown capture
  the **TI** peak regardless of how far past it you train, **but the final CIFAR number is
  budget-dependent when the TI signal has not plateaued** (TI keeps rising → TI-best lands later
  → CIFAR transfer regresses; measured: the seed-42 campaign dips 1.8pp at 4k–8k, with a
  4.7pp kNN-level oracle gap at 8k — see paper §4.4). "Non-critical" holds only for splits
  whose TI signal peaks within budget.
  If a split is still rising at its budget, **extend it** with `--resume_from` (see §7). This
  is *why* we can afford rough budgets. Following Cole et al. 2022 (arXiv:2105.05837) on
  data-quantity SSL (they used constant 1000 epochs across sizes). [report:
  how-researchers-pick-epochs]
- **Time-budget rule of thumb:** supervisor ceiling ~8h/run; our whole 8-split sweep is ~9h.

---

## 6. Learning rate (selected by ablation)

- **LR = 5e-4** (constant, AdamW, batch 250). **Other optimizer HP (verified in code,
  previously undocumented here): weight decay 1e-4 (`--weight_decay`), gradient clipping 1.0
  (`--grad_clip`)** — both must be disclosed in the paper's recipe table.
- **Why AdamW, not the BT paper's LARS:** LARS targets *large-batch CNN* training (batch
  1k–32k); at **batch 250** it offers nothing and would fight the transformer. Every ViT-SSL
  method uses AdamW (DINO, MoCo-v3, MAE). BT's LARS recipe (ResNet-50, batch 2048) doesn't
  transfer. *Consequence:* the comparable LRs are the **AdamW-ViT-SSL @ batch-256** ones, not
  BT's LARS LR.
- **Ablation (one-off, on split 8k):** LR ∈ {1e-4, 3e-4, 5e-4, 7e-4, 1.5e-3}. **Exact numbers
  extracted from W&B summaries 2026-06-10 (full runs at commit 905506c, single seed):** best
  kNN(TI) at selected checkpoint = 13.9 / 14.7 / 15.3 / 15.8 / 3.9 %; budget-end effective
  rank = 55 / 100 / 94 / 81 / 8. NB: **7e-4 is nominally highest** (+0.5pp over 5e-4) — do NOT
  describe the band as "a plateau within probe noise" alone; the honest pick rationale is
  mid-band + near-peak effective rank (declines beyond 3e-4) + DINO's batch-256 reference
  (`main_dino.py --lr 5e-4`). 1.5e-3 **fails outright** (best step 256 — inside warmup; rank
  8/192). Now reported in the paper as Table 1 (§4.1), disclosed as single-seed at an earlier
  code revision.
- **One LR for all splits** (don't vary with N → single-variable / change-one-variable, RP9).
- **Defense line:** *"LR chosen by a one-off ablation over the AdamW-ViT-SSL range; an
  inverted-U with a 3e-4–7e-4 plateau and degradation at 1.5e-3; 5e-4 = plateau center and
  DINO's reference."*

---

## 7. Reproducibility, resumability, run identity

- **Run tag = git short SHA** (`-dirty` if uncommitted); custom labels get the SHA prefixed
  (`e97d41c-curve`). Every checkpoint stores `git_sha`. Indexing is by **(commit, split,
  seed)**.
- **Two checkpoints per run:** `_best` (peak; for cooldown) and `_stable` (fully resumable:
  model + optimizer + epoch + RNG states + best-tracking). `cooldown.py` produces `_final`
  (annealed; the eval target).
- **Resume (`pretrain.py --resume_from <_stable>`):** continues the stable phase to a larger
  `--epochs`. Enabled by constant LR (no committed horizon — WSD). Manual per-step LR (pure
  function of `global_step`) makes resume trivial — no scheduler state to restore.
- **Cross-commit guard:** on resume, if the checkpoint's `git_sha` differs from the current
  one, a loud `[WARN]` — a hybrid old-code/new-code trajectory isn't attributable to one
  commit. Resume writes back to the *source* dir so the `_best`/`_stable` lineage stays
  together. *Rule:* commit before resumable runs; resume on the same clean commit.
- **Determinism:** fixed seeds (data subset + run); cuDNN deterministic; **TF32** for
  pre-training matmuls/convs (~1.5–2× faster, bit-reproducible on the same machine; numerics
  differ from fp32). Linear-probe eval stays **fp32**.

---

## 8. Batch size = 250 (not 256)

- **Why:** all splits are multiples of 1000 and 1000/250 = 4, so **250 divides every split**
  → with `drop_last`, **1 epoch = exactly 1 exposure/image** for every split. Batch 256 drops
  up to **23%** of 1k each epoch, and equal epochs ≠ equal exposure — which matters now that
  budgets are framed in epochs.
- **Power-of-2 is convention, not a hardware requirement.** NVIDIA's real rule is divisibility
  by 8 (FP16)/4 (TF32)/16 (INT8) for Tensor Cores, and the **batch dimension barely enters tile
  quantization** (effective GEMM rows = batch×tokens ≈ 16k ≫ tile). Benchmarks show
  power-of-2 vs nearby ≈ 0% (Raschka/Lightning; W&B). 256→250 costs sub-1%, in noise. [report:
  power-of-2-batch-sizes]
- **Defense line:** *"250 so every split forms whole epochs (1 epoch = 1 exposure); power-of-2
  is convention, and the batch dim doesn't hit Tensor-Core quantization."*

---

## 9. Augmentations

- **Two-view BYOL transform** (lightly `BYOLView1/2Transform`), input 64, `min_scale=0.25`
  — a **disclosed deviation** from the published/lightly default 0.08, chosen a priori (never
  ablated; NO measured "0.08 collapses" claim exists — no logged run used 0.08): `scale` is an
  *area* fraction, so 0.08 at 64×64 allows ~18×18 px crops (vs ~63×63 px at the 224px
  resolution the recipe was tuned at) — about 2×2 of our 8-px ViT patches, leaving the two
  views almost no shared content to align (InfoMin: over-aggressive views destroy
  task-relevant information). 0.25 keeps minimum crops at ~32×32 px. `cj_strength=1.0`
  (default), `gaussian_blur=0`.
  RandomResizedCrop ratio (0.75, 1.333) (torchvision default — mild aspect stretch exists).
- **Why these:** the published recipe, reused verbatim so the baseline is verifiably standard;
  augmentation experiments (if pursued) vary `cj_strength`/`min_scale` against it.
- **Augmentation as a lever (future/discussion):** stronger color jitter delayed 1k's
  over-training and raised its peak slightly in a long-run probe — but augmentation *defines
  invariances*, and aggressive ones destroy task-relevant info (InfoMin reverse-U, arXiv:
  2005.10243; Xiao et al. 2020 "what should not be contrastive," arXiv:2008.05659). For a
  single-downstream-task study, claims are scoped to CIFAR-10 transfer. Augmentation =
  artificial data enlargement, more important at small N (Moutakanni 2024, arXiv:2406.09294;
  Steiner 2021 AugReg, arXiv:2106.10270). [reports: ssl-augmentations-instance-discrimination,
  ssl-small-data-overtraining-augmentations]

---

## 10. Downstream evaluation (Leo protocol, group-shared)

- **Linear probe on CIFAR-10** (`leo_protocol_eval.py`, group-shared "Leo protocol"): freeze
  the encoder, train a single `nn.Linear(192, 10)` on the frozen CLS embedding. **Verified
  hyperparameters:**
  - **SGD**, momentum **0.9**, **weight decay 0.0**; LR = **0.1 × batch/256 = 0.2** at batch
    **512** (linear scaling rule); **CosineAnnealingLR → 0 over 100 epochs**; cross-entropy.
  - CIFAR-10 resized **32→64** (bilinear), ImageNet-normalized (mean/std .485/.456/.406,
    .229/.224/.225). **Train transform adds `RandomHorizontalFlip(0.5)`** — a *documented
    deviation* matching Leo's actual run, not strict protocol B.2; test transform is
    resize+normalize only. Train loader `drop_last=True`.
  - Report **best test top-1 across the 100 epochs** (the metric Jan flagged — see below).
  - **Eval seed = 0**, fixed, independent of the pre-training seed (which is chosen via
    `--pretrain_seed`, default 42).
  - CSV (`evaluation/results.csv`) schema: `method, fraction, seed, epochs, eval_dataset,
    eval_method, accuracy, checkpoint_path, wallclock_pretrain_s, wallclock_eval_s, git_sha,
    timestamp`. **NB: the `git_sha` column is currently written EMPTY** by the eval script —
    only the checkpoint payload carries the SHA; fix if you cite SHAs from the eval rows.
  - Default sweep = `[random-init, 1k, 2k, 4k, 8k, 16k, 32k]`; **64k/100k must be passed via
    `--splits`**. Random-init baseline logs as method `barlow_twins_random_init`, fraction 0.
- **Eval the `_final` (annealed) checkpoint** (default `--ckpt final`), not `_best` (high-LR,
  un-annealed); `--ckpt best` is for analysis/oracle-stopping. Requires running `cooldown.py`
  first to produce `_final`.
- **Known concern (Jan, meeting 5):** "best test top-1" peeks at the test set during probe
  selection → record final-epoch accuracy alongside and report the gap (planned).
- **Diagnostics:** `effective_rank` (Roy & Vetterli 2007; exp-entropy of the feature
  covariance spectrum) as a dimensional-collapse signal (Jing et al. 2022, arXiv:2110.09348) —
  logged only, never used for selection.

---

## 11. Experiment tracking (W&B)

- Online W&B; metrics under `train/*`, `knn/*` (`knn/acc_ti`, `knn/smoothed_acc_ti`,
  `knn/best_acc`, `knn/effective_rank_ti`; optional `knn/acc_cifar`), cooldown under
  `cooldown/*`.
- **Resumed runs use W&B grouping** (each segment a fresh run sharing a `group`), not run-id
  reuse — segments sit on disjoint `global_step` ranges so the group view reads continuous.
  (NB: a wandb.ai **backend ingestion incident** on Jun 9 2026 delayed history display for ALL
  runs; it is a server-side outage, not our code/version — verified by API + status page.
  Local `.wandb` data is always complete; charts backfill on recovery. [report: wandb-fix])

---

## 12. Threats to validity / limitations (anticipate these questions)

- **No convergence claim** — we report the best-val checkpoint within a *fixed budget*; the
  budget may not reach convergence for every split. Disclose per-split whether validation
  plateaued; do not call the curve "converged" or "best-achievable."
- **Task-aware selection** — *corrected* by switching to the standard in-domain TI probe (we
  had wrongly used a CIFAR probe earlier); this is fixing our own mistake, not a contribution.
- **best-on-test eval** — record final-epoch too; disclose.
- **Model-selection (winner's-curse) inflation** of the selected kNN value — mitigated by
  smoothing; magnitude open on a 200-way/5k-query probe.
- **Probe regime unvalidated** — 200-way kNN at 25 gallery imgs/class is far sparser than any
  published protocol; absolute kNN numbers are low (≈ a few %); used only as a *relative*
  selection signal.
- **Single downstream task (CIFAR-10)** — claims scoped to CIFAR-10 transfer; a VTAB-style
  multi-task suite (CIFAR-100/SVHN/EuroSAT/PCAM/Clevr-count, linear probe on cached features)
  is scoped as an extension. [report: transfer-eval-suites-small-scale]
- **El-Nouby positioning** — embedding-SSL (BT-family) is reported *less* small-data-robust
  than masked/denoising methods (arXiv:2112.10740 abstract: denoising autoencoders "are more
  robust to the type and size of the pre-training data than popular self-supervised methods
  trained by comparing image embeddings"; comparative, NOT "least"; BT itself was not tested
  there). Yan *chose* BT (not assigned — corrected 2026-06-10); any BT-on-small-data weakness
  is expected from the family positioning, which makes characterizing it interesting.
  **Deep-read caveat (2026-06-10, docs/research/elnouby-deep-read.md):** their evidence is
  DINO-only and **fine-tuning-only** (iNat-2019, Table 1); their own §5.4 reports DAEs fall
  behind joint-embedding methods **under linear probing** (our regime) — always carry the
  fine-tuning qualifier when citing the robustness claim.
- **Transfer of large-scale findings** — WSD/cooldown-shape, RankMe, cosine-horizon evidence is
  LLM/large-vision scale; transfer to 1k-image ViT-Tiny BT is *open* and disclosed.
- **Cross-environment numbers** — random-init floor shifted with a torch/GPU change
  (40.29→41.45); compare only within the same environment.
- **Augmentation experiments n=1** if pursued lightly — frame as discussion or run 3 seeds.

---

## 13. Key results so far (subject to the final campaign)

- LR ablation (8k): inverted-U, plateau 3e-4–7e-4, 1.5e-3 degrades → **5e-4**. (Measured.)
- An earlier 3-seed campaign differed from an equal-compute baseline (higher at 1k–4k, parity
  from 8k) — an *observed difference between two training-length policies*, reported as such;
  **not** a claim that either policy is "correct" or that we corrected the literature.
- In a long 1k run, downstream accuracy peaked then declined even with annealing → motivates
  best-val selection over end-of-training. (Observed; the *mechanism* — overfitting — is an
  interpretation, not measured.)

*(Final numbers come from the campaign at commit `e97d41c`, run tag `e97d41c-curve`, seed 42
first, then 43/44. None of the above is a "converged" measurement.)*

---

## 14. Citation index

- Barlow Twins — Zbontar 2021, arXiv:2103.03230
- DINO — Caron 2021, arXiv:2104.14294 (kNN protocol, reference LR)
- MoCo-v3 — Chen 2021, arXiv:2104.02057 ; MAE — He 2021, arXiv:2111.06377 (AdamW ViT-SSL)
- WSD / cooldown shape — Hägele 2024, arXiv:2405.18392 ; MiniCPM — Hu 2024, arXiv:2404.06395
- Cosine-horizon — Hoffmann (Chinchilla) 2022, arXiv:2203.15556 ; SGDR — Loshchilov 2016,
  arXiv:1608.03983 ; horizon-free vision — Singh & Janson 2025, arXiv:2503.02844
- Data-quantity SSL — Cole 2022, arXiv:2105.05837 ; data-constrained scaling — Muennighoff
  2023, arXiv:2305.16264
- Small-data SSL robustness — El-Nouby 2021, arXiv:2112.10740
- Effective rank — Roy & Vetterli 2007 ; dimensional collapse — Jing 2022, arXiv:2110.09348
- Augmentation — InfoMin arXiv:2005.10243 ; Xiao 2020 arXiv:2008.05659 ; Moutakanni 2024
  arXiv:2406.09294 ; Steiner 2021 (AugReg) arXiv:2106.10270 ; HaoChen 2021 arXiv:2106.04156
- Model-selection bias — Cawley & Talbot 2010 (JMLR 11:2079)
- Batch size — NVIDIA Matrix-Multiplication / GPU-Performance guides ; Raschka 2022
- Label-free model selection (considered, not adopted) — RankMe arXiv:2210.02885

---

## 15. Deferred / future work

- Final 3-seed campaign (seeds 42/43/44) with the convergence methodology.
- VTAB-style transfer suite (multi-task linear probe on cached features).
- PEFT fine-tuning vs linear probe (project brief; ~order-of-magnitude cost analysis exists).
- Augmentation intervention (cj_strength / multi-crop / i-Mix), if time.
- Grade-hardening per van Gemert's guidelines: storyline doc, same-env baseline, kNN↔probe
  correlation figure, oracle-stopping comparison (all in `docs/research/` + the plan file).

*All research reports backing these decisions are in `docs/research/`. Code: `pretrain.py`
(stable phase + resume), `cooldown.py` (anneal → `_final`), `run_all_splits.py` (driver),
`leo_protocol_eval.py` (linear probe), `tiny_imagenet_features.py` / `cifar10_features.py`
(probes), `two_view_dataset.py` (augmented loader), `models/barlow_twins.py`.*
