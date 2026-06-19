# Defense notes — recurring questions and the worked-out answers

Companion to `paper-judgement-calls.md` (decisions) and
`docs/research/*` (verifications). Add every meeting/defense question here.

## Q: "If the goal is data efficiency, do specific HPs matter? Isn't
## fixed-HP accuracy comparison the point, since HPs can be tuned forever?"
(meeting 8 era, asked by supervisors)

Three-layer answer:
1. **Conventional HPs (LR, λ, augs): agree — tune minimally, freeze,
   disclose.** We tuned ONE knob (LR), once, on one split, anchored to
   DINO's published 5e-4; purpose was avoiding a failure regime (1.5e-3 →
   rank-8 collapse would fake "BT can't learn"), not optimization. Within
   the sane band LR is worth ≤1 kNN point. Tuning is explicitly not the
   contribution.
2. **The premise breaks for protocol parameters.** "Same HPs across N" is
   incoherent for budget/stopping/selection: equal epochs = 25× different
   steps; equal steps = wildly different exposures. Some N-dependent choice
   is FORCED. Our headline result measures the consequence: the
   stopping/selection policy moves mid-N readings by up to ~6 probe points
   (>> the ≤1-point LR band) and flips the curve monotone↔dipping in every
   seed. The unfixable "HPs" dominate the tunable ones.
3. **Converse trap.** Equal-HP cross-method comparison silently favors the
   method whose published defaults fit the scale (relevant for the 5-method
   group curve). No HP-free measurement exists — only disclosed, frozen,
   externally-anchored ones.

One-liner: "We froze what can be frozen; our result is that what CANNOT be
frozen across N moves the curve more than anything tunable — so a
data-efficiency claim that doesn't pin its protocol isn't measuring data."

## Q: "Why 5e-4 and not the best-kNN (7e-4) or best-rank (3e-4) LR?"
Band 3–7e-4 equivalent within single-seed noise; 7e-4's rank already
declining toward the 1.5e-3 cliff; argmax-picking a noisy sweep =
winner's curse; 5e-4 = band center + DINO's batch-256 reference (external
anchor chosen before our numbers).

## Q: "Is WSD better than cosine?"
Same, per cited evidence (Hägele: matches well-tuned cosine's final loss,
LLM scale; Zhai: horizon-committed slightly better if duration known). Its
value here is checkpoint comparability + extendability, which best-val
selection requires. Same-protocol cosine baseline in OUR setup: untested,
disclosed as an assumption (§6).

## Q: "What does effective rank do / why for BT?"
exp-entropy of feature-covariance eigenvalue shares = #dimensions
effectively carrying variance (1=constant output, 192=even spread). For BT
specifically: (a) audits whether the loss's decorrelation (applied to the
discarded 1024-d projector output) propagates to the kept 192-d CLS space;
(b) dimensional collapse is BT's one remaining collapse mode (complete
collapse excluded by construction); (c) canary for our a-priori λ/width
choices — and it works (LR 1.5e-3 → rank 8). Measures spread, not
usefulness; logged, never selected on.

## Q: "Why projector 1024 with embedding 192 (5.3:1), when BT has 8192/2048 (4:1)?
## Why not keep the ratio?"
**There is no ratio in BT — the premise is the error.** Verified from the BT
paper (Zbontar 2021, local PDF):
- §4: "BARLOW TWINS performs better when the dimensionality of the projector
  output is very large … keeps improving with all output dimensionality
  tested (Fig. 4)." → 8192 = biggest tested, still climbing; an ABSOLUTE
  compute-bounded choice.
- "the output of the ResNet is kept fixed to 2048, which acts as a
  dimensionality bottleneck." → 2048 is just ResNet-50's native output, not
  chosen as half of 8192.
So 8192:2048=4:1 is coincidental (no other method shares it: SimCLR/BYOL
project DOWN to 128-256, DINO UP to 65536).
- **Load-bearing quantity = absolute D** (loss is D×D cross-correlation,
  redundancy reduction over D dims); BT says more D helps.
- **Why 1024:** full-width 8192 projector ≈ 136M params ≈ 25× the 5.4M
  ViT-Tiny encoder (indefensible for a small-compute study); 8192 was tuned
  for ImageNet+ResNet-50, over-parameterized for 1k-100k images. Our 1024
  projector = 2.3M (~0.4× encoder).
- **Twist:** by ratio we're slightly LARGER not smaller (5.3:1 vs 4:1), so no
  under-provisioning concern. We kept DEPTH=3 (BT ablation: saturates at 3
  layers — the part with an optimum) and only cut WIDTH (unbounded-but-
  expensive).
- **Caveat = the strong answer:** width is the a-priori-fixed, un-ablated knob
  = THE planned projector-width study. "Whether 1024 is right, or width should
  scale with N, is exactly the open question we're about to measure."

## Old-campaign facts (cross-campaign questions)
Cosine-era runs (fe00dad, long-base, aug-cj): peak LR 3e-4, CIFAR-kNN
selection, different budgets → THREE variables differ vs the final
protocol; cross-campaign deltas are corroborating only. LR component of
that delta is ≤0.6 kNN points per the ablation.
