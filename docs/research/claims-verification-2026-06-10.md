# Primary-source verification log — schedule & augmentation claims (2026-06-10)

Agent deep-reads of four primary sources backing §2/§3 claims. Quotes verbatim
from the papers (ar5iv/arXiv HTML full texts). Use the "approved phrasing"
lines in the paper; deviations need a re-check.

## Chinchilla (Hoffmann 2022, arXiv:2203.15556) — cosine-horizon claim

- App. B "Optimal cosine cycle length": "setting the cosine cycle length too
  much longer than the target number of training steps results in
  sub-optimally trained models, as shown in Figure A1."
- App. D.1: "when the cosine cycle overshoots the number of training steps by
  more than 25%, performance is noticeably degraded."
- Sec. 2: "the intermediate loss estimates (for D′ ≪ 130B) are therefore
  overestimates of the loss of a model trained with a schedule length
  matching D′."
- **Nuance:** degradation is *clear* beyond ~25% overshoot; tiny overshoots
  (≤1.1–1.25×) cost little. Mid-training readouts (our best-val case) are far
  past 25% — claim holds there.
- **Approved phrasing:** "a checkpoint read out well before the cosine horizon
  attains higher loss than a model whose schedule was matched to that
  duration (App. B, Fig. A1)". Avoid unqualified "systematically worse".

## Hägele 2024 (arXiv:2405.18392, NeurIPS) — WSD claims

- A (WSD ≈ cosine): SUPPORTED. v3 Takeaway 1: "offers significant convenience
  by not requiring the number of training steps to be specified in advance,
  and provides similar performance compared to a well tuned cosine schedule."
  Metric is validation LOSS (v3 adds 1B-scale downstream matches, Fig. 8) —
  prefer "final loss" over "accuracy" when citing.
- B ((1−sqrt) shape): SUPPORTED. Shapes ablated: linear, 1−sqrt, cosine,
  mirror-cosine, 1−square (App. B.1, Fig. 16/17). Fig. 16: "(1-sqrt) being
  the most effective"; it beats linear AND the (untuned) cosine cooldown.
  "Fast initial drop, long low-LR tail" is OUR gloss (correct but not their
  words). Their caveat: "the order might change for substantially different
  cooldown lengths; we focus on 10% and 20%."
- C (cooldown length): benefits "plateau at around 20%" of steps; cooldown
  surpasses cosine between 10–20% (Fig. 5); ~5% with (1−sqrt) nearly matches
  cosine for long runs (Fig. 6). **The "2.5% falls short, ~10% suffices"
  quote is NOT in Hägele (any version)** — it is MiniCPM §4.3 (below).
- Scale/domain: decoder-only LLMs, 33M–360M core ablations (210M for shapes),
  v3 adds 1B (100B/460B tokens) + short 8B run. Vision precedent they cite:
  Zhai et al. 2022 "Scaling Vision Transformers" (arXiv:2106.04560, CVPR
  2022). **Hägele's gloss ("find reciprocal square-root with cooldown to
  perform best") OVERSTATES the primary source** — Zhai's own §3.5: "the
  linear schedule is still preferable when one knows the training duration in
  advance … all three alternatives come reasonably close, with the advantage
  of allowing" evaluation from one run; "We therefore opt for the reciprocal
  square-root schedule." Cite Zhai as: infinite schedules + cooldown come
  close to horizon-committed linear while serving all durations from one run
  (verified against Zhai full text 2026-06-10).

## MiniCPM (Hu 2024, arXiv:2404.06395) — decay length & branching

- §4.3 "10% Steps are Enough": "having a decay of 10% of the total tokens is
  sufficient to achieve the best results, while a decay of 2.5% of total
  tokens falls short." (Percent of TOKENS, not steps; 0.036B models, branches
  from 40N/60N/80N stable checkpoints; only 2.5% and 10% tested.)
- Decay shape: exponential annealing, half-life 5000 steps (§6.2) — NOT
  (1−sqrt); our shape comes from Hägele.
- Branch-from-stable precedent: §4.3 "we can reuse the model before decay and
  continue training with the previous high learning rate … then perform
  annealing"; §7.2 branches MiniCPM-128K off the 2.4B stable checkpoint.
  (They branch FORWARD; our rewind-to-best variant remains a disclosed
  adaptation.)

## Moutakanni 2024 (arXiv:2406.09294) — augmentations as data enlargement

- The small-data link is THEIR claim, not just our inference. Contribution
  (i): "the impact of data-augmentations invariance enforcement … is
  secondary to the impact of their artificial increase in dataset size and
  distribution." §4.1: "the real impact of data-augmentation is to
  artificially increase the number of samples and allow JEA to reach good
  performance with less data." Conclusion: "The data augmentations merely
  influence training by increasing the dataset size."
- Minimal recipe: RandomCrop WITHOUT resizing + iBOT patch masking (masking
  itself optional at ~1–2% cost, Table 4). ViT-L/14 'Crop' on IN-22k: 84.0
  IN1k linear probe (Table 1). Gaps close as data grows (Fig. 2: <1% at
  LVD-142M).
- §4.2 (relevant to OUR over-training question): "training longer on smaller
  scale dataset is harmful for performances when we don't use the dataset
  size increase property of data-augmentations. This is probably a
  consequence of overfitting."
- **Caveats:** smallest tested scale is ImageNet-1k (1.3M images) — far above
  our 1k–100k; gap also grows with MODEL size (data relative to
  model/compute); v2 adds OOD domains where augmentations are harmful.
- **Approved phrasing:** augmentations "act chiefly by artificially enlarging
  the dataset rather than by enforcing useful invariances"; never claim they
  tested below 1.3M images.
