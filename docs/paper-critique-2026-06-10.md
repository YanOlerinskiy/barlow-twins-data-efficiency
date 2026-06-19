# Critical review of the draft — 2026-06-10

Sources: a savage-reviewer agent pass over the built PDF (van Gemert checklist
in hand), a literature-verification agent (solo-learn / BT-paper / BT-on-ViT
sweep), my own decision-by-decision scrutiny, and Jan's meeting feedback.
Items marked ✎ are edits (cheap); ⚗ are small experiments (GPU-hours given);
✋ need Yan's decision. Severity: 🔴 defense-losing if unfixed, 🟡 weakens the
paper, 🟢 polish.

---

## 1. Defense-critical reasoning fixes

### 🔴 R1 — "N is our only controlled variable" is literally false ✎(+⚗ free)
§1 and §4.1 both say it; but epoch budgets, total steps (~25× across the
curve), and cooldown lengths all vary with N. An examiner quotes this and
asks why the curve isn't a compute curve. Our real defense is good — every
point is a post-peak best-validation checkpoint, so extra steps can't
advantage anyone — but the sentence as written hands over the opening.
**Fix:** (a) reword to "the only *manipulated* variable; budgets vary with N
but the best-checkpoint protocol decouples the reading from budget length";
(b) **free experiment from existing logs**: a matched-step slice — at a fixed
optimizer-step count (e.g. 4k steps), read probe accuracy across all N from
the logged probes. One supplementary figure, zero GPU-hours, directly
separates fresh-data benefit from training-length benefit. Strongly
recommended.

### 🔴 R2 — "We adopt Barlow Twins unchanged" contradicts our own §3 ✎
§2 says "unchanged --- reference loss implementation and published
augmentations" while §3 discloses five changes (λ, projector width, blur,
crop bound, optimizer). Each change is fine; the sentence is the bug (RM3).
**Fix:** "We adopt the Barlow Twins *objective* and reference loss
implementation unchanged, with disclosed adaptations of the recipe to 64×64
inputs and small-batch ViT training (Section 3)."

### 🔴 R3 — "label-free validation signal" oversells ✎
The kNN selector uses Tiny-ImageNet *labels* (held-out, in-domain). §2
correctly reserves "fully label-free" for RankMe; §1 and §3's "label-free"
is an internal inconsistency an examiner will spot. **Fix:** call it
"downstream-blind" / "in-domain validation signal that never sees the
downstream task"; reserve label-free for RankMe.

### 🔴 R4 — The evaluated model is not the selected model ✎+⚗(~1h)
We select t_best, then train 0.2×t_best *further* steps (annealing) and
evaluate the annealed model — on exactly the splits where we say post-peak
training hurts. We have ONE measurement (+1.5pp from annealing at 16k,
methods doc §3.3) and it's not in the paper. **Fix:** (a) state the gap in
§3.4; (b) cheap experiment: linear-probe the *un-annealed* t_best checkpoint
alongside `_final` on the smallest and largest splits (probes only, no
pre-training; `--ckpt best` already exists). This validates our most
original methodological move (rewind-and-anneal) and is already half-planned
as the oracle-stopping comparison.

### 🟡 R5 — λ=1e-2 has a timestamp, not a reason ✎(+⚗ ~1h optional)
"Chosen a priori, not ablated" is honest but motivates nothing (RM10).
**Fix:** (a) one mechanism sentence: the off-diagonal sum grows with D², so
narrowing the projector 8192→1024 shifts the invariance/redundancy balance;
we raised λ to roughly compensate, without tuning on the measurement
variable. (b) Optional hardening: ONE run at λ=5e-3 on one split (~1 GPU-h)
turns the inevitable defense question into a result. ✋ decide if worth it
after the projector sweep.

### 🟡 R6 — Peak-then-decline is asserted in §3.4 before §4.3 asks it ✎
Reviewer called it circular. It isn't — we measured a ~7pp decline from peak
in a 1k pilot (methods doc §3.3) — but the paper doesn't say so, so it reads
circular. **Fix:** "in preliminary runs the 1k probe accuracy declined
several points past its peak; the rewind protects against annealing a
degraded model" — pilot observation, attributed as such. This also explains
where the budget numbers came from (currently unexplained).

### 🟡 R7 — Intro ¶2's equal-steps dismissal argues the wrong direction ✎
"Equal steps would turn the comparison into a compute question" reads as
dismissing the control a critic wants. The stronger true argument: under
best-checkpoint selection with generous budgets, each N is read at its own
peak, so unequal budgets cannot *advantage* small N; equal steps would
instead under-train large N or over-train small N. Rewrite in that
direction (and see R1's matched-step figure).

### 🟡 R8 — Headline metric peeks at the test curve ✋
Best-test-top-1 is group protocol, disclosed, final-epoch gap promised. The
reviewer (and Jan, meeting 5) still flag it. **Options:** (a) keep
best-test as headline for group comparability, final-epoch in an appendix
column; (b) make final-epoch OUR headline and best-test the group-table
number. I lean (b) — cleaner science, zero compute — but it's a
group-coordination question only Yan can settle.

### 🟡 R9 — "The two datasets share no images" (Fig. 3 caption) is unverified ✎
TI derives from ImageNet, CIFAR-10 from 80M Tiny Images — distinct sources,
but cross-dataset near-duplicates are documented in the literature and we ran
no check. Hard-rule violation (empirical sentence, no measurement).
**Fix:** "are drawn from disjoint source collections" (or run a perceptual-
hash dedup if we want to keep the strong claim — not worth it now).

---

## 2. Literature positioning (verification agent; all sources checked)

Our narrow claim ("the cited studies don't measure BT") is **fully safe** —
Cole = SimCLR/MoCo/BYOL; El-Nouby = BEiT/SplitMask/DINO; Moutakanni = DINOv2;
none mentions BT. The broader "scarcely characterized" claim is **defensible
but should be sharpened to be bulletproof**:

- **solo-learn** (JMLR 2022) benchmarks BT on CIFAR-10/100 (92.1/70.9,
  ResNet-18), ImageNet-100 — full dataset size only, no size sweep, no ViT.
- **Mixed Barlow Twins** (arXiv:2312.02151) runs BT on Tiny-ImageNet —
  ResNet, full size only.
- **The BT paper's own 1%/10% experiments are label-efficiency of the
  fine-tuning set, NOT pre-training-size efficiency** (pre-training always on
  full ImageNet) — confirmed from the full text; worth one explicit sentence
  in §2 because examiners may conflate the two.
- **No published work sweeps BT accuracy vs pre-training set size on any
  natural-image benchmark, on any backbone**; the only BT-on-ViT data point
  found is a workshop paper (DinoTwins, 256 COCO images, no curve).

✎ **Recommended §2 addition (one or two sentences):** "Barlow Twins itself
appears in small-dataset benchmarks --- CIFAR and Tiny-ImageNet with ResNet
backbones at full dataset size [solo-learn; optionally Mixed-BT] --- and its
original evaluation subsamples only the downstream labels, not the
pre-training set [zbontar2021barlow]; no study characterizes its accuracy as
a function of pre-training set size, nor on a small ViT." This converts a
vague "scarcely described" into a precise, checkable statement. Requires
adding the solo-learn bib entry (verify via JMLR page).

---

## 3. Jan's feedback — coverage audit

| Jan's item | Status | Gap / action |
|---|---|---|
| why this SSL method | ✓ §1 (chosen; collapse-proof yet family data-hungry) | sharpen per F1 below |
| effect of augmentation | ◐ principled non-choice (§5) | no experiment; defensible via Moutakanni; lowest-priority optional ⚗ |
| effect of backbone | ✓ fixed by protocol (§3, §6) | — |
| effect of fine-tuning | ◐ future work | acceptable; PEFT named in §7 |
| **compare losses of pre to fine** | ✗ **not addressed anywhere** | ⚗ FREE: extend §4.5 to correlate the logged BT training loss with kNN(TI) and probe accuracy across checkpoints — answers it from existing logs |
| minimize compute | ✓ §5 compute paragraph | numbers PENDING |
| random baselines | ✓ random-init floor | — |
| **upper bound (ViT on CIFAR)** | ◐ hedged "(if run)" | ⚗ ~1–2 GPU-h: commit to ONE supervised ViT-Tiny/CIFAR-10 run; a floor without a ceiling makes the gap uninterpretable. Recommend committing. |
| HP tuning (LR/seed/epochs) | ✓ LR ablation, 3 seeds, budget framing | promote LR ablation per F4 |
| **random data splits for stddev** | ◐ disclosed limitation only | ⚗ ~1 GPU-h total: 2 extra 1k splits from different shuffle seeds; converts our weakest limitation into a result. Recommend. |
| pre/fine loss comparison | see above | — |
| visual abstracts ×2, dataset images | ✓ Figs 1–3 scripted | polish pipeline text size (#23) |
| caption conclusions | ✓ all captions end in conclusions / PENDING slots | — |
| individual topic per person | ✓ projector study | — |
| method-specific HP tuning | ✓ projector sweep planned | λ spot-check optional (R5) |

---

## 4. Flow / structure edits (all ✎)

- **F1 — The best hook is buried.** Intro ¶4 contains a genuine tension:
  collapse-proof by construction, yet the family is reported data-hungry —
  two competing predictions for the small-data end. Lead the paragraph with
  that tension and write the expected answers into the §4 stubs (RP6). Right
  now the motivation skirts "scarcely described" (WI5 risk).
- **F2 — Contribution 3 (projector) isn't set up by the RQ or intro prose.**
  Add one intro sentence motivating the capacity knob (and consider a
  sub-question clause). ✋ minor wording choice.
- **F3 — Duplicated arguments.** The cosine-horizon argument and the
  Cawley selection-bias point each appear nearly verbatim in §2 AND §3.4.
  Keep the why in §2; let §3.4 state the choice + cite. Buys ~15 lines.
- **F4 — The LR ablation is a result hiding in §4.1.** Either promote to a
  question-headed "Experiment 0" with its (existing) figure, or compress to
  one sentence + appendix table. I lean promote — it's our only tuning
  experiment and shows methodological care.
- **F5 — Intro ¶2 front-loads defensive methodology** before the pipeline
  exists (uses "checkpoint/validation/budget" pre-definition). Slim it to
  two sentences; the detail already lives in §3.4.
- **F6 — Compute-stance inconsistency.** §5: "not a contribution"; §7: "part
  of the research question". Align: the small-compute *regime* is part of the
  question; the recipe's cheapness is not a contribution.
- **F7 — Gloss "effective rank" and "projector" in the contribution bullets**
  (most-read sentences; non-DL examiner per WG2).

## 5. Mechanical guideline edits (🟢, batch in one pass)

- WF6 brackets carrying load-bearing content: §3.4 warmup numbers, intro
  "(ViT-Tiny…5.4M)", "(disclosed, not claimed standard)" → promote to clauses.
- WS5: §3.4 "It also leaves runs extendable" → "A constant rate also…".
- WS3: ~9 "(Section X)" pointers; keep ≤3 that genuinely help.
- "best performer" → "attained the lowest final loss" (precision, WF8-spirit).
- WI2 opener: current "expensive twice over" passed Yan's review; optional
  sharpening to a practitioner vignette. Leave unless Yan wants.

## 6. Cheap experiments, ranked (respecting committed scope)

Committed scope stays: 3-seed campaign + projector sweep. Everything below
is additive, ordered by (defense value / GPU-cost):

1. **Matched-step slice figure** — 0 GPU-h (existing logs). Kills the
   compute-confound attack (R1). DO IT.
2. **kNN(TI) ↔ probe ↔ BT-loss correlation (§4.5) — make it mandatory, not
   optional** — 0 GPU-h (logs + diagnostic pass). It validates the selection
   mechanism the whole curve depends on, and answers Jan's pre/fine-loss
   item. WE4 says validate first.
3. **t_best (un-annealed) vs _final probes on 1k and 100k** — ~1 GPU-h.
   Evidence for the rewind-and-anneal adaptation (R4).
4. **Supervised ceiling: ViT-Tiny on CIFAR-10** — ~1–2 GPU-h. Jan asked
   explicitly; makes the curve interpretable (R8 in agent report, D3).
5. **2 resampled 1k splits** — ~1 GPU-h. Converts the data-composition
   limitation into a measurement; Jan asked explicitly.
6. **λ=5e-3 single run on one small split** — ~1 GPU-h. Optional; only if
   time after the projector sweep.
7. **cj_strength single point** — lowest priority; the non-choice argument
   stands without it.

Items 1–2 are free and should go in regardless; 3–5 total ≈ 3–4 GPU-hours —
roughly half of one campaign seed. ✋ Yan picks the cutoff.

## 7. Where I'd push back / consciously accept

- The reviewer's "circularity" charge (R6) dissolves once pilots are
  attributed — no redesign needed.
- Keeping best-test-top-1 *somewhere* is right (group comparability); the
  only question is which number is the headline (R8).
- The Euclidean-unweighted kNN (vs DINO's cosine-weighted) stays as-is:
  disclosed, used as a relative signal, and re-running selection under
  cosine would cost a campaign. Accepted, documented.
- "Scarcely characterized" stays as the positioning — the literature agent
  confirmed no BT size-curve exists anywhere — but with §2 sharpened per
  item 2 so the claim is precise rather than vibe-y.
- The Discussion's augmentation non-choice is the strongest reasoning in the
  paper post-verification; don't dilute it with a token augmentation run
  unless everything else is done.

## 8. Bottom line

No fatal flaws; the structure is genuinely strong (the reviewer agent said
so too, grudgingly). The fixes that matter most are the four internal
contradictions (R1–R4) — all quotable by an examiner, all fixable in a day —
plus making §4.5 mandatory and adding the two free figures. The committed
experiment plan already covers Jan's list except the pre/fine-loss
correlation (free), the supervised ceiling, and the split-resample (cheap);
those three would close his feedback table completely.
