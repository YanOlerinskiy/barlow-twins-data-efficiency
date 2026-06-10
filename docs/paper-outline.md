# Paper outline — final-draft scaffold (Barlow Twins data efficiency)

**Honesty rule (overrides everything):** this is a thesis — every empirical sentence needs a
measurement behind it; interpretations are labelled as interpretations. We make **no** of
these claims: "trained to convergence", "novel methodology", "in-domain validation is our
contribution", "fixed-budget curves confound the field / we corrected the literature." What we
have is a **careful characterization** of Barlow Twins (BT) in the small-data / small-compute
regime. Methods backbone: `docs/methods-decisions.md`. Citations: `docs/research/*.md`.

---

## What to write NOW vs what waits

| Section | Status | Why |
|---|---|---|
| §1 Introduction (motivation, RQ, contributions) | **WRITE NOW** | stable; framing below |
| §2 Related work | **WRITE NOW** | citations don't change |
| §3 Method & setup | **WRITE NOW** | factual description of the pipeline |
| §5 Discussion (augmentation non-choice, compute) | **WRITE NOW** | reasoning, not results |
| §6 Limitations | **WRITE NOW** | mostly stable |
| Figures: visual abstracts, dataset images, schedule schematic | **WRITE NOW** | no results |
| §4 Results (curve, small-data behaviour, projector) | **PENDING** | needs the runs |
| Abstract headline number, Conclusion's specific answer | **PENDING** | needs results |

**Committed experiments (the ~8h scope):** (1) the data-efficiency campaign already running
(seed 42, then 43/44); (2) **projector-dim sweep on the smaller splits** (the extension).
**Cheap add-ons we already have or can get:** random-init floor (have); effective-rank traces
(logged); kNN(TI)↔CIFAR correlation (from the diagnostic pass). **Optional if time:**
supervised upper bound (ViT on CIFAR-10); otherwise mention as future work.

---

## Storyline (the spine — write first, ~8 bullets; nothing here over-claims)

1. **Context.** SSL "foundation" models usually need large data *and* compute → inaccessible.
   Understanding SSL with *little data* on *little compute* matters. (Visual abstract: why.)
2. **Object.** Pre-train a small ViT (ViT-Tiny/8) with **Barlow Twins** on subsets of
   Tiny-ImageNet of increasing size, evaluate by CIFAR-10 linear probe. (Group: 5 students
   compare 5 SSL methods on one shared data-efficiency curve; BT is mine.)
3. **What's thin (scoped honestly — no "first ever").** Small-data SSL is a well-studied
   benchmark area, but it is dominated by contrastive / masked methods; **Barlow Twins
   specifically** is comparatively little-characterized at this small scale, and it belongs to
   the joint-embedding family that El-Nouby (2112.10740) reports as the *least* data-robust —
   making its small-data behaviour both under-described and a-priori interesting.
4. **What we do.** Train BT across split sizes with a standard WSD recipe and standard in-domain
   kNN model selection, then *characterize* the resulting curve and the small-data regime, and
   probe one BT-specific knob (projector width vs N).
5. **What we report** (all observational, conditional on the runs): the BT data-efficiency
   curve with a random-init floor (+ optional supervised ceiling); the small-data behaviour
   (does it over-train? is it collapse?); and the projector-width × N effect.

---

## Title & abstract `[headline PENDING]`
- Title conveys: *data efficiency of Barlow Twins for a small ViT at small scale* (descriptive,
  no novelty puffery).
- Abstract (write last): context → the focused question (BT's small-data behaviour) → what we
  do (train across N, standard selection, projector study) → headline result `[PENDING]` →
  one honest takeaway.

## 1. Introduction  **[WRITE NOW]**  (WI3: contributions bulleted at the end)
- Motivate accessibility; define **data efficiency** (accuracy vs #images) and state we study
  the data axis, not compute (so: not equal-steps; epochs held generous).
- One paragraph: SSL pre-train → linear-probe pipeline / what a "foundation model" setup is here
  (visual abstract).
- **RQ:** *How does Barlow Twins' downstream accuracy scale with pre-training set size for a
  small ViT in a small-compute regime, and how does it behave at the smallest scales?*
- **Contributions (state modestly):**
  1. A characterization of BT's data-efficiency curve (TinyImageNet→CIFAR-10) for a ViT-Tiny,
     with a random-init floor (and, if run, a supervised upper bound).
  2. An empirical look at BT's small-data regime: whether it over-trains, and whether that
     coincides with representation collapse (via effective rank).
  3. A study of one BT-specific knob — projector width vs dataset size.

## 2. Related work  **[WRITE NOW]**  (themed paragraphs, each ending in "…so we…")
- **SSL & Barlow Twins:** redundancy-reduction loss; collapse-resistant by construction (no
  negatives/asymmetry). Zbontar 2021 (2103.03230). Contrast DINO 2104.14294 / MoCo-v3
  2104.02057 / MAE 2111.06377.
- **Small-data SSL (acknowledge it's well-studied):** CIFAR/STL/Tiny-ImageNet are standard SSL
  benchmarks; El-Nouby 2021 (2112.10740, joint-embedding methods least data-robust); Cole 2022
  (2105.05837, SSL across dataset sizes); Moutakanni 2024 (2406.09294, augmentation = data
  enlargement). → so BT *specifically* at small scale is the under-described slice we look at.
- **Training schedules:** WSD constant-LR + cooldown (Hägele 2024 2405.18392; MiniCPM
  2404.06395); cosine-horizon (Chinchilla 2203.15556; SGDR 1608.03983). → so we use a standard
  WSD recipe (cited, not claimed novel).
- **SSL model selection / monitoring:** in-domain kNN (DINO); label-free RankMe 2210.02885
  (mentioned, not used); model-selection bias Cawley & Talbot 2010 (motivates smoothing).
- **Collapse & augmentation invariance:** dimensional collapse Jing 2022 (2110.09348); effective
  rank Roy & Vetterli 2007; augmentation defines invariances — InfoMin 2005.10243, Xiao 2020
  (2008.05659) → motivates studying *capacity* rather than stronger augmentation for a
  foundation model.

## 3. Method & experimental setup  **[WRITE NOW]**  (condense from `methods-decisions.md`)
- Backbone ViT-Tiny/8 @64px (shared, fixed — answers Jan's "effect of backbone": held constant
  by design). BT loss; projector 192→1024→1024; λ=1e-2 (disclose vs lightly's 5e-3).
- Data & nested-stratified splits {1k…100k}; dataset sample images (Jan).
- **Training recipe (describe, don't claim):** warmup(500 steps) → constant LR (5e-4, AdamW,
  batch 250) → best in-domain-val checkpoint → short (1−√t) cooldown. Justify each knob briefly
  (constant LR, best-val + cooldown, AdamW-not-LARS, batch 250, LR-by-ablation) with citations;
  full rationale in the methods doc. **State explicitly we use a generous fixed budget and do
  NOT claim convergence.**
- **Model selection:** standard in-domain Tiny-ImageNet kNN probe (k=20, smoothed best-val) —
  note we use the in-domain probe as is standard (we had earlier mistakenly used a CIFAR probe).
- **Downstream eval:** CIFAR-10 linear probe (Leo protocol — exact HP in methods doc, incl. the
  RandomHorizontalFlip and best-test-top-1 choices).
- **Reproducibility & HP (Jan):** seeds 42/43/44; (commit, split, seed) indexing; the LR
  ablation; what was tuned vs fixed-by-protocol.

## 4. Results  **[ALL PENDING — write as `[PENDING]`, conditional, interpretations flagged]**
(WE1: question-headed subsections; WT1: caption ends in the conclusion; RP9: change one variable)

### 4.1 How does BT's downstream accuracy scale with pre-training set size?
- The data-efficiency curve + **random-init floor** (have) + **supervised upper bound** (if run).
- Report shape, the small↔large gap, % of any ceiling reached. `[PENDING]`
- **Disclose per split whether validation plateaued within budget** (no convergence claim).
- Fig: curve (x=#images log, y=CIFAR top-1, mean±band over seeds) + floor (+ ceiling).

### 4.2 At the smallest scales, does BT over-train — and is it collapse?
- Long-run accuracy + effective-rank traces on the smallest split(s).
- Report: does accuracy peak-then-decline? does effective rank stay high? `[PENDING]`
- If yes+high-rank: state the *observation* (decline without rank collapse) firmly; label
  "overfitting to the small set" explicitly as an **interpretation we did not isolate**.
- Fig: accuracy + effective rank vs step.

### 4.3 Does projector width interact with dataset size? (BT-specific knob — the extension)
- Sweep projector dim {256, 512, 1024, 2048} on the smaller splits (re-pretrain, ≥2 seeds).
- Report the accuracy-vs-width curve per split; whether the best width shifts with N. `[PENDING]`
- Honest both ways: a shift → "smaller projectors suit smaller N"; no shift → report the null.
- Fig: accuracy vs projector dim, one line per split.

### 4.4 (Optional) Does the in-domain selection signal track downstream accuracy?
- kNN(TI) ↔ kNN(CIFAR) ↔ linear-probe correlation across checkpoints (diagnostic data). `[PENDING]`

## 5. Discussion  **[WRITE NOW for the reasoning parts]**
- **Why not stronger augmentation as the small-data lever:** it works by adding invariances,
  undesirable for a foundation model (Xiao 2020; InfoMin) — a principled non-choice. (Reasoning,
  writable now; if we have the `cj` long-run data, cite it as supporting, carefully.)
- **Compute:** report total/per-split wall-clock; one neutral paragraph (Jan's "minimize
  compute"), not a claim.
- **Implications** for small-data BT practice — phrased tentatively, tied to the observations.

## 6. Limitations  **[WRITE NOW]**  (from methods-decisions §12 — be explicit)
- **No convergence claim** (best-val within a fixed budget; disclose non-plateaued splits).
- best-on-test eval (record final-epoch gap); model-selection inflation (smoothed); probe regime
  unvalidated (200-way/sparse, low absolute kNN — relative signal only); **single downstream
  task** (CIFAR-10); **stddev over training seeds, not data subsets** (nested-prefix design —
  disclose; resample 1–2 splits only if time); transfer of large-scale schedule findings to
  1k-image BT is open.

## 7. Conclusion  **[skeleton now, answer PENDING]**
- One-paragraph answer to the RQ + honest takeaway + 2 future directions (transfer suite, PEFT,
  projector-scaling).

---

## Figure / table inventory (captions END in the conclusion — WT1/WT2)
1. Visual abstract: why data efficiency. **[now]**
2. Visual abstract: foundation-model / pre-train→probe setup. **[now]**
3. Dataset sample images (TinyImageNet, CIFAR-10). **[now]**
4. Schedule schematic (warmup→constant→best-val→cooldown). **[now]**
5. Main curve + floor (+ ceiling). **[pending]**
6. Over-training: accuracy + effective-rank traces. **[pending]**
7. Projector-dim × N. **[pending]**
8. kNN(TI)↔probe correlation. **[pending, optional]**

## Jan's feedback → where addressed (most already covered)
| ask | where |
|---|---|
| why this SSL method | §1/§3 (chose BT; collapse-resistant, family reported less data-robust) |
| effect of augmentation | §5 (principled non-choice) |
| effect of backbone | §3 (fixed by protocol) |
| fine-tuning / pre-vs-fine loss | §7 future work |
| minimize compute | §5 |
| random baselines | §4.1 (random-init floor) — **have** |
| upper bound (ViT on CIFAR) | §4.1 — **optional run** |
| HP tuning (LR/seed/epochs) | §3/§4 (LR ablation, seeds, budgets) |
| random data splits for stddev | §6 (disclose; optional resample) |
| visual abstracts / dataset images / "so what" captions | Figs 1–4; all captions |
| individual topic / method-specific HP | §4.3 (projector width) |
