# Paper judgement calls — log for later revisiting

Every writing decision that was debatable enough to show Yan, with its status.
Revisit before the final read-through. (Format: status = OPEN / RESOLVED date.)

## §1 Introduction

1. **Opening sentence** — definition-first ("SSL trains image representation
   models...") vs tension-first ("expensive twice over: data and compute").
   Borderline WI2 (generic first sentence). **RESOLVED 2026-06-10: tension-first**,
   per Yan.
2. **Parenthetical section pointers** ("(Section 3)" etc., 3 instances) — WS3
   discourages; kept minimal because they support scoping claims (e.g. "well
   studied (Section 2)" backs WI5). **OPEN** — drop or keep at final pass.
3. **"The method assigned to us"** — WRONG: Yan selected Barlow Twins himself.
   Was copied from methods-decisions §12 / outline ("assigned"), both now
   corrected. Paper says the method was chosen. **RESOLVED 2026-06-10.**
   Defense angle: choosing the reportedly least data-robust family makes its
   small-data behaviour *more* worth characterizing, not less.
4. **El-Nouby positioning** — outline said "least data-robust" (superlative);
   the paper's abstract only supports a *comparative*: joint-embedding methods
   ("trained by comparing image embeddings") are *less robust to the type and
   size of pre-training data* than denoising autoencoders (BEiT/SplitMask).
   Also: El-Nouby never tested Barlow Twins itself — hence our phrasing "the
   family it belongs to has been reported to...". **RESOLVED 2026-06-10:
   comparative phrasing only; never "least".**
   **Deep-read addendum (tables read, see docs/research/elnouby-deep-read.md):**
   evidence = DINO only, FINE-TUNING eval only (iNat-2019; Table 1: DINO 70.1 @
   IN-1% vs BEiT 74.1/SplitMask 74.8; DINO −8.3 on COCO); their Sec. 5.4 says
   DAEs *fall behind joint-embedding under linear probing* — our eval regime.
   §2 must carry the fine-tuning qualifier; intro's hedged "has been reported"
   stays. **RESOLVED 2026-06-10.**
5. **Academic voice** — "we" throughout (single-author thesis). Yan did not
   object. **OPEN (default: keep "we").**

## §3 Method (pre-logged for drafting)

6. **min_scale=0.25 rationale** — NO measured "0.08 collapses" exists (no
   logged run used 0.08); BT resists *complete* collapse by construction, so
   collapse language would also contradict §4.2. Use the a-priori geometric
   argument: scale is an area fraction; 0.08 of 64² ≈ 18×18 px ≈ 2×2 ViT
   patches of shared content (vs ~63×63 px at 224px where the recipe was
   tuned); cite InfoMin; disclose "set a priori, not ablated".
   **RESOLVED 2026-06-10.** Reserve the word "collapse" for dimensional
   collapse (effective rank) only.
7. **Disclosed deviations list** (must all appear in §3): λ=1e-2 (lightly
   default 5e-3); min_scale 0.25 (default 0.08); gaussian_blur off;
   weight_decay 1e-4 + grad_clip 1.0 (were undocumented); AdamW not LARS;
   batch 250 not 256; probe's RandomHorizontalFlip + best-test-top-1 (Leo
   protocol deviations); rewind-to-best cooldown branch (vs branch-forward in
   Hägele/MiniCPM). **STANDING.**
8. **Param count** — 5,388,480 encoder / 7,687,360 with projector (counted
   from code; "~5.7M" was the patch-16@224 figure). Paper says 5.4M.
   **RESOLVED 2026-06-10.**

## §5 Discussion (pre-logged for drafting)

12. **Linear-probe vs El-Nouby's fine-tuning evidence** — two optional
    Discussion remarks, decide when drafting §5: (a) one line framing it as a
    strength: whether the fine-tuning-based robustness ranking transfers to
    linear-probe evaluation at small scale is open, and our BT curve is one
    data point on it; (b) the group-level curve contains an MAE (denoising)
    arm, giving a probe-regime test of El-Nouby's ranking — phrase carefully
    (cross-student comparability caveats). Linear-probe choice itself:
    group-shared protocol + standard representation eval; fine-tuning
    deferred for time/compute — goes to future work, not Limitations
    hand-wringing. **OPEN.**

## §5 Discussion / §6 Limitations / §7 RR (autonomous stretch, 2026-06-10 evening)

13. **Color-jitter long-run anecdote OMITTED from §5.** The "stronger cj
    delayed 1k over-training" observation is n=1, from a dirty-commit run
    (`f374768-dirty-long-cj20`), W&B-only. The principled argument
    (Moutakanni + Xiao + InfoMin, all abstract-verified) carries §5 without
    it; citing an unreproducible run is a defense liability. The outline
    marked it optional ("if we have the data, cite carefully").
    **RESOLVED 2026-06-10: omitted.** Revisit only if the run is re-done
    clean for the camera-ready.
14. **El-Nouby Discussion remarks** — included BOTH options from call #12:
    (a) probe-regime openness as one tentative paragraph, and (b) the group
    MAE arm as a [PENDING] sentence with an explicit cross-student-tuning
    caveat. **RESOLVED 2026-06-10.**
15. **Compute paragraph** — wall-clock numbers [PENDING]; the "~9 hours per
    8-split sweep" figure is the methods-doc calibration estimate, marked as
    an estimate in the text. Replace with measured totals from campaign logs.
    **OPEN until results.**
16. **Limitations phrasing** — initially wrote "both biases shift the curve
    rather than reshape it"; self-caught as an overclaim (bias magnitude need
    not be uniform across N) and softened to "inflate in the same direction;
    magnitude need not be identical across N". **RESOLVED 2026-06-10.**
17. **DINO gallery density figure** — "roughly 1,300 per class" = ImageNet
    train (~1,281/class) used as DINO's kNN gallery; arithmetic + methods
    doc. **RESOLVED.**
18. **Floor shift wording** — paper says "about 1.2 points" (41.45−40.29)
    without printing environment-specific values; raw numbers live in
    results.csv. **RESOLVED.**
19. **"Code available in the project repository" (§7 RR)** — assumes the
    code/splits will be public or submitted alongside the thesis. **OPEN —
    Yan must confirm** (make repo public, or attach an archive, or reword to
    "available from the author / thesis repository").

## Figures (Phase 2, autonomous stretch)

20. **Schedule schematic geometry** — drawn with the 1k-split proportions
    (4 steps/epoch, 4000-step budget) so warmup/cooldown are visible;
    t_best=2800 is ILLUSTRATIVE, not a measurement; the LR curves replicate
    the exact pretrain.py/cooldown.py functions. Caption discloses the
    geometry. **RESOLVED 2026-06-10.**
21. **Visual abstract shows a "?" instead of a sketched curve** — a sketched
    rising curve could be read as a result before results exist; the curve
    panel shows axes + random-init floor + a question mark. Replace with the
    real curve in the final version if desired. **RESOLVED 2026-06-10.**
22. **Dataset samples drawn with seed 42** — deterministic, regenerable;
    CIFAR row uses the same 32→64 bilinear resize as the probe pipeline.
    One Tiny-ImageNet sample contains a person (dataset-representative;
    swap the seed if undesired). **OPEN (minor).**
23. **Pipeline figure inner text is small at column width** — legible in
    print but worth a polish pass (fewer words in boxes) before submission.
    **OPEN (polish).**

## Critique pass (2026-06-10, see docs/paper-critique-2026-06-10.md)

24. **Full critical review delivered** — savage-reviewer agent + literature
    verification + Jan-feedback audit. Defense-critical items R1–R4 (the
    "only variable" overclaim, "adopt unchanged" contradiction, "label-free"
    oversell, annealed≠selected gap), the §2 sharpening with solo-learn /
    Mixed-BT acknowledgment, and the ranked cheap-experiment list (matched-
    step slice and §4.5-mandatory are free). **OPEN — Yan reviews the
    critique and picks the edit/experiment cutoff.**

## Style pass (2026-06-10, after external style feedback)

25. **Hedge consolidation + dedup pass applied.** One home per argument:
    cosine-horizon → §3.4 (clause in §2); Cawley smoothing-why → §3.4
    (clause in §2); no-convergence → §3.4 (one short sentence in intro;
    interpretation angle stays in §6); honest-reporting → §6 (§7 compressed
    to a cross-reference paragraph); per-item "disclosed / a priori / not
    ablated" tails removed from §3 in favour of ONE consolidated deviations
    sentence in §3's intro + the §6 list. §2 ¶3–4 compressed to clause-level.
    RQ paraphrased (not repeated verbatim) in the conclusion. Captions
    trimmed to content + one takeaway. "Principled non-choice" and similar
    self-pleased phrasings removed. Sentence rhythm varied (the §4.1 budgets
    paragraph was the canonical fused example). Paper went 9 → 8 pages with
    zero content removed. Folded in the three pure-wording critique fixes:
    R1a ("only variable we manipulate" + peak-comparability argument), R2
    ("objective and reference loss unchanged, recipe adapted"), R3
    ("downstream-blind"/"in-domain", label-free reserved for RankMe), plus
    R6's three-word pilot attribution and R9's "disjoint source collections".
    **RESOLVED 2026-06-10.** NOT yet applied (await Yan's cutoff): R4 anneal
    gap, R5 λ mechanism sentence, F1 tension-led intro ¶4, F2 contribution-3
    setup, F4 LR-ablation promotion, §2 solo-learn sentence, experiments 1–7.

## Rubric pass (2026-06-10, vs CSE3000 assessment rubric)

26. **LR ablation promoted to a reported finding (Table 1, §4.1)** — numbers
    extracted from W&B summaries of the completed `905506c-lrabl-*` runs:
    kNN(TI) 13.9/14.7/15.3/15.8/3.9 %, budget-end effective rank
    55/100/94/81/8 for {1,3,5,7,15}×10⁻⁴. Disclosed as single-seed at an
    earlier code revision. IMPORTANT data-vs-doc correction: 7e-4 is
    nominally HIGHEST (+0.5pp over 5e-4) — "plateau within probe noise" was
    not supportable; the pick rationale is now mid-band + near-peak effective
    rank + DINO reference. 1.5e-3 "degrades" was an understatement: it never
    improves past warmup (best step 256) and ends at rank 8/192 — "fails
    outright". methods-decisions §6 synced. **RESOLVED 2026-06-10.**
27. **Spelling unified to American English** (behaviour→behavior,
    labelled→labeled, neighbour→neighbor, unlabelled→unlabeled,
    over-fitting→overfitting); aspell pass clean (remaining flags are
    technical terms). **RESOLVED 2026-06-10.**
28. **Rubric dependencies that prose cannot fix (submission blockers):**
    abstract, §4.2–4.5 results, conclusion answer = campaign-gated;
    "code available in the project repository" (#19) must be made true;
    Process category (20%) is graded on behavior — project plan + meeting
    notes exist in the repo as evidence. **OPEN.**

## Campaign findings (rolling)

29. **TI↑ / final-linear-probe↓ divergence under budget extension** (Yan,
    2026-06-10, clarified: the END-TO-END protocol output — best-TI-val →
    cooldown → linear probe — regressed ~1pp (60→59%) when a run was
    extended (reading: 1k vs 6k EPOCH budgets on one split; split/seed to
    confirm) while TI validation kept improving). Implication: with a
    non-plateauing TI signal, the curve point is BUDGET-DEPENDENT — this
    contradicts the methods-doc claim that budgets are non-critical (true
    only for the TI peak). Decisions: (a) the pre-registered rule decides
    the curve number — the written protocol includes extend-while-rising,
    so the EXTENDED result is the curve point and the shorter-budget number
    is reported as budget-sensitivity analysis; do not choose the rule by
    the nicer CIFAR number; (b) "still rising" must be made concrete
    (smoothed best not improved for K consecutive probes + hard cap) and
    applied uniformly to all splits; (c) verify ±1pp against seeds 43/44
    before narrating; (d) report in §4.5 (decoupling at the final-probe
    level), §4.2 per-split effective budgets, one Limitations sentence
    (stopping-rule sensitivity ~1pp). Methods-doc "budgets non-critical"
    claim needs correcting once confirmed. **OPEN — pending split/seed
    confirmation + seed check.**

## Draft-deadline session (2026-06-10 evening, autonomous)

30. **#29 RESOLVED — the "60→59" is the cross-split dip**, confirmed by Yan
    against the probe sweep output and by the CSV: e97d41c seed-42 curve =
    61.35 / 61.76 / 59.92 / 59.58 / 63.34 / 66.48 / 69.86 / 71.45, floor
    41.45. Draft curve FROZEN at tabulated budgets, seed 42 (Yan's OK);
    extensions/cfsel runs reported as in-progress, not swapped in.
31. **Decoupling promoted to contribution 3** (replacing the projector study,
    which moves to future work for the final version — Yan's instruction).
    Evidence chain: within-run kNN oracle gap (smoothed, consistent
    estimator): 0.6/1.6/3.7/4.7/3.5/1.4/0.4/0.0 across N — inverted-U
    peaking at 8k where TI-best (step 28160) lies 24k steps past the CIFAR
    diagnostic peak (4416). Oracle-line monotone, selected-line dips ⇒ "the
    dip is a selection effect, not a data effect, at the diagnostic level."
    Probe-level oracle comparison pending (cfsel _best_cifar checkpoints).
32. **fe00dad (old 3-seed campaign) identified from configs/payloads as
    TASK-AWARE-selected** (CIFAR kNN probe: best_acc 50.5% = CIFAR scale;
    db 5000/val 10000) with different budgets ({1k:5000, 8k:640, 100k:200}
    epochs). Used in §4.4 ONLY as corroboration with the two-variable
    confound disclosed. Its seed ranges (0.5–2.9pp) quoted in §6 as the
    seed-band reference.
33. **Single-seed draft disclosures** added in abstract ("completed
    single-seed campaign"), intro scoping line, §4.1, §6. Cross-revision
    code disclosure (e97d41c → dirty → 9a26560; diagnostic-only diff,
    verified) in §6. methods-doc "budgets non-critical" claim corrected.
    Repo sentence reworded to "kept under version control" (#19 resolved —
    true regardless of submission policy).
34. **Compute numbers**: 10.2 GPU-h = sum of the eight stable-phase
    wallclocks from checkpoint payloads (36,661 s); probe ≈375 s from CSV;
    cooldown "up to ~25 min" is DERIVED (0.2×best_step / ~13 steps/s) —
    labeled approximate in §5.

## Course rules-of-thumb audit (2026-06-10, late)

35. **Course rules saved to writing-guidelines.md** (front-page contents,
    8-page two-column cap excluding title/references/RR, no appendices,
    conciseness first, reference workload, audience). Audit of the draft:
    page count OK (~6.6 countable pages of 8); student number absent ✓;
    29 refs OK ("possibly even more" allowed, depth exists). Fixed: EMAIL
    added to title page (**used yan.olerinskiy@gmail.com — swap to the
    @student.tudelft.nl address if preferred**); appendix removed entirely
    (file deleted, inputs dropped from both main files); effective-rank
    gloss added to the intro contribution bullet (audience rule).
    Email set to Y.Olerinskiy@student.tudelft.nl. **RESOLVED.**

## Dual-review pass (2026-06-10, expert + novice subagent reviews)

36. **Abstract reframed around the robust finding** (expert F2/F9): the
    selection divergence (4.7pp, exceeds the quoted seed band) is now the
    headline; the dip is presented as the symptom, explicitly flagged as
    within seed variance and awaiting confirmation. Jargon removed per the
    novice review ("diagnostic points against a downstream oracle" →
    "score worse on a CIFAR-10 monitoring metric than the best checkpoint
    in hindsight").
37. **Causal claim softened everywhere** (expert F1/F10/F17): Fig. 6 caption
    no longer says "a selection effect, not a data effect" — now
    "coincides with selection-and-budget effects ... at the diagnostic
    level", plus an explicit metric warning (kNN diagnostic ≠ Fig. 4's
    probe); §5 and §8 takeaways anchored to "at the diagnostic level".
38. **INTEGRITY FIX** (expert F6): the claim that final-epoch probe accuracy
    "is recorded alongside" was FALSE — leo_protocol_eval.py persists only
    best_test_acc (verified in code). §4.1 and §6 now state the script does
    not yet persist it and the gap is unquantified in this draft.
    Final-version TODO: persist final-epoch accuracy and report the gap.
39. **Matched-step control promoted** (expert F5) into §5 as "the sharpest
    evidence the data is not at fault"; §4.4's number recitation trimmed to
    shape + peak (F16); LR-as-alternative-mechanism sentence added to §4.3
    (F3); budgets disclosed as a second N-correlated protocol variable with
    pilot-run provenance (F4); LR-ablation split choice motivated
    ("mid-range", F7); budget-sensitivity re-run added to future work (F8).
40. **Novice glosses added**: "frozen --- its weights no longer update",
    "in-domain ... computed on the pre-training dataset", effective-rank
    intuition (192 = equal variance; ~1 = constant output). Final-version
    ideas logged, not done: background-glossary box, BT-objective diagram,
    de-jargoned Related Work schedules paragraph.

## Rubric grading pass (2026-06-10, late — subagent graded vs official rubric)

41. **Graded outcome: Content 8.0, Writing 9.0 → ≈8.4 on the
    paper-attributable 70%.** Row ticks: related work EXCELLENT, methodology
    EXCELLENT, scientific method SUFFICIENT-upper (evidence volume: n=1 seed
    on the headline dip dominates the row), interpretation EXCELLENT,
    responsible engineering EXCELLENT; all four writing rows EXCELLENT
    (held at 9.0 by prose density only). Path to Content 8.5–9.0 = seeds
    43/44 + probe-level oracle comparison (already the declared next steps;
    grader: "report the dip's fate either way — a null result would still
    be a publishable finding").
    **Deliberate non-action tonight:** no further prose edits — the two
    remaining polish items (decompress §3.4 cooldown paragraph, cap em-dash
    interpolations at one per sentence; stronger artifact-availability
    statement pending Yan's repo decision) are final-version work; churning
    a 9.0-writing draft unreviewed under deadline risks more than it gains.
    **OPEN for final version.**

## Post-submission edit (2026-06-10, Yan's request)

42. **"Task-aware" self-judgment removed; data kept.** §4.4 now describes
    the earlier campaign neutrally ("an earlier three-seed campaign, run
    with different budgets and with checkpoints selected by the CIFAR
    diagnostic itself"); §5's "the task-aware protocol this study replaced"
    framing is gone. All numbers, the two-variable confound disclosure, and
    §6's seed-band reference stay — they anchor the abstract's "within seed
    variance" qualifier. **DECLINED** Yan's alternative ("past results
    indicate the results will be much better"): an unsupported
    forward-looking claim that would invite the exact protocol question the
    edit avoids. The honest forward statement stays: oracle headroom
    quantified (≤4.7 diagnostic points) + probe-level comparison in
    progress. NOTE: the submitted PDF predates this edit — this is in the
    working copy for the final version (or a resubmission if the portal
    allows). **RESOLVED 2026-06-10.**

## Overnight results (2026-06-11, 9b11e5d-cfsel: seeds 43/44, 1k-16k, dual checkpoints)

43. **Findings (CSV-verified):**
    (a) DIP, sharpened: non-monotonicity replicates in ALL 3 seeds, but its
    locus is 8k — "8k below 2k" holds per-seed (−2.18/−2.56/−0.77) and "8k
    below 4k" too (−0.34/−3.46/−1.62); "below 1k" does NOT replicate
    (−1.77/−0.01/+0.13 — seed 42 drew a high 1k). Abstract's hedge was
    correct.
    (b) PROBE-LEVEL ORACLE, confirmed and larger: oracle−blind gap at the
    linear probe = +1.5/+2.1/+2.3/+5.9/+3.4 (means, 1k→16k), peaking at 8k
    (+6.75 seed 43) — bigger than the diagnostic's 4.7. Oracle curve is
    MONOTONE in both seeds; blind curve dips at 8k in both. The paper's
    central claim upgrades from diagnostic-level to probe-level with
    replication.
    (c) EXTENSIONS (seed 42, large splits, TI still rising): every extended
    split LOST probe accuracy: 16k −0.76, 32k −1.93, 64k −2.28, 100k −1.92.
    The "at large N the budget ends first" interpretation confirmed.
    Caveats: cfsel runs recorded git 9b11e5d-DIRTY (uncommitted changes at
    launch — disclose); extensions are single-seed.
44. **Final-version text consequences:** §4.2/§6 "lower bounds under this
    budget / on what longer training might reach" is now contradicted by (c)
    — under TI-selection, longer training DECREASED the probe number; reword
    to direction-neutral ("budget-dependent readings"). Dip locus narrows to
    8k; "4k and 8k score below 1k" must become the 3-seed statement. Oracle
    §4.4 upgrade + Fig. 6 probe-level replacement. PIN: figure scripts now
    take the FIRST CSV row per split (frozen numbers) — extension re-evals
    appended later would otherwise silently change Fig. 4. HOUSEKEEPING:
    extension cooldowns OVERWROTE the frozen _final checkpoints in
    e97d41c dirs (CSV/W&B remain the record) — archive checkpoint dirs
    before future reruns. Meeting figure: figures/meeting_3seed.pdf.

## Title page / global

9. **Working title** — "How Does the Downstream Accuracy of Barlow Twins Scale
   with Pre-training Set Size?" — review once results are in. **OPEN.**
10. **Supervisor + examiner names** — placeholders on the title page.
    **OPEN — waiting on Yan.**
11. **Citation policy** — venue cited only when verified against the
    proceedings/journal page; otherwise always-true "arXiv preprint" form
    (El-Nouby, Moutakanni, MiniCPM, Chinchilla currently preprint-form).
    Krizhevsky CIFAR TR has NO Hinton co-author; Tiny ImageNet author is
    "Ya Le". **STANDING.**
