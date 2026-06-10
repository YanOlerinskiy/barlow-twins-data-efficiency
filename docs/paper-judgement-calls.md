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
