# Epoch-Budget Selection & Validation-Driven LR Scheduling — Survey

*Research-agent report, 2026-06-05. Web-verified against primary sources. Companion to `ssl-stopping-and-lr-schedules.md` (Hägele/WSD, anneal-before-stop) and `ssl-training-budgets-steps-vs-epochs.md` (El-Nouby constant-iterations, Muennighoff repetition). This report covers what those don't: **how the budget number itself is chosen**, and **prior art for the student's "rewind-to-best-then-anneal" proposal**. Tags: [established] = stated/measured in a primary source; [folklore] = widespread practice without published justification; [open] = no evidence found.*

## Q1. How do published works choose the epoch *number*?

Honest answer: **almost nobody derives it; the number is convention or a compute ceiling, then validated post-hoc by an ablation showing "more is better until saturation."** No SSL paper found derives epochs from dataset size or a convergence criterion.

- **Barlow Twins** (Zbontar 2021, arXiv:2103.03230): 1000 ep is inherited SimCLR/BYOL convention; the paper gives **no derivation** of 1000 [folklore].
- **SimCLR** (Chen 2020, arXiv:2002.05709, §B/Fig.9): explicitly *post-hoc* — *"trained for 1000 epochs to ensure convergence"*; ablation shows longer training and bigger batches help, and the batch-size gap **shrinks with more epochs**. So 1000 = "enough to converge," not a derived optimum [established that longer helps; folklore that 1000 specifically].
- **MoCo v3** (Chen 2021, arXiv:2104.02057): 300 ep default; ablation shows **saturation** (ViT-S 72.5→73.4% at 300→600). Number is a measured plateau, picked by ablation not theory [established].
- **MAE** (He 2021, arXiv:2111.06377, §A/Fig.7): 800 ep default, tested to 1600; *"accuracy improves steadily… we have not observed saturation even at 1600 epochs."* Budget is a **compute ceiling**, not convergence; also warns epochs aren't comparable across methods (75% masking → encoder sees ~25% patches/epoch) [established].
- **DINO** (Caron 2021, arXiv:2104.14294): 100/300/800-ep configs; states longer training helps ViT-S but gives **no budget justification**; repo issue #145 confirms hyperparameters differ between 100 and 800 (teacher_temp, clip_grad), i.e. ad-hoc per-budget tuning [folklore].
- **Supervised vision**: same pattern — ResNet/ViT "90/300 epochs" are conventions, justified at most by "accuracy saturated."
- **LLMs / Chinchilla** (Hoffmann 2022, arXiv:2203.15556) is the **one regime with a derived budget**: tokens (≈ the "epoch" analogue, since LLM pretraining is ~1 epoch) are set to ≈20× parameters via fitted compute-optimal scaling laws — derived from a *fixed compute budget* C, allocating C between N and D. This derives *tokens-per-parameter*, not *epochs over a fixed dataset*; it does **not** transfer to "how many passes over a 1k-image set" [established but out-of-regime].

Implication for the thesis: there is **no citable rule "dataset size N → E epochs."** El-Nouby's constant-iterations-with-a-cap (arXiv:2112.10740, see prior report) is the closest defensible operating principle. Choosing a budget by *watching a probe and stopping* is therefore not a deviation from a standard — there is no standard to deviate from.

## Q2. Prior art for the student's scheme (rough-budget → validate → best checkpoint → anneal)

**(a) ReduceLROnPlateau (decay-when-validation-stalls).** Canonical published origin: **Bengio 2012, "Practical Recommendations for Gradient-Based Training of Deep Architectures"** (arXiv:1206.5533, §3.2.1) — reduce LR when validation error stops improving; predates the framework callbacks (Keras/PyTorch `ReduceLROnPlateau`). It was **standard in the supervised pre-2018 era** (e.g. seq2seq, ASR, many ResNet recipes) [established]. **In modern SSL pretraining it is essentially absent** — every major method (Q1) uses *open-loop* warmup+cosine on a fixed budget, not a closed-loop plateau trigger. No mainstream SSL pretraining recipe using ReduceLROnPlateau was found [open]. The student's "trigger anneal when the kNN probe stalls" is structurally a ReduceLROnPlateau variant (single large reduction = a cooldown, instead of step-factor ×0.1), applied to SSL where it is non-standard.

**(b) WSD / checkpoint branching.** **MiniCPM** (Hu 2024, arXiv:2404.06395) is the clearest precedent for "stable phase + branch the decay": *"we can reuse the model before decay and continue training with the previous high learning rate"* — they fork decays from **intermediate stable-phase checkpoints** (at 40N/60N/80N tokens), decay ≈10% of tokens (10% suffices, 2.5% falls short), **exponential** shape (×0.5 per period) [established]. **Hägele 2024** (arXiv:2405.18392): constant-LR + cooldown matches cosine, **(1−√t) beats linear**, *"cooldown can be initiated at any time."* **Crucially, both branch FORWARD from the current/intermediate checkpoint — neither rewinds to a best-validation checkpoint** [established]. DeepSeek-LLM (arXiv:2401.02954) uses a multi-step (not plateau) schedule, also no rewind.

**(c) SGDR / cyclical LR.** **SGDR** (Loshchilov & Hutter 2016, arXiv:1608.03983): cosine cycles with warm *restarts*; improves *anytime* performance (any restart endpoint is a usable model). **Snapshot Ensembles** (Huang 2017, arXiv:1704.00109) collect the cycle-minima checkpoints. **Cyclical LR** (Smith 2015, arXiv:1506.01186): triangular bounds. Relevance to the thesis is **weak**: these *raise* LR to escape minima and want *many* good endpoints, the opposite of "find one peak and anneal down to lock it in." They are restart-then-re-anneal, not rewind-to-peak.

**(d) Explicit "train past peak → rewind to best checkpoint → anneal from there."** **No precedent found** [open]. The two families that look adjacent both differ: WSD/MiniCPM branch *forward* (no rewind); plateau-decay (ReduceLROnPlateau) decays *in place from the current point* (= what the loop does now). "Discard the overshoot, revert to the best checkpoint, then anneal" appears to be **unattested in the literature**. Report this honestly: it is a plausible-but-novel combination, not a cited technique.

## Q3. Cosine-to-bound pitfall — is a wrong/never-reached horizon harmful?

**Yes, this is the single best-established LR-scheduling result relevant here** [established]:

- **Hoffmann/Chinchilla** (arXiv:2203.15556, §3 / App.): the cosine cycle length must be set ≈ the number of training steps; *setting the cosine cycle longer than the target steps yields sub-optimally trained models, and stopping a long-horizon cosine early (before it decays) under-trains the model at that step count.* This is precisely the loop's failure mode: a cosine aimed at a generous bound, truncated when the trigger fires, has only partially decayed → the model is evaluated/annealed from a point where LR is still high relative to a correctly-horizoned cosine.
- **Hägele 2024** (arXiv:2405.18392) makes the same point as the *motivation for WSD*: cosine *"is suboptimal during training and underestimates the model's performance for the same token count,"* and forces *"train multiple models for different lengths, from scratch"* to fit scaling laws. Their fix: constant-LR + late cooldown removes the need to commit to a horizon.
- **Beyond Cosine Decay** (Singh & Janson 2025, arXiv:2503.02844) extends this to **vision/MAE**: an "infinite LR schedule" (warmup → constant → optional short cooldown) **outperforms repeated cosine for MAE pretraining** and *"is not restricted to a fixed iteration budget."* This is direct vision-SSL evidence that horizon-free schedules work for MAE [established for MAE; untested for Barlow Twins].

So the modern answer to "duration unknown a priori" is decisively **constant/stable LR + late cooldown (WSD)**, not cosine. The loop's truncated-cosine tail inherits exactly the misspecified-horizon pathology these papers identify.

## Q4. Recommendation for this thesis (splits 1k–100k, ~10× variance in convergence time, unknown a priori)

Compare three designs on grounding / risk / simplicity:

**(a) Current: truncated cosine → triggered (1−√t) tail.** *Grounding:* weak — the cosine horizon is the arbitrary "generous bound" that Chinchilla/Hägele warn against; the realized schedule is an undocumented hybrid. *Risk:* the trigger fires while LR is partway down a mis-horizoned cosine, so the annealing start-LR varies uncontrolledly with how generous the bound was. *Simplicity:* worst — two coupled mechanisms (cosine shape + bound + trigger + tail length).

**(b) Constant LR (post-warmup) + triggered cooldown = pure WSD.** *Grounding:* strongest — Hägele (matches cosine, trigger-anytime, 1−√t shape) **and now Singh 2025 for MAE-vision specifically**. *Risk:* lowest; removes the cosine-horizon free parameter entirely (the only horizon-dependent quantity becomes cooldown length, ≈10–20% of steps-so-far). *Simplicity:* best. **This is the recommended default.** It is also the smallest honest change from the current loop: drop the cosine, hold LR flat after warmup, keep the existing kNN-stagnation trigger and (1−√t) tail. Note Hägele's gains plateau beyond ~20% cooldown, so `max(0.2×steps, 500)` is already in-range.

**(c) Student's rewind-to-best-then-anneal.** *Grounding:* none direct (Q2d) — combines ReduceLROnPlateau (in-SSL: novel) with a checkpoint rewind (unattested). *Risk — discuss honestly:*
   - **Noisy peak.** With a noisy kNN probe, "best so far" at trigger time is often a **noise spike**, not the true optimum (this is the classic early-stopping selection-bias problem; the argmax of a noisy series is upward-biased). Rewinding *to that spike* then annealing locks in a fluke. A smoothed/EMA probe or "best over a window" mitigates but doesn't eliminate this. Trigger-from-current (b) is more robust because it doesn't *select* a single noisy point — it just decides *when* to start cooling.
   - **Optimizer state.** The common worry ("rewind discards Adam moment estimates") is **avoidable if checkpoints save optimizer state** (standard in PyTorch `state_dict`; do verify the loop's checkpointer saves `optimizer.state_dict()` and the LR-scheduler/step counter). If it does, the rewind is *state-preserving* and the only true cost is wasted compute on the overshoot steps between peak and trigger.
   - **Wasted compute.** By construction (c) trains past the peak and throws those steps away — directly opposed to the thesis's data-/compute-efficiency framing. (b) wastes nothing.
   - **Where (c) could win:** if SSL representation quality genuinely *degrades* late (some evidence for dense tasks, arXiv:2510.17299; not established for linear/kNN on classification), annealing from a pre-degradation checkpoint beats annealing from the degraded current point. For ViT-Tiny/Barlow-Twins/linear-probe this degradation is **unverified**, so (c)'s upside is speculative.

**Bottom line.** Recommend **(b) pure WSD** as the principled, best-grounded, simplest default — it eliminates the cosine-horizon arbitrariness (Q3) that is the only *established* defect of the current loop, while keeping the existing trigger and tail. Keep **fixed-budget cosine** as the canonical baseline. Treat **(c)** as an optional ablation, defensible only with (i) a smoothed probe to avoid locking onto a noise spike, (ii) confirmed optimizer-state-preserving checkpoints, and (iii) explicit acknowledgment that it is a novel, uncited construction whose extra overshoot compute cuts against the data-efficiency thesis.

## Sources

- Bengio 2012, Practical Recommendations — arXiv:1206.5533 (§3.2.1, plateau LR reduction)
- Smith 2015, Cyclical LR — arXiv:1506.01186
- Loshchilov & Hutter 2016, SGDR — arXiv:1608.03983; Huang 2017, Snapshot Ensembles — arXiv:1704.00109
- Chen 2020, SimCLR — arXiv:2002.05709 (§B, Fig.9); Zbontar 2021, Barlow Twins — arXiv:2103.03230
- Chen 2021, MoCo v3 — arXiv:2104.02057; Caron 2021, DINO — arXiv:2104.14294 (repo issue #145); He 2021, MAE — arXiv:2111.06377 (§A)
- Hoffmann 2022, Chinchilla — arXiv:2203.15556 (cosine-length-must-match-steps); El-Nouby 2021 — arXiv:2112.10740
- Hu 2024, MiniCPM/WSD — arXiv:2404.06395; Hägele 2024 — arXiv:2405.18392; Singh & Janson 2025, Beyond Cosine Decay (MAE infinite-LR) — arXiv:2503.02844
- Dense-task late degradation — arXiv:2510.17299
