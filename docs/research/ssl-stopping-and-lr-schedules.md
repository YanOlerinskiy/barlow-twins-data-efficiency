# SSL Training Duration, Stopping & LR Annealing — Survey

*Research-agent report, 2026-06-04. Web-verified against primary sources. Motivated the v2 schedule design: cosine to a generous bound + kNN-stagnation-triggered (1−√t) terminal decay.*

## Q1. Canonical vision SSL: duration & LR schedule

All major methods use a **fixed epoch budget with linear warmup + cosine decay to (near-)zero at the budget end** — no validation-based stopping. Confirmed:
- **Barlow Twins** (Zbontar et al. 2021): 1000 epochs, LARS, 10-epoch warmup, cosine decay (LR ÷1000).
- **MoCo v3** (Chen et al. 2021, *An Empirical Study of Training Self-Supervised ViTs*): ViT-S/B 300 ep, 40 warmup, AdamW, cosine. 300→600 ep gives small gains (ViT-S 72.5→73.4%).
- **SimCLR / BYOL** (Chen 2020; Grill 2020): LARS, linear warmup + cosine decay.
- **DINO** (Caron et al. 2021): 100–300+ ep, 10-ep warmup, cosine LR, cosine weight-decay 0.04→0.4.
- **MAE** (He et al. 2022): 800 ep default, up to 1600; cosine; linear-probe keeps improving to 1600 (no saturation).
- **I-JEPA** (Assran et al. 2023): ViT-H/14 300 ep on IN-1K; cosine (exact warmup/shape unverified from primary text — uncertain).

Takeaway: cosine-to-fixed-budget is the universal convention; none early-stop.

## Q2. Early stopping / monitoring for SSL specifically

SSL loss is a known-poor convergence/quality proxy; the literature monitors representations:
- **Xu, Lowe & Trappenberg 2024**, *Label-free Monitoring of Self-Supervised Learning Progress* (arXiv:2409.06612): label-free metrics (k-means silhouette, clustering agreement, embedding entropy) to decide "how much longer to keep training." Correlate with linear-probe accuracy for SimCLR/MoCo-v2 but **not SimSiam** — proxy reliability is method-dependent.
- Online **kNN monitoring** is standard practice (DINO), but for evaluation, not formalized stopping.
- *Exploring Structural Degradation in Dense Representations for SSL* (arXiv:2510.17299): dense-task performance **declines at later training stages** even while loss converges and classification improves (more relevant to dense tasks; uncertain for linear probing).

**No standard, validated patience-based early-stopping recipe for SSL pre-training was found** — our kNN-patience approach is non-standard (a contribution to defend, not a convention to cite).

## Q3. LR schedules for stopping at an arbitrary point (WSD / cooldown)

- **Hägele et al. 2024**, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (NeurIPS spotlight, arXiv:2405.18392): constant-LR + cooldown **matches cosine**; cooldown gains plateau beyond ~20% of steps; a **(1−√t) decay shape beats linear**. Crucially: *"the cooldown can be initiated at any time to observe model behavior and decide whether to stop"* — endorses a signal-triggered cooldown. Evidence is from LLM/cross-entropy training; transfer to BT/ViT was untested (our smoke run + 16k eval subsequently validated it empirically in our setup).
- **WSD origin**: popularized by **MiniCPM** (Hu et al. 2024); stable phase ~80–90%, cooldown ~10–20%.

## Q4. Small-data SSL budgeting conventions

Weak/no firm convention. *Rethinking SSL: Small is Beautiful* (Cao et al. 2021) shows small-data/small-model SSL is viable. Budgets typically expressed in gradient steps; no verified rule mapping dataset size → epochs.

## How this shaped our design

Adopted: cosine to a generous per-split epoch bound (≈WSD when the bound is generous) + kNN-stagnation trigger + (1−√t) terminal decay over max(0.2 × steps-so-far, 500). **Cite Hägele for the decay shape and trigger-at-any-time only** — our adaptations (cosine not constant stable phase; adaptive kNN trigger; decay length as fraction of steps-so-far) must be disclosed as such. Fixed-budget cosine (v1) remains the canonical baseline to compare against.
