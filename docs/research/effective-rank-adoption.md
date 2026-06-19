# Effective rank in DL research — adoption verification (2026-06-10, agent-verified)

For the defense question "is this common practice?" All items verified against
primary sources (URLs in the agent log; key ones below).

## Verified facts

- **RankMe (Garrido, Balestriero, Najman, LeCun; ICML 2023, PMLR 202)** uses
  EXACTLY our estimator — exp-entropy of normalized singular values, citing
  Roy & Vetterli 2007 by name (their Eq. 1). ~139 citations. It goes further
  than we do: label-free *hyperparameter selection* by effective rank.
- **Follow-ups / contemporaries** (label-free spectral SSL quality):
  - α-ReQ — Agrawal, Mondal, Ghosh, Richards, "α-ReQ: Assessing
    Representation Quality... by measuring eigenspectrum decay",
    **NeurIPS 2022** (eigenspectrum power-law decay coefficient; notes BT
    hyperparameters modulate it).
  - LiDAR — Thilak et al. (Apple), "LiDAR: Sensing Linear Probing
    Performance in Joint Embedding SSL Architectures", **ICLR 2024**,
    arXiv:2312.04000 — treats RankMe as the baseline to beat.
- **Dimensional collapse is a named, mainstream phenomenon**: Jing et al.
  ICLR 2022 (~480 citations; covariance singular-value spectra as primary
  evidence) and **Hua et al., "On Feature Decorrelation in Self-Supervised
  Learning", ICCV 2021 Oral** (names dimensional collapse; directly relevant
  to BT since it studies decorrelation methods).
- **Beyond SSL**: Huh et al. "Low-Rank Simplicity Bias" (arXiv:2103.10427,
  TMLR) uses Roy & Vetterli effective rank; Kumar et al. ICLR 2021
  (deep RL) tracks a threshold-variant feature rank.

## Defense-ready sentence (agent-suggested, verified)

"Examining the eigenspectrum of the embedding covariance is the standard way
to diagnose dimensional collapse in SSL [Hua ICCV 2021; Jing ICLR 2022], and
recent work uses the effective rank [Roy & Vetterli 2007] of this spectrum
not just as a diagnostic but as a label-free model-selection criterion
[Garrido ICML 2023; cf. Agrawal NeurIPS 2022; Thilak ICLR 2024]."

Avoid "THE established diagnostic" (alternatives coexist: α-ReQ's decay
coefficient, threshold ranks, LiDAR's LDA rank). Bonus defensive point: we
make a strictly WEAKER use of the metric than the published literature
(logged diagnostic only, never selection).

## Possible final-version additions (NOT in the draft)

Hua et al. (ICCV 2021) would strengthen §2's collapse paragraph (it's the
decorrelation-family collapse study — closest to BT). α-ReQ/LiDAR could join
the RankMe clause. Verify bib entries against proceedings pages before adding.
