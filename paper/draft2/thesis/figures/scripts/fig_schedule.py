"""Schedule schematic: warmup -> constant LR -> best-val checkpoint -> (1-sqrt) cooldown.

The LR curves replicate the exact pure functions from pretrain.py
(warmup_constant_factor) and cooldown_factor(shape="sqrt"); proportions follow
the 1k-split geometry (4 steps/epoch, 1000-epoch budget = 4000 steps) so every
phase is visible at figure scale.

Usage: python fig_schedule.py  (writes ../schedule.pdf)
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "schedule.pdf")

LR = 5e-4
WARMUP = 500          # optimizer steps (pretrain.py --warmup_steps default)
BUDGET = 4000         # 1k split: 4 steps/epoch x 1000 epochs
BEST = 2800           # illustrative best-validation step (~70% of budget)
COOLDOWN = max(500, int(0.2 * BEST))  # cooldown.py: 0.2 x best_step, floor 500


def warmup_constant_factor(step: int, warmup_steps: int) -> float:
    # pretrain.py:103
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    return 1.0


def cooldown_factor(t: int, length: int) -> float:
    # pretrain.py:110, shape="sqrt"
    frac = min(1.0, t / max(1, length))
    return 1.0 - math.sqrt(frac)


def main() -> None:
    steps = np.arange(0, BUDGET + 1)
    stable = np.array([LR * warmup_constant_factor(s, WARMUP) for s in steps])
    cd_steps = np.arange(0, COOLDOWN + 1)
    cooldown = np.array([LR * cooldown_factor(t, COOLDOWN) for t in cd_steps])

    fig, ax = plt.subplots(figsize=(6.0, 2.6))

    ax.plot(steps, stable, color="#1f77b4", lw=2.2,
            label="stable phase (warmup $\\rightarrow$ constant LR)")
    ax.plot(BEST + cd_steps, cooldown, color="#d62728", lw=2.2, ls="--",
            label="cooldown branch: $(1-\\sqrt{t})$, $0.2\\times t_{best}$")

    # Periodic kNN probes (every 4 epochs = 16 steps at the 1k geometry); draw
    # sparse ticks so the band reads as "periodic probes", not data.
    probe_steps = np.arange(WARMUP, BUDGET, 320)
    ax.plot(probe_steps, np.full_like(probe_steps, LR, dtype=float), ls="none",
            marker="|", ms=9, color="#7f7f7f", label="periodic kNN probes")

    ax.plot([BEST], [LR], marker="*", ms=16, color="#d62728", ls="none",
            zorder=5, label="best smoothed-probe checkpoint $t_{best}$")

    ax.axvspan(0, WARMUP, color="#1f77b4", alpha=0.08)
    ax.annotate("warmup\n(500 steps)", xy=(WARMUP / 2, LR * 0.45),
                ha="center", va="center", fontsize=9)
    ax.annotate("budget end\n(no anneal here)", xy=(BUDGET, LR),
                xytext=(BUDGET - 80, LR * 0.74), ha="right", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("rewind to $t_{best}$,\nanneal to 0",
                xy=(BEST + COOLDOWN * 0.34,
                    LR * cooldown_factor(int(COOLDOWN * 0.34), COOLDOWN)),
                xytext=(BEST + 800, LR * 0.40), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.0))

    ax.set_xlabel("optimizer step", fontsize=10)
    ax.set_ylabel("learning rate", fontsize=10)
    ax.set_ylim(0, LR * 1.18)
    ax.set_xlim(0, BUDGET * 1.02)
    ax.set_yticks([0, LR])
    ax.set_yticklabels(["0", "$5\\times10^{-4}$"], fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.13, 0.04),
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
