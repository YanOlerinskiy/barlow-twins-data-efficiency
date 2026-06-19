"""Result figures for the e97d41c campaign (seed 42, tabulated budgets):

  curve.pdf       - CIFAR-10 linear-probe accuracy vs pre-training set size N,
                    with the random-init floor; non-plateaued splits marked.
  overtraining.pdf- 1k split: TI kNN accuracy + effective rank vs step.
  decoupling.pdf  - per split: CIFAR kNN at the TI-selected checkpoint vs at
                    its own (smoothed) peak.

Data: evaluation/results.csv (linear probes) + data/campaign_histories.json
(W&B probe histories, fetched by fetch_run_data.py). CPU only.

Usage: python fig_results.py
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/yolerinskiy/Studies/TUDelft/research-project/reducing-data-in-visual-ai"
CSV = os.path.join(REPO, "evaluation", "results.csv")
CACHE = os.path.join(HERE, "data", "campaign_histories.json")
OUT = lambda name: os.path.join(HERE, "..", name)

CAMPAIGN_DIRS = ("e97d41c-curve", "e97d41c-dirty-curve")
NON_PLATEAUED = {32000, 64000, 100000}  # TI-best within the final probes
SMOOTH_W = 3  # same smoothing window as training


def load_curve():
    """(split -> linear-probe acc) for the campaign rows, + the latest floor."""
    points, floor = {}, None
    with open(CSV) as f:
        for row in csv.DictReader(f):
            if row["method"] == "barlow_twins_random_init":
                floor = float(row["accuracy"])  # latest row wins (same env)
            if row["method"] == "barlow_twins" and any(
                d in row["checkpoint_path"] for d in CAMPAIGN_DIRS
            ):
                # FROZEN campaign numbers = first eval per split (2026-06-10
                # 12:49-13:33 batch); later rows are budget-extension re-evals.
                points.setdefault(int(row["fraction"]), float(row["accuracy"]))
    return points, floor


def smooth(vals, w=SMOOTH_W):
    return [sum(vals[max(0, i - w + 1): i + 1]) / (i - max(0, i - w + 1) + 1)
            for i in range(len(vals))]


def fig_curve():
    points, floor = load_curve()
    ns = sorted(points)
    accs = [points[n] for n in ns]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(ns, accs, "-", color="#1f77b4", lw=1.8, zorder=2)
    filled = [n for n in ns if n not in NON_PLATEAUED]
    ax.plot(filled, [points[n] for n in filled], "o", color="#1f77b4", ms=7,
            label="plateaued within budget", zorder=3)
    open_ns = [n for n in ns if n in NON_PLATEAUED]
    ax.plot(open_ns, [points[n] for n in open_ns], "o", mfc="white",
            color="#1f77b4", ms=7, mew=1.8,
            label="validation still rising at budget end", zorder=3)
    ax.axhline(floor, ls=":", color="#666666", lw=1.5)
    ax.text(ns[0], floor + 0.5, f"random-init floor ({floor:.1f}%)",
            fontsize=8.5, color="#555555")

    for n, a in points.items():
        dy = -1.9 if n in (4000, 8000) else 1.1
        ax.annotate(f"{a:.1f}", (n, a), textcoords="offset points",
                    xytext=(0, 14 * dy / abs(dy) * 0.6), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k", "32k", "64k", "100k"],
                       fontsize=9)
    ax.set_xlabel("pre-training set size $N$ (log scale)", fontsize=10)
    ax.set_ylabel("CIFAR-10 linear-probe top-1 (%)", fontsize=10)
    ax.set_ylim(38, 75)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.16),
              frameon=True, framealpha=1.0, edgecolor="none")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT("curve.pdf"), bbox_inches="tight")
    print("curve.pdf:", {n: a for n, a in points.items()}, "floor", floor)


def fig_overtraining():
    cache = json.load(open(CACHE))
    rows = sorted(cache["1000"]["rows"], key=lambda r: r["_step"])
    steps = [r["_step"] for r in rows]
    ti = [100 * r["knn/acc_ti"] for r in rows]
    sti = smooth(ti)
    rank = [r["knn/effective_rank_ti"] for r in rows]
    i_best = max(range(len(sti)), key=lambda i: sti[i])

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.plot(steps, ti, color="#1f77b4", lw=1.0, alpha=0.45)
    ax.plot(steps, sti, color="#1f77b4", lw=2.0,
            label="TI validation kNN top-1 (smoothed)")
    ax.plot(steps[i_best], sti[i_best], "*", color="#d62728", ms=15, zorder=5,
            label=f"selected checkpoint (step {steps[i_best]})")
    ax.set_xlabel("optimizer step", fontsize=10)
    ax.set_ylabel("TI kNN top-1 (%)", fontsize=10, color="#1f77b4")
    ax.tick_params(labelsize=9)
    ax.set_ylim(0, 12.5)

    ax2 = ax.twinx()
    ax2.plot(steps, rank, color="#2ca02c", lw=1.8, ls="--",
             label="effective rank (right axis)")
    ax2.set_ylabel("effective rank (of 192)", fontsize=10, color="#2ca02c")
    ax2.set_ylim(0, 192)
    ax2.tick_params(labelsize=9)

    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, fontsize=8, loc="lower right",
              frameon=False)
    fig.tight_layout()
    fig.savefig(OUT("overtraining.pdf"), bbox_inches="tight")
    print(f"overtraining.pdf: 1k peak {max(sti):.2f}%@{steps[i_best]}, "
          f"end {sti[-1]:.2f}%, rank end {rank[-1]:.0f}")


def fig_decoupling():
    cache = json.load(open(CACHE))
    ns, at_ti, at_cf = [], [], []
    for split in sorted(cache, key=int):
        rows = sorted(cache[split]["rows"], key=lambda r: r["_step"])
        scf = smooth([100 * r["knn/acc_cifar"] for r in rows])
        sti = smooth([100 * r["knn/acc_ti"] for r in rows])
        i_ti = max(range(len(sti)), key=lambda i: sti[i])
        i_cf = max(range(len(scf)), key=lambda i: scf[i])
        ns.append(int(split))
        at_ti.append(scf[i_ti])
        at_cf.append(scf[i_cf])

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(ns, at_cf, "s-", color="#2ca02c", ms=6, lw=1.6,
            label="at its own peak (downstream oracle)")
    ax.plot(ns, at_ti, "o-", color="#d62728", ms=6, lw=1.6,
            label="at the TI-selected checkpoint")
    ax.fill_between(ns, at_ti, at_cf, color="#d62728", alpha=0.12)
    for n, a, b in zip(ns, at_ti, at_cf):
        if b - a >= 1.0:
            ax.annotate(f"$-${b - a:.1f}", (n, (a + b) / 2), fontsize=8,
                        ha="center", color="#a03030",
                        textcoords="offset points", xytext=(14, -3))

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k", "32k", "64k", "100k"],
                       fontsize=9)
    ax.set_xlabel("pre-training set size $N$ (log scale)", fontsize=10)
    ax.set_ylabel("CIFAR-10 kNN top-1 (%)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT("decoupling.pdf"), bbox_inches="tight")
    print("decoupling.pdf:", [(n, round(b - a, 2)) for n, a, b in zip(ns, at_ti, at_cf)])


if __name__ == "__main__":
    fig_curve()
    fig_overtraining()
    fig_decoupling()
