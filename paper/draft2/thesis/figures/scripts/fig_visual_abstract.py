"""Visual abstract / pipeline figure: nested unlabeled subsets -> Barlow Twins
pre-training -> frozen encoder -> CIFAR-10 linear probe -> the measured
data-efficiency curve (real seed-42 numbers from evaluation/results.csv).

Usage: python fig_visual_abstract.py  (writes ../pipeline.pdf)
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "pipeline.pdf")
CSV = ("/home/yolerinskiy/Studies/TUDelft/research-project/"
       "reducing-data-in-visual-ai/evaluation/results.csv")
CAMPAIGN_DIRS = ("e97d41c-curve", "e97d41c-dirty-curve")


def load_curve():
    points, floor = {}, None
    with open(CSV) as f:
        for row in csv.DictReader(f):
            if row["method"] == "barlow_twins_random_init":
                floor = float(row["accuracy"])
            if row["method"] == "barlow_twins" and any(
                d in row["checkpoint_path"] for d in CAMPAIGN_DIRS
            ):
                # FROZEN campaign numbers = first eval per split (2026-06-10
                # 12:49-13:33 batch); later rows are budget-extension re-evals.
                points.setdefault(int(row["fraction"]), float(row["accuracy"]))
    ns = sorted(points)
    return ns, [points[n] for n in ns], floor


def box(ax, x, y, w, h, text, fc, fontsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="#444444", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.4, color="#444444"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Nested subsets: three offset rectangles behind a front box (text only on
    # the front box, so nothing overflows).
    for off in (0.030, 0.015):
        ax.add_patch(FancyBboxPatch((0.025 + off, 0.16 + off * 1.6), 0.205, 0.66,
                                    boxstyle="round,pad=0.008",
                                    fc="white", ec="#44679a", lw=1.0))
    ax.add_patch(FancyBboxPatch((0.025, 0.16), 0.205, 0.66,
                                boxstyle="round,pad=0.008",
                                fc="#dbe9f6", ec="#44679a", lw=1.2))
    ax.text(0.128, 0.49, "Tiny-ImageNet\nsubsets\n$1k \\subset \\ldots \\subset 100k$\n(no labels)",
            ha="center", va="center", fontsize=7.5, linespacing=1.4)

    arrow(ax, 0.245, 0.49, 0.29, 0.49)

    box(ax, 0.295, 0.14, 0.255, 0.70,
        "Barlow Twins\npre-training\nViT-Tiny/8 + projector\n(two views,\ncross-correlation loss)",
        "#fdebd3", fontsize=7.5)

    arrow(ax, 0.56, 0.49, 0.605, 0.49)

    box(ax, 0.61, 0.42, 0.125, 0.30, "frozen\nencoder", "#e8e8e8", fontsize=7.5)
    box(ax, 0.61, 0.04, 0.125, 0.26, "linear probe\non CIFAR-10", "#ddeedd", fontsize=7.5)
    arrow(ax, 0.6725, 0.41, 0.6725, 0.315)

    arrow(ax, 0.745, 0.18, 0.795, 0.26)

    # The measured curve (seed 42), small but real.
    ns, accs, floor = load_curve()
    ax2 = fig.add_axes([0.795, 0.30, 0.185, 0.50])
    ax2.plot(ns, accs, "o-", color="#1f77b4", ms=2.5, lw=1.3)
    ax2.axhline(floor, ls=":", lw=1.0, color="#888888")
    ax2.set_xscale("log")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim(36, 76)
    ax2.set_xlabel("$N$ (log)", fontsize=7.5, labelpad=1)
    ax2.set_ylabel("CIFAR-10 top-1", fontsize=7.5, labelpad=1)
    ax2.text(0.97, 0.10, "random-init floor", fontsize=6, ha="right",
             transform=ax2.transAxes, color="#666666")
    ax2.set_title("data-efficiency curve", fontsize=7.5, pad=2)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {os.path.abspath(OUT)}; curve points {len(ns)}, floor {floor}")


if __name__ == "__main__":
    main()
