"""Render the Barlow Twins learning curve from a results CSV."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(_REPO_ROOT, "evaluation")
DEFAULT_CSV = os.path.join(EVAL_DIR, "results.csv")
DEFAULT_PNG = os.path.join(EVAL_DIR, "learning_curve.png")


def read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=DEFAULT_CSV)
    p.add_argument("--out", type=str, default=DEFAULT_PNG)
    p.add_argument("--method", type=str, default="barlow_twins",
                   help="Filter rows to this method.")
    p.add_argument("--baseline_method", type=str,
                   default="barlow_twins_random_init",
                   help="Plot rows from this method as a horizontal dashed "
                        "line; pass an empty string to disable.")
    p.add_argument("--title", type=str,
                   default="Barlow Twins data-efficiency on CIFAR-10")
    p.add_argument("--annotate", action="store_true", default=True,
                   help="Annotate each data point with its accuracy value.")
    p.add_argument("--x_scale", type=str, default="categorical",
                   choices=["categorical", "log"],
                   help="'categorical' = equally-spaced; 'log' = log10 scale.")
    return p.parse_args()


def _format_count(n: int) -> str:
    """1000 -> '1k', 100000 -> '100k', 500 -> '500'."""
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def main() -> None:
    args = parse_args()
    all_rows = read_rows(args.csv)

    rows = [r for r in all_rows if r["method"] == args.method]
    if not rows:
        raise SystemExit(f"No rows for method={args.method} in {args.csv}")

    by_fraction: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        by_fraction[float(r["fraction"])].append(float(r["accuracy"]))
    fractions = sorted(by_fraction.keys())
    means = [sum(by_fraction[f]) / len(by_fraction[f]) for f in fractions]

    # X positions: either log-numeric (actual values) or categorical (indices).
    if args.x_scale == "categorical":
        x_positions = list(range(len(fractions)))
    else:
        x_positions = fractions

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        x_positions, means,
        marker="o", markersize=9, linewidth=2.5,
        color="#1f77b4",
        label=args.method.replace("_", " ").title(),
        zorder=3,
    )

    if args.annotate:
        for x, y in zip(x_positions, means):
            ax.annotate(f"{y:.1f}", (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color="#1f77b4")

    if args.baseline_method:
        baseline_rows = [r for r in all_rows if r["method"] == args.baseline_method]
        if not baseline_rows:
            print(f"WARN: no rows for baseline method {args.baseline_method}; "
                  "skipping baseline.")
        else:
            baseline_acc = sum(float(r["accuracy"]) for r in baseline_rows) / len(baseline_rows)
            ax.axhline(
                baseline_acc, linestyle="--", color="#d62728", linewidth=1.8,
                alpha=0.85,
                label=f"Random-init baseline ({baseline_acc:.2f}%)",
                zorder=2,
            )
            ax.fill_between(
                x_positions, baseline_acc, means,
                where=[m > baseline_acc for m in means],
                color="#1f77b4", alpha=0.08, zorder=1,
            )

    if args.x_scale == "categorical":
        ax.set_xticks(x_positions)
        ax.set_xticklabels([_format_count(int(f)) for f in fractions])
    else:
        ax.set_xscale("log")
    ax.set_xlabel("Pre-training images (Tiny ImageNet)", fontsize=12)
    ax.set_ylabel("CIFAR-10 linear-probe accuracy (%)", fontsize=12)
    ax.set_title(args.title, fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
