"""Meeting figure (NOT in the paper): 1k-16k linear-probe accuracy under
downstream-blind (TI) selection across seeds 42/43/44 vs the downstream-oracle
checkpoints (seeds 43/44), plus the seed-42 budget-extension effect on the
large splits. Writes ../meeting_3seed.pdf.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = ("/home/yolerinskiy/Studies/TUDelft/research-project/"
       "reducing-data-in-visual-ai/evaluation/results.csv")
OUT = os.path.join(HERE, "..", "meeting_3seed.pdf")
NS = [1000, 2000, 4000, 8000, 16000]
LABELS = ["1k", "2k", "4k", "8k", "16k"]


def main() -> None:
    rows = list(csv.DictReader(open(CSV)))

    def acc(method, n, sub):
        return [float(r["accuracy"]) for r in rows
                if r["method"] == method and r["fraction"] == str(n)
                and sub in r["checkpoint_path"]]

    ti = {n: [acc("barlow_twins", n, "e97d41c")[0]] + acc("barlow_twins", n, "9b11e5d")
          for n in NS}
    orc = {n: acc("barlow_twins_cifarsel", n, "9b11e5d") for n in NS}

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ti_mean = [sum(ti[n]) / 3 for n in NS]
    ax.fill_between(NS, [min(ti[n]) for n in NS], [max(ti[n]) for n in NS],
                    color="#d62728", alpha=0.15)
    ax.plot(NS, ti_mean, "o-", color="#d62728", lw=2,
            label="downstream-blind (TI) selection, mean of seeds 42/43/44")
    orc_mean = [sum(orc[n]) / 2 for n in NS]
    ax.fill_between(NS, [min(orc[n]) for n in NS], [max(orc[n]) for n in NS],
                    color="#2ca02c", alpha=0.15)
    ax.plot(NS, orc_mean, "s-", color="#2ca02c", lw=2,
            label="downstream-oracle checkpoints, mean of seeds 43/44")

    for n, m_ti, m_or in zip(NS, ti_mean, orc_mean):
        ax.annotate(f"+{m_or - m_ti:.1f}", (n, (m_ti + m_or) / 2), fontsize=8,
                    ha="center", color="#1a6b1a",
                    textcoords="offset points", xytext=(15, -3))

    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_xlabel("pre-training set size $N$ (log scale)", fontsize=10)
    ax.set_ylabel("CIFAR-10 linear-probe top-1 (%)", fontsize=10)
    ax.set_title("Probe-level replication: oracle is monotone in every seed;\n"
                 "blind selection dips at 8k in every seed", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("ti means:", [round(x, 2) for x in ti_mean])
    print("oracle means:", [round(x, 2) for x in orc_mean])
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
