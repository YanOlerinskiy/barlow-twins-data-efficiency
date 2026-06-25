"""VTAB-1k data-efficiency curve split into the three VTAB groups
(natural / specialized / structured) — one line per group.

Reads the per-job eval CSVs written by vtab_protocol_eval.py (default: the campaign
backup's evaluation/vtab_s*.csv), aggregates per group over whatever pretrain seeds
are present (mean +/- std), and emits a table, a CSV, and a plot. Re-run as more
seeds finish — it uses every seed it finds.

    python barlow_twins_experiments/analyze_vtab_groups.py \
        --eval_glob '~/Studies/TUDelft/campaign-backup/evaluation/vtab_s*.csv' \
        --selection final --out_dir ~/Studies/TUDelft/campaign-backup/analysis
"""
import argparse
import csv
import glob
import os
import statistics as st

VTAB_TASK_GROUPS = {
    "natural": ["caltech101", "cifar", "dtd", "oxford_flowers102",
                "oxford_iiit_pet", "sun397", "svhn"],
    "specialized": ["eurosat", "patch_camelyon", "resisc45", "diabetic_retinopathy"],
    "structured": ["clevr_count", "clevr_dist", "dmlab", "kitti",
                   "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"],
}
SPLITS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
SEL_METHOD = {"final": "barlow_twins", "final_vtab": "barlow_twins_vtabsel"}
RANDOM_METHOD = "barlow_twins_random_init"


def load_rows(eval_glob):
    """Load + dedupe eval rows keyed by (method, fraction, pretrain_seed, eval_dataset)."""
    rows = {}
    files = glob.glob(os.path.expanduser(eval_glob))
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(r["method"], r["fraction"], r["pretrain_seed"], r["eval_dataset"])] = r
    return list(rows.values()), files


def acc_lookup(rows):
    idx = {}
    for r in rows:
        idx[(r["method"], int(r["fraction"]), int(r["pretrain_seed"]), r["eval_dataset"])] = float(r["accuracy"])
    return idx


def group_mean(idx, method, split, seed, tasks):
    vals = [idx.get((method, split, seed, t)) for t in tasks]
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if len(vals) == len(tasks) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_glob", default="~/Studies/TUDelft/campaign-backup/evaluation/vtab_s*.csv")
    p.add_argument("--selection", default="final", choices=["final", "final_vtab"],
                   help="Which cooled checkpoint's eval to curve: 'final' = TI-selected, "
                        "'final_vtab' = transfer-(VTAB)-selected oracle.")
    p.add_argument("--out_dir", default="~/Studies/TUDelft/campaign-backup/analysis")
    cli = p.parse_args()
    out_dir = os.path.expanduser(cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    method = SEL_METHOD[cli.selection]

    rows, files = load_rows(cli.eval_glob)
    if not rows:
        raise SystemExit(f"No eval rows found from {cli.eval_glob}")
    idx = acc_lookup(rows)
    seeds = sorted({int(r["pretrain_seed"]) for r in rows
                    if r["method"] == method and int(r["pretrain_seed"]) >= 0})
    print(f"Loaded {len(files)} CSV(s); selection='{cli.selection}' (method={method}); seeds={seeds}")

    # random-init floor per group (seed -1, single value)
    rand = {}
    for g, tasks in VTAB_TASK_GROUPS.items():
        rand[g] = group_mean(idx, RANDOM_METHOD, 0, -1, tasks)

    # per group/split: per-seed group-mean -> mean,std
    table = {g: {} for g in VTAB_TASK_GROUPS}
    for g, tasks in VTAB_TASK_GROUPS.items():
        for s in SPLITS:
            per_seed = [group_mean(idx, method, s, sd, tasks) for sd in seeds]
            per_seed = [v for v in per_seed if v is not None]
            if per_seed:
                table[g][s] = (st.mean(per_seed),
                               st.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                               len(per_seed))

    # ---- print + CSV ----
    csv_path = os.path.join(out_dir, f"vtab_groups_{cli.selection}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "split", "mean_acc", "std_acc", "n_seeds", "random_floor"])
        for g in VTAB_TASK_GROUPS:
            print(f"\n[{g}]  random={rand[g]:.2f}" if rand[g] is not None else f"\n[{g}]")
            print(f"  {'split':>7} {'mean':>7} {'std':>6} {'n':>2}")
            for s in SPLITS:
                if s in table[g]:
                    m, sd, n = table[g][s]
                    print(f"  {s:>7} {m:>7.2f} {sd:>6.2f} {n:>2}")
                    w.writerow([g, s, f"{m:.4f}", f"{sd:.4f}", n, f"{rand[g]:.4f}"])
    print(f"\nwrote {csv_path}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable: {e}; CSV written, skipping plot)")
        return
    colors = {"natural": "tab:blue", "specialized": "tab:orange", "structured": "tab:green"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in VTAB_TASK_GROUPS:
        xs = [s for s in SPLITS if s in table[g]]
        if not xs:
            continue
        ms = [table[g][s][0] for s in xs]
        sds = [table[g][s][1] for s in xs]
        ax.plot(xs, ms, "-o", color=colors[g], label=f"{g}")
        ax.fill_between(xs, [m - d for m, d in zip(ms, sds)], [m + d for m, d in zip(ms, sds)],
                        color=colors[g], alpha=0.15)
        if rand[g] is not None:
            ax.axhline(rand[g], color=colors[g], ls=":", alpha=0.6, lw=1)
    ax.set_xscale("log")
    ax.set_xticks(SPLITS)
    ax.set_xticklabels([f"{s//1000}k" for s in SPLITS])
    ax.set_xlabel("pretraining set size (images)")
    ax.set_ylabel("VTAB-1k linear-probe top-1 (%)")
    ax.set_title(f"BT ViT-Tiny transfer by VTAB group ({cli.selection}-selected, "
                 f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''})\n"
                 f"dotted = random-init floor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"vtab_groups_{cli.selection}.{ext}"), dpi=150)
    print(f"wrote {out_dir}/vtab_groups_{cli.selection}.{{pdf,png}}")


if __name__ == "__main__":
    main()
