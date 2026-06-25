"""Driver: pre-train (stable phase) Barlow Twins across the splits.

Pre-training only — the cooldown (`cooldown.py`) and evaluation
(`leo_protocol_eval.py`) are separate, on-demand steps. Calls pretrain.py as a
subprocess per split so each run gets a fresh CUDA context and W&B process
(also easier to recover/resume if one split fails).
"""
import argparse
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(_REPO_ROOT, "barlow_twins_experiments")
CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "checkpoints")

sys.path.append(_REPO_ROOT)
from barlow_twins_experiments.pretrain import default_run_tag  # noqa: E402

# Per-split epoch budget, TIME-TARGETED. All 8 splits on ONE GPU in ~9 h (under a
# ~12 h target, leaving margin to resume any split still rising). Capped at 1000
# epochs/split. No early stop: each split trains its full budget under a constant LR
# (stable phase); pretrain.py selects the best-val (kNN) checkpoint and cooldown.py
# anneals it on demand. Small splits sit at the 1000 cap (cheap, and they peak latest
# in epochs); big splits are generous past their (early) peaks. Budgets are resumable —
# a split still rising at its budget can be extended via `pretrain.py --resume_from
# <_stable>` (constant LR commits no horizon).
#
# Timing: calibrated from a real 8k run (500 ep / 16k steps = 1440 s -> ~13 steps/s,
# t_step ~0.078 s). per-epoch ~= (split/250)*0.078 + diagnostic-probe overhead.
# RE-CALIBRATE if the setup changes; over/undershoot is self-correcting via resume.
#
# Validation cadence is set in epochs via --knn_every_epochs (default 4), forwarded
# below and converted to steps inside pretrain.py (batch 250 makes that exact).
#
#   split: epoch_budget       # ~9 h total for all 8 on one GPU
SPLIT_CONFIGS = {
    1000: 1000,
    2000: 1000,
    4000: 1000,
    8000: 1000,
    16000: 600,
    32000: 400,
    64000: 300,
    100000: 250,
}


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=None,
                   help="Data subset seed forwarded to every split (defaults to --seed inside "
                        "pretrain.py, i.e. paired). For a paired replicate run --seed D "
                        "--data_seed D; needs data/generate_splits.py --data_seed D first.")
    p.add_argument("--batch_size", type=int, default=250,
                   help="250 divides every split into whole epochs (1 epoch = 1 "
                        "exposure/image); see pretrain.py for rationale.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--knn_every_epochs", type=float, default=4.0,
                   help="Validation probe cadence in epochs, forwarded to every split "
                        "(pretrain.py converts to steps). Default 4.")
    p.add_argument("--wandb_project", type=str, default="barlow-twins-data-efficiency")
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="Forwarded to every split. Use 'disabled' on a headless rented box with "
                        "no W&B login (else the default 'online' tries to authenticate and stalls).")
    p.add_argument("--max_split", type=int, default=None,
                   help="Only run splits with `split <= max_split` (e.g. 32000).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Uniform epoch budget for EVERY split, overriding the per-split "
                        "SPLIT_CONFIGS defaults (which cap big splits below 1000 to fit a "
                        "time target). Set 1000 for the paired-seed campaign so all splits "
                        "train to the same horizon.")
    p.add_argument("--run_tag", type=str, default=None,
                   help="Run index shared by the whole sweep. Computed once at "
                        "driver start (git SHA) so a mid-sweep code edit cannot "
                        "fork the tag between splits.")
    p.add_argument("--diagnostic_cifar_probe", action="store_true",
                   help="Forward --diagnostic_cifar_probe to every split so each run "
                        "also logs the CIFAR-10 kNN diagnostic alongside the in-domain "
                        "TI probe (for the saturation pass / correlation figure).")
    p.add_argument("--vtab_probe", action="store_true",
                   help="Forward --vtab_probe to every split: log the VTAB-1k kNN transfer "
                        "signal (and save a _best_vtab oracle) across training.")
    p.add_argument("--vtab_every_epochs", type=float, default=None,
                   help="Forward a custom VTAB probe cadence (epochs) to every split.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    run_tag = args.run_tag if args.run_tag is not None else default_run_tag()

    splits = sorted(SPLIT_CONFIGS)
    if args.max_split is not None:
        splits = [s for s in splits if s <= args.max_split]
    print(f"[driver] run_tag={run_tag}, splits={splits}")

    python = sys.executable
    timings: list[tuple[int, float]] = []
    for split in splits:
        epochs = args.epochs if args.epochs is not None else SPLIT_CONFIGS[split]
        t0 = time.time()
        cmd = [
            python, os.path.join(EXPERIMENTS_DIR, "pretrain.py"),
            "--split", str(split),
            "--epochs", str(epochs),
            "--knn_every_epochs", str(args.knn_every_epochs),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--run_tag", run_tag,
            "--wandb_project", args.wandb_project,
        ]
        if args.data_seed is not None:
            cmd += ["--data_seed", str(args.data_seed)]
        if args.diagnostic_cifar_probe:
            cmd.append("--diagnostic_cifar_probe")
        if args.vtab_probe:
            cmd.append("--vtab_probe")
        if args.vtab_every_epochs is not None:
            cmd += ["--vtab_every_epochs", str(args.vtab_every_epochs)]
        run(cmd)
        elapsed = time.time() - t0
        timings.append((split, elapsed))
        print(f"[driver] split={split} done in {elapsed / 60:.1f} min")

    total = sum(t for _, t in timings)
    print(f"\n[driver] SWEEP COMPLETE — {total / 60:.1f} min total (run_tag={run_tag})")
    print(f"{'split':>8} {'minutes':>9}")
    for split, t in timings:
        print(f"{split:>8} {t / 60:>9.1f}")


if __name__ == "__main__":
    main()
