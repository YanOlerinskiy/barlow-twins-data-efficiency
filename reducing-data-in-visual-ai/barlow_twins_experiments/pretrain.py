"""Barlow Twins stable-phase pre-training on a Tiny ImageNet split (constant-LR,
WSD-style), with best-validation checkpoint selection and resumable checkpoints.

This script runs ONLY the stable phase: linear warmup (in optimizer steps) ->
constant LR for the per-split epoch budget. There is no early stop, no trigger,
and no automatic cooldown — the terminal cooldown that produces the eval model
is a separate, on-demand step (`cooldown.py`), and resuming/extending a run is a
separate invocation (`--resume_from`).

Why this split (WSD, Hägele et al. 2024 arXiv:2405.18392; MiniCPM, Hu et al.
2024 arXiv:2404.06395): a constant stable LR commits to NO training horizon, so
(a) the stable checkpoint is a reusable artifact you can resume and extend later,
and (b) the cooldown can be initiated from any checkpoint at any time. Constant
LR (vs cosine) also makes every checkpoint schedule-comparable, so best-val
selection is not confounded by where it sits on a decay curve (Chinchilla
cosine-horizon result, arXiv:2203.15556; SGDR, arXiv:1608.03983).

Outputs two checkpoints per (tag, split, seed) — three with the CIFAR probe on:
- `_best`  — the best-smoothed-val snapshot (the peak); `cooldown.py` anneals it.
- `_stable`— the last state, fully resumable (model + optimizer + epoch + RNG +
             best-val tracking); `--resume_from` continues from it.
- `_best_cifar` (only with --diagnostic_cifar_probe) — the best-smoothed snapshot
             under the CIFAR-10 TRAIN-only kNN probe: target-aware selection. The
             in-domain and transfer optima diverge under over-training (TI keeps
             rising while CIFAR peaks early and declines), so this second
             checkpoint captures the transfer peak the TI signal cannot see.
             CIFAR test stays untouched. Never drives `_best`.

Validation uses an in-domain Tiny-ImageNet kNN probe: gallery + query are
stratified-disjoint partitions of the labelled TI `valid` split (k=20, 5k/5k
default), held out from `train` for every split — so the shipped checkpoint is
not selected by peeking at the downstream (CIFAR-10) domain. The CIFAR-10 probe
remains available as an optional logged-only diagnostic (--diagnostic_cifar_probe).

Cadence is set in epochs (--knn_every_epochs) and converted to steps from the
loader length (batch 250 divides every split, so the conversion is exact); an
explicit --knn_every_steps overrides it. Per-split *generous* budgets (not equal
steps) keep this a DATA-efficiency study, following Cole et al. 2022
(arXiv:2105.05837).
"""
import argparse
import csv
import math
import os
import random
import subprocess
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from lightly.loss import BarlowTwinsLoss
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import AdamW
from tqdm import tqdm

import wandb

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO_ROOT)
from models.barlow_twins import BarlowTwinsViT  # noqa: E402

from barlow_twins_experiments.cifar10_features import (  # noqa: E402
    extract_features,
    get_knn_db_val_loaders,
)
from barlow_twins_experiments.tiny_imagenet_features import (  # noqa: E402
    get_tiny_imagenet_knn_loaders,
)
from barlow_twins_experiments.vtab_features import (  # noqa: E402
    get_vtab_knn_loaders,
    resolve_vtab_tasks,
)
from barlow_twins_experiments.two_view_dataset import get_two_view_loader  # noqa: E402

CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "checkpoints")
STABLE_SAVE_EVERY_EPOCHS = 50  # crash-insurance cadence for the resumable checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Stability/reproducibility on consumer Blackwell cards (RTX 50xx):
    # fixed algorithms, no autotuning.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # TF32 for pretraining matmuls/convs: ~1.5-2x faster, still deterministic
    # (same config + machine -> bit-identical run; numerics differ from fp32).
    # Pretraining only — the linear-probe eval (leo_protocol_eval.py) stays fp32.
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"


def append_probe_row(path: str, header: list[str], row: dict) -> None:
    """Append one probe row to a local CSV (header written once), flushed so a crash
    keeps prior rows. This persists the full per-probe trace independently of W&B —
    the source of truth for the later TI-vs-transfer peak-divergence analysis.

    If a prior segment wrote a DIFFERENT column set (e.g. a resume launched with a
    different --vtab_tasks), the existing file is archived to `<path>.legacy-N` and a
    fresh header is written, rather than silently misaligning appended rows or raising
    a DictWriter error mid-run."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as f:
            first = f.readline().rstrip("\n\r")
        if first != ",".join(header):
            n = 1
            while os.path.exists(f"{path}.legacy-{n}"):
                n += 1
            os.rename(path, f"{path}.legacy-{n}")
            print(f"[probe-csv] {os.path.basename(path)} column set changed; archived -> "
                  f"{os.path.basename(path)}.legacy-{n}, starting fresh.")
    existed = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not existed:
            w.writeheader()
        w.writerow(row)
        f.flush()


def default_run_tag() -> str:
    """Short git SHA of the shared repo; '-dirty' if uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "nogit"


def warmup_constant_factor(step: int, warmup_steps: int) -> float:
    """LR factor for the stable phase: linear warmup, then a constant 1.0."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    return 1.0


def cooldown_factor(t: int, length: int, shape: str = "sqrt") -> float:
    """LR factor for the terminal cooldown, 1.0 -> 0.0 over `length` steps.

    `shape="sqrt"` (default) is the (1 - sqrt(t)) decay of Hägele et al. 2024
    (arXiv:2405.18392) — fast initial drop, long low-LR tail. `"cosine"` and
    `"linear"` are provided for the (small, LLM-derived) shape ablation. Used by
    `cooldown.py`.
    """
    frac = min(1.0, t / max(1, length))
    if shape == "sqrt":
        return 1.0 - math.sqrt(frac)
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * frac))
    if shape == "linear":
        return 1.0 - frac
    raise ValueError(f"cooldown_shape must be sqrt/cosine/linear, got {shape!r}")


def _infinite_batches(loader):
    """Yield batches from `loader` forever, re-iterating (reshuffling) each pass."""
    while True:
        for batch in loader:
            yield batch


def _capture_rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    # RNG states must be CPU ByteTensors. Loading a checkpoint with
    # map_location=cuda moves them onto the GPU, so coerce them back.
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [s.cpu().to(torch.uint8) for s in state["cuda"]]
        )
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _effective_rank(features: np.ndarray) -> float:
    """Effective rank (Roy & Vetterli 2007) of the feature covariance:
    exp(entropy of the normalized eigenvalue spectrum), in [1, dim].

    Diagnostic for dimensional collapse (Jing et al. 2022, arXiv:2110.09348).
    Logged only — never used for stopping or selection.
    """
    centered = features - features.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(centered) - 1)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = eig.sum()
    if total <= 0:
        return 0.0
    p = eig / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def knn_probe(
    model: nn.Module,
    device: torch.device,
    db_loader,
    val_loader,
    k: int,
) -> tuple[float, float]:
    """Run a kNN probe over prebuilt (gallery, query) loaders; return (top-1
    accuracy on the query set, effective rank of the query CLS features).

    Loader-agnostic: the gallery/query come from the in-domain Tiny-ImageNet
    val partition (selection signal) or the CIFAR-10 diagnostic — built once and
    reused across probes.
    """
    db_X, db_y = extract_features(model, db_loader, device)
    q_X, q_y = extract_features(model, val_loader, device)
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(db_X, db_y)
    return float(clf.score(q_X, q_y)), _effective_rank(q_X)


def run_probe(
    model: nn.Module,
    device: torch.device,
    ti_db,
    ti_val,
    k: int,
    cifar_db=None,
    cifar_val=None,
) -> tuple[float, float, float | None]:
    """Run the in-domain TI probe (drives selection/logging) and, when CIFAR
    loaders are supplied, the CIFAR diagnostic probe. Returns
    (ti_acc, ti_eff_rank, cifar_acc) — cifar_acc is None with the diagnostic
    off. Leaves the model in train() mode (the probes set eval()).
    """
    ti_acc, ti_rank = knn_probe(model, device, ti_db, ti_val, k)
    cifar_acc = None
    if cifar_db is not None:
        cifar_acc, _ = knn_probe(model, device, cifar_db, cifar_val, k)
    model.train()
    return ti_acc, ti_rank, cifar_acc


def resolve_run_tag(run_tag: str | None) -> str:
    """Mirror the run-tag convention: default to the git SHA; a custom label gets
    the SHA prefixed so every checkpoint dir stays traceable to the code."""
    auto_tag = default_run_tag()
    if run_tag is None:
        return auto_tag
    if auto_tag.split("-")[0] not in run_tag:
        return f"{auto_tag}-{run_tag}"
    return run_tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100,
                   help="Per-split training budget in epochs; runs in FULL (no early "
                        "stop). With --resume_from, the new (larger) target. Generous — "
                        "best-val selection + the (separate) cooldown absorb over-training, "
                        "so the exact value is non-critical.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a `_stable` checkpoint to continue training from (model + "
                        "optimizer + epoch + RNG + best-val tracking restored); train on to "
                        "--epochs. Enabled by the constant-LR schedule (no horizon committed).")
    p.add_argument("--batch_size", type=int, default=250,
                   help="250 divides every split (all multiples of 1000) into whole "
                        "epochs, so 1 epoch = exactly 1 exposure/image across splits "
                        "(256 drops up to 23%% via drop_last at 1k). Power-of-2 is "
                        "convention, not a HW requirement — the batch dim does not hit "
                        "Tensor-Core tile quantization at our token count.")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Constant stable-phase LR (AdamW, batch 250). Selected by a "
                        "one-off ablation over {1e-4..1.5e-3} on split 8000: inverted-U "
                        "with a 3e-4–7e-4 plateau (within probe noise) and degradation at "
                        "1.5e-3; 5e-4 = plateau center and DINO's batch-256 reference.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_param", type=float, default=1e-2)
    p.add_argument("--projector_dim", type=int, default=1024,
                   help="Barlow Twins projector hidden AND output width (symmetric "
                        "192->dim->dim). Default 1024 (publication-style). The BT-specific "
                        "knob for the projector-width x N study; the backbone embedding "
                        "(192-d, what the linear probe reads) is unchanged, so checkpoints "
                        "at different widths stay probe-comparable. Saved in args; "
                        "cooldown.py / leo_protocol_eval.py rebuild at this width.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear LR warmup, in optimizer steps (uniform across splits); "
                        "the LR is then held constant for the rest of the budget.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=None,
                   help="Selects WHICH images form each split (independent of the model "
                        "seed, which controls init/shuffle/augmentation RNG). Defaults to "
                        "--seed, so paired (model, data) replicates are the natural default "
                        "(e.g. --seed 43 -> data_seed 43). Requires the matching index file "
                        "from data/generate_splits.py --data_seed N. The TI-valid kNN probe "
                        "set is fixed across data seeds (a constant evaluation ruler).")
    p.add_argument("--knn_every_epochs", type=float, default=4.0,
                   help="kNN probe cadence in epochs (converted to steps from the loader "
                        "length). Small splits run more epochs so they get more probes. "
                        "0 disables probing. Overridden by --knn_every_steps if that is >0.")
    p.add_argument("--knn_every_steps", type=int, default=0,
                   help="Explicit probe cadence in optimizer steps; 0 (default) means "
                        "derive it from --knn_every_epochs. Override for direct/debug runs.")
    p.add_argument("--knn_db_size", type=int, default=5000,
                   help="kNN gallery size — a class-balanced partition of the TI "
                        "val split (must be divisible by 200). 5000 -> 25 imgs/class.")
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--knn_val_size", type=int, default=5000,
                   help="kNN query size — the disjoint class-balanced TI val partition "
                        "(divisible by 200; db+val <= 10000 = 50/class). 5000 -> 25/class, "
                        "200-way SE ~0.4-0.6pp.")
    p.add_argument("--diagnostic_cifar_probe", action="store_true",
                   help="Also run the CIFAR-10 kNN probe (train-only, disjoint db/val; "
                        "test untouched) each cadence: logs knn/acc_cifar and saves a "
                        "second `_best_cifar` checkpoint at its smoothed peak "
                        "(target-aware selection). Never drives `_best`.")
    p.add_argument("--vtab_probe", action="store_true",
                   help="Also run an in-training VTAB-1k kNN transfer probe (gallery=train800, "
                        "query=val200 per task; test untouched), logging knn/acc_vtab_mean (+ "
                        "per-task) and saving a `_best_vtab` oracle at its smoothed peak. Same "
                        "kNN protocol as the TI probe; cadence via --vtab_every_epochs. Never "
                        "drives `_best`. Used to show TI-selection != transfer-optimal.")
    p.add_argument("--vtab_data_root", type=str, default=None,
                   help="VTAB-1k bundle root (default: <repo>/vtab-1k).")
    p.add_argument("--vtab_tasks", type=str, default="all",
                   help="Comma-separated VTAB task subset, or 'all' (default) for all 19.")
    p.add_argument("--vtab_every_epochs", type=float, default=8.0,
                   help="VTAB probe cadence in epochs (coarser than the TI probe: 19 tasks "
                        "cost ~2-3x a TI probe). Converted to steps from loader length.")
    p.add_argument("--val_smooth_window", type=int, default=3,
                   help="Moving-average window (in probes) for best-val selection; "
                        "smooths the noisy probe so a fluke spike is not selected.")
    p.add_argument("--cj_strength", type=float, default=1.0,
                   help="Color-jitter strength multiplier for both BYOL views "
                        "(1.0 = the BYOL/publication default).")
    p.add_argument("--min_scale", type=float, default=0.25,
                   help="RandomResizedCrop minimum area scale "
                        "(0.25 = the publication default; 0.08 collapses at 64px).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    p.add_argument("--run_tag", type=str, default=None,
                   help="Run label for checkpoints/W&B (checkpoints/<tag>/...). "
                        "Defaults to the short git SHA ('-dirty' if uncommitted); "
                        "a custom label gets the SHA prefixed automatically "
                        "(e.g. 'cj20' -> 'a1b2c3d-cj20') so every tag stays "
                        "traceable to the code that ran it.")
    p.add_argument("--wandb_project", type=str, default="barlow-twins-data-efficiency")
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def main() -> None:
    t_run_start = time.time()
    args = parse_args()
    args.run_tag = resolve_run_tag(args.run_tag)
    if args.data_seed is None:
        args.data_seed = args.seed  # paired by default; recorded in vars(args) for provenance
    git_sha = default_run_tag()  # recorded in checkpoints; checked on resume
    set_seed(args.seed)
    if args.data_seed != args.seed:
        print(
            f"[WARN] data_seed ({args.data_seed}) != model seed ({args.seed}). Checkpoint "
            f"filenames are keyed by the MODEL seed only, so use a distinct --run_tag for this "
            f"run to avoid overwriting a same-model-seed run on a different data subset."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_ckpt = None
    if args.resume_from is not None:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)

    if resume_ckpt is not None:
        # Keep the resumed run's checkpoints in the SAME directory as the source so the
        # _best/_stable lineage stays together even if the (current) run tag differs.
        ckpt_dir = os.path.dirname(os.path.abspath(args.resume_from))
    else:
        ckpt_dir = os.path.join(args.checkpoint_dir, args.run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"bt_split{args.split}_seed{args.seed}_best.pt")
    stable_path = os.path.join(ckpt_dir, f"bt_split{args.split}_seed{args.seed}_stable.pt")
    best_cifar_path = os.path.join(
        ckpt_dir, f"bt_split{args.split}_seed{args.seed}_best_cifar.pt"
    )
    best_vtab_path = os.path.join(
        ckpt_dir, f"bt_split{args.split}_seed{args.seed}_best_vtab.pt"
    )
    # Durable local probe trace (independent of W&B). The TI/CIFAR probe and the
    # (coarser) VTAB probe fire on different cadences with different column sets, so
    # they go to two append-only CSVs; resumed segments append to the same files.
    probes_csv_path = os.path.join(
        ckpt_dir, f"bt_split{args.split}_seed{args.seed}_probes.csv"
    )
    vtab_probes_csv_path = os.path.join(
        ckpt_dir, f"bt_split{args.split}_seed{args.seed}_vtab_probes.csv"
    )

    run_name = f"bt_{args.run_tag}_seed{args.seed}_split{args.split}"

    loader = get_two_view_loader(
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_scale=args.min_scale,
        cj_strength=args.cj_strength,
        data_seed=args.data_seed,
    )
    steps_per_epoch = len(loader)

    # Cadence: explicit steps override; else derive from epochs (exact at batch 250).
    if args.knn_every_steps > 0:
        knn_every_steps = args.knn_every_steps
    elif args.knn_every_epochs > 0:
        knn_every_steps = max(1, round(args.knn_every_epochs * steps_per_epoch))
    else:
        knn_every_steps = 0  # probing (and best-val selection) disabled

    # VTAB transfer probe runs on its own (coarser) cadence; gated on knn probing.
    vtab_every_steps = 0
    if args.vtab_probe and knn_every_steps > 0 and args.vtab_every_epochs > 0:
        vtab_every_steps = max(1, round(args.vtab_every_epochs * steps_per_epoch))

    # W&B GROUPING for resumable runs (NOT run-id resume: reusing an id hits a wandb
    # 0.27.0 bug that leaves the resumed run's history table empty server-side). Every
    # segment of a (tag, seed, split) lineage shares one `group`; each segment is its own
    # run (so history always materializes), and since segments sit on disjoint global_step
    # ranges, the group view reads as one continuous curve.
    wandb_group = run_name
    if resume_ckpt is not None:
        seg_name = f"{run_name}_resume{resume_ckpt.get('epoch', '?')}-{args.epochs}ep"
    else:
        seg_name = f"{run_name}_0-{args.epochs}ep"
    wandb.init(
        project=args.wandb_project,
        name=seg_name,
        group=wandb_group,
        config=vars(args),
        mode=args.wandb_mode,
    )
    wandb_run_id = getattr(wandb.run, "id", None)  # recorded for provenance (not reused)

    model = BarlowTwinsViT(
        projector_hidden=args.projector_dim, projector_out=args.projector_dim
    ).to(device)
    loss_fn = BarlowTwinsLoss(lambda_param=args.lambda_param)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    start_epoch = 0
    best_smoothed_acc = -1.0   # best smoothed val; gates best-checkpoint saving
    best_knn_acc = -1.0        # raw val at the best-smoothed step (for the record)
    best_knn_step = -1
    best_cifar_smoothed = -1.0  # target-aware twin of the above (CIFAR-train probe)
    best_cifar_acc = -1.0
    best_cifar_step = -1
    best_vtab_smoothed = -1.0   # transfer oracle on the VTAB-1k mean kNN
    best_vtab_acc = -1.0
    best_vtab_step = -1
    prior_wallclock = 0.0

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        global_step = resume_ckpt["global_step"]
        start_epoch = resume_ckpt["epoch"]
        best_smoothed_acc = resume_ckpt.get("best_smoothed_acc", -1.0)
        best_knn_acc = resume_ckpt.get("best_knn_acc", -1.0)
        best_knn_step = resume_ckpt.get("best_knn_step", -1)
        best_cifar_smoothed = resume_ckpt.get("best_cifar_smoothed", -1.0)
        best_cifar_acc = resume_ckpt.get("best_cifar_acc", -1.0)
        best_cifar_step = resume_ckpt.get("best_cifar_step", -1)
        best_vtab_smoothed = resume_ckpt.get("best_vtab_smoothed", -1.0)
        best_vtab_acc = resume_ckpt.get("best_vtab_acc", -1.0)
        best_vtab_step = resume_ckpt.get("best_vtab_step", -1)
        prior_wallclock = resume_ckpt.get("wallclock_pretrain_s", 0.0)
        _restore_rng(resume_ckpt.get("rng_state"))
        if args.epochs <= start_epoch:
            raise SystemExit(
                f"--epochs ({args.epochs}) must exceed the resumed epoch ({start_epoch})."
            )
        saved_sha = resume_ckpt.get("git_sha")
        if saved_sha and saved_sha.split("-")[0] != git_sha.split("-")[0]:
            print(
                f"[WARN] resuming across commits: checkpoint built at {saved_sha}, current code "
                f"{git_sha}. The run would span two code versions — training dynamics may differ "
                f"and the result is no longer attributable to a single commit. Resume is meant for "
                f"extending the SAME code; abort (Ctrl-C) unless this change is intentional/benign."
            )
        elif "dirty" in git_sha:
            print(
                f"[WARN] resuming with a dirty working tree ({git_sha}); the code may differ from "
                f"the checkpoint's even at the same commit. Commit before resumable runs."
            )
        print(
            f"[resume] from {args.resume_from}: epoch {start_epoch}, step {global_step}, "
            f"best-val={best_knn_acc:.4f}@{best_knn_step} (preserved so continued training "
            f"only overwrites _best on a genuine new peak)."
        )

    print(
        f"[schedule] split={args.split} (data_seed={args.data_seed}): "
        f"{steps_per_epoch} steps/epoch, "
        f"budget={args.epochs} epochs (= {args.epochs * steps_per_epoch} steps), "
        f"warmup={args.warmup_steps} steps then constant LR={args.lr}, "
        f"probe every {knn_every_steps} steps (= {args.knn_every_epochs} epochs), "
        f"val window={args.val_smooth_window}. Cooldown is a separate step (cooldown.py)."
    )

    # Probe loaders, built once and reused across every probe (the model changes,
    # the data does not). The in-domain TI val partition drives selection; the
    # CIFAR loaders are built only for the optional logged-only diagnostic.
    ti_db = ti_val = cf_db = cf_val = None
    vtab_loaders: list[tuple[str, object, object]] = []  # (task, gallery_loader, query_loader)
    if knn_every_steps > 0:
        ti_db, ti_val = get_tiny_imagenet_knn_loaders(
            db_size=args.knn_db_size,
            val_size=args.knn_val_size,
            num_workers=min(2, args.num_workers),
        )
        if args.diagnostic_cifar_probe:
            cf_db, cf_val = get_knn_db_val_loaders(
                db_size=5000, val_size=10000, stratified=True, num_workers=2,
            )
        if vtab_every_steps > 0:
            vtab_root = args.vtab_data_root  # None -> module default (<repo>/vtab-1k)
            vtab_tasks = resolve_vtab_tasks(args.vtab_tasks)
            for task in vtab_tasks:
                kw = {"task": task, "num_workers": min(2, args.num_workers)}
                if vtab_root is not None:
                    kw["data_root"] = vtab_root
                vdb, vq, _nc = get_vtab_knn_loaders(**kw)
                vtab_loaders.append((task, vdb, vq))
            print(f"[vtab] transfer probe on {len(vtab_loaders)} tasks every "
                  f"{vtab_every_steps} steps (= {args.vtab_every_epochs} epochs)")

    val_window: deque[float] = deque(maxlen=max(1, args.val_smooth_window))
    cf_window: deque[float] = deque(maxlen=max(1, args.val_smooth_window))
    vtab_window: deque[float] = deque(maxlen=max(1, args.val_smooth_window))
    if resume_ckpt is not None:
        # Restore the moving-average windows too. The best-* SCALARS are restored above,
        # but if the windows restart empty the post-resume smoothed value is computed over
        # 1-2 probes instead of `val_smooth_window`, so a single fluke probe can exceed the
        # restored best and overwrite a good _best. Repopulating keeps smoothing continuous.
        val_window.extend(resume_ckpt.get("val_window", []))
        cf_window.extend(resume_ckpt.get("cf_window", []))
        vtab_window.extend(resume_ckpt.get("vtab_window", []))

    def save_stable(epoch_done: int) -> None:
        """Write the resumable stable checkpoint (overwrites in place)."""
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch_done,
                "rng_state": _capture_rng(),
                "best_smoothed_acc": best_smoothed_acc,
                "best_knn_acc": best_knn_acc,
                "best_knn_step": best_knn_step,
                "best_cifar_smoothed": best_cifar_smoothed,
                "best_cifar_acc": best_cifar_acc,
                "best_cifar_step": best_cifar_step,
                "best_vtab_smoothed": best_vtab_smoothed,
                "best_vtab_acc": best_vtab_acc,
                "best_vtab_step": best_vtab_step,
                # Moving-average windows: restored on resume so smoothing stays continuous
                # and a thin post-resume window can't fluke-overwrite a good _best.
                "val_window": list(val_window),
                "cf_window": list(cf_window),
                "vtab_window": list(vtab_window),
                "knn_every_steps": knn_every_steps,
                "args": vars(args),
                "probe_dataset": "tiny_imagenet_valid",
                "wandb_run_id": wandb_run_id,
                "git_sha": git_sha,
                "wallclock_pretrain_s": prior_wallclock + (time.time() - t_run_start),
            },
            stable_path,
        )

    epoch_times: list[float] = []  # per-epoch wallclock (incl. in-epoch probes), for calibration
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        t0 = time.time()
        pbar = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{args.epochs} (split={args.split})",
            leave=False,
        )
        for view1, view2 in pbar:
            # Manual LR (pure function of global_step): warmup -> constant. Keeping the
            # LR step-addressable (no scheduler object) makes resume trivial.
            lr = args.lr * warmup_constant_factor(global_step, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            z1 = model(view1)
            z2 = model(view2)
            loss = loss_fn(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            global_step += 1

            loss_val = float(loss.detach())
            epoch_loss_sum += loss_val
            epoch_batches += 1
            wandb.log({"train/loss_step": loss_val, "train/lr": lr}, step=global_step)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if knn_every_steps > 0 and global_step % knn_every_steps == 0:
                knn_acc, eff_rank, cifar_acc = run_probe(
                    model, device, ti_db, ti_val, args.knn_k, cf_db, cf_val,
                )

                val_window.append(knn_acc)
                smoothed = sum(val_window) / len(val_window)
                if smoothed > best_smoothed_acc:
                    best_smoothed_acc = smoothed
                    best_knn_acc = knn_acc
                    best_knn_step = global_step
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),
                            "global_step": global_step,
                            "best_knn_acc": best_knn_acc,
                            "best_knn_step": best_knn_step,
                            "probe_dataset": "tiny_imagenet_valid",
                            "git_sha": git_sha,
                        },
                        best_path,
                    )

                # Target-aware twin selection on the CIFAR-train probe: same
                # smoothed-best rule, separate checkpoint. Captures the transfer
                # peak, which sits (much) earlier than the TI peak at small N.
                if cifar_acc is not None:
                    cf_window.append(cifar_acc)
                    cf_smoothed = sum(cf_window) / len(cf_window)
                    if cf_smoothed > best_cifar_smoothed:
                        best_cifar_smoothed = cf_smoothed
                        best_cifar_acc = cifar_acc
                        best_cifar_step = global_step
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "args": vars(args),
                                "global_step": global_step,
                                "best_knn_acc": best_cifar_acc,
                                "best_knn_step": best_cifar_step,
                                "probe_dataset": "cifar10_train_knn",
                                "git_sha": git_sha,
                            },
                            best_cifar_path,
                        )

                probe_log = {
                    "knn/acc_ti": knn_acc,
                    "knn/smoothed_acc_ti": smoothed,
                    "knn/effective_rank_ti": eff_rank,
                    "knn/best_acc": best_knn_acc,
                    "knn/best_step": best_knn_step,
                }
                if cifar_acc is not None:
                    probe_log["knn/acc_cifar"] = cifar_acc
                    probe_log["knn/best_acc_cifar"] = best_cifar_acc
                    probe_log["knn/best_step_cifar"] = best_cifar_step
                wandb.log(probe_log, step=global_step)
                append_probe_row(
                    probes_csv_path,
                    ["global_step", "epoch", "knn_acc_ti", "eff_rank_ti", "knn_acc_cifar"],
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "knn_acc_ti": f"{knn_acc:.6f}",
                        "eff_rank_ti": f"{eff_rank:.6f}",
                        "knn_acc_cifar": "" if cifar_acc is None else f"{cifar_acc:.6f}",
                    },
                )
                tqdm.write(
                    f"[step {global_step}] knn_ti={knn_acc:.4f} "
                    f"smoothed={smoothed:.4f} "
                    f"best={best_smoothed_acc:.4f}@{best_knn_step}"
                )

            # VTAB-1k transfer probe (own coarser cadence). Per-task kNN (train800
            # gallery, val200 query) averaged into one transfer signal; saves a
            # `_best_vtab` transfer oracle. Same kNN protocol as TI. Never drives _best.
            if vtab_every_steps > 0 and global_step % vtab_every_steps == 0:
                vtab_accs = {}
                for task, vdb, vq in vtab_loaders:
                    v_acc, _ = knn_probe(model, device, vdb, vq, args.knn_k)
                    vtab_accs[task] = v_acc
                model.train()  # knn_probe leaves the model in eval()
                vtab_mean = sum(vtab_accs.values()) / max(1, len(vtab_accs))
                vtab_window.append(vtab_mean)
                vtab_smoothed = sum(vtab_window) / len(vtab_window)
                if vtab_smoothed > best_vtab_smoothed:
                    best_vtab_smoothed = vtab_smoothed
                    best_vtab_acc = vtab_mean
                    best_vtab_step = global_step
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),
                            "global_step": global_step,
                            "best_knn_acc": best_vtab_acc,
                            "best_knn_step": best_vtab_step,
                            "probe_dataset": "vtab1k_val200_knn_mean",
                            "git_sha": git_sha,
                        },
                        best_vtab_path,
                    )
                vtab_log = {
                    "knn/acc_vtab_mean": vtab_mean,
                    "knn/smoothed_acc_vtab": vtab_smoothed,
                    "knn/best_acc_vtab": best_vtab_acc,
                    "knn/best_step_vtab": best_vtab_step,
                }
                for task, v in vtab_accs.items():
                    vtab_log[f"knn/acc_vtab_{task}"] = v
                wandb.log(vtab_log, step=global_step)
                vtab_csv_header = (
                    ["global_step", "epoch", "vtab_mean"]
                    + [f"acc_{task}" for task, _, _ in vtab_loaders]
                )
                vtab_csv_row = {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "vtab_mean": f"{vtab_mean:.6f}",
                }
                for task, v in vtab_accs.items():
                    vtab_csv_row[f"acc_{task}"] = f"{v:.6f}"
                append_probe_row(vtab_probes_csv_path, vtab_csv_header, vtab_csv_row)
                tqdm.write(
                    f"[step {global_step}] vtab_mean={vtab_mean:.4f} "
                    f"smoothed={vtab_smoothed:.4f} best={best_vtab_smoothed:.4f}@{best_vtab_step}"
                )

        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        wandb.log(
            {
                "train/loss_epoch": avg_loss,
                "train/epoch_time_s": epoch_time,
                "train/data_repetitions": global_step * args.batch_size / args.split,
                "epoch": epoch + 1,
            },
            step=global_step,
        )
        if (epoch + 1) % STABLE_SAVE_EVERY_EPOCHS == 0:
            save_stable(epoch + 1)  # crash insurance; epoch-aligned for clean resume

    save_stable(args.epochs)
    wallclock_s = prior_wallclock + (time.time() - t_run_start)
    wandb.log({"train/wallclock_total_s": wallclock_s}, step=global_step + 1)

    print(f"Saved resumable stable checkpoint to {stable_path}")
    print(
        f"Completed {args.epochs} epochs ({global_step} steps). "
        f"best-val kNN={best_knn_acc:.4f} at step {best_knn_step}."
    )
    print(
        f"Best-val checkpoint at {best_path} — run cooldown.py on it to produce the "
        f"annealed eval (_final) model; or resume with --resume_from {stable_path}."
    )
    if best_cifar_step > 0:
        print(
            f"Target-aware checkpoint at {best_cifar_path} "
            f"(CIFAR-train kNN={best_cifar_acc:.4f} at step {best_cifar_step}) — "
            f"cooldown.py accepts it the same way."
        )
    if best_vtab_step > 0:
        print(
            f"Transfer-oracle checkpoint at {best_vtab_path} "
            f"(VTAB-1k val kNN mean={best_vtab_acc:.4f} at step {best_vtab_step}) — "
            f"cooldown.py accepts it the same way."
        )
    print(
        f"Wallclock (incl. any resumed time): {wallclock_s:.0f}s ({wallclock_s / 60:.1f} min)"
    )
    # Clean per-epoch timing for the campaign calibration: drop epoch 0 (one-time cache /
    # loader / cudnn warmup) so the mean reflects steady-state cost (training + amortized
    # in-epoch probes). Parsed by calibrate_campaign.sh; excludes process startup entirely.
    steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    if steady:
        mean_epoch_s = sum(steady) / len(steady)
        print(
            f"MEAN_EPOCH_S split={args.split} epochs_measured={len(steady)} "
            f"mean_epoch_s={mean_epoch_s:.4f}"
        )
    wandb.finish()


if __name__ == "__main__":
    main()
