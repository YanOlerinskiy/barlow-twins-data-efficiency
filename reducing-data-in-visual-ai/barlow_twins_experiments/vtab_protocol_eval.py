"""Evaluate Barlow Twins checkpoints on VTAB-1k linear probing, across splits + a
random-init baseline -> a VTAB-1k transfer data-efficiency curve.

Mirrors `leo_protocol_eval.py` (CIFAR-10) but runs the 19-task VTAB-1k suite using
the same linear-probe recipe as the group's VTAB notebook (`yan-vtab-finetuning.ipynb`,
itself Makar's protocol), so the numbers are comparable across the group's methods:
per task, freeze the backbone, extract pre-logit (CLS, 192-d) features once for
`train800val200` (1000 imgs) and `test`, train a fresh linear head (SGD + cosine,
base LR 0.1 * batch/256, momentum 0.9, weight-decay 0, 90 epochs), and report the
BEST test top-1 across epochs. Test split is the canonical VTAB-1k eval set.

Run from the repo root, e.g.:
    python barlow_twins_experiments/vtab_protocol_eval.py \
        --run_tag e97d41c-curve --splits 1000,2000,4000,8000,16000
"""
import argparse
import csv
import os
import pathlib
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO_ROOT)

# Reuse the backbone loader (handles random-init, projector width, freeze, eval) and
# the eval-seed setter from the CIFAR protocol; reuse the VTAB dataset + task list.
from barlow_twins_experiments.leo_protocol_eval import build_backbone, set_seed  # noqa: E402
from barlow_twins_experiments.pretrain import default_run_tag  # noqa: E402
from barlow_twins_experiments.vtab_features import (  # noqa: E402
    VTAB_DATA_ROOT,
    VtabImageDataset,
    VTAB_EVAL_TRANSFORM,
    resolve_vtab_tasks,
)

# ---- Protocol constants (match the VTAB notebook / Makar) -------------
EVAL_EPOCHS = 90
EVAL_BATCH = 256
EVAL_LR_BASE = 0.1
EVAL_LR = EVAL_LR_BASE * EVAL_BATCH / 256   # = 0.1 at batch 256
EVAL_MOMENTUM = 0.9
EVAL_WD = 0.0
EVAL_SEED = 1205                            # the notebook's eval seed (Makar-comparable)
EXTRACT_BATCH = 512                         # feature extraction only (no effect on result)
NUM_WORKERS = 4
EMBED_DIM = 192
METHOD_NAME = "barlow_twins"
TRAIN_SPLIT = "train800val200"
TEST_SPLIT = "test"

CHECKPOINT_DIR = pathlib.Path(_REPO_ROOT) / "checkpoints"
EVAL_DIR = pathlib.Path(_REPO_ROOT) / "evaluation"
RESULTS_CSV = EVAL_DIR / "vtab_results.csv"

SPLITS_TO_EVAL = [None, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]

CSV_HEADER = [
    "method", "fraction", "seed", "pretrain_seed", "epochs",
    "eval_dataset", "eval_method",
    "accuracy", "checkpoint_path",
    "wallclock_eval_s", "git_sha", "timestamp",
]


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module, device: torch.device):
    feats, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        feats.append(model.embed(images).cpu())
        labels.append(target)
    return torch.cat(feats), torch.cat(labels)


def _iterate(feats, labels, batch_size, shuffle):
    n = feats.shape[0]
    idx = torch.randperm(n, device=feats.device) if shuffle else torch.arange(n, device=feats.device)
    for start in range(0, n, batch_size):  # keep all samples (1000-image train split)
        sel = idx[start:start + batch_size]
        yield feats[sel], labels[sel]


def _cosine_lr(optimizer, init_lr, epoch, total_epochs):
    import math
    cur = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    for g in optimizer.param_groups:
        g["lr"] = cur


@torch.no_grad()
def _test_top1(classifier, feats, labels, batch_size):
    classifier.eval()
    correct = total = 0
    for x, y in _iterate(feats, labels, batch_size, shuffle=False):
        preds = classifier(x).argmax(-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


def train_linear_probe(train_feats, train_labels, test_feats, test_labels,
                       num_classes: int, device: torch.device) -> float:
    """Train a fresh linear head on cached features; return BEST test top-1 across epochs."""
    classifier = nn.Linear(train_feats.shape[1], num_classes).to(device)
    classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.bias.data.zero_()
    optim = torch.optim.SGD(classifier.parameters(), EVAL_LR,
                            momentum=EVAL_MOMENTUM, weight_decay=EVAL_WD)
    criterion = nn.CrossEntropyLoss().to(device)
    best = 0.0
    for epoch in range(EVAL_EPOCHS):
        _cosine_lr(optim, EVAL_LR, epoch, EVAL_EPOCHS)
        classifier.train()
        for x, y in _iterate(train_feats, train_labels, EVAL_BATCH, shuffle=True):
            loss = criterion(classifier(x), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        best = max(best, _test_top1(classifier, test_feats, test_labels, EVAL_BATCH))
    return best


def _rotate_if_stale_schema(path: pathlib.Path, header: list[str]) -> None:
    """If the CSV exists with a different header (e.g. the pre-`pretrain_seed` schema),
    archive it to a `.legacy-N.csv` backup so a fresh file uses the new header. Rename,
    not delete — avoids silently misaligned appended rows."""
    if not (path.exists() and path.stat().st_size > 0):
        return
    with open(path, encoding="utf-8") as f:
        first = f.readline().rstrip("\n\r")
    if first == ",".join(header):
        return
    n = 1
    while True:
        backup = path.with_name(f"{path.stem}.legacy-{n}.csv")
        if not backup.exists():
            break
        n += 1
    path.rename(backup)
    print(f"  [schema] {path.name} had an old header; archived -> {backup.name}, starting fresh.")


def append_csv_row(method, fraction, task, accuracy_pct, checkpoint_path,
                   wallclock_eval_s, git_sha, pretrain_seed):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_if_stale_schema(RESULTS_CSV, CSV_HEADER)
    existed = RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0
    row = {
        "method": method, "fraction": fraction, "seed": EVAL_SEED,
        "pretrain_seed": pretrain_seed, "epochs": EVAL_EPOCHS,
        "eval_dataset": task, "eval_method": "linear_probe",
        "accuracy": f"{accuracy_pct:.4f}", "checkpoint_path": checkpoint_path,
        "wallclock_eval_s": f"{wallclock_eval_s:.1f}", "git_sha": git_sha,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not existed:
            w.writeheader()
        w.writerow(row)


def _loaders_for(task: str, data_root: str):
    tr = VtabImageDataset(data_root, task, TRAIN_SPLIT, VTAB_EVAL_TRANSFORM)
    te = VtabImageDataset(data_root, task, TEST_SPLIT, VTAB_EVAL_TRANSFORM)
    num_classes = max(tr.num_classes, te.num_classes)
    pin = torch.cuda.is_available()
    trl = DataLoader(tr, batch_size=EXTRACT_BATCH, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=pin)
    tel = DataLoader(te, batch_size=EXTRACT_BATCH, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=pin)
    return trl, tel, num_classes, len(tr), len(te)


def run_one_split(split, device, ckpt_dir: pathlib.Path, ckpt_kind: str,
                  pretrain_seed: int, tasks: list[str], data_root: str, git_sha: str):
    if split is None:
        method = "barlow_twins_random_init"
        fraction = 0
        ckpt_path = None
        ckpt_relpath = ""
    else:
        sel = "_vtabsel" if "vtab" in ckpt_kind else "_cifarsel" if "cifar" in ckpt_kind else ""
        method = METHOD_NAME + sel
        fraction = split
        ckpt_path = ckpt_dir / f"bt_split{split}_seed{pretrain_seed}_{ckpt_kind}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP split={split}: no checkpoint at {ckpt_path}")
            return None
        ckpt_relpath = str(ckpt_path.relative_to(pathlib.Path(_REPO_ROOT).parent))

    # Random-init is independent of the pretrain seed (eval seed is fixed), so tag it -1
    # instead of the passed seed — otherwise each per-seed eval re-emits an identical,
    # differently-labeled random row, creating spurious per-seed baseline groups.
    row_seed = -1 if split is None else pretrain_seed

    print("\n" + "=" * 64)
    print(f"  Split={split if split is not None else 'random_init'}  (method={method})")
    print("=" * 64)
    set_seed(EVAL_SEED)
    backbone, _ = build_backbone(ckpt_path, device)

    per_task = {}
    t0 = time.time()
    for task in tasks:
        trl, tel, num_classes, n_tr, n_te = _loaders_for(task, data_root)
        tr_f, tr_y = extract_features(trl, backbone, device)
        te_f, te_y = extract_features(tel, backbone, device)
        tr_f, tr_y = tr_f.to(device), tr_y.to(device)
        te_f, te_y = te_f.to(device), te_y.to(device)
        set_seed(EVAL_SEED)  # deterministic classifier init/order per task
        acc = train_linear_probe(tr_f, tr_y, te_f, te_y, num_classes, device)
        per_task[task] = acc
        print(f"  {task:<22} {num_classes:>4}cls  {n_tr}->{n_te:<6}  best@1={acc:5.2f}%")
        append_csv_row(method, fraction, task, acc, ckpt_relpath, time.time() - t0, git_sha,
                       row_seed)
        del tr_f, tr_y, te_f, te_y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_acc = sum(per_task.values()) / max(1, len(per_task))
    append_csv_row(method, fraction, "vtab1k_mean", mean_acc, ckpt_relpath,
                   time.time() - t0, git_sha, row_seed)
    print(f"  -> VTAB-1k mean ({len(per_task)} tasks) = {mean_acc:.2f}%   "
          f"({(time.time() - t0) / 60:.1f} min)")
    return mean_acc


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=str, default=None,
                   help="Comma-separated split sizes (e.g. '1000,2000'). Omit for the full sweep.")
    p.add_argument("--no-random", action="store_true",
                   help="Skip the random-init baseline (already in the CSV).")
    p.add_argument("--run_tag", type=str, default=None,
                   help="Checkpoint dir checkpoints/<tag>/ (default: current short git SHA).")
    p.add_argument("--ckpt", type=str, default="final",
                   choices=["final", "best", "final_cifar", "best_cifar", "final_vtab", "best_vtab"],
                   help="Which checkpoint kind to probe. Default 'final' = the cooled best.")
    p.add_argument("--pretrain_seed", type=int, default=42,
                   help="Pretraining seed whose checkpoints to load (eval seed is fixed = 1205).")
    p.add_argument("--tasks", type=str, default="all",
                   help="Comma-separated VTAB task subset, or 'all' (default).")
    p.add_argument("--vtab_data_root", type=str, default=None,
                   help="VTAB-1k bundle root (default: <repo>/vtab-1k).")
    p.add_argument("--results_csv", type=str, default=None,
                   help="Override the output CSV path (default: evaluation/vtab_results.csv). Use a "
                        "per-job path when two GPU jobs share one clone so their appends don't race.")
    return p.parse_args()


def main() -> None:
    global RESULTS_CSV
    cli = parse_cli()
    if cli.results_csv is not None:
        RESULTS_CSV = pathlib.Path(cli.results_csv)
    if cli.splits is not None:
        splits: list = [int(s) for s in cli.splits.split(",") if s.strip()]
        if not cli.no_random:
            splits = [None] + splits
    else:
        splits = [s for s in SPLITS_TO_EVAL if not (cli.no_random and s is None)]

    tasks = resolve_vtab_tasks(cli.tasks)
    data_root = cli.vtab_data_root or VTAB_DATA_ROOT
    run_tag = cli.run_tag if cli.run_tag is not None else default_run_tag()
    ckpt_dir = CHECKPOINT_DIR / run_tag
    git_sha = default_run_tag()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Checkpoints: {ckpt_dir} (kind={cli.ckpt}, pretrain_seed={cli.pretrain_seed})")
    print(f"VTAB root: {data_root} | tasks: {len(tasks)} | eval seed: {EVAL_SEED}")
    print(f"Linear probe: SGD lr={EVAL_LR:.3f} mom={EVAL_MOMENTUM} wd={EVAL_WD} "
          f"epochs={EVAL_EPOCHS} batch={EVAL_BATCH} (train800val200 -> test, best top-1)")
    print(f"Results -> {RESULTS_CSV}")
    print(f"Splits: {splits}")

    summary = []
    for split in splits:
        m = run_one_split(split, device, ckpt_dir, cli.ckpt, cli.pretrain_seed,
                          tasks, data_root, git_sha)
        summary.append((split, m))

    print("\n" + "=" * 44)
    print(f"  VTAB-1k SWEEP COMPLETE (run_tag={run_tag})")
    print("=" * 44)
    print(f"{'condition':>14} {'vtab mean@1':>12}")
    for split, m in summary:
        name = "random_init" if split is None else f"bt-{split}"
        print(f"{name:>14} {('%.2f%%' % m) if m is not None else 'skipped':>12}")


if __name__ == "__main__":
    main()
