"""Evaluate Barlow Twins checkpoints using Leo's group-shared linear-probe
protocol (LINEAR_PROBE_PROTOCOL.md / dino_eval_linear_probe.ipynb).

Protocol highlights:
- Classifier trained with SGD + cosine schedule for 100 epochs.
- LR = 0.1 * batch_size / 256 (linear scaling rule).
- RandomHorizontalFlip on train transform (matches Leo's actual run, not strict B.2).
- Reports BEST test top-1 across 100 epochs.
- Probes the `final` (annealed) checkpoint by default; `--ckpt best` is for
  analysis only.
- Uses seed=0 (Leo's eval seed) regardless of pretraining seed.
- Writes rows to the shared `evaluation/results.csv` with the group schema.

Run from the repo root:
    python barlow_twins_experiments/leo_protocol_eval.py [--run_tag <tag>]
"""
import argparse
import csv
import os
import pathlib
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO_ROOT)
from models.barlow_twins import BarlowTwinsViT  # noqa: E402

from barlow_twins_experiments.pretrain import default_run_tag  # noqa: E402

# ---- Protocol constants (from LINEAR_PROBE_PROTOCOL.md) ---------------
EVAL_EPOCHS = 100
EVAL_BATCH = 512                    # Leo's A100 setting; 5060 Ti handles it fine
EVAL_LR_BASE = 0.1
EVAL_LR = EVAL_LR_BASE * EVAL_BATCH / 256   # = 0.2 at batch=512
EVAL_MOMENTUM = 0.9
EVAL_WD = 0.0
RESCALE = 64
SEED = 0                            # eval-time seed (independent of pretraining seed)
NUM_WORKERS = 4
METHOD_NAME = "barlow_twins"

# ---- Paths ------------------------------------------------------------
CHECKPOINT_DIR = pathlib.Path(_REPO_ROOT) / "checkpoints"
EVAL_DIR = pathlib.Path(_REPO_ROOT) / "evaluation"
RESULTS_CSV = EVAL_DIR / "results.csv"
CIFAR10_ROOT = os.environ.get("CIFAR10_ROOT", os.path.join(_REPO_ROOT, "cifar10_data"))

# ---- Conditions to evaluate -------------------------------------------
# None = random-init baseline; integers = pretraining sample count (matches
# what's in checkpoints/, NOT powers-of-2 like Leo's protocol doc — we keep
# our actual splits and let the meeting reconcile labels.)
SPLITS_TO_EVAL = [None, 1000, 2000, 4000, 8000, 16000, 32000]

# ImageNet normalization (matches pretraining)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(RESCALE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(RESCALE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# ---- CSV schema (from LINEAR_PROBE_PROTOCOL.md) -----------------------
CSV_HEADER = [
    "method", "fraction", "seed", "pretrain_seed", "epochs",
    "eval_dataset", "eval_method",
    "accuracy", "checkpoint_path",
    "wallclock_pretrain_s", "wallclock_eval_s",
    "git_sha", "timestamp",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders() -> tuple[DataLoader, DataLoader]:
    train_set = CIFAR10(
        root=CIFAR10_ROOT, train=True, download=True, transform=TRAIN_TRANSFORM
    )
    test_set = CIFAR10(
        root=CIFAR10_ROOT, train=False, download=True, transform=TEST_TRANSFORM
    )
    train_loader = DataLoader(
        train_set, batch_size=EVAL_BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=EVAL_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, test_loader


def build_backbone(
    checkpoint_path: pathlib.Path | None, device: torch.device
) -> tuple[nn.Module, dict | None]:
    """Return (frozen backbone, checkpoint metadata dict or None)."""
    # Load first (if any) so the model is rebuilt at the checkpoint's projector
    # width — otherwise load_state_dict shape-mismatches on non-1024 projectors.
    # The probe only reads the 192-d backbone embedding, but the full state dict
    # (incl. projector) must still load cleanly.
    ckpt = None
    proj_dim = 1024
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        proj_dim = ckpt.get("args", {}).get("projector_dim", 1024)
    model = BarlowTwinsViT(projector_hidden=proj_dim, projector_out=proj_dim).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint from {checkpoint_path.name} (projector_dim={proj_dim})")
    else:
        print("  Random-init baseline (no checkpoint loaded)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


@torch.no_grad()
def evaluate(model: nn.Module, classifier: nn.Linear, loader: DataLoader,
             device: torch.device) -> float:
    classifier.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feat = model.embed(imgs)
        logits = classifier(feat)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def _rotate_if_stale_schema(path: pathlib.Path, header: list[str]) -> None:
    """If the CSV exists with a different header (e.g. the pre-`pretrain_seed` schema),
    rename it to a `.legacy-N.csv` backup so a fresh file is written with the new header.
    Non-destructive (rename, not delete) — prevents silently misaligned appended rows."""
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


def append_csv_row(
    method: str, fraction: int, accuracy_pct: float, checkpoint_path: str,
    wallclock_eval_s: float, pretrain_seed: int, wallclock_pretrain_s: str = "",
) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_if_stale_schema(RESULTS_CSV, CSV_HEADER)
    file_existed = RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0
    row = {
        "method": method,
        "fraction": fraction,
        "seed": SEED,
        "pretrain_seed": pretrain_seed,
        "epochs": EVAL_EPOCHS,
        "eval_dataset": "cifar10",
        "eval_method": "linear_probe",
        "accuracy": f"{accuracy_pct:.4f}",
        "checkpoint_path": checkpoint_path,
        "wallclock_pretrain_s": wallclock_pretrain_s,
        "wallclock_eval_s": f"{wallclock_eval_s:.1f}",
        "git_sha": default_run_tag(),  # record the eval-code commit for provenance
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_existed:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Appended row -> {RESULTS_CSV.name}: "
          f"method={method} fraction={fraction} accuracy={accuracy_pct:.4f}")


def run_one(split: int | None, train_loader: DataLoader, test_loader: DataLoader,
            device: torch.device, ckpt_dir: pathlib.Path, ckpt_kind: str,
            pretrain_seed: int) -> tuple[float, float]:
    set_seed(SEED)

    if split is None:
        method_name = "barlow_twins_random_init"
        fraction = 0
        ckpt_path = None
        ckpt_relpath = ""
    else:
        # Tag alternatively-selected checkpoints in the method column so the
        # selection rules stay distinguishable in the shared CSV.
        sel_suffix = "_cifarsel" if "cifar" in ckpt_kind else "_vtabsel" if "vtab" in ckpt_kind else ""
        method_name = METHOD_NAME + sel_suffix
        fraction = split
        ckpt_path = ckpt_dir / f"bt_split{split}_seed{pretrain_seed}_{ckpt_kind}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found at {ckpt_path}")
            return 0.0, 0.0
        ckpt_relpath = str(ckpt_path.relative_to(pathlib.Path(_REPO_ROOT).parent))

    print()
    print("=" * 60)
    print(f"  Evaluating: method={method_name}, fraction={fraction}")
    print("=" * 60)

    backbone, ckpt_meta = build_backbone(ckpt_path, device)
    pretrain_s = (ckpt_meta or {}).get("wallclock_pretrain_s")
    wallclock_pretrain = f"{pretrain_s:.1f}" if pretrain_s else ""
    classifier = nn.Linear(192, 10).to(device)
    optim = torch.optim.SGD(
        classifier.parameters(),
        lr=EVAL_LR, momentum=EVAL_MOMENTUM, weight_decay=EVAL_WD,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EVAL_EPOCHS, eta_min=0
    )

    print(f"  SGD lr={EVAL_LR:.4f} momentum={EVAL_MOMENTUM} wd={EVAL_WD}")
    print(f"  Cosine schedule over {EVAL_EPOCHS} epochs -> 0")

    t0 = time.time()
    best_test_acc = 0.0

    for epoch in range(EVAL_EPOCHS):
        classifier.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                feat = backbone.embed(imgs)

            logits = classifier(feat)
            loss = F.cross_entropy(logits, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(-1) == labels).sum().item()
            running_total += labels.size(0)

        sched.step()
        train_acc = 100.0 * running_correct / running_total
        train_loss = running_loss / running_total
        test_acc = evaluate(backbone, classifier, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EVAL_EPOCHS - 1:
            print(f"  epoch {epoch+1:3d}/{EVAL_EPOCHS}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                  f"test_acc={test_acc:.2f}% best={best_test_acc:.2f}%")

    wallclock = time.time() - t0
    print(f"  Best test top-1: {best_test_acc:.2f}%   wallclock: {wallclock:.1f}s")

    append_csv_row(
        method=method_name, fraction=fraction,
        accuracy_pct=best_test_acc,
        checkpoint_path=ckpt_relpath,
        wallclock_eval_s=wallclock,
        # Random-init is independent of the pretrain seed (eval seed is fixed), so tag it -1
        # instead of the passed seed — otherwise each per-seed eval re-emits an identical,
        # differently-labeled random row, creating spurious per-seed baseline groups.
        pretrain_seed=(-1 if split is None else pretrain_seed),
        wallclock_pretrain_s=wallclock_pretrain,
    )

    del backbone, classifier, optim, sched
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_test_acc, wallclock


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated split sizes to evaluate (e.g. '64000,100000'). "
             "If omitted, the default 7-condition sweep runs.",
    )
    p.add_argument(
        "--no-random",
        action="store_true",
        help="Skip the random-init baseline (useful when it's already in the CSV).",
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Run index of the pretraining sweep (checkpoints/<tag>/...). "
             "Defaults to the current short git SHA ('-dirty' if uncommitted).",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="final",
        choices=["final", "best", "final_cifar", "best_cifar", "final_vtab", "best_vtab"],
        help="Which pretraining checkpoint to probe. Headline numbers use "
             "'final' — the annealed model produced by cooldown.py from the best-val "
             "checkpoint; 'best' is the pre-cooldown best-val snapshot (high LR, "
             "for analysis / oracle-stopping only). '*_cifar' are the target-aware "
             "twins selected on the CIFAR-train kNN probe; '*_vtab' are the transfer "
             "oracles selected on the VTAB-1k val kNN mean ('final_*' = annealed via "
             "cooldown.py from the matching '_best_*'). Run cooldown.py first to create "
             "the '_final*' variants.",
    )
    p.add_argument(
        "--pretrain_seed",
        type=int,
        default=42,
        help="Seed of the pretraining run whose checkpoints to load "
             "(eval itself always uses seed 0 per protocol).",
    )
    return p.parse_args()


def main() -> None:
    cli = parse_cli()

    if cli.splits is not None:
        splits_to_eval: list[int | None] = [int(s) for s in cli.splits.split(",") if s.strip()]
        if not cli.no_random:
            splits_to_eval = [None] + splits_to_eval
    else:
        splits_to_eval = [None] + [s for s in SPLITS_TO_EVAL if s is not None]
        if cli.no_random:
            splits_to_eval = [s for s in splits_to_eval if s is not None]

    run_tag = cli.run_tag if cli.run_tag is not None else default_run_tag()
    ckpt_dir = CHECKPOINT_DIR / run_tag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Effective LR: {EVAL_LR:.4f} (= {EVAL_LR_BASE} * {EVAL_BATCH} / 256)")
    print(f"Checkpoints: {ckpt_dir} (kind={cli.ckpt}, pretrain_seed={cli.pretrain_seed})")
    print(f"Results will be appended to: {RESULTS_CSV}")
    print(f"Splits to evaluate: {splits_to_eval}")

    train_loader, test_loader = get_loaders()
    print(f"Train batches: {len(train_loader)} "
          f"({len(train_loader.dataset)} samples, batch={EVAL_BATCH}, drop_last=True)")
    print(f"Test batches:  {len(test_loader)} ({len(test_loader.dataset)} samples)")

    sweep_start = time.time()
    summary = []
    for split in splits_to_eval:
        acc, wc = run_one(split, train_loader, test_loader, device,
                          ckpt_dir, cli.ckpt, cli.pretrain_seed)
        summary.append((split, acc, wc))

    total = time.time() - sweep_start
    print()
    print("=" * 60)
    print(f"  SWEEP COMPLETE  --  {total:.1f}s  ({total/60:.1f} min)")
    print("=" * 60)
    print(f"{'condition':>20} {'accuracy':>10} {'wallclock':>12}")
    print("-" * 44)
    for split, acc, wc in summary:
        name = "random_init" if split is None else f"bt-{split}"
        print(f"{name:>20} {acc:>9.2f}% {wc:>10.1f}s")


if __name__ == "__main__":
    main()
