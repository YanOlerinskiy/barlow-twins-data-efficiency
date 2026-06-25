"""Anneal a stable-phase checkpoint into an eval-ready model (the WSD cooldown).

`pretrain.py` runs the constant-LR stable phase and saves a `_best` (peak) and a
`_stable` (resumable) checkpoint but does NOT anneal. This script performs the
terminal cooldown on demand: it loads a checkpoint (the `_best` peak by default),
continues Barlow Twins training for a short phase while decaying the LR to ~0 with
the (1 - sqrt(t)) shape (Hägele et al. 2024, arXiv:2405.18392), and writes the
annealed `_final` checkpoint that `leo_protocol_eval.py` probes.

Cooldown length = `cooldown_frac * best_step` (clamped to `cooldown_min_steps`),
i.e. ~20% of the training that produced the peak — sized to the rewind point, not
the full budget, so an early peak does not get an over-long cooldown.

All the heavy lifting (model, loss, loader, probe, schedule shape) is reused from
`pretrain.py`; this file only orchestrates the cooldown.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from lightly.loss import BarlowTwinsLoss
from torch.optim import AdamW
from tqdm import tqdm

import wandb

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO_ROOT)  # so `python barlow_twins_experiments/cooldown.py` works from the repo root

from barlow_twins_experiments.cifar10_features import get_knn_db_val_loaders  # noqa: E402
from barlow_twins_experiments.pretrain import (  # noqa: E402
    _infinite_batches,
    cooldown_factor,
    run_probe,
    set_seed,
)
from barlow_twins_experiments.tiny_imagenet_features import (  # noqa: E402
    get_tiny_imagenet_knn_loaders,
)
from barlow_twins_experiments.two_view_dataset import get_two_view_loader  # noqa: E402
from models.barlow_twins import BarlowTwinsViT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to the stable-phase checkpoint to anneal (normally the "
                        "`_best` peak; `_stable` also works). Writes `_final` alongside it.")
    p.add_argument("--cooldown_frac", type=float, default=0.2,
                   help="Cooldown length as a fraction of the checkpoint's step "
                        "(the training that produced the peak). ~20%% per Hägele 2024.")
    p.add_argument("--cooldown_min_steps", type=int, default=500,
                   help="Lower bound on the cooldown length, in steps.")
    p.add_argument("--cooldown_shape", type=str, default="sqrt",
                   choices=["sqrt", "cosine", "linear"],
                   help="LR decay shape; 'sqrt' = (1 - sqrt(t)) per Hägele 2024.")
    p.add_argument("--knn_db_size", type=int, default=5000)
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--knn_val_size", type=int, default=5000)
    p.add_argument("--diagnostic_cifar_probe", action="store_true",
                   help="Also log the CIFAR-10 kNN diagnostic during/after cooldown.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb_project", type=str, default="barlow-twins-data-efficiency")
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def _final_path_for(ckpt_path: str) -> str:
    # Qualified bests keep their qualifier (-> `_final_cifar.pt` / `_final_vtab.pt`), so
    # the in-domain, target-aware, and transfer-oracle finals coexist in one directory.
    for q in ("_best_cifar.pt", "_best_vtab.pt"):
        if ckpt_path.endswith(q):
            return ckpt_path[: -len(q)] + q.replace("_best_", "_final_")
    for suffix in ("_best.pt", "_stable.pt"):
        if ckpt_path.endswith(suffix):
            return ckpt_path[: -len(suffix)] + "_final.pt"
    return ckpt_path[:-3] + "_final.pt" if ckpt_path.endswith(".pt") else ckpt_path + "_final.pt"


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck = ckpt["args"]  # the training args saved in the checkpoint (a dict)
    seed = ck.get("seed", 42)
    set_seed(seed)  # also enables the same TF32/cudnn config as pretraining
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_step = ckpt["global_step"]
    cooldown_steps = max(int(args.cooldown_frac * best_step), args.cooldown_min_steps)
    base_lr = ck["lr"]
    grad_clip = ck.get("grad_clip", 1.0)

    loader = get_two_view_loader(
        split=ck["split"],
        batch_size=ck["batch_size"],
        num_workers=args.num_workers,
        # .get with the publication defaults so cooling a pre-augmentation-arg checkpoint
        # (older runs that didn't save these) doesn't KeyError; current runs always save them.
        min_scale=ck.get("min_scale", 0.25),
        cj_strength=ck.get("cj_strength", 1.0),
        # Cool on the SAME data subset the run trained on; without this a seed-43/44
        # checkpoint would silently anneal on the seed-42 subset (default).
        data_seed=ck.get("data_seed", ck.get("seed", 42)),
    )

    proj_dim = ck.get("projector_dim", 1024)
    model = BarlowTwinsViT(projector_hidden=proj_dim, projector_out=proj_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    loss_fn = BarlowTwinsLoss(lambda_param=ck["lambda_param"])
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=ck["weight_decay"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = base_lr  # cooldown starts from the constant-phase LR

    ti_db, ti_val = get_tiny_imagenet_knn_loaders(
        db_size=args.knn_db_size,
        val_size=args.knn_val_size,
        num_workers=min(2, args.num_workers),
    )
    cf_db = cf_val = None
    if args.diagnostic_cifar_probe:
        cf_db, cf_val = get_knn_db_val_loaders(
            db_size=5000, val_size=10000, stratified=True, num_workers=2,
        )
    probe_every = max(1, cooldown_steps // 10)  # ~10 probes across the cooldown

    run_name = f"bt_{ck.get('run_tag', 'run')}_seed{seed}_split{ck['split']}_cooldown"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={**ck, **vars(args), "best_step": best_step, "cooldown_steps": cooldown_steps},
        mode=args.wandb_mode,
    )

    print(
        f"[cooldown] {args.cooldown_shape} anneal of {args.ckpt} @ step {best_step}, "
        f"over {cooldown_steps} steps (= {args.cooldown_frac}*best_step, "
        f">= {args.cooldown_min_steps}), base lr={base_lr}, "
        f"data_seed={ck.get('data_seed', ck.get('seed', 42))} split={ck['split']}"
    )

    t0 = time.time()
    model.train()
    batches = _infinite_batches(loader)
    global_step = best_step
    for t in tqdm(range(cooldown_steps), desc=f"cooldown (split={ck['split']})", leave=False):
        lr = base_lr * cooldown_factor(t, cooldown_steps, args.cooldown_shape)
        for group in optimizer.param_groups:
            group["lr"] = lr

        view1, view2 = next(batches)
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        z1 = model(view1)
        z2 = model(view2)
        loss = loss_fn(z1, z2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        global_step += 1

        wandb.log(
            {"cooldown/loss_step": float(loss.detach()), "cooldown/lr": lr},
            step=global_step,
        )
        if (t + 1) % probe_every == 0:
            cd_acc, cd_rank, cd_cifar = run_probe(
                model, device, ti_db, ti_val, args.knn_k, cf_db, cf_val,
            )
            cd_log = {"cooldown/knn_acc_ti": cd_acc, "cooldown/effective_rank_ti": cd_rank}
            if cd_cifar is not None:
                cd_log["cooldown/knn_acc_cifar"] = cd_cifar
            wandb.log(cd_log, step=global_step)
            tqdm.write(f"[cooldown step {global_step}] knn_ti={cd_acc:.4f} lr={lr:.2e}")

    final_acc, final_rank, final_cifar = run_probe(
        model, device, ti_db, ti_val, args.knn_k, cf_db, cf_val,
    )
    final_log = {"knn/final_acc_ti": final_acc, "knn/final_effective_rank_ti": final_rank}
    if final_cifar is not None:
        final_log["knn/final_acc_cifar"] = final_cifar
    wandb.log(final_log, step=global_step + 1)

    cooldown_wallclock = time.time() - t0
    # Carry the stable-phase wallclock (saved only in the sibling `_stable`) so the
    # eval CSV records total pretraining time; fall back to cooldown-only if absent.
    pretrain_wc = 0.0
    stable_sibling = (
        args.ckpt.replace("_best_cifar.pt", "_stable.pt")
        .replace("_best_vtab.pt", "_stable.pt")
        .replace("_best.pt", "_stable.pt")
    )
    if stable_sibling != args.ckpt and os.path.exists(stable_sibling):
        try:
            pretrain_wc = torch.load(
                stable_sibling, map_location="cpu", weights_only=False
            ).get("wallclock_pretrain_s", 0.0) or 0.0
        except (OSError, RuntimeError, KeyError):
            pretrain_wc = 0.0

    final_path = _final_path_for(args.ckpt)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": ck,
            "global_step": global_step,
            "best_knn_acc": ckpt.get("best_knn_acc"),
            "best_knn_step": ckpt.get("best_knn_step"),
            "final_knn_acc": final_acc,
            "cooldown_steps": cooldown_steps,
            "cooldown_shape": args.cooldown_shape,
            "wallclock_pretrain_s": pretrain_wc + cooldown_wallclock,
            "probe_dataset": "tiny_imagenet_valid",
            "source_ckpt": args.ckpt,
        },
        final_path,
    )
    best_acc = ckpt.get("best_knn_acc")
    best_str = f"{best_acc:.4f}" if isinstance(best_acc, (int, float)) else "n/a"
    print(
        f"Saved annealed eval checkpoint to {final_path}. "
        f"final kNN={final_acc:.4f} (pre-cooldown best-val={best_str}); "
        f"cooldown {cooldown_wallclock:.0f}s."
    )
    wandb.finish()


if __name__ == "__main__":
    main()
