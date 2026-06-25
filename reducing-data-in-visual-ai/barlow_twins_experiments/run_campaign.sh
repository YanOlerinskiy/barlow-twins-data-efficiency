#!/usr/bin/env bash
# Per-GPU campaign runner for one seed over an explicit set of splits:
#   pretrain each split -> 1000 epochs (TI + VTAB in-training probes; no CIFAR)
#   -> cooldown _best (TI-selected) and _best_vtab (VTAB-selected)
#   -> VTAB-1k linear eval of final + final_vtab.
#
# Pin to a GPU with CUDA_VISIBLE_DEVICES. Two jobs can share ONE clone safely: the run tag and
# the eval CSV are namespaced by GPU index, so checkpoints, probe traces, and results never collide.
#
#   export WANDB_API_KEY=...    # optional -> live W&B dashboards (else runs with W&B disabled)
#   # 2-GPU box, one seed, balanced across both GPUs:
#   CUDA_VISIBLE_DEVICES=0 bash barlow_twins_experiments/run_campaign.sh 42 1000,2000,4000,8000,32000
#   CUDA_VISIBLE_DEVICES=1 bash barlow_twins_experiments/run_campaign.sh 42 16000,64000,100000
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

SEED="${1:-}"; SPLITS_CSV="${2:-}"
[ -n "$SEED" ] && [ -n "$SPLITS_CSV" ] || {
  echo "usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <seed> <split1,split2,...>"; exit 2; }

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
TAG="${SHA}-s${SEED}-gpu${GPU}"            # SHA-prefixed -> pretrain's resolve_run_tag keeps it as-is
RESULTS="evaluation/vtab_s${SEED}_gpu${GPU}.csv"
WANDB="--wandb_mode disabled"; [ -n "${WANDB_API_KEY:-}" ] && WANDB="--wandb_mode online"
SPLITS="${SPLITS_CSV//,/ }"

echo "[campaign] seed=$SEED gpu=$GPU tag=$TAG splits=[$SPLITS] wandb=${WANDB#--wandb_mode } -> $RESULTS"

# 1) PRETRAIN each assigned split to 1000 epochs (TI + VTAB probes; CIFAR dropped)
for S in $SPLITS; do
  echo ">>> pretrain split=$S seed=$SEED"
  $PY barlow_twins_experiments/pretrain.py \
    --split "$S" --epochs 1000 --seed "$SEED" --data_seed "$SEED" \
    --knn_every_epochs 4 --vtab_probe --vtab_every_epochs 8 \
    --run_tag "$TAG" $WANDB
done

# 2) COOLDOWN the TI best and the VTAB best of each split
for S in $SPLITS; do
  for KIND in best best_vtab; do
    CK="checkpoints/$TAG/bt_split${S}_seed${SEED}_${KIND}.pt"
    if [ -f "$CK" ]; then
      echo ">>> cooldown $CK"
      $PY barlow_twins_experiments/cooldown.py --ckpt "$CK" $WANDB
    else
      echo "[warn] missing $CK -- skipping cooldown"
    fi
  done
done

# 3) VTAB-1k linear eval: final (TI-selected) + final_vtab (VTAB-selected). Per-job CSV (no race).
echo ">>> eval final (TI-selected) + random-init"
$PY barlow_twins_experiments/vtab_protocol_eval.py \
  --run_tag "$TAG" --pretrain_seed "$SEED" --splits "$SPLITS_CSV" --results_csv "$RESULTS"
echo ">>> eval final_vtab (VTAB-selected)"
$PY barlow_twins_experiments/vtab_protocol_eval.py \
  --run_tag "$TAG" --pretrain_seed "$SEED" --splits "$SPLITS_CSV" --ckpt final_vtab --no-random --results_csv "$RESULTS"

echo "[DONE] seed=$SEED gpu=$GPU"
echo "  results : $RESULTS"
echo "  traces  : checkpoints/$TAG/bt_split*_seed${SEED}_{probes,vtab_probes}.csv"
echo "  ckpts   : checkpoints/$TAG/bt_split*_seed${SEED}_final{,_vtab}.pt"
