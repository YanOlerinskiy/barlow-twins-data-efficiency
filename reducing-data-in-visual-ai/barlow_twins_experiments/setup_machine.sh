#!/usr/bin/env bash
# Post-clone setup on a fresh GPU box (run AFTER git clone, from the repo root):
#   bash barlow_twins_experiments/setup_machine.sh
# Creates the venv, installs deps, fetches the VTAB-1k bundle (gitignored, ~5.5 GB), and warms the
# Tiny-ImageNet cache so concurrent GPU jobs don't race on first download. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

echo "=== disk before setup ===" ; df -h . | tail -1

echo "=== [1/3] venv + deps ==="
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install -q --no-cache-dir --upgrade pip
# --no-cache-dir: don't keep downloaded wheels (the torch wheel alone is multi-GB) — these boxes
# often have only ~32 GB disk and the VTAB-1k extraction below needs the headroom.
pip install -q --no-cache-dir -r requirements.txt
pip install -q --no-cache-dir gdown   # not in requirements; needed for the VTAB-1k download

echo "=== [2/3] VTAB-1k bundle (~5.5 GB, gitignored) ==="
if [ "$(ls vtab-1k 2>/dev/null | wc -l)" -ge 19 ]; then
  echo "  already present ($(ls vtab-1k | wc -l) tasks)"
else
  gdown 1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p -O vtab-1k.zip
  unzip -q -o vtab-1k.zip
  rm -f vtab-1k.zip
  n="$(ls vtab-1k 2>/dev/null | wc -l)"
  [ "$n" -ge 19 ] || {
    echo "ERROR: vtab-1k has $n task dirs (expected 19)."
    echo "  Either the bundle layout differs, or the gdown id changed. Fallback: rsync it from your"
    echo "  local box:  rsync -az --info=progress2 ~/Studies/TUDelft/research-project/reducing-data-in-visual-ai/vtab-1k/ root@<host>:<repo>/vtab-1k/"
    exit 1; }
  echo "  extracted ($n tasks)"
fi

echo "=== [3/3] warm Tiny-ImageNet cache (avoids a concurrent-download race between GPU jobs) ==="
python - <<'PY' || echo "  (could not prefetch; it will download on the first training run)"
from datasets import load_dataset
load_dataset("Maysee/tiny-imagenet")
print("  Tiny-ImageNet cached.")
PY

echo
echo "Setup complete. Next (example, 2-GPU box, seed 42):"
echo "  export WANDB_API_KEY=...   # optional, for live dashboards"
echo "  tmux new -s g0 'CUDA_VISIBLE_DEVICES=0 bash barlow_twins_experiments/run_campaign.sh 42 1000,2000,4000,8000,32000; bash'"
echo "  tmux new -s g1 'CUDA_VISIBLE_DEVICES=1 bash barlow_twins_experiments/run_campaign.sh 42 16000,64000,100000; bash'"
