#!/usr/bin/env bash
# Campaign calibration + smoke test.
#
# Purpose (run this ONCE per machine type before committing to the paired-seed campaign):
#   PHASE 1  correctness smoke  — exercises the full chain on a NON-42 data seed (the path the
#                                 cooldown data_seed bug lived on): generate splits -> pretrain
#                                 (TI+VTAB probes) -> resume -> cooldown(_best,_best_vtab)
#                                 -> VTAB eval -> assert every artifact exists, incl. the durable
#                                 probe CSVs and the new pretrain_seed eval column.
#   PHASE 2  pretrain timing    — clean per-epoch time (startup excluded) on 3 representative
#                                 splits, with the EXACT campaign probe flags, then a linear fit
#                                 extrapolated to all 8 splits x 1000 epochs.
#   PHASE 3  eval timing        — time ONE full 19-task VTAB condition, scaled to all conditions.
#   -> prints a PER-SEED and x3-CAMPAIGN wallclock estimate (pretrain + cooldown + eval).
#   CIFAR-10 is NOT used (no in-training CIFAR kNN probe, no CIFAR downstream eval) — VTAB-1k is
#   the sole transfer signal. Re-add --diagnostic_cifar_probe / leo_protocol_eval.py to restore it.
#
# The user launches this in tmux. It only invokes the project's own entry points. Timing runs use
# --wandb_mode disabled (removes network/login as a confound and a dependency on a fresh box).
#
# Usage:   bash barlow_twins_experiments/calibrate_campaign.sh [smoke|timing|all]   (default: all)
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root
PY=".venv/bin/python"
PHASE="${1:-all}"

SHA="$($PY - <<'PY'
import subprocess
try:
    print(subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip())
except Exception:
    print("nogit")
PY
)"
# Pre-prefix the SHA so pretrain's resolve_run_tag() leaves the tag as-is AND the eval scripts
# (which take --run_tag verbatim) look in the same dir. Isolated from the real campaign tag (<sha>).
SMOKE_TAG="${SHA}-smoke"
CAL_TAG="${SHA}-caltiming"
SMOKE_SEED=44
SMOKE_DIR="checkpoints/${SMOKE_TAG}"

CAL_EPOCHS=25                    # steady window = 24 epochs = 6 TI probes + 3 VTAB probes (exact campaign ratio)
CAMPAIGN_EPOCHS=1000

banner(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
assert_file(){ [ -s "$1" ] && echo "  [ok] $1" || { echo "  [FAIL] missing/empty: $1"; exit 1; }; }

# ---- ensure the seed-43/44 index files exist (idempotent) ----------------------------
ensure_split_file(){
  local s="$1" f
  if [ "$s" = 42 ]; then f="data/tiny_imagenet_shuffled_indices.json"; else f="data/tiny_imagenet_shuffled_indices_seed${s}.json"; fi
  if [ -s "$f" ]; then echo "  [ok] data seed $s present ($f)"; else
    echo "  [gen] data seed $s"; $PY data/generate_splits.py --data_seed "$s"; fi
}

# ======================================================================================
# PHASE 1 — correctness smoke
# ======================================================================================
run_smoke(){
  banner "PHASE 1: correctness smoke (tag=$SMOKE_TAG, seed=$SMOKE_SEED)"
  ensure_split_file "$SMOKE_SEED"

  local base="${SMOKE_DIR}/bt_split1000_seed${SMOKE_SEED}"

  echo "-- pretrain 12 epochs (TI+VTAB probes) --"
  $PY barlow_twins_experiments/pretrain.py \
    --split 1000 --epochs 12 --seed $SMOKE_SEED --data_seed $SMOKE_SEED \
    --knn_every_epochs 4 --vtab_probe --vtab_every_epochs 8 --vtab_tasks cifar,dtd \
    --run_tag "$SMOKE_TAG" --wandb_mode disabled
  assert_file "${base}_best.pt"
  assert_file "${base}_best_vtab.pt"
  assert_file "${base}_stable.pt"
  assert_file "${base}_probes.csv"
  assert_file "${base}_vtab_probes.csv"

  echo "-- resume 12 -> 20 epochs (must restore smoothing windows; not clobber _best) --"
  local best_step_before
  best_step_before="$($PY - "$base" <<'PY'
import sys,torch
b=torch.load(sys.argv[1]+"_best.pt",map_location="cpu",weights_only=False)
print(b.get("best_knn_step"))
PY
)"
  $PY barlow_twins_experiments/pretrain.py \
    --split 1000 --epochs 20 --seed $SMOKE_SEED --data_seed $SMOKE_SEED \
    --knn_every_epochs 4 --vtab_probe --vtab_every_epochs 8 --vtab_tasks cifar,dtd \
    --run_tag "$SMOKE_TAG" --wandb_mode disabled \
    --resume_from "${base}_stable.pt"
  $PY - "$base" "$best_step_before" <<'PY'
import sys,torch
base,before=sys.argv[1],sys.argv[2]
st=torch.load(base+"_stable.pt",map_location="cpu",weights_only=False)
for k in ("val_window","cf_window","vtab_window"):
    assert k in st, f"_stable missing {k}"
    print(f"  [ok] _stable has {k} (len {len(st[k])})")
b=torch.load(base+"_best.pt",map_location="cpu",weights_only=False)
print(f"  best_knn_step before={before} after={b.get('best_knn_step')} "
      f"({'preserved/raised' if int(b.get('best_knn_step'))>=int(before) else 'REGRESSED!'})")
assert int(b.get("best_knn_step"))>=int(before), "best_knn_step went backwards -> _best clobbered"
PY

  echo "-- cooldown _best (TI) and _best_vtab (must print data_seed=$SMOKE_SEED) --"
  $PY barlow_twins_experiments/cooldown.py --ckpt "${base}_best.pt" --wandb_mode disabled
  $PY barlow_twins_experiments/cooldown.py --ckpt "${base}_best_vtab.pt" --wandb_mode disabled
  assert_file "${base}_final.pt"
  assert_file "${base}_final_vtab.pt"

  echo "-- VTAB eval on the smoke checkpoints (2 tasks; checks pretrain_seed column) --"
  $PY barlow_twins_experiments/vtab_protocol_eval.py \
    --run_tag "$SMOKE_TAG" --pretrain_seed $SMOKE_SEED --splits 1000 --tasks cifar,dtd --no-random
  $PY - <<'PY'
import csv,pathlib
p=pathlib.Path("evaluation")/"vtab_results.csv"
hdr=next(csv.reader(open(p)))
assert "pretrain_seed" in hdr, "vtab_results.csv missing pretrain_seed column"
print("  [ok] vtab_results.csv has pretrain_seed column")
PY
  echo "  [PHASE 1 PASS] full chain works on seed $SMOKE_SEED."
}

# ======================================================================================
# PHASE 2 — pretrain timing
# ======================================================================================
measure_epoch(){   # echoes mean steady-state epoch seconds for a split (fails loudly on a bad parse)
  local split="$1" log val
  log="$(mktemp)"
  $PY barlow_twins_experiments/pretrain.py \
    --split "$split" --epochs $CAL_EPOCHS --seed $SMOKE_SEED --data_seed $SMOKE_SEED \
    --knn_every_epochs 4 --vtab_probe --vtab_every_epochs 8 --vtab_tasks all \
    --run_tag "$CAL_TAG" --wandb_mode disabled >"$log" 2>&1 || { cat "$log"; exit 1; }
  val="$(grep -oP 'mean_epoch_s=\K[0-9.]+' "$log" | tail -1 || true)"
  if [ -z "$val" ]; then
    echo "[FATAL] no MEAN_EPOCH_S line for split=$split; pretrain output below:" >&2
    cat "$log" >&2; rm -f "$log"; exit 1
  fi
  rm -f "$log"
  printf '%s' "$val"
}

run_timing(){
  banner "PHASE 2: pretrain timing (tag=$CAL_TAG, $CAL_EPOCHS epochs, full 19-task VTAB probe)"
  ensure_split_file "$SMOKE_SEED"
  echo "  (each run is short; the 100k run is the longest — a few minutes)"
  E1000="$(measure_epoch 1000)"     || exit 1; echo "  split   1000: ${E1000}s/epoch"
  E16000="$(measure_epoch 16000)"   || exit 1; echo "  split  16000: ${E16000}s/epoch"
  E100000="$(measure_epoch 100000)" || exit 1; echo "  split 100000: ${E100000}s/epoch"

  banner "PHASE 3: eval timing (1 full 19-task VTAB condition)"
  # Reuse the smoke _final (split 1000, seed 44) — eval cost is dominated by decoding the 19
  # test sets, ~independent of how the backbone was trained, so 1 condition scales to all.
  [ -s "${SMOKE_DIR}/bt_split1000_seed${SMOKE_SEED}_final.pt" ] || { echo "  run the smoke phase first (need a _final ckpt)"; exit 1; }
  local t0 t1
  t0=$(date +%s)
  $PY barlow_twins_experiments/vtab_protocol_eval.py \
    --run_tag "$SMOKE_TAG" --pretrain_seed $SMOKE_SEED --splits 1000 --tasks all --no-random >/dev/null
  t1=$(date +%s); VTAB_1COND=$((t1-t0)); echo "  VTAB  (19 tasks, 1 condition): ${VTAB_1COND}s"

  banner "ESTIMATE"
  E1000="$E1000" E16000="$E16000" E100000="$E100000" \
  VTAB_1COND="$VTAB_1COND" CAMPAIGN_EPOCHS="$CAMPAIGN_EPOCHS" \
  $PY - <<'PY'
import os
e1,e16,e100 = float(os.environ["E1000"]),float(os.environ["E16000"]),float(os.environ["E100000"])
vtab1 = float(os.environ["VTAB_1COND"])
EP = int(os.environ["CAMPAIGN_EPOCHS"])
splits=[1000,2000,4000,8000,16000,32000,64000,100000]

# Per-epoch time vs split: PIECEWISE-LINEAR interpolation over the 3 measured points (every
# campaign split lies in [1000,100000], so no extrapolation and no slope-sign assumption — this
# is robust to noisy short runs where a global linear fit could go negative).
pts=sorted([(1000.0,e1),(16000.0,e16),(100000.0,e100)])
def t_epoch(s):
    s=float(s)
    if s<=pts[0][0]: return max(0.0,pts[0][1])
    for (x0,y0),(x1,y1) in zip(pts,pts[1:]):
        if s<=x1: return max(0.0, y0+(y1-y0)*(s-x0)/(x1-x0))
    return max(0.0,pts[-1][1])

print(f"  measured s/epoch: 1k={e1:.2f}  16k={e16:.2f}  100k={e100:.2f}  (piecewise-interpolated between)")
print(f"\n  {'split':>7} {'s/epoch':>9} {'1000ep (h)':>12}")
pre=0.0
for s in splits:
    te=t_epoch(s); T=EP*te; pre+=T
    print(f"  {s:>7} {te:>9.2f} {T/3600:>12.2f}")
# Cooldown UPPER BOUND: cooldown length = 0.2*best_step, best_step <= EP*spe (peak at the very end),
# and cooldown's per-step cost <= a full pretrain epoch's amortized cost. So 0.2*pretrain-time is a
# safe upper bound for the TI bests (which peak late); the VTAB bests peak early -> hit the 500-step
# floor -> negligible. Positive by construction (pre>=0).
cool = 0.2*pre
nv = 2*len(splits)+1     # VTAB eval: final + final_vtab over 8 splits, + 1 random
evalt = nv*vtab1
print(f"\n  pretrain / seed          : {pre/3600:6.2f} h")
print(f"  cooldown / seed (UB)     : {cool/3600:6.2f} h   (0.2x pretrain; TI bests, worst-case late peak)")
print(f"  VTAB eval / seed (~{nv} cond): {evalt/3600:6.2f} h")
per = pre+cool+evalt
print(f"\n  >>> PER SEED  : {per/3600:6.2f} h")
print(f"  >>> CAMPAIGN  : {3*per/3600:6.2f} h  (3 seeds)")
print("\n  Notes: single-GPU, wandb-disabled (online adds <1%). Cooldown is an UPPER bound (assumes")
print("  the TI peak sits at the end). VTAB-only eval (final + final_vtab); no CIFAR. Re-measure on")
print("  the rented GPU. To trim further, raise --vtab_every_epochs (fewer in-training VTAB probes).")
PY
}

case "$PHASE" in
  smoke)  run_smoke ;;
  timing) run_timing ;;
  all)    run_smoke; run_timing ;;
  *) echo "usage: $0 [smoke|timing|all]"; exit 2 ;;
esac
echo; echo "Done ($PHASE)."
