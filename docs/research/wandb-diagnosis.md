# W&B metrics not displaying — diagnosis (run `chlj4j2z`)

Date: 2026-06-10. Repo: `reducing-data-in-visual-ai`. Entity/project:
`yan-olerinskiy-tu-delft/barlow-twins-data-efficiency`.

## TL;DR root cause
The history stream for **resumed** runs is not being materialized into the
queryable history table server-side, even though the line count and key
metadata are recorded and `wandb sync` reports success. This is a wandb-core
**0.27.0 resume bug** (same run id reused via `resume="allow"`), not a UI/x-axis
problem and not a code bug in your logging. Non-resumed runs in the same project
have full history and charts. The empty System tab is a related, milder symptom
(only ~4 system samples were collected for a 64 s run, and they share the same
broken history table).

## Evidence

### Versions (task 1)
- `wandb` 0.27.0 (CLI + `wandb-core` 0.27.0, from `debug-internal.log`:
  `"stream: starting","core version":"0.27.0"`).
- `torch==2.12.0`, `numpy==2.4.6`, CUDA 13.2, GPU **RTX 5070 Ti (Blackwell)**.
- Known-issue search: multiple confirmed wandb resume issues match the symptom —
  "charts not updated when re-running with same run_id in online mode, though
  logs/metadata/summary update" (community), and resume summary/history desync
  (GitHub #9684, #5396, #11348). No fix specific to 0.27.0 surfaced; the class of
  bug is well documented. The "monotonically increasing" warning is GitHub
  #10628 / #9449 (benign here — see task 4).

### Server-side truth (task 2 — decisive)
`api.run(".../chlj4j2z")`:
- `state = finished`; **summary fully populated** (`_step=321`, `train/loss_step`,
  `knn/acc_ti`, `epoch=80`, etc.).
- `r._attrs["historyLineCount"] == 321` and `historyKeys` lists every metric with
  correct type-counts (`_step` count 321 monotonic, `knn/*` count 20,
  `system/cpu` count 4, `system/disk*` count 4).
- BUT every history read returns **0 rows**: `r.history(samples=10000)` →
  `(0,0)`; `r.history(keys=["train/loss_step"])` → empty; `scan_history(...)` → 0;
  `_sampled_history(...)` → `[]`.
- Control runs queried via the SAME API return full history:
  `7392skr4` → `(6521,14)`, `38vluvad` → `(7801,17)`. Those were single
  (non-resumed) runs. So the API and account are fine; only the resumed run's
  history table is empty.

Conclusion: the server **counted** the history lines and recorded key metadata,
but the actual history table/parquet was never materialized for this run.
History present-count but unreadable ⇒ server-side materialization failure, not a
display/x-axis issue.

### Local data integrity (task 3)
- `.wandb` files are healthy and non-empty: fresh `188 KB`, resume `192 KB`; both
  have `run-chlj4j2z.wandb.synced` markers.
- Decoding the local transaction logs (`datastore` + protobuf) shows **clean,
  monotonic history locally**:
  - FRESH dir: 161 history records, `_step` 1→161.
  - RESUME dir: 160 history records, `_step` 162→321. Continuous, no collision.
- `debug-internal.log` (both runs) shows filestream uploads succeeding:
  fresh `history_lines: 75 + 74 + 12` (offset 0→161); resume
  `history_lines: 71 + 73 + 16` (offset 161→321) — all `"request sent","status":"200 OK"`.
- Only notable log lines:
  - Resume: `WARN "handler: ignoring partial history record","step":161,"current":162`
    and the user-facing `wandb: WARNING Tried to log to step 161 that is less than
    the current step 162 ... this data will be ignored.`
  - Both runs end with `"handler: operation stats","stats":{}` (empty stats).
- During `wandb sync`, `debug-cli.*.log` shows it only re-uploaded
  `output.log`, `requirements.txt`, `wandb-metadata.json`, `config.yaml`,
  `wandb-summary.json` then `"file stream finish is done"` — it did **not** replay
  history (the run was already marked finished/synced), so re-running `wandb sync`
  cannot repair this run.

### Step / resume sanity (task 4)
Your code is correct. The fresh run logged `_step` 1→161 and the resume continued
162→321 with no overlap (verified from the decoded local files). The
"step 161 < 162 / data ignored" warning is the resume handshake reconciling the
restored cursor (161) against the first new step (162); it drops at most that one
boundary record and is **not** the cause of the empty table — the table is
entirely empty, not missing one row. Passing explicit `step=global_step` with a
continuing counter is fine and does not cause a collision here. (Minor: the first
log is at `step=1`, never 0, because `global_step += 1` precedes the first
`wandb.log` — harmless.)

## Why "restarting everything" sometimes fixes it
The failure is flaky and lives in the wandb-core agent / server materialization
path for resumed runs, not in deterministic user code — consistent with a
backend/agent race rather than a config error.

## Fixes (ranked)

1. **Stop reusing the run id for resumes (most reliable).** The bug is specific to
   `resume="allow"` + reused id. Start a **fresh** wandb run on each
   `--resume_from` invocation (it will get full, working history/charts), and
   stitch the segments in the UI by grouping on a shared key. Concretely, in
   `pretrain.py` `wandb.init(...)`: drop `id=resume_wandb_id` / `resume=...`, and
   instead pass `group=run_name` (stable across segments) and a fresh per-segment
   name, e.g. `name=f"{run_name}-s{global_step}"`. You lose a single continuous
   line but gain reliable data; the segments group together and concatenate on a
   shared x-axis. Apply the same to `cooldown.py`.

2. **Upgrade wandb** (`pip install -U wandb` in the venv) to pick up resume/
   history-materialization fixes past 0.27.0, then re-test a fresh+resume pair and
   re-check `api.run(...).history()` is non-empty. If you must keep id-reuse, this
   is the path most likely to make it work end-to-end.

3. **For the immediate already-broken run:** the data only exists locally. Don't
   bother re-running `wandb sync` (it won't replay history). To recover charts,
   create a NEW run id and re-push the local history, e.g. read the two `.wandb`
   files and `wandb.log` their rows into a fresh run, or simply re-run training to
   a fresh id now that fix #1/#2 is in place. The local `.wandb` files are intact
   if you need the numbers.

4. **Make the x-axis robust (defensive, cheap).** Add once after `wandb.init`:
   `wandb.define_metric("train/loss_step", step_metric="_step")` (and similar), or
   log `epoch`/`global_step` as explicit metrics and select them as the chart
   x-axis. This guards against out-of-order/step issues independent of the resume
   bug. It does NOT fix the current empty-table run, but is good hygiene given the
   "Try a different X axis" message.

5. **System tab:** with a ~60 s smoke run you only get ~4 system samples (sampler
   default cadence), so the System graphs look empty even when working. On real
   long runs they will populate. If they stay empty after fix #1/#2, set
   `WANDB_X_STATS_SAMPLING_INTERVAL=5` and confirm `system/*` rows appear via the
   API. Blackwell/pynvml looked fine here (GPU correctly detected as RTX 5070 Ti
   in `wandb-metadata.json`; `system/*` keys were recorded).

6. **Housekeeping:** no stale `wandb-core` processes were found, so a `pkill` is
   not needed now; keep it in mind for the flaky cases.
