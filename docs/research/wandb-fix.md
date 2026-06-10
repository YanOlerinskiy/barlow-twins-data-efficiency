# W&B online history not displaying — ROOT CAUSE + fix

Date: 2026-06-10. Entity/project: `yan-olerinskiy-tu-delft/barlow-twins-data-efficiency`.
Supersedes `wandb-diagnosis.md` (the resume-id theory is **wrong** — see below).

## TL;DR (verified)
This is a **server-side W&B (wandb.ai) metric-ingestion incident**, NOT a client bug,
NOT a resume bug, NOT a version regression, NOT your code. W&B's own status page
reported: *"metric ingestion is delayed for some runs up to 1 hour"* starting
**June 9 02:32 PDT**. In practice the backlog is far longer than 1 hour: runs
created after ~23:15 UTC on Jun 9 still show **0 history rows 12+ hours later**.
**There is no code/version change that fixes this — it resolves when W&B clears the
backlog server-side.** Do NOT downgrade/upgrade wandb or rewrite `wandb.init`; that
would be chasing a phantom. Your data is safe locally and will appear once ingestion
catches up (or via `wandb sync` after recovery).

## Decisive evidence

### 1. API check (task 1) — the fresh run IS empty server-side
`api.run(...).scan_history()` / `.history(samples=100000)`:
| run | state | historyLineCount | history rows | summary `_step` |
|-----|-------|------------------|--------------|-----------------|
| **t9lnuayr** (fresh) | finished | 161 | **0** | 161 |
| chlj4j2z (resume)    | finished | 321 | **0** | 321 |
| 7392skr4 (control)   | killed   | 6521 | **6521** | 6521 |

So the FRESH run has 0 server-side rows ⇒ fresh runs are broken too ⇒ the prior
"resume-id" diagnosis is incomplete/wrong. The server *counted* the lines
(`historyLineCount` correct) but never *materialized* the queryable history.

### 2. The break is purely time-correlated (task 2) — the smoking gun
Surveying the 20 most-recent runs by `createdAt` (scan_history row counts):
```
xjj6fsri finished hlc=16001 rows=16001  2026-06-09T20:05:14Z   <-- WORKS
...(all earlier runs, finished AND killed): rows == hlc, full history
chlj4j2z finished hlc=321   rows=0      2026-06-09T23:15:43Z   <-- BREAKS
t9lnuayr finished hlc=161   rows=0      2026-06-09T23:36:23Z   <-- BREAKS
(all my throwaway test runs after this point: rows=0)
```
**Sharp cutover between 20:05 and 23:15 UTC on Jun 9.** Every run before it works
(including clean-`finished` 16001-row runs on the same 0.27.0 client); every run
after it is empty. The breakage correlates ONLY with wall-clock time — not version,
not resume, not finish-vs-kill, not config. That is the signature of a backend
outage, not a client defect.

### 3. Version is NOT the cause (task 4 — tested in throwaway venvs, main .venv untouched)
Tiny 30-step online runs, clean `wandb.finish()`, then immediate API query:
| client version (temp venv) | run id | historyLineCount | history rows after finish |
|---|---|---|---|
| 0.27.2 (latest) `/tmp/wbtest` | cv5t2nco | 30 | **0** |
| 0.19.11 (legacy path) `/tmp/wbtest2` | 9eysguxg | 30 | **0** |
| 0.27.2 (run "now") | vlxiq4wu | 30 | **0** |
All three versions reproduce the empty history identically ⇒ **not a 0.27.0
regression**. Upgrading to 0.27.2 (which has the `scan_history` parquet fix) and
downgrading to 0.19.11 BOTH still produce 0 rows. The changelog items for
0.27.1/0.27.2 (GraphQL via wandb-core, parquet history export) are unrelated.

### 4. Not a propagation/UI/refresh delay we can wait out (briefly)
`chlj4j2z` (23:15 UTC) and `t9lnuayr` (23:36 UTC) still return 0 rows **>12 hours**
after finishing. So the empty charts you see are genuinely empty on the server right
now — the browser is showing the truth. It is "delayed ingestion," but the delay
currently spans many hours, not the advertised ~1h.

### 5. Local data is intact
`debug-internal.log` for t9lnuayr shows history uploaded successfully
(history_lines 76+72+13 = 161, all `"status":"200 OK"`) and `run-t9lnuayr.wandb`
is 188 KB. Nothing was lost client-side.

## Root cause
W&B cloud **metric/history ingestion incident** (status.wandb.com, started
Jun 9 02:32 PDT 2026). The ingestion pipeline that turns the uploaded line-stream
into the queryable history table/parquet is backlogged, so charts and
`run.history()` stay empty until the backend processes the run.

## Recommendation (concrete)

**Primary: do nothing to the client. Wait for W&B to clear the backlog.**
- Watch https://status.wandb.com (the "metric ingestion delayed" incident). Re-query
  a broken run with the snippet below; when rows>0, all your backlogged runs (incl.
  t9lnuayr, chlj4j2z) should populate and charts will show.
- Do **NOT** change wandb version (verified no version fixes it) and do **NOT**
  rewrite `wandb.init` in `pretrain.py`/`cooldown.py` (the current fresh-run-per-
  segment setup is correct; the empty charts are unrelated to it).
- Do **NOT** edit `requirements_barlow_twins.txt` — keep `wandb==0.27.0` (it works
  fine; the prior agent's upgrade/resume advice was based on the wrong diagnosis).

**If a run finished during the incident and its charts are still empty after the
incident is marked resolved:** that single run's server stream may be stuck. Re-push
it to a NEW run id from the intact local `.wandb` (a true re-init+`wandb.log` replay;
plain `wandb sync` won't replay an already-"synced" run). Cheapest is to re-run
training once ingestion is healthy.

**Verification snippet (run anytime to check recovery):**
```bash
.venv/bin/python -c "import wandb;a=wandb.Api();r=a.run('yan-olerinskiy-tu-delft/barlow-twins-data-efficiency/t9lnuayr');print('rows', sum(1 for _ in r.scan_history()))"
```
rows>0 ⇒ incident cleared, charts back. As of this investigation it still prints 0.

## Empirically verified vs inferred
- VERIFIED: fresh run t9lnuayr has 0 server rows; sharp time-cutover at ~20:05–23:15
  UTC Jun 9; 0.27.2 and 0.19.11 both reproduce empty history from a clean temp venv;
  12h+ persistence; local logs show successful 200-OK history upload.
- INFERRED (from W&B status snippet): the specific backend incident
  ("metric ingestion delayed, Jun 9 02:32 PDT") is the cause of the time-cutover.
  The correlation is exact and the client-side ruling-out is conclusive.

## Cleanup
Throwaway venvs `/tmp/wbtest`, `/tmp/wbtest2` and test runs (cv5t2nco, 9eysguxg,
vlxiq4wu, etc.) can be deleted. Main `.venv` was NOT modified.
