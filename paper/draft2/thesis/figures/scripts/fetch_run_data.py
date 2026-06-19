"""Fetch probe histories for the e97d41c campaign from W&B into a local JSON
cache (figures/scripts/data/campaign_histories.json), so figures regenerate
offline. CPU/network only; touches no GPU.

Usage: python fetch_run_data.py
"""

import json
import os

import wandb

RUNS = {  # split -> (run_id, tag)
    1000:   ("txzmjwss", "e97d41c-curve"),
    2000:   ("l6ftnqkw", "e97d41c-curve"),
    4000:   ("avxmvbs0", "e97d41c-curve"),
    8000:   ("tv08ce0b", "e97d41c-curve"),
    16000:  ("lk70r48x", "e97d41c-curve"),
    32000:  ("x4mf234b", "e97d41c-dirty-curve"),
    64000:  ("fyfd8kr9", "e97d41c-dirty-curve"),
    100000: ("wmjrpxg9", "e97d41c-dirty-curve"),
}
ENTITY_PROJECT = "yan-olerinskiy-tu-delft/barlow-twins-data-efficiency"
KEYS = ["knn/acc_ti", "knn/smoothed_acc_ti", "knn/acc_cifar", "knn/effective_rank_ti"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "campaign_histories.json")


def main() -> None:
    api = wandb.Api(timeout=120)
    cache = {}
    for split, (rid, tag) in RUNS.items():
        run = api.run(f"{ENTITY_PROJECT}/{rid}")
        rows = list(run.history(keys=KEYS, samples=5000, pandas=False))
        cache[str(split)] = {
            "run_id": rid, "tag": tag, "name": run.name, "state": run.state,
            "rows": rows,
        }
        print(f"split {split}: {len(rows)} probes, state={run.state}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(cache, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
