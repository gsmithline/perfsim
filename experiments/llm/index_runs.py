"""Master run index: walk runs/pokec_gated_lm/*/config.json, print an
inventory grouped by dataset x model x batch family, write runs_index.csv.

Run from the perfsim root. Older configs lack the dataset key; those fall
back to the tag-stem map below.
"""
import csv
import json
from pathlib import Path

RUNS = Path("runs/pokec_gated_lm")
OUT_CSV = Path("experiments/llm/figs/runs_index.csv")

# tag-stem -> dataset for runs from before config.json recorded it
STEM_DATASET = {
    "e2d": "pokec", "gcore": "pokec", "ggam": "pokec", "gca": "pokec",
    "gcr": "pokec", "gcan": "pokec", "gnf": "pokec", "gfa": "pokec",
    "gfp": "pokec", "gdir": "pokec", "gfr": "pokec", "ginn": "pokec",
    "e2ds": "pokec", "e2dr": "pokec", "e2df": "pokec",
    "e2dnf": "pokec", "e2dv2": "pokec", "e2dnfv2": "pokec",
    "mla": "movielens", "mlat": "movielens",
    "ylp": "yelp",
}


def stem_dataset(tag):
    stem = tag.split("_")[0]
    for pref, ds in sorted(STEM_DATASET.items(), key=lambda kv: -len(kv[0])):
        if stem.startswith(pref):
            return ds
    return "?"


def model_short(base_model):
    return (base_model or "?").split("/")[-1].replace("-Instruct", "")


rows = []
for d in sorted(RUNS.iterdir() if RUNS.exists() else []):
    cfg_path = d / "config.json"
    if not d.is_dir() or not cfg_path.exists():
        continue
    cfg = json.loads(cfg_path.read_text())
    tel = d / "telemetry.json"
    rounds_done = sum(1 for _ in open(tel)) if tel.exists() else 0
    rows.append({
        "tag": d.name,
        "family": d.name.split("_")[0],
        "dataset": cfg.get("dataset", stem_dataset(d.name)),
        "model": model_short(cfg.get("base_model")),
        "pop": cfg.get("pop_model"),
        "eps": cfg.get("eps"), "eps_ai": cfg.get("eps_ai"),
        "beta": cfg.get("kl_beta"), "seed": cfg.get("seed"),
        "data_regime": cfg.get("data_regime"),
        "fresh": int(bool(cfg.get("fresh_each_round"))),
        "pfrac": cfg.get("pristine_frac", 0.0),
        "gamma": cfg.get("gamma_bias"), "w": cfg.get("w_plat"),
        "n_rounds": cfg.get("n_rounds"),
        "rounds_done": rounds_done,
        "done": int((d / "trajectory.pt").exists()),
    })

groups = {}
for r in rows:
    key = (r["dataset"], r["model"], r["family"])
    g = groups.setdefault(key, {"n": 0, "done": 0, "betas": set(), "seeds": set()})
    g["n"] += 1
    g["done"] += r["done"]
    g["betas"].add(r["beta"])
    g["seeds"].add(r["seed"])

print(f"{'dataset':>10} {'model':>14} {'family':>10} | {'runs':>4} {'done':>4} | betas / seeds")
for (ds, m, fam), g in sorted(groups.items()):
    betas = ",".join(str(b) for b in sorted(g["betas"]))
    seeds = ",".join(str(s) for s in sorted(g["seeds"]))
    print(f"{ds:>10} {m:>14} {fam:>10} | {g['n']:>4} {g['done']:>4} | {betas} / s{seeds}")
print(f"\n{len(rows)} runs total, {sum(r['done'] for r in rows)} complete")

if rows:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")
