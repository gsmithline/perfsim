"""ML atlas slab reader: cells x regimes x betas x seeds -> summary table + CSV.

Run from the perfsim root after pulling runs. Missing runs print as dashes,
so it works mid-campaign. Legacy tags cover the 14 pre-slab seed-0 combos.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

RUNS = Path("runs/pokec_gated_lm")
OUT_CSV = Path("experiments/llm/figs/atlas_slab_summary.csv")
TAIL = 5

CELLS = ["e040_a010", "e040_a020", "e040_a040", "e020_a040"]
REGIMES = ["rep", "acc", "pri", "frep", "facc"]
BETAS = ["b0", "b0p5", "b1"]
SEEDS = [0, 42, 43]

# regime -> (fresh_each_round, data_regime, pristine_frac) expected in config
REGIME_SPEC = {
    "rep":  (False, "replace", 0.0),
    "acc":  (False, "accumulate", 0.0),
    "pri":  (False, "accumulate", 0.5),
    "frep": (True, "replace", 0.0),
    "facc": (True, "accumulate", 0.0),
}

LEGACY = {}
for c in CELLS:
    LEGACY[(c, "rep", "b0", 0)] = f"mla2dv2_{c}_b0_s0"
for c in ["e040_a010", "e040_a040", "e020_a040"]:
    for b in ["b0p5", "b1"]:
        LEGACY[(c, "rep", b, 0)] = f"mla2bv2_{c}_{b}_s0"
LEGACY[("e040_a040", "acc", "b0", 0)] = "mla2drv2_e040_a040_b0_acc_s0"
LEGACY[("e040_a040", "pri", "b0", 0)] = "mla2drv2_e040_a040_b0_pri_s0"
LEGACY[("e040_a040", "frep", "b0", 0)] = "mla2dfv2_e040_a040_b0_rep_s0"
LEGACY[("e040_a040", "facc", "b0", 0)] = "mla2dfv2_e040_a040_b0_acc_s0"


def stats(tag, regime):
    d = torch.load(RUNS / tag / "trajectory.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((RUNS / tag / "config.json").read_text())
    fresh, dmode, pfrac = REGIME_SPEC[regime]
    got = (bool(cfg.get("fresh_each_round")), cfg["data_regime"],
           float(cfg.get("pristine_frac", 0.0)))
    if got != (fresh, dmode, pfrac):
        print(f"  !! {tag}: config regime {got} != tag-implied {(fresh, dmode, pfrac)}")
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    op_stdF = op[-TAIL:].std(1).mean()
    return dict(
        dr=float(op_stdF / (inn.std() + 1e-9)),
        vr=float(pr[-TAIL:].std(1).mean() / (op_stdF + 1e-9)),
        tru=float(np.sqrt(((pr[-TAIL:] - inn[None]) ** 2).mean())),
        app=float(np.sqrt(((pr[-TAIL:] - op[-TAIL:]) ** 2).mean())),
        bias=float(op[-TAIL:].mean() - inn.mean()),
        pmed=float(np.median(ppl[-TAIL:])),
        ce=float(np.log(np.maximum(ppl[-TAIL:], 1e-9)).mean()),
        rounds=int(op.shape[0]))


def find_tag(cell, regime, beta, seed):
    tag = f"mlat_{cell}_{regime}_{beta}_s{seed}"
    if (RUNS / tag / "trajectory.pt").exists():
        return tag
    leg = LEGACY.get((cell, regime, beta, seed))
    if leg and (RUNS / leg / "trajectory.pt").exists():
        return leg
    return None


rows_out = []
hdr = (f"{'regime':>6} {'b':>4} {'s':>3} | {'dr':>5} {'vr':>6} | {'true':>6} "
       f"{'app':>6} | {'bias':>7} | {'pmed':>8} {'ce':>6}")
for cell in CELLS:
    print(f"\n== {cell} ==")
    print(hdr)
    for regime in REGIMES:
        for beta in BETAS:
            for seed in SEEDS:
                tag = find_tag(cell, regime, beta, seed)
                if tag is None:
                    continue
                s = stats(tag, regime)
                print(f"{regime:>6} {beta:>4} {seed:>3} | {s['dr']:>5.2f} {s['vr']:>6.2f} | "
                      f"{s['tru']:>6.3f} {s['app']:>6.3f} | {s['bias']:>7.3f} | "
                      f"{s['pmed']:>8.1f} {s['ce']:>6.2f}")
                rows_out.append({"cell": cell, "regime": regime, "beta": beta,
                                 "seed": seed, "tag": tag, **s})

expected = len(CELLS) * len(REGIMES) * len(BETAS) * len(SEEDS)
print(f"\n{len(rows_out)}/{expected} combos present")
if rows_out:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {OUT_CSV}")
