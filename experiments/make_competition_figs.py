"""Figures for the 3-LLM competition KL phase line. Reads
runs/pokec_fj_competition/comp_kl{beta}_t05_s{seed}/trajectory.json and makes:
 (A) divergence vs round per beta  -- the arrest: beta=0 keeps falling, beta>=1 plateaus
 (B) population op_std vs round per beta
 (C) final divergence vs beta      -- phase line with per-seed scatter
Run (on the cluster, where the data is): python experiments/make_competition_figs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("runs/pokec_fj_competition")
OUT = ROOT / "figs"


def load():
    runs: dict[int, list] = {}
    for d in sorted(ROOT.glob("comp_kl*_t05_s*")):
        tj = d / "trajectory.json"
        m = re.match(r"comp_kl(\d+)_t05_s(\d+)", d.name)
        if not (tj.exists() and m):
            continue
        t = json.load(open(tj))
        rec = {
            "div": [r["pred_divergence"] for r in t],
            "gap": [r["position_gap"] for r in t],
            "ostd": [r["op_std"] for r in t],
        }
        runs.setdefault(int(m.group(1)), []).append(rec)
    return runs


def mean_curve(seeds, key):
    n = min(len(s[key]) for s in seeds)
    cols = [[s[key][i] for s in seeds] for i in range(n)]
    return [sum(c) / len(c) for c in cols]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load()
    betas = sorted(runs)
    if not betas:
        print(f"no runs found under {ROOT}/comp_kl*_t05_s*"); return
    colors = plt.cm.viridis([i / max(len(betas) - 1, 1) * 0.9 for i in range(len(betas))])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for b, c in zip(betas, colors):
        seeds = runs[b]
        div = mean_curve(seeds, "div")
        ostd = mean_curve(seeds, "ostd")
        ax[0].plot(div, color=c, label=f"beta={b}")
        ax[1].plot(ostd, color=c, label=f"beta={b}")
    ax[0].set_title("inter-platform divergence vs round\n(beta=0 keeps falling; beta>=1 arrests)")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("pred divergence")
    ax[1].set_title("population op_std vs round")
    ax[1].set_xlabel("round"); ax[1].set_ylabel("op_std")

    for b, c in zip(betas, colors):
        finals = [s["div"][-1] for s in runs[b]]
        ax[2].scatter([b] * len(finals), finals, color=c, s=30, zorder=3)
        ax[2].scatter([b], [sum(finals) / len(finals)], color="black", marker="_", s=400, zorder=4)
    ax[2].axhline(runs[betas[0]][0]["div"][0], ls=":", color="gray", lw=1, label="initial div")
    ax[2].set_title("final divergence vs beta (KL prior)\n(higher = more diversity preserved)")
    ax[2].set_xlabel("beta (KL)"); ax[2].set_ylabel("final pred divergence")
    ax[2].set_xscale("symlog"); ax[2].legend(fontsize=8)

    for a in ax:
        a.legend(fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("3 competing LLMs over Pokec FJ: the KL prior arrests homogenization",
                 fontweight="bold")
    fig.savefig(OUT / "kl_phase_line.png", dpi=150)

    print(f"{'beta':>5s} | {'final div (per seed)':>28s} | mean")
    for b in betas:
        fin = [s["div"][-1] for s in runs[b]]
        print(f"{b:>5d} | {str([round(x,3) for x in fin]):>28s} | {sum(fin)/len(fin):.3f}")
    print(f"[figs] -> {OUT / 'kl_phase_line.png'}")


if __name__ == "__main__":
    main()
