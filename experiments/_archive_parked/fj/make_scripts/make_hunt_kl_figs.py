"""Figures for the all-hunters KL phase line (strategic counterpart of the
competition SFT phase line). Reads
runs/pokec_fj_hunt_kl/hunt_kl{beta}_t05_s{seed}/trajectory.json and makes:
 (A) divergence vs round per beta  -- arrest: does the prior hold hunters apart?
 (B) distance-to-consensus (std of means) vs round per beta
 (C) final divergence vs beta      -- phase line with per-seed scatter
Run: python experiments/make_hunt_kl_figs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("runs/pokec_fj_hunt_kl")
OUT = ROOT / "figs"


def load():
    runs: dict[int, list] = {}
    for d in sorted(ROOT.glob("hunt_kl*_t05_s*")):
        tj = d / "trajectory.json"
        m = re.match(r"hunt_kl(\d+)_t05_s(\d+)", d.name)
        if not (tj.exists() and m):
            continue
        t = json.load(open(tj))
        std_means = [(sum((x - sum(p["platform_means"]) / 3) ** 2
                          for x in p["platform_means"]) / 3) ** 0.5 for p in t]
        runs.setdefault(int(m.group(1)), []).append({
            "div": [r["pred_divergence"] for r in t],
            "gap": std_means,
            "ostd": [r["op_std"] for r in t],
        })
    return runs


def mean_curve(seeds, key):
    n = min(len(s[key]) for s in seeds)
    return [sum(s[key][i] for s in seeds) / len(seeds) for i in range(n)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load()
    betas = sorted(runs)
    if not betas:
        print(f"no runs under {ROOT}/hunt_kl*_t05_s*"); return
    colors = plt.cm.viridis([i / max(len(betas) - 1, 1) * 0.9 for i in range(len(betas))])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for b, c in zip(betas, colors):
        ax[0].plot(mean_curve(runs[b], "div"), color=c, label=f"beta={b}")
        ax[1].plot(mean_curve(runs[b], "gap"), color=c, label=f"beta={b}")
    ax[0].set_title("inter-platform divergence vs round\n(all hunters; does the prior arrest the merge?)")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("pred divergence")
    ax[1].set_title("distance to consensus (std of means) vs round")
    ax[1].set_xlabel("round"); ax[1].set_ylabel("std of platform means")

    xs = list(range(len(betas)))
    for x, b, c in zip(xs, betas, colors):
        fin = [s["div"][-1] for s in runs[b]]
        ax[2].scatter([x] * len(fin), fin, color=c, s=30, zorder=3)
        ax[2].scatter([x], [sum(fin) / len(fin)], color="black", marker="_", s=400, zorder=4)
    ax[2].axhline(runs[betas[0]][0]["div"][0], ls=":", color="gray", lw=1, label="initial div")
    ax[2].set_xticks(xs); ax[2].set_xticklabels([str(b) for b in betas])
    ax[2].set_title("final divergence vs beta (KL leash on hunters)")
    ax[2].set_xlabel("beta (KL)"); ax[2].set_ylabel("final pred divergence"); ax[2].legend(fontsize=8)

    for a in ax:
        a.legend(fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Three KL-anchored hunters over Pokec FJ: does the prior survive strategic competition?",
                 fontweight="bold")
    fig.savefig(OUT / "hunt_kl_phase_line.png", dpi=150)

    print(f"{'beta':>5s} | {'final div (per seed)':>28s} | mean")
    for b in betas:
        fin = [s["div"][-1] for s in runs[b]]
        print(f"{b:>5d} | {str([round(x,3) for x in fin]):>28s} | {sum(fin)/len(fin):.3f}")
    print(f"[figs] -> {OUT / 'hunt_kl_phase_line.png'}")


if __name__ == "__main__":
    main()
