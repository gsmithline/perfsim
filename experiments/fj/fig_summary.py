"""Four-panel summary of the FJ platform-competition experiments.
Run: python experiments/fj/fig_summary.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "experiments/fj")
import run_hotelling as rh

OUT = Path("runs/fj_beach")
GRID = torch.linspace(0.0, 1.0, 401)
START3 = (0.30, 0.50, 0.70)
BLUE, TEAL, ORANGE, GRAY = "#3b6fd8", "#3bb0a0", "#d8633b", "#777777"


def gradient_run(temp, lr, rounds, seed=0, b0=START3):
    setup = gradient_run.setup
    innate = setup["innate"]
    n = innate.shape[0]
    world = rh.FJWorld(innate, setup["W"], setup["peer_sus"],
                       platform_sus=setup["platform_sus"])
    world.reset(seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    b = torch.tensor(b0)
    x = innate.clone()
    out = {"gap": [], "std": []}
    for _ in range(rounds):
        probs = torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / temp, dim=1)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        data = world.run(rh.FixedPredictions(b[assign]), n_steps=10)
        x = data["y"].reshape(-1)
        probs = torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / temp, dim=1)
        g = torch.zeros(len(b))
        for p in range(len(b)):
            mask = assign == p
            if int(mask.sum()):
                g[p] = ((1 - probs[mask, p]) * -torch.sign(b[p] - x[mask]) / temp).sum() / n
        b = (b + lr * g).detach()
        out["gap"].append(float(b.max() - b.min()))
        out["std"].append(float(x.std()))
    return out


def best_response(p, b, x, temp):
    others = torch.stack([b[q] for q in range(len(b)) if q != p])
    d_others = -(others.unsqueeze(0) - x.unsqueeze(1)).abs() / temp
    d_cand = -(GRID.unsqueeze(0) - x.unsqueeze(1)).abs() / temp
    denom = torch.exp(d_others).sum(dim=1, keepdim=True)
    share = (torch.exp(d_cand) / (torch.exp(d_cand) + denom)).mean(dim=0)
    return GRID[int(share.argmax())]


def br_run(beta, temp=0.02, rounds=300, seed=0, churn=0.0):
    setup = gradient_run.setup
    innate = setup["innate"]
    n = innate.shape[0]
    world = rh.FJWorld(innate, setup["W"], setup["peer_sus"], platform_sus=float(beta))
    world.reset(seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    b = torch.tensor(START3)
    x = innate.clone()
    traj = []
    for _ in range(rounds):
        probs = torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / temp, dim=1)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        data = world.run(rh.FixedPredictions(b[assign]), n_steps=10)
        x = data["y"].reshape(-1)
        if churn > 0:
            reborn = torch.rand(n, generator=gen) < churn
            x[reborn] = innate[reborn]
            world._state["opinion"] = x.clone()
        b = torch.stack([best_response(p, b, x, temp) for p in range(3)])
        traj.append(sorted(float(v) for v in b))
    late = torch.tensor(traj[-100:])
    amp = float(((late.max(dim=0).values - late.min(dim=0).values) / 2).max())
    gap = float((late[:, 2] - late[:, 0]).mean())
    return amp, gap, float(x.std())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gradient_run.setup = rh.load_pokec()
    innate_std = float(gradient_run.setup["innate"].std())
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)

    # (a) joint flow to the homogeneous equilibrium
    ax = axes[0, 0]
    starts = [(0.25, 0.45, 0.75), (0.30, 0.50, 0.70), (0.35, 0.55, 0.80),
              (0.22, 0.40, 0.62), (0.40, 0.60, 0.72)]
    cmap = plt.cm.viridis
    for i, b0 in enumerate(starts):
        r = gradient_run(0.05, 0.01, 800, b0=b0)
        c = cmap(i / (len(starts) - 1))
        ax.plot(r["gap"], r["std"], color=c, lw=1.5, alpha=0.9)
        ax.scatter(r["gap"][0], r["std"][0], facecolors="none", edgecolors=c, s=45)
    ax.scatter(0, r["std"][-1], marker="*", s=320, color=ORANGE,
               edgecolors="black", zorder=5, label="homogeneous equilibrium")
    ax.axhline(innate_std, ls=":", color="black", lw=1, alpha=0.6)
    ax.text(0.30, innate_std + 0.003, "innate diversity", fontsize=8, alpha=0.7)
    ax.set_xlabel("market differentiation (max platform gap)")
    ax.set_ylabel("population diversity (opinion std)")
    ax.set_title("(a) one attractor: market and population homogenize together")
    ax.legend(fontsize=8, loc="lower right")

    # (b) choice-noise regime map for gradient platforms
    ax = axes[0, 1]
    taus = (0.005, 0.01, 0.02, 0.05, 0.10)
    gaps, stds = [], []
    for t in taus:
        r = gradient_run(t, 0.01, 800)
        gaps.append(sum(r["gap"][-50:]) / 50)
        stds.append(sum(r["std"][-50:]) / 50)
    xs = range(len(taus))
    ax.plot(xs, gaps, "o-", color=BLUE, label="final platform gap")
    ax.plot(xs, stds, "s-", color=ORANGE, label="final population diversity")
    ax.axhline(innate_std, ls=":", color="black", lw=1, alpha=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_xlabel("choice noise tau  (loyalists -> flippers)")
    ax.set_title("(b) regime map: frozen segmentation vs total merge")
    ax.text(0.4, 0.36, "freeze", fontsize=10, color=BLUE)
    ax.text(3.3, 0.06, "merge", fontsize=10, color=BLUE)
    ax.legend(fontsize=8)

    # (c) trust gate: collapse scales with trust, restlessness dies
    ax = axes[1, 0]
    betas = (0.0, 0.1, 0.3, 0.6, 0.9)
    amps, stds_b = [], []
    for beta in betas:
        amp, gap, std = br_run(beta)
        amps.append(amp)
        stds_b.append(std)
    ax.plot(betas, stds_b, "s-", color=ORANGE, label="population diversity")
    ax.plot(betas, amps, "o-", color=TEAL, label="platform restlessness (cycle amplitude)")
    ax.axhline(innate_std, ls=":", color="black", lw=1, alpha=0.6)
    ax.set_xlabel("platform trust beta  (best-response platforms)")
    ax.set_title("(c) trust gate: drift collapses diversity and stills the market")
    ax.legend(fontsize=8)

    # (d) churn gate: renewal restores a diversity floor, market unmoved
    ax = axes[1, 1]
    churns = (0.0, 0.02, 0.10, 0.25)
    stds_c, gaps_c = [], []
    for ch in churns:
        amp, gap, std = br_run(0.9, churn=ch)
        stds_c.append(std)
        gaps_c.append(gap)
    ax.plot(churns, stds_c, "s-", color=ORANGE, label="population diversity")
    ax.plot(churns, gaps_c, "o-", color=BLUE, label="platform gap (market unmoved)")
    ax.set_xlabel("population churn per round  (trust beta = 0.9)")
    ax.set_title("(d) churn gate: newcomers restore diversity, not competition")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("Platform competition over an FJ population (Pokec, N = 2163): "
                 "the homogeneous attractor and its boundaries", fontweight="bold")
    fig.savefig(OUT / "fj_summary.png", dpi=150)
    print(f"[fj] figure -> {OUT / 'fj_summary.png'}")


if __name__ == "__main__":
    main()
