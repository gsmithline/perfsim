"""Scalar Hotelling: each platform is ONE position b_p on the opinion line
(no MLP, no curve-splitting). Pure share-gradient hunters, simultaneous play.
Morph the population unimodal->bimodal, 15 seeds, locate the merge->segment
basin. Clean re-pin of run_density_sweep.py with single-location firms.
Run: python experiments/fj/run_density_sweep_scalar.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from perfsim.environments.dynamics.fj import FJWorld, normalize_adjacency
from _pokec import FixedPredictions

OUT = Path("runs/density_sweep_scalar")
N = 200
ROUNDS = 80
K_INNER = 10
ANCHOR_Q = (0.1, 0.5, 0.9)
STEPS = 8
LR = 0.05
TAU = 0.1
BETA_TRUST = 0.6
SPREAD = 0.3
SEG = 0.3


def make_population(sep, seed):
    gen = torch.Generator(); gen.manual_seed(seed)
    community = (torch.arange(N) >= N // 2).long()
    centers = torch.where(community == 0, -sep, sep)
    innate = centers + SPREAD * torch.randn(N, generator=gen)
    p = torch.where(community.unsqueeze(0) == community.unsqueeze(1), 0.10, 0.01)
    adj = (torch.rand(N, N, generator=gen) < p).float()
    adj = torch.triu(adj, diagonal=1); adj = adj + adj.t()
    return innate, normalize_adjacency(adj), community


def run_market(sep, seed):
    innate, graph, community = make_population(sep, seed)
    world = FJWorld(innate, graph, torch.full((N,), 0.5), platform_sus=BETA_TRUST)
    world.reset(seed=seed)
    gen = torch.Generator(); gen.manual_seed(seed + 1000)
    b = torch.quantile(innate, torch.tensor(ANCHOR_Q)).clone()
    n_p = b.shape[0]
    mode_sep = float((innate[community == 1].mean() - innate[community == 0].mean()).abs())
    for _ in range(ROUNDS):
        op0 = world.state["opinion"].float()
        probs = torch.softmax(-(b.unsqueeze(0) - op0.unsqueeze(1)).abs() / TAU, dim=1)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        data = world.run(FixedPredictions(b[assign]), n_steps=K_INNER)
        op = data["y"].reshape(-1)
        b_start = b.clone()
        for p in range(n_p):
            rival_w = torch.stack([torch.exp(-(b_start[q] - op).abs() / TAU)
                                   for q in range(n_p) if q != p]).sum(dim=0)
            for _ in range(STEPS):
                bp = b[p].clone().requires_grad_(True)
                own = torch.exp(-(bp - op).abs() / TAU)
                cap = own / (own + rival_w)
                (g,) = torch.autograd.grad(-cap.mean(), bp)
                b[p] = (b[p] - LR * g).detach()
    div = sum(float((b[a] - b[c]).abs())
              for a in range(n_p) for c in range(n_p) if a != c) / (n_p * (n_p - 1))
    means = sorted(float(v) for v in b)
    return {"mode_sep": mode_sep, "div": div, "means": means,
            "pop_std": float(op.std())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = range(15)
    seps = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3)
    rows, allpts = [], []
    print(f"{'mode_sep':>9s} | {'frac_seg':>8s} {'mean_div':>9s}  per-seed div")
    for sep in seps:
        runs = [run_market(sep, s) for s in seeds]
        msep = sum(r["mode_sep"] for r in runs) / len(runs)
        divs = [r["div"] for r in runs]
        frac = sum(d > SEG for d in divs) / len(divs)
        rows.append((msep, frac, sum(divs) / len(divs)))
        allpts += [(msep, d) for d in divs]
        print(f"{msep:>9.3f} | {frac:>8.2f} {sum(divs)/len(divs):>9.3f}  "
              f"{[round(d, 2) for d in divs]}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax[0].scatter([p[0] for p in allpts], [p[1] for p in allpts],
                  s=18, alpha=0.5, color="#3b6fd8")
    ax[0].axhline(SEG, ls=":", color="gray", lw=1, label=f"segment cutoff {SEG}")
    ax[0].set_xlabel("innate mode separation"); ax[0].set_ylabel("final platform div |b_p - b_q|")
    ax[0].set_title("per-seed outcome (15 seeds): bistable band"); ax[0].legend(fontsize=8)
    ax[1].plot([r[0] for r in rows], [r[1] for r in rows], "o-", color="#d8633b")
    ax[1].set_xlabel("innate mode separation"); ax[1].set_ylabel("fraction of seeds that segment")
    ax[1].set_title("basin fraction vs density")
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Scalar Hotelling, density bifurcation: merge -> bistable -> segment",
                 fontweight="bold")
    fig.savefig(OUT / "basin_fraction.png", dpi=150)
    print(f"[density_sweep_scalar] figure -> {OUT / 'basin_fraction.png'}")


if __name__ == "__main__":
    main()
