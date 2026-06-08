"""Competing prior-anchored platforms over one FJ population, single-homing;
do the beach's modes end at innate community means or at the priors?
Run: python experiments/fj/run_beach_competition.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from perfsim.environments.dynamics.fj import FJWorld
from run_beach import make_population
from _pokec import FixedPredictions

OUT = Path("runs/fj_beach")
SEEDS = (0, 1, 2)
N = 200
ROUNDS = 40
K_INNER = 10
BETA_TRUST = 0.6
ETA_MOB = 0.5
PRIORS = (-0.8, 0.8)
LAMS = (0.0, 0.5, 4.0)


def run_market(priors: tuple[float, ...], lam: float, seed: int) -> dict:
    innate, graph, community = make_population(seed)
    world = FJWorld(innate, graph, torch.full((N,), 0.5), platform_sus=BETA_TRUST)
    world.reset(seed=seed)
    n_p = len(priors)
    b = torch.tensor(priors)
    prior_t = torch.tensor(priors)
    shares = torch.full((N, n_p), 1.0 / n_p)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    x = innate.clone()
    out = {"std": [], "sep": [], "b": [], "conc": []}
    for _ in range(ROUNDS):
        assign = torch.multinomial(shares, 1, generator=gen).squeeze(1)   # (N,)
        data = world.run(FixedPredictions(b[assign]), n_steps=K_INNER)
        x = data["y"].reshape(-1)
        b_next = b.clone()
        for p in range(n_p):
            mask = assign == p
            n_served = int(mask.sum())
            if n_served > 0 or lam > 0:
                b_next[p] = (x[mask].sum() + lam * prior_t[p]) / (n_served + lam)
        b = b_next
        agree = -(b.unsqueeze(0) - x.unsqueeze(1)).abs()        # (N, P)
        shares = shares * torch.exp(ETA_MOB * (agree - agree.max(dim=1, keepdim=True).values))
        shares = shares / shares.sum(dim=1, keepdim=True)
        out["std"].append(float(x.std()))
        out["sep"].append(float((x[community == 1].mean() - x[community == 0].mean()).abs()))
        out["b"].append([float(v) for v in b])
        out["conc"].append(float(shares.max(dim=1).values.mean()))
    out["x_final"] = x
    out["innate"] = innate
    out["community"] = community
    out["innate_means"] = [float(innate[community == c].mean()) for c in (0, 1)]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {lam: [run_market(PRIORS, lam, s) for s in SEEDS] for lam in LAMS}
    mono = [run_market((0.8,), 4.0, s) for s in SEEDS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    colors = {0.0: "#3b6fd8", 0.5: "#3bb0a0", 4.0: "#d8633b"}

    ax = axes.flat[0]
    for lam, rs in runs.items():
        curves = torch.tensor([r["sep"] for r in rs])
        ax.plot(curves.mean(dim=0), color=colors[lam], label=f"duopoly lam={lam:g}")
    ax.plot(torch.tensor([r["sep"] for r in mono]).mean(dim=0), color="#888888",
            label="monopoly lam=4")
    ax.set_title("community separation")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    ax = axes.flat[1]
    for lam, rs in runs.items():
        traj = torch.tensor([r["b"] for r in rs]).mean(dim=0)   # (T, P)
        for p in range(traj.shape[1]):
            ax.plot(traj[:, p], color=colors[lam],
                    label=f"lam={lam:g}" if p == 0 else None)
    for c, m in enumerate(runs[LAMS[0]][0]["innate_means"]):
        ax.axhline(m, color="black", lw=0.7, ls=":")
    for pr in PRIORS:
        ax.axhline(pr, color="black", lw=0.7, ls="--")
    ax.set_title("platform positions (dashed = priors, dotted = innate means)")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    ax = axes.flat[2]
    for lam, rs in runs.items():
        curves = torch.tensor([r["conc"] for r in rs])
        ax.plot(curves.mean(dim=0), color=colors[lam], label=f"lam={lam:g}")
    ax.set_title("trust concentration (mean max share)")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    ax = axes.flat[3]
    r0 = runs[4.0][0]
    ax.hist(r0["innate"].numpy(), bins=40, alpha=0.4, color="#888888", label="innate beach")
    ax.hist(r0["x_final"].numpy(), bins=40, alpha=0.6, color="#d8633b",
            label="final beach (lam=4)")
    for pr in PRIORS:
        ax.axvline(pr, color="black", lw=0.9, ls="--")
    for m in r0["innate_means"]:
        ax.axvline(m, color="black", lw=0.9, ls=":")
    ax.set_title("the beach, before and after (seed 0)")
    ax.legend(fontsize=8)

    fig.suptitle("Prior capture: do the beach's modes end at innate means or at the priors?",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "beach_competition.png", dpi=140)

    print(f"{'condition':>16s} | {'final b':>16s} {'sep':>6s} {'conc':>6s}  innate means")
    for lam, rs in runs.items():
        bf = torch.tensor([r["b"][-1] for r in rs]).mean(dim=0)
        sep = sum(r["sep"][-1] for r in rs) / len(rs)
        conc = sum(r["conc"][-1] for r in rs) / len(rs)
        im = rs[0]["innate_means"]
        print(f"{'duopoly lam=' + format(lam, 'g'):>16s} | "
              f"{', '.join(f'{v:+.2f}' for v in bf):>16s} {sep:6.3f} {conc:6.3f}  "
              f"{im[0]:+.2f}, {im[1]:+.2f}")
    bf = torch.tensor([r["b"][-1] for r in mono]).mean(dim=0)
    sep = sum(r["sep"][-1] for r in mono) / len(mono)
    print(f"{'monopoly lam=4':>16s} | {f'{float(bf):+.2f}':>16s} {sep:6.3f}")
    print(f"[fj] figure -> {OUT / 'beach_competition.png'}  (seeds {SEEDS})")


if __name__ == "__main__":
    main()
