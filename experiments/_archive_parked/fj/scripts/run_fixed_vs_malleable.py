"""Fixed vs malleable demand, the control that ties this to the prior
'competition -> diversity' work. Same scalar Hotelling competition, same
densities; in 'fixed' the demand never moves (classic Hotelling), in
'malleable' the population drifts toward what it is fed (performative). The
prediction: fixed preserves the innate separation (differentiation), malleable
erodes it (merge), so the curves split in a middle band.
Run: python experiments/fj/run_fixed_vs_malleable.py
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
from run_density_sweep_scalar import ANCHOR_Q, BETA_TRUST, TAU, make_population
from _pokec import FixedPredictions

OUT = Path("runs/fixed_vs_malleable")
N = 200
ROUNDS = 80
K_INNER = 10
STEPS = 8
LR = 0.02
SEEDS = range(8)
SEPS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3)


def hunt_step(b, op, gen):
    n_p = b.shape[0]
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
    return b


def run_market(sep, seed, malleable):
    innate, graph, community = make_population(sep, seed)
    world = FJWorld(innate, graph, torch.full((N,), 0.5), platform_sus=BETA_TRUST)
    world.reset(seed=seed)
    gen = torch.Generator(); gen.manual_seed(seed + 1000)
    b = torch.quantile(innate, torch.tensor(ANCHOR_Q)).clone()
    n_p = b.shape[0]
    mode_sep = float((innate[community == 1].mean() - innate[community == 0].mean()).abs())
    op = innate.clone()
    for _ in range(ROUNDS):
        if malleable:
            probs = torch.softmax(-(b.unsqueeze(0) - op.unsqueeze(1)).abs() / TAU, dim=1)
            assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            op = world.run(FixedPredictions(b[assign]), n_steps=K_INNER)["y"].reshape(-1)
        b = hunt_step(b, op, gen)
    div = sum(float((b[a] - b[c]).abs())
              for a in range(n_p) for c in range(n_p) if a != c) / (n_p * (n_p - 1))
    return {"mode_sep": mode_sep, "div": div, "pop_std": float(op.std())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'mode_sep':>9s} | {'fixed div':>10s} {'malleable div':>14s} "
          f"{'fixed popstd':>13s} {'mall popstd':>12s}")
    xs, fixed_div, mall_div, fixed_std, mall_std = [], [], [], [], []
    for sep in SEPS:
        rf = [run_market(sep, s, False) for s in SEEDS]
        rm = [run_market(sep, s, True) for s in SEEDS]
        msep = sum(r["mode_sep"] for r in rf) / len(rf)
        fd = sum(r["div"] for r in rf) / len(rf)
        md = sum(r["div"] for r in rm) / len(rm)
        fs = sum(r["pop_std"] for r in rf) / len(rf)
        ms = sum(r["pop_std"] for r in rm) / len(rm)
        xs.append(msep); fixed_div.append(fd); mall_div.append(md)
        fixed_std.append(fs); mall_std.append(ms)
        print(f"{msep:>9.3f} | {fd:>10.3f} {md:>14.3f} {fs:>13.3f} {ms:>12.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax[0].plot(xs, fixed_div, "o-", color="#3b6fd8", label="fixed demand")
    ax[0].plot(xs, mall_div, "o-", color="#d8633b", label="malleable demand")
    ax[0].plot(xs, xs, ls=":", color="gray", lw=1, label="div = mode sep (platforms at modes)")
    ax[0].set_xlabel("innate mode separation"); ax[0].set_ylabel("final platform div |b_p - b_q|")
    ax[0].set_title("platform differentiation: fixed preserves, malleable erodes")
    ax[0].legend(fontsize=8)
    ax[1].plot(xs, fixed_std, "o-", color="#3b6fd8", label="fixed demand")
    ax[1].plot(xs, mall_std, "o-", color="#d8633b", label="malleable demand")
    ax[1].set_xlabel("innate mode separation"); ax[1].set_ylabel("final population std")
    ax[1].set_title("population spread: malleable contracts it")
    ax[1].legend(fontsize=8)
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Fixed vs malleable demand: malleability flips competition from diversifying to homogenizing",
                 fontweight="bold")
    fig.savefig(OUT / "fixed_vs_malleable.png", dpi=150)
    print(f"[fixed_vs_malleable] figure -> {OUT / 'fixed_vs_malleable.png'}")


if __name__ == "__main__":
    main()
