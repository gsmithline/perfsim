"""The model, nothing else:
  - population: plain FJ
  - agents: multiplicative-weights choice over platforms (sticky)
  - platforms: Hotelling, each positions to maximize captured share
Run on the real Pokec graph, sweeping injected mode separation to test whether
modality drives merge vs segment.
Run: python experiments/fj/run_fj_mw_hotelling.py
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
from _pokec import FixedPredictions, load_pokec

OUT = Path("runs/fj_mw_hotelling")
SEEDS = range(6)
ROUNDS = 100
K_INNER = 10
ANCHOR_Q = (0.1, 0.5, 0.9)
STEPS = 8
LR = 0.02
TAU = 0.05          # Hotelling choice sharpness
ETA = 5.0           # MW rate
SPREAD = 0.10
CENTER = 0.5
SEPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
SEG = 0.15


def fiedler_split(adj):
    deg = adj.sum(dim=1)
    _, vecs = torch.linalg.eigh(torch.diag(deg) - adj)
    return (vecs[:, 1] > 0).long()


def make_innate(sep, community, seed):
    gen = torch.Generator(); gen.manual_seed(seed)
    sign = torch.where(community == 0, -1.0, 1.0)
    return (CENTER + sign * (sep / 2) + SPREAD * torch.randn(community.shape[0], generator=gen)).clamp(0, 1)


def run_market(setup, community, sep, seed):
    innate = make_innate(sep, community, seed)
    n = innate.shape[0]
    world = FJWorld(innate, setup["W"], setup["peer_sus"], platform_sus=setup["platform_sus"])
    world.reset(seed=seed)
    gen = torch.Generator(); gen.manual_seed(seed + 1000)
    b = torch.quantile(innate, torch.tensor(ANCHOR_Q)).clone()
    n_p = b.shape[0]
    w = torch.full((n, n_p), 1.0 / n_p)                       # MW weights
    mode_sep = float((innate[community == 1].mean() - innate[community == 0].mean()).abs())
    for _ in range(ROUNDS):
        follow = torch.multinomial(w, 1, generator=gen).squeeze(1)
        op = world.run(FixedPredictions(b[follow]), n_steps=K_INNER)["y"].reshape(-1)
        b_start = b.clone()
        for p in range(n_p):                                  # platforms play Hotelling
            rival = torch.stack([torch.exp(-(b_start[q] - op).abs() / TAU)
                                 for q in range(n_p) if q != p]).sum(dim=0)
            for _ in range(STEPS):
                bp = b[p].clone().requires_grad_(True)
                own = torch.exp(-(bp - op).abs() / TAU)
                share = own / (own + rival)
                (g,) = torch.autograd.grad(-share.mean(), bp)
                b[p] = (b[p] - LR * g).detach()
        agree = -(b.unsqueeze(0) - op.unsqueeze(1)).abs()     # agents MW-update
        w = w * torch.exp(ETA * (agree - agree.max(dim=1, keepdim=True).values))
        w = (w.clamp_min(1e-3))
        w = w / w.sum(dim=1, keepdim=True)
    div = sum(float((b[a] - b[c]).abs())
              for a in range(n_p) for c in range(n_p) if a != c) / (n_p * (n_p - 1))
    return {"mode_sep": mode_sep, "div": div, "pop_std": float(op.std()),
            "conc": float(w.max(dim=1).values.mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup = load_pokec()
    community = fiedler_split(setup["adj"])
    print(f"pokec N={community.shape[0]}  tau={TAU} eta={ETA}  (FJ + MW agents + Hotelling)")
    print(f"{'mode_sep':>9s} | {'mean_div':>9s} {'frac_seg':>8s} {'pop_std':>8s} "
          f"{'lock-in':>8s}  per-seed div")
    xs, dv, st = [], [], []
    for sep in SEPS:
        runs = [run_market(setup, community, sep, s) for s in SEEDS]
        msep = sum(r["mode_sep"] for r in runs) / len(runs)
        dvs = [r["div"] for r in runs]
        frac = sum(d > SEG for d in dvs) / len(dvs)
        pstd = sum(r["pop_std"] for r in runs) / len(runs)
        conc = sum(r["conc"] for r in runs) / len(runs)
        xs.append(msep); dv.append(sum(dvs) / len(dvs)); st.append(pstd)
        print(f"{msep:>9.3f} | {sum(dvs)/len(dvs):>9.3f} {frac:>8.2f} {pstd:>8.3f} "
              f"{conc:>8.3f}  {[round(d, 2) for d in dvs]}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax[0].plot(xs, dv, "o-", color="#3b6fd8")
    ax[0].plot(xs, xs, ls=":", color="gray", lw=1, label="platforms at modes")
    ax[0].set_xlabel("injected mode separation"); ax[0].set_ylabel("final platform div |b_p - b_q|")
    ax[0].set_title("merge -> segment (FJ + MW + Hotelling, real Pokec)"); ax[0].legend(fontsize=8)
    ax[1].plot(xs, st, "o-", color="#d8633b")
    ax[1].set_xlabel("injected mode separation"); ax[1].set_ylabel("final population std")
    ax[1].set_title("population spread")
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("FJ population + MW platform choice + Hotelling platforms, on real Pokec",
                 fontweight="bold")
    fig.savefig(OUT / "fj_mw_hotelling.png", dpi=150)
    print(f"[fj_mw_hotelling] figure -> {OUT / 'fj_mw_hotelling.png'}")


if __name__ == "__main__":
    main()
