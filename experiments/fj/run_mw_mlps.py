"""Three mixed-objective MLP platforms with MW trust mobility over a bimodal
FJ population. Local analogue of the LLM pipeline.
Run: python experiments/fj/run_mw_mlps.py
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
from perfsim.models import MLPModel
from run_beach import make_population
from run_hotelling import FixedPredictions

OUT = Path("runs/fj_beach")
SEEDS = (0, 1, 2)
N = 200
ROUNDS = 200
K_INNER = 10
ANCHORS = (-0.8, 0.0, 0.8)
TAU = 0.1
ETA_MOB = 0.5
MIX_LAMBDA = 1.0
LAMBDA_ACQ = float(os.environ.get("LAMBDA_ACQ", "0.5"))
ANCHOR_COST = float(os.environ.get("ANCHOR_COST", "0"))
STUBBORN = float(os.environ.get("STUBBORN", "0"))
STEPS = 8
BATCH = 64
LR = 1e-3
BETA_TRUST = 0.6


def make_platforms(innate: torch.Tensor) -> tuple[list[MLPModel], list]:
    models, opts = [], []
    x = innate.unsqueeze(-1)
    for p, anchor in enumerate(ANCHORS):
        m = MLPModel(1, (32, 32), init_seed=p)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        tgt = torch.full((N, 1), anchor)
        for _ in range(200):
            opt.zero_grad()
            ((m(x) - tgt) ** 2).mean().backward()
            opt.step()
        models.append(m)
        opts.append(opt)
    return models, opts


def run_market(seed: int) -> dict:
    innate, graph, community = make_population(seed)
    plat_sus = torch.full((N,), BETA_TRUST)
    if STUBBORN > 0:
        gen_s = torch.Generator()
        gen_s.manual_seed(seed + 99)
        plat_sus[torch.rand(N, generator=gen_s) < STUBBORN] = 0.0
    world = FJWorld(innate, graph, torch.full((N,), 0.5), platform_sus=plat_sus)
    world.reset(seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    models, opts = make_platforms(innate)
    n_p = len(models)
    if os.environ.get("WARM", "0") == "1":
        with torch.no_grad():
            f0 = torch.stack([m(innate.unsqueeze(-1)).reshape(-1) for m in models])
        shares = torch.softmax(-(f0 - innate.unsqueeze(0)).abs().t() / TAU, dim=1)
    else:
        shares = torch.full((N, n_p), 1.0 / n_p)
    x = innate.clone()
    feat = innate.unsqueeze(-1)
    out = {k: [] for k in ("means", "gap", "div", "std", "conc")}
    for _ in range(ROUNDS):
        with torch.no_grad():
            preds = torch.stack([m(feat).reshape(-1) for m in models])     # (P, N)
        assign = torch.multinomial(shares, 1, generator=gen).squeeze(1)
        data = world.run(FixedPredictions(preds[assign, torch.arange(N)]), n_steps=K_INNER)
        op = data["y"].reshape(-1)

        for p in range(n_p):
            served = (assign == p).nonzero().reshape(-1)
            rivals = torch.stack([preds[q] for q in range(n_p) if q != p])
            rival_w = torch.exp(-(rivals - op.unsqueeze(0)).abs() / TAU).sum(dim=0)
            others = (assign != p).nonzero().reshape(-1)
            for _ in range(STEPS):
                loss = torch.tensor(0.0)
                if served.numel():
                    ssel = served[torch.randperm(served.numel(), generator=gen)[:BATCH]]
                    f_s = models[p](feat[ssel]).reshape(-1)
                    loss = loss + ((f_s - op[ssel]) ** 2).mean()
                    own_s = torch.exp(-(f_s - op[ssel]).abs() / TAU)
                    keep = own_s / (own_s + rival_w[ssel])
                    loss = loss + MIX_LAMBDA * -keep.mean()
                if others.numel():
                    osel = others[torch.randperm(others.numel(), generator=gen)[:BATCH]]
                    f_o = models[p](feat[osel]).reshape(-1)
                    own_o = torch.exp(-(f_o - op[osel]).abs() / TAU)
                    grab = own_o / (own_o + rival_w[osel])
                    loss = loss + MIX_LAMBDA * LAMBDA_ACQ * -grab.mean()
                if ANCHOR_COST > 0:
                    asel = torch.randperm(N, generator=gen)[:BATCH]
                    fa = models[p](feat[asel]).reshape(-1)
                    loss = loss + ANCHOR_COST * ((fa - ANCHORS[p]) ** 2).mean()
                opts[p].zero_grad()
                loss.backward()
                opts[p].step()

        agree = -(preds - op.unsqueeze(0)).abs().t()
        shares = shares * torch.exp(ETA_MOB * (agree - agree.max(dim=1, keepdim=True).values))
        shares = shares.clamp_min(1e-3)
        shares = shares / shares.sum(dim=1, keepdim=True)

        means = preds.mean(dim=1)
        div = sum(float((preds[a] - preds[b]).abs().mean())
                  for a in range(n_p) for b in range(n_p) if a != b) / (n_p * (n_p - 1))
        out["means"].append([float(v) for v in means])
        out["gap"].append(float(means.max() - means.min()))
        out["div"].append(div)
        out["std"].append(float(op.std()))
        out["conc"].append(float(shares.max(dim=1).values.mean()))
    out["innate_std"] = float(innate.std())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [run_market(s) for s in SEEDS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("#d8633b", "#3bb0a0", "#3b6fd8")
    ax = axes[0, 0]
    traj = torch.tensor(runs[0]["means"])
    for p in range(3):
        ax.plot(traj[:, p], color=colors[p], label=f"platform {p + 1}")
    for a in ANCHORS:
        ax.axhline(a, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_title("platform mean positions (seed 0, dashed = initial anchors)")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    panels = (("gap", "position gap"), ("std", "population std"),
              ("conc", "trust concentration"))
    for ax, (key, title) in zip((axes[0, 1], axes[1, 0], axes[1, 1]), panels):
        curves = torch.tensor([r[key] for r in runs])
        ax.plot(curves.mean(dim=0), color="#3b6fd8")
        ax.fill_between(range(ROUNDS), curves.min(dim=0).values,
                        curves.max(dim=0).values, color="#3b6fd8", alpha=0.15)
        if key == "std":
            ax.axhline(runs[0]["innate_std"], ls=":", color="black", lw=1)
        ax.set_title(title)
        ax.set_xlabel("round")

    fig.suptitle("Three mixed MLP platforms, MW trust, bimodal FJ population",
                 fontweight="bold")
    fig.savefig(OUT / "mw_mlps.png", dpi=140)

    for key in ("gap", "div", "std", "conc"):
        vals = [r[key] for r in runs]
        first = sum(v[0] for v in vals) / len(vals)
        last = sum(v[-1] for v in vals) / len(vals)
        print(f"{key:>5s}: {first:.3f} -> {last:.3f}")
    print(f"final means (seed 0): {[round(v, 3) for v in runs[0]['means'][-1]]}")
    print(f"[fj] figure -> {OUT / 'mw_mlps.png'}  (seeds {SEEDS})")


if __name__ == "__main__":
    main()
