"""Pure MLP hunters (share-gradient) over the bimodal FJ population. Each
platform maximizes expected capture vs fixed rivals; positions start anchored.
Saves trajectory.pt (preds_raw [T,P,N], op_raw, innate) for analyze_matrix.py.
Run: python experiments/fj/run_mlp_hunters.py
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
from _pokec import FixedPredictions

OUT = Path("runs/mlp_hunters")
SEEDS = (0, 1, 2)
N = 200
ROUNDS = 80
K_INNER = 10
ANCHORS = (-0.8, 0.0, 0.8)
STEPS = 8
BATCH = 64
LR = 1e-3
BETA_TRUST = 0.6


def make_platforms(innate, seed):
    models, opts = [], []
    x = innate.unsqueeze(-1)
    for p, anchor in enumerate(ANCHORS):
        m = MLPModel(1, (32, 32), init_seed=seed * 10 + p)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        tgt = torch.full((N, 1), anchor)
        for _ in range(200):
            opt.zero_grad()
            ((m(x) - tgt) ** 2).mean().backward()
            opt.step()
        models.append(m)
        opts.append(opt)
    return models, opts


def run_market(tag, tau, mobility, eta_mob, seed):
    innate, graph, community = make_population(seed)
    world = FJWorld(innate, graph, torch.full((N,), 0.5), platform_sus=BETA_TRUST)
    world.reset(seed=seed)
    gen = torch.Generator(); gen.manual_seed(seed + 1000)
    models, opts = make_platforms(innate, seed)
    n_p = len(models)
    shares = torch.full((N, n_p), 1.0 / n_p)
    feat = innate.unsqueeze(-1)
    preds_raw, op_raw = [], []
    out = {k: [] for k in ("gap", "div", "std")}
    for _ in range(ROUNDS):
        with torch.no_grad():
            preds = torch.stack([m(feat).reshape(-1) for m in models])
        if mobility == "mw":
            assign = torch.multinomial(shares, 1, generator=gen).squeeze(1)
        else:
            probs = torch.softmax(-(preds - world.state["opinion"].float().unsqueeze(0)).abs().t() / tau, dim=1)
            assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        data = world.run(FixedPredictions(preds[assign, torch.arange(N)]), n_steps=K_INNER)
        op = data["y"].reshape(-1)

        for p in range(n_p):
            rivals = torch.stack([preds[q] for q in range(n_p) if q != p])
            rival_w = torch.exp(-(rivals - op.unsqueeze(0)).abs() / tau).sum(dim=0)
            for _ in range(STEPS):
                sel = torch.randperm(N, generator=gen)[:BATCH]
                f = models[p](feat[sel]).reshape(-1)
                own = torch.exp(-(f - op[sel]).abs() / tau)
                capture = own / (own + rival_w[sel])
                opts[p].zero_grad()
                (-capture.mean()).backward()
                opts[p].step()

        if mobility == "mw":
            agree = -(preds - op.unsqueeze(0)).abs().t()
            shares = shares * torch.exp(eta_mob * (agree - agree.max(dim=1, keepdim=True).values))
            shares = (shares.clamp_min(1e-3))
            shares = shares / shares.sum(dim=1, keepdim=True)

        means = preds.mean(dim=1)
        div = sum(float((preds[a] - preds[b]).abs().mean())
                  for a in range(n_p) for b in range(n_p) if a != b) / (n_p * (n_p - 1))
        out["gap"].append(float(means.max() - means.min()))
        out["div"].append(div)
        out["std"].append(float(op.std()))
        preds_raw.append(preds)
        op_raw.append(op)

    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"preds_raw": torch.stack(preds_raw), "op_raw": torch.stack(op_raw),
                "innate": innate, "trajectory": [], "config": {"tag": tag, "tau": tau,
                "mobility": mobility, "eta_mob": eta_mob, "seed": seed}},
               d / "trajectory.pt")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [
        ("memoryless_t05", 0.10, "memoryless", 0.0),
        ("memoryless_t01", 0.02, "memoryless", 0.0),
        ("mw_t05", 0.10, "mw", 0.5),
        ("mw_t01", 0.02, "mw", 0.5),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    colors = plt.cm.tab10.colors
    print(f"{'cell':>16s} | {'gap f->l':>14s} {'div f->l':>14s} {'std f->l':>14s}")
    for i, (tag, tau, mob, eta) in enumerate(cells):
        runs = [run_market(f"{tag}_s{s}", tau, mob, eta, s) for s in SEEDS]
        for ax, key in zip(axes, ("gap", "div", "std")):
            curves = torch.tensor([r[key] for r in runs])
            ax.plot(curves.mean(dim=0), color=colors[i], label=tag)
        m = lambda k: (sum(r[k][0] for r in runs) / len(runs), sum(r[k][-1] for r in runs) / len(runs))
        g0, g1 = m("gap"); d0, d1 = m("div"); s0, s1 = m("std")
        print(f"{tag:>16s} | {g0:.3f}->{g1:.3f}   {d0:.3f}->{d1:.3f}   {s0:.3f}->{s1:.3f}")
    for ax, t in zip(axes, ("position gap", "inter-platform div", "population std")):
        ax.set_title(t); ax.set_xlabel("round")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle("MLP hunters over bimodal FJ population (mean of seeds)", fontweight="bold")
    fig.savefig(OUT / "summary.png", dpi=150)
    print(f"[mlp_hunters] figure -> {OUT / 'summary.png'}")


if __name__ == "__main__":
    main()
