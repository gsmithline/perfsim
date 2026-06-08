"""Mixed-objective platforms (retain your served base + LAMBDA_ACQ step toward
acquiring rivals' users) with MW population updating, on the real Pokec graph.
Sweep LAMBDA_ACQ: does retention + lock-in hold separate bases on the unimodal
density, and where does acquisition pressure tip it back to a merge?
Run: python experiments/fj/run_mw_mlps_pokec.py
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
from _pokec import FixedPredictions, load_pokec

OUT = Path("runs/mw_mlps_pokec")
SEEDS = (0, 1, 2)
ROUNDS = 120
K_INNER = 10
ANCHOR_Q = (0.1, 0.5, 0.9)
TAU = 0.05
ETA_MOB = 5.0
MIX_LAMBDA = 1.0
STEPS = 8
BATCH = 64
LR = 1e-3
ACQS = (0.0, 0.1, 0.5, 1.0, 2.0)


def make_platforms(innate, anchors, n, seed):
    models, opts = [], []
    x = innate.unsqueeze(-1)
    for p, anchor in enumerate(anchors):
        m = MLPModel(1, (32, 32), init_seed=seed * 10 + p)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        tgt = torch.full((n, 1), float(anchor))
        for _ in range(200):
            opt.zero_grad()
            ((m(x) - tgt) ** 2).mean().backward()
            opt.step()
        models.append(m)
        opts.append(opt)
    return models, opts


def run_market(tag, lambda_acq, seed):
    setup = load_pokec()
    innate, W = setup["innate"], setup["W"]
    n = innate.shape[0]
    world = FJWorld(innate, W, setup["peer_sus"], platform_sus=setup["platform_sus"])
    world.reset(seed=seed)
    gen = torch.Generator(); gen.manual_seed(seed + 1000)
    anchors = torch.quantile(innate, torch.tensor(ANCHOR_Q))
    models, opts = make_platforms(innate, anchors, n, seed)
    n_p = len(models)
    shares = torch.full((n, n_p), 1.0 / n_p)
    feat = innate.unsqueeze(-1)
    preds_raw, op_raw = [], []
    out = {k: [] for k in ("means", "gap", "div", "std", "conc")}
    for _ in range(ROUNDS):
        with torch.no_grad():
            preds = torch.stack([m(feat).reshape(-1) for m in models])
        assign = torch.multinomial(shares, 1, generator=gen).squeeze(1)
        data = world.run(FixedPredictions(preds[assign, torch.arange(n)]), n_steps=K_INNER)
        op = data["y"].reshape(-1)

        for p in range(n_p):
            served = (assign == p).nonzero().reshape(-1)
            others = (assign != p).nonzero().reshape(-1)
            rivals = torch.stack([preds[q] for q in range(n_p) if q != p])
            rival_w = torch.exp(-(rivals - op.unsqueeze(0)).abs() / TAU).sum(dim=0)
            for _ in range(STEPS):
                loss = torch.tensor(0.0)
                if served.numel():
                    ssel = served[torch.randperm(served.numel(), generator=gen)[:BATCH]]
                    f_s = models[p](feat[ssel]).reshape(-1)
                    loss = loss + ((f_s - op[ssel]) ** 2).mean()
                    own_s = torch.exp(-(f_s - op[ssel]).abs() / TAU)
                    keep = own_s / (own_s + rival_w[ssel])
                    loss = loss + MIX_LAMBDA * -keep.mean()
                if others.numel() and lambda_acq > 0:
                    osel = others[torch.randperm(others.numel(), generator=gen)[:BATCH]]
                    f_o = models[p](feat[osel]).reshape(-1)
                    own_o = torch.exp(-(f_o - op[osel]).abs() / TAU)
                    grab = own_o / (own_o + rival_w[osel])
                    loss = loss + MIX_LAMBDA * lambda_acq * -grab.mean()
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
        preds_raw.append(preds)
        op_raw.append(op)

    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"preds_raw": torch.stack(preds_raw), "op_raw": torch.stack(op_raw),
                "innate": innate, "trajectory": [], "config": {"tag": tag,
                "lambda_acq": lambda_acq, "seed": seed}}, d / "trajectory.pt")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    colors = plt.cm.viridis(torch.linspace(0, 0.9, len(ACQS)).tolist())
    print(f"{'acq':>6s} | {'div f->l':>14s} {'gap f->l':>14s} "
          f"{'std f->l':>14s} {'conc f->l':>14s}  final means (s0)")
    for i, acq in enumerate(ACQS):
        runs = [run_market(f"acq{acq:g}_s{s}", acq, s) for s in SEEDS]
        for ax, key in zip(axes, ("div", "gap", "std", "conc")):
            curves = torch.tensor([r[key] for r in runs])
            ax.plot(curves.mean(dim=0), color=colors[i], label=f"acq={acq:g}")
        m = lambda k: (sum(r[k][0] for r in runs) / len(runs), sum(r[k][-1] for r in runs) / len(runs))
        d0, d1 = m("div"); g0, g1 = m("gap"); s0, s1 = m("std"); c0, c1 = m("conc")
        fm = [round(v, 3) for v in runs[0]["means"][-1]]
        print(f"{acq:>6g} | {d0:.3f}->{d1:.3f}   {g0:.3f}->{g1:.3f}   "
              f"{s0:.3f}->{s1:.3f}   {c0:.3f}->{c1:.3f}  {fm}")
    titles = ("inter-platform div", "position gap", "population std",
              "lock-in (mean max share)")
    for ax, t in zip(axes, titles):
        ax.set_title(t); ax.set_xlabel("round")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle("Pokec, retention + acquisition with MW lock-in: where does merge tip?",
                 fontweight="bold")
    fig.savefig(OUT / "summary.png", dpi=150)
    print(f"[mw_mlps_pokec] figure -> {OUT / 'summary.png'}")


if __name__ == "__main__":
    main()
