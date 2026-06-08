"""MW-population sweep on Pokec: the population does multiplicative-weight
updating over platforms (each user keeps a persistent share vector, reweighted
by agreement). Sweep the MW rate eta_mob to ask whether fast lock-in can beat
the hunters' merge force on the unimodal density. Saves trajectory.pt per cell.
Run: python experiments/fj/run_mlp_hunters_pokec_mw.py
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

OUT = Path("runs/mlp_hunters_pokec_mw")
SEEDS = (0, 1, 2)
ROUNDS = 80
K_INNER = 10
ANCHOR_Q = (0.1, 0.5, 0.9)
STEPS = 8
BATCH = 64
LR = 1e-3
TAU = 0.05
ETAS = (0.5, 2.0, 5.0, 10.0, 20.0)


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


def run_market(tag, eta_mob, seed):
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
    out = {k: [] for k in ("gap", "div", "std", "conc")}
    for _ in range(ROUNDS):
        with torch.no_grad():
            preds = torch.stack([m(feat).reshape(-1) for m in models])
        assign = torch.multinomial(shares, 1, generator=gen).squeeze(1)
        data = world.run(FixedPredictions(preds[assign, torch.arange(n)]), n_steps=K_INNER)
        op = data["y"].reshape(-1)

        for p in range(n_p):
            rivals = torch.stack([preds[q] for q in range(n_p) if q != p])
            rival_w = torch.exp(-(rivals - op.unsqueeze(0)).abs() / TAU).sum(dim=0)
            for _ in range(STEPS):
                sel = torch.randperm(n, generator=gen)[:BATCH]
                f = models[p](feat[sel]).reshape(-1)
                own = torch.exp(-(f - op[sel]).abs() / TAU)
                capture = own / (own + rival_w[sel])
                opts[p].zero_grad()
                (-capture.mean()).backward()
                opts[p].step()

        agree = -(preds - op.unsqueeze(0)).abs().t()
        shares = shares * torch.exp(eta_mob * (agree - agree.max(dim=1, keepdim=True).values))
        shares = shares.clamp_min(1e-3)
        shares = shares / shares.sum(dim=1, keepdim=True)

        means = preds.mean(dim=1)
        div = sum(float((preds[a] - preds[b]).abs().mean())
                  for a in range(n_p) for b in range(n_p) if a != b) / (n_p * (n_p - 1))
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
                "eta_mob": eta_mob, "seed": seed}}, d / "trajectory.pt")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    colors = plt.cm.viridis(torch.linspace(0, 0.9, len(ETAS)).tolist())
    print(f"{'eta_mob':>8s} | {'div f->l':>14s} {'gap f->l':>14s} "
          f"{'std f->l':>14s} {'conc f->l':>14s}")
    for i, eta in enumerate(ETAS):
        runs = [run_market(f"eta{eta:g}_s{s}", eta, s) for s in SEEDS]
        for ax, key in zip(axes, ("div", "gap", "std", "conc")):
            curves = torch.tensor([r[key] for r in runs])
            ax.plot(curves.mean(dim=0), color=colors[i], label=f"eta={eta:g}")
        m = lambda k: (sum(r[k][0] for r in runs) / len(runs), sum(r[k][-1] for r in runs) / len(runs))
        d0, d1 = m("div"); g0, g1 = m("gap"); s0, s1 = m("std"); c0, c1 = m("conc")
        print(f"{eta:>8g} | {d0:.3f}->{d1:.3f}   {g0:.3f}->{g1:.3f}   "
              f"{s0:.3f}->{s1:.3f}   {c0:.3f}->{c1:.3f}")
    titles = ("inter-platform div", "position gap", "population std",
              "lock-in (mean max share)")
    for ax, t in zip(axes, titles):
        ax.set_title(t); ax.set_xlabel("round")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle("Pokec with MW population updating: does fast lock-in beat the merge?",
                 fontweight="bold")
    fig.savefig(OUT / "summary.png", dpi=150)
    print(f"[mlp_hunters_pokec_mw] figure -> {OUT / 'summary.png'}")


if __name__ == "__main__":
    main()
