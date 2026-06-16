"""Block 22: the 3-way phase cube as LAYERED SURFACES colored by l_c0.

Axes = the three knobs: eps (trust) x lam (innate) x anchor (beta-analogue).
For each anchor value we draw a flat surface sheet at its height on the anchor
axis, spanning the eps x lam plane, with color = l_c0 (collapse depth). Stacked,
the sheets fill the cube. Collapse (bright) sits at high eps + low lam + low anchor.

Features are deliberately UNINFORMATIVE (SF large): this is the faithful MLP
analogue of the LLM information ceiling (uninformative profile features), which is
what makes the model predict the mode and collapse. With informative features the
MLP cannot collapse and l_c0 is flat.

Run: python experiments/competition/22_phase_cube.py
"""

import importlib.util
import os
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_spec = importlib.util.spec_from_file_location("b9", "experiments/competition/09_ab_mlp_loop.py")
b9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b9)

N = 300
W = 0.3
GAMMA = 1.5
P0 = 0.7
ROUNDS = 60
TRAIN = 10
SF = 0.6                                   # uninformative features (LLM info-ceiling analogue)

EPS = np.linspace(0.1, 0.4, 6)
LAM = np.linspace(0.0, 0.3, 6)
ANCH = [0.0, 0.5, 1.0, 2.0]


def mlp():
    return nn.Sequential(nn.Linear(3, 24), nn.Tanh(), nn.Linear(24, 1), nn.Sigmoid())


def feats(x, rng):
    noise = rng.normal(0, 1.0, (len(x), 2))
    return np.stack([x + rng.normal(0, SF, len(x)), noise[:, 0], noise[:, 1]], 1)


def base_net(rng, seed):
    torch.manual_seed(seed)
    y = np.clip(rng.normal(P0, 0.18, 3000), 0.01, 0.99)
    F = torch.tensor(feats(y, rng), dtype=torch.float32)
    T = torch.tensor(y, dtype=torch.float32)
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        ((net(F).squeeze(1) - T) ** 2).mean().backward()
        opt.step()
    return net


def run(eps, lam, anchor, seed=0):
    pop = b9.build_pop(eps, GAMMA, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    F = torch.tensor(feats(x0, rng), dtype=torch.float32)
    y0 = torch.tensor(x0, dtype=torch.float32)
    base = base_net(rng, seed)
    bp = base(F).squeeze(1).detach()
    net = base_net(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    peak = 0.0
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if lam > 0:
            x = (1 - lam) * x + lam * x0
            for i in range(N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
        tgt = torch.tensor(x, dtype=torch.float32)
        for _ in range(TRAIN):
            opt.zero_grad()
            p = net(F).squeeze(1)
            (((p - tgt) ** 2).mean() + anchor * ((p - bp) ** 2).mean()).backward()
            opt.step()
        with torch.no_grad():
            m = net(F).squeeze(1).numpy()
            lc0 = ((net(F).squeeze(1) - y0) ** 2).mean().item()
        peak = max(peak, lc0)
        g = np.abs(m - x) < eps
        x = np.where(g, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
    return peak


def main():
    E, L = np.meshgrid(EPS, LAM)
    grids = {}
    for a in ANCH:
        grids[a] = np.array([[run(e, l, a) for e in EPS] for l in LAM])
        print(f"anchor={a} done  (l_c0 {grids[a].min():.3f}-{grids[a].max():.3f})", flush=True)

    vmin = min(g.min() for g in grids.values())
    vmax = max(g.max() for g in grids.values())
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.inferno

    fig = plt.figure(figsize=(9.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    for a in ANCH:
        Z = np.full_like(E, a)
        ax.plot_surface(E, L, Z, facecolors=cmap(norm(grids[a])),
                        rstride=1, cstride=1, shade=False, alpha=0.9, linewidth=0)
    ax.set(xlabel="eps (trust)", ylabel="innate lambda", zlabel="anchor (beta-analogue)")
    ax.set_title("3-way phase cube (layered surfaces): color = l_c0 (collapse depth)\n"
                 "collapse brightest at HIGH eps + LOW lambda + LOW anchor")
    ax.view_init(elev=18, azim=-122)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="l_c0 (collapse depth)", shrink=0.6, pad=0.12)
    fig.savefig("experiments/competition/figs/phase_cube.png", dpi=135)
    print("saved experiments/competition/figs/phase_cube.png")


if __name__ == "__main__":
    main()
