"""Diffusion-style snapshot figures: population scatter + 3D density surface.

Plane = (current opinion x_t, initial opinion x_0). At t=0 everything sits on
the diagonal; the platform drags mass horizontally toward its prior, stranded
holdouts stay on the diagonal. Top row: scatter at snapshot times. Bottom row:
smoothed 2D density rendered as a 3D surface (viridis), one column per time.

Cells: synthetic AB healing (eps=.3, gamma=1.5) with and without the platform,
and PolitiSky24 partisan capture (P0=.8) at eps=.2 (flip) vs eps=.1 (strand).

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/fig_diffusion_3d.py
"""

import importlib.util
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b9 = _load("b9", "experiments/competition/09_ab_mlp_loop.py")

SNAPS = [0, 5, 15, 50, 300]
P0_PART = 0.8


def run_ab_recorded(eps, gamma, w, seed=0):
    pop = b9.build_pop(eps, gamma, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(b9.N)])
    feats = torch.tensor(b9.make_features(x0, rng), dtype=torch.float32)
    net = b9.pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    snaps = {0: x0.copy()}
    for t in range(1, max(SNAPS) + 1):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(b9.N)])
        target = torch.tensor(x, dtype=torch.float32)
        for _ in range(b9.TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() \
                + b9.ANCHOR_W * ((pred - b9.P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        if w > 0:
            gate = np.abs(m - x) < eps
            x = np.where(gate, (1 - w) * x + w * m, x)
            for i in range(b9.N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
        if t in SNAPS:
            snaps[t] = x.copy()
    return x0, snaps


def run_politisky_recorded(p0, eps, gamma, w, seed=0):
    b15 = _load("b15", "experiments/competition/15_politisky.py")
    rng = np.random.default_rng(seed)
    x0 = b15.X0.copy()
    pop = b15.build_pop(eps, gamma, seed, x0)
    feats = torch.tensor(b15.make_features(x0, rng), dtype=torch.float32)
    net = b15.pretrain_base(rng, seed, p0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    snaps = {0: x0.copy()}
    for t in range(1, max(SNAPS) + 1):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(b15.N)])
        target = torch.tensor(x, dtype=torch.float32)
        for _ in range(b15.TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() \
                + b15.ANCHOR_W * ((pred - p0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        if w > 0:
            gate = np.abs(m - x) < eps
            x = np.where(gate, (1 - w) * x + w * m, x)
            b15.write_opinions(pop, x)
        if t in SNAPS:
            snaps[t] = x.copy()
    return x0, snaps


def density(xt, x0, bins=60, sigma=1.8):
    h, _, _ = np.histogram2d(xt, x0, bins=bins, range=[[0, 1], [0, 1]])
    return gaussian_filter(h.T / len(xt), sigma)


def render(x0, snaps, title, prior, fname):
    cols = len(SNAPS)
    fig = plt.figure(figsize=(3.1 * cols, 6.2))
    for j, t in enumerate(SNAPS):
        xt = snaps[t]
        ax = fig.add_subplot(2, cols, j + 1)
        ax.scatter(xt, x0, s=2, alpha=0.25, c="#1f77b4", linewidths=0)
        ax.axvline(prior, c="#d62728", ls="--", lw=0.9)
        ax.plot([0, 1], [0, 1], c="gray", lw=0.6, alpha=0.5)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"t = {t}", xticks=[], yticks=[])
        if j == 0:
            ax.set_ylabel("initial opinion $x_0$")
        ax.set_xlabel("opinion $x_t$", fontsize=8)

        d = density(xt, x0)
        ax3 = fig.add_subplot(2, cols, cols + j + 1, projection="3d")
        g = np.linspace(0, 1, d.shape[0])
        gx, gy = np.meshgrid(g, g)
        ax3.plot_surface(gx, gy, d, cmap="viridis", rstride=1, cstride=1,
                         linewidth=0, antialiased=True)
        ax3.set(xticks=[], yticks=[], zticks=[])
        ax3.view_init(elev=35, azim=-60)
        ax3.set_axis_off()
        if j == 0:
            ax3.text2D(-0.02, 0.5, "$q(x_t)$", transform=ax3.transAxes,
                       fontsize=11, rotation=90, va="center")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"experiments/competition/figs/{fname}.png", dpi=130)
    plt.close(fig)
    print(f"saved experiments/competition/figs/{fname}.png", flush=True)


def main():
    x0, snaps = run_ab_recorded(0.3, 1.5, 0.0)
    render(x0, snaps, "AB gamma=1.5, no platform: fragmentation persists",
           b9.P0, "diff3d_ab_frozen")
    x0, snaps = run_ab_recorded(0.3, 1.5, 0.3)
    render(x0, snaps, "AB gamma=1.5, platform w=0.3: healed at the prior",
           b9.P0, "diff3d_ab_heal")
    x0, snaps = run_politisky_recorded(P0_PART, 0.2, 1.5, 0.3)
    render(x0, snaps, "PolitiSky24, partisan prior 0.8, eps=0.2: total flip",
           P0_PART, "diff3d_politisky_flip")
    x0, snaps = run_politisky_recorded(P0_PART, 0.1, 1.5, 0.3)
    render(x0, snaps, "PolitiSky24, partisan prior 0.8, eps=0.1: stranded camp",
           P0_PART, "diff3d_politisky_strand")


if __name__ == "__main__":
    main()
