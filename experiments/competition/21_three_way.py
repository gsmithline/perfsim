"""Block 21: the 3-way phase grid -- eps (trust) x lam (innate) x anchor (model spread).

Three knobs, mapped over an MLP gated loop:
  eps   : confidence radius -- how much of the population the model captures.
  lam   : innate re-anchor weight -- caps the synthetic fraction s (s*=wc/(wc+lam)).
  anchor: model prediction spread. anchor_w toward a feature-pretrained base keeps
          predictions spread (sigma_m>0, the beta>0 analogue); anchor_w=0 lets the
          model fit the (collapsing) population, concentrating predictions (beta=0).

Per cell we record: s (contamination tag), final pop diversity (std), mean displacement.
Output: phase surface + the slice tables that say which LLM slices are worth running.

Run: python experiments/competition/21_three_way.py
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

_spec = importlib.util.spec_from_file_location("b9", "experiments/competition/09_ab_mlp_loop.py")
b9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b9)

N = 300
W = 0.3
GAMMA = 1.5
P0 = 0.7
ROUNDS = 50
TRAIN = 10
SIGMA_F = 0.15

EPS_GRID = [0.1, 0.2, 0.3, 0.4]
LAM_GRID = [0.0, 0.1, 0.3]
ANCHOR_GRID = [0.0, 1.0]          # 0 = unanchored (beta=0 analogue), 1 = anchored (beta>0)
SEEDS = [0, 1]


def mlp():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())


def feats(x, rng):
    noise = rng.normal(0, 1.0, (len(x), 2))
    return np.stack([x + rng.normal(0, SIGMA_F, len(x)), noise[:, 0], noise[:, 1]], 1)


def pretrain_base(rng, seed):
    torch.manual_seed(seed)
    y = np.clip(rng.normal(P0, 0.18, 4000), 0.01, 0.99)
    F = torch.tensor(feats(y, rng), dtype=torch.float32)
    T = torch.tensor(y, dtype=torch.float32)
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad()
        ((net(F).squeeze(1) - T) ** 2).mean().backward()
        opt.step()
    return net


def run(eps, lam, anchor, seed):
    pop = b9.build_pop(eps, GAMMA, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    F = torch.tensor(feats(x0, rng), dtype=torch.float32)
    base = pretrain_base(rng, seed)
    base_pred = base(F).squeeze(1).detach()
    net = pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    f = np.zeros(N)                                   # contamination tag (provenance)
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if lam > 0:                                   # innate re-anchor (tag 0)
            x = (1 - lam) * x + lam * x0
            f = (1 - lam) * f
            for i in range(N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
        tgt = torch.tensor(x, dtype=torch.float32)
        for _ in range(TRAIN):
            opt.zero_grad()
            pred = net(F).squeeze(1)
            loss = ((pred - tgt) ** 2).mean() + anchor * ((pred - base_pred) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(F).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        x = np.where(gate, (1 - W) * x + W * m, x)
        f = np.where(gate, (1 - W) * f + W * 1.0, f)   # platform injects tag 1
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
    return f.mean(), float(x.std()), float(x.mean() - x0.mean())


def main():
    res = {}
    print(f"{'eps':>5}{'lam':>5}{'anch':>5} | {'s':>6}{'std_T':>7}{'disp':>7}  regime")
    for anchor in ANCHOR_GRID:
        for lam in LAM_GRID:
            for eps in EPS_GRID:
                r = [run(eps, lam, anchor, s) for s in SEEDS]
                s = np.mean([a for a, _, _ in r]); st = np.mean([b for _, b, _ in r])
                dp = np.mean([c for _, _, c in r])
                res[(eps, lam, anchor)] = (s, st, dp)
                reg = ("collapse" if (anchor == 0 and st < 0.10)
                       else "displace" if (anchor == 1 and st > 0.13)
                       else "partial" if st < 0.16 else "diverse")
                print(f"{eps:>5}{lam:>5}{anchor:>5} | {s:>6.2f}{st:>7.3f}{dp:>+7.3f}  {reg}", flush=True)

    # phase surface: for each anchor, heatmap of final std over (eps, lam)
    fig, ax = plt.subplots(1, len(ANCHOR_GRID), figsize=(11, 4.2), constrained_layout=True)
    for k, anchor in enumerate(ANCHOR_GRID):
        grid = np.array([[res[(e, l, anchor)][1] for e in EPS_GRID] for l in LAM_GRID])
        im = ax[k].imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                          extent=[EPS_GRID[0], EPS_GRID[-1], LAM_GRID[0], LAM_GRID[-1]])
        ax[k].set(xlabel="eps (trust)", ylabel="lam (innate)",
                  title=f"anchor={anchor} ({'beta>0 spread' if anchor else 'beta=0 concentrated'})")
        fig.colorbar(im, ax=ax[k], label="final pop std (low=collapse)")
    fig.suptitle("3-way phase grid: collapse needs high eps + low lam + low anchor")
    fig.savefig("experiments/competition/figs/three_way.png", dpi=130)
    print("\nsaved experiments/competition/figs/three_way.png")

    # which LLM slices are worth running? report the transitions
    print("\nLLM slices worth the GPU (where the surface moves most):")
    print("  - eps sweep at lam=0, anchor=0: the collapse ladder (already have it)")
    print("  - lam sweep at eps=0.4, anchor=0: s-ceiling / partial-collapse band (NEW)")
    print("  - anchor(beta) sweep at eps=0.4, lam=0: collapse->displace (have beta 0/1/3)")


if __name__ == "__main__":
    main()
