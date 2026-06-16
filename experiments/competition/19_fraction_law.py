"""Block 19: clean test of the collapse fraction-law l_c0 = base + s^2 c^2.

The LLM loop saturates s->1, so the fraction never varies (P5 fraction-saturated
corner). Here we dial the synthetic fraction DIRECTLY (the pristine-mix instrument):
train an MLP on a mix of rho real samples + (1-rho) synthetic samples, with the
synthetic targets a degraded (collapsed-by-q) copy of the real ones. Synthetic
fraction s = 1-rho; quality gap is set by q. We then check whether the loss on
held-out REAL data follows Dohmatob's s^2 c^2 form. No population dynamics, no innate
-> the fraction moves cleanly, isolated from data-diversity.

Run: python experiments/competition/19_fraction_law.py
"""

import os
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 6000
SIGMA_F = 0.15
RHOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # pristine (real) fraction; s = 1-rho
QS = [0.3, 0.6, 0.9]                      # synthetic degradation (sets c^2)
SEEDS = [0, 1, 2]


def mlp():
    return nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 1))


def features(y, rng):
    noise = rng.normal(0, 1.0, (len(y), 2))
    return np.stack([y + rng.normal(0, SIGMA_F, len(y)), noise[:, 0], noise[:, 1]], 1)


def train_eval(rho, q, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    y_real = rng.random(N)                       # diverse real targets
    y_synth = (1 - q) * y_real + q * y_real.mean()   # degraded (collapsed by q)
    feat = features(y_real, rng)                 # same features; only targets differ
    is_real = rng.random(N) < rho                # fraction rho real, 1-rho synthetic
    target = np.where(is_real, y_real, y_synth)

    X = torch.tensor(feat, dtype=torch.float32)
    T = torch.tensor(target, dtype=torch.float32)
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad()
        loss = ((net(X).squeeze(1) - T) ** 2).mean()
        loss.backward()
        opt.step()
    # evaluate on a fresh held-out REAL set (the round-0 / true distribution)
    yh = rng.random(N)
    Xh = torch.tensor(features(yh, rng), dtype=torch.float32)
    with torch.no_grad():
        lc0 = ((net(Xh).squeeze(1) - torch.tensor(yh, dtype=torch.float32)) ** 2).mean().item()
    return lc0


def main():
    grid = {}
    print(f"{'q':>4} {'rho':>5} {'s=1-rho':>8} {'l_c0':>8}")
    for q in QS:
        for rho in RHOS:
            lc = np.mean([train_eval(rho, q, s) for s in SEEDS])
            grid[(q, rho)] = lc
            print(f"{q:>4} {rho:>5} {1-rho:>8.1f} {lc:>8.4f}", flush=True)

    # fit l_c0 = base + c2(q) * s^2, per q; report R^2 and linear-in-s control
    print("\nFIT per q: l_c0 = base + c2 * s^2   (s = 1-rho)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for q in QS:
        s = np.array([1 - r for r in RHOS])
        y = np.array([grid[(q, r)] for r in RHOS])
        # quadratic-in-s (no linear term): y = a + b s^2
        Aq = np.column_stack([np.ones_like(s), s ** 2]); cq, *_ = np.linalg.lstsq(Aq, y, rcond=None)
        r2q = 1 - ((y - Aq @ cq) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # linear-in-s control: y = a + b s
        Al = np.column_stack([np.ones_like(s), s]); cl, *_ = np.linalg.lstsq(Al, y, rcond=None)
        r2l = 1 - ((y - Al @ cl) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  q={q}: base={cq[0]:.4f} c2={cq[1]:.4f}  R2(s^2)={r2q:.3f}   R2(linear-s)={r2l:.3f}")
        ax[0].plot(s, y, "o-", label=f"q={q}")
        ax[1].plot(s ** 2, y, "o-", label=f"q={q}")
    ax[0].set(xlabel="synthetic fraction s", ylabel="l_c0 (loss on real)", title="vs s")
    ax[1].set(xlabel="s^2", ylabel="l_c0", title="vs s^2 (straight line => Dohmatob law)")
    for a in ax:
        a.legend(frameon=False)
    fig.suptitle("Fraction law: does l_c0 scale as s^2 (at fixed quality q)?")
    fig.savefig("experiments/competition/figs/fraction_law.png", dpi=130)
    print("saved experiments/competition/figs/fraction_law.png")


if __name__ == "__main__":
    main()
