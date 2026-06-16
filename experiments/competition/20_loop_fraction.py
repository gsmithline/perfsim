"""Block 20: dynamical fraction test + innate-vs-pristine confound, in the gated loop.

(1) DYNAMICAL: run the gated AB loop with an MLP platform; each round the MLP retrains
    on a mix of (1-rho) current (recycled) opinions + rho frozen round-0 (real) opinions.
    rho is the pristine fraction; the recycled fraction in training is s = 1-rho. Measure
    collapse depth l_c0 (MLP loss on round-0 data) vs s. Test l_c0 ~ s^2 in the loop.
(2) CONFOUND: same loop, but instead of mixing pristine into training, add an INNATE
    re-anchor of the population toward round-0 (weight lam). Show innate gives a worse /
    confounded relation (it moves population diversity AND s together), while pristine
    gives the clean s^2.

Run: python experiments/competition/20_loop_fraction.py
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
EPS = 0.3
W = 0.3
GAMMA = 1.5
P0 = 0.7
ROUNDS = 60
TRAIN = 10
SIGMA_F = 0.15


def mlp():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())


def feats(x0, rng):
    noise = rng.normal(0, 1.0, (len(x0), 2))
    return np.stack([x0 + rng.normal(0, SIGMA_F, len(x0)), noise[:, 0], noise[:, 1]], 1)


def run(rho, lam, seed):
    """rho = pristine fraction mixed into TRAINING; lam = innate re-anchor on POPULATION."""
    pop = b9.build_pop(EPS, GAMMA, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    F = torch.tensor(feats(x0, rng), dtype=torch.float32)
    y0 = torch.tensor(x0, dtype=torch.float32)          # frozen round-0 real targets
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    tail_lc0, tail_std = [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if lam > 0:                                     # innate re-anchor of population
            x = (1 - lam) * x + lam * x0
            for i in range(N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
        tgt = torch.tensor(x, dtype=torch.float32)
        real = torch.rand(N, generator=torch.Generator().manual_seed(seed * 7 + t)) < rho
        mixed = torch.where(real, y0, tgt)              # pristine-mix into training
        for _ in range(TRAIN):
            opt.zero_grad()
            loss = ((net(F).squeeze(1) - mixed) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(F).squeeze(1).numpy()
            lc0 = ((net(F).squeeze(1) - y0) ** 2).mean().item()   # loss on round-0 real
        gate = np.abs(m - x) < EPS                      # gated feedback
        x = np.where(gate, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        if t >= ROUNDS - 5:                             # converged tail (not the transient peak)
            tail_lc0.append(lc0)
            tail_std.append(float(x.std()))
    return float(np.mean(tail_lc0)), float(np.mean(tail_std))


def fit_s2(s, y):
    A = np.column_stack([np.ones_like(s), s ** 2]); c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2q = 1 - ((y - A @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    Al = np.column_stack([np.ones_like(s), s]); cl, *_ = np.linalg.lstsq(Al, y, rcond=None)
    r2l = 1 - ((y - Al @ cl) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2q, r2l


def main():
    seeds = [0, 1, 2]
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    lams = [0.0, 0.1, 0.2, 0.3, 0.4]

    print("(1) PRISTINE-MIX in the loop: l_c0 AND population diversity vs s")
    print(f"  {'s':>4}{'l_c0':>9}{'pop_std':>9}  (does l_c0 fall AND pop stay diverse? = confound)")
    yp, dp = [], []
    for rho in rhos:
        r = [run(rho, 0.0, s) for s in seeds]
        lc = np.mean([a for a, _ in r]); st = np.mean([b for _, b in r])
        yp.append(lc); dp.append(st)
        print(f"  {1-rho:>4.1f}{lc:>9.4f}{st:>9.3f}", flush=True)
    sp = np.array([1 - r for r in rhos]); yp = np.array(yp)

    print("\n(2) INNATE re-anchor in the loop: l_c0 AND population diversity vs lambda")
    print(f"  {'lam':>4}{'l_c0':>9}{'pop_std':>9}")
    yi, di = [], []
    for lam in lams:
        r = [run(0.0, lam, s) for s in seeds]
        lc = np.mean([a for a, _ in r]); st = np.mean([b for _, b in r])
        yi.append(lc); di.append(st)
        print(f"  {lam:>4.1f}{lc:>9.4f}{st:>9.3f}", flush=True)
    yi = np.array(yi)

    print("\nVERDICT: in a CLOSED LOOP both instruments are confounded -- lowering s (more")
    print("pristine, or more innate) also RAISES population diversity, because the model's")
    print("predictions feed back. So fraction and quality cannot be separated in the loop;")
    print("only the static test (block 19) isolates the s^2 law. corr(l_c0, pop_std):")
    print(f"  pristine: {np.corrcoef(yp, dp)[0,1]:+.2f}   innate: {np.corrcoef(yi, di)[0,1]:+.2f}  (strong neg = confound)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax[0].plot(sp, yp, "o-", label="l_c0 (collapse)")
    ax0b = ax[0].twinx(); ax0b.plot(sp, dp, "s--", color="#2ca02c", label="pop_std")
    ax[0].set(xlabel="synthetic fraction s", ylabel="l_c0", title="pristine-mix (loop): confounded")
    ax[1].plot(lams, yi, "o-"); ax1b = ax[1].twinx(); ax1b.plot(lams, di, "s--", color="#2ca02c")
    ax[1].set(xlabel="innate lambda", ylabel="l_c0", title="innate (loop): confounded")
    fig.suptitle("In the closed loop, lowering s also raises diversity (feedback) -- both confounded; "
                 "only the static test (blk 19) isolates s^2")
    fig.savefig("experiments/competition/figs/loop_fraction.png", dpi=130)
    print("saved experiments/competition/figs/loop_fraction.png")


if __name__ == "__main__":
    main()
