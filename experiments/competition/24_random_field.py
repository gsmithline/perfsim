"""Block 24: control stack for the MMHD loop (local MLP, no LLM).

Decomposes the phenomenon into three separable mechanisms via five arms, an
alignment metric, and three environments.

arms (same gated AB population, same eps/gamma/seed):
  model     : MLP retrained on the population each round (the real loop).
  frozen    : MLP trained once (round 0), then NO retraining (static field).
  shuffled  : each round permute the model's predictions across agents -- keep the
              marginal prediction distribution, destroy person-level alignment.
  rand_model: random field matched to the model's per-round mean/variance.
  rand_pop  : random field matched to the current population's mean/std.

align = corr(current opinion_i, injected field_i): the Part-B Cov term made
explicit. High align preserves variance; ~0 align contracts it.

environments:
  preserve   : eps=0.3, informative features (model maps people to themselves).
  collapse   : eps=0.3, uninformative features (model can only predict the mode).
  unsat-gate : eps=0.1, informative (contact NOT saturated -> s should depend on values).

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/24_random_field.py
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
GAMMA = 1.5
P0 = 0.7
W = 0.3
ROUNDS = 60
TRAIN = 10
SEEDS = [0, 1, 2]
ARMS = ["model", "observer", "frozen", "shuffled", "rand_model", "rand_pop"]
ENVS = [("preserve", 0.3, 0.15), ("collapse", 0.3, 1.2), ("unsat-gate", 0.1, 0.15)]


def mlp():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())


def feats(x, rng, sf):
    noise = rng.normal(0, 1.0, (len(x), 2))
    return np.stack([x + rng.normal(0, sf, len(x)), noise[:, 0], noise[:, 1]], 1)


def base_net(rng, seed, sf):
    torch.manual_seed(seed)
    y = np.clip(rng.normal(P0, 0.18, 3000), 0.01, 0.99)
    F = torch.tensor(feats(y, rng, sf), dtype=torch.float32)
    T = torch.tensor(y, dtype=torch.float32)
    net = mlp(); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad(); ((net(F).squeeze(1) - T) ** 2).mean().backward(); opt.step()
    return net


def corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(arm, seed, eps, sf, moments=None):
    pop = b9.build_pop(eps, GAMMA, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    F = torch.tensor(feats(x0, rng, sf), dtype=torch.float32)
    y0 = torch.tensor(x0, dtype=torch.float32)
    net = base_net(rng, seed, sf); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    f = np.zeros(N); rec_mom, traj = [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if arm in ("model", "frozen", "shuffled", "observer"):
            if arm != "frozen" or t == 0:
                tgt = torch.tensor(x, dtype=torch.float32)
                for _ in range(TRAIN):
                    opt.zero_grad(); (((net(F).squeeze(1) - tgt) ** 2).mean()).backward(); opt.step()
            m = net(F).squeeze(1).detach().numpy()
            if arm == "shuffled":
                m = m[rng.permutation(N)]
        elif arm == "rand_model":
            mu, sd = moments[t]
            m = np.clip(rng.normal(mu, max(sd, 1e-6), N), 0.0, 1.0)
        else:
            m = np.clip(rng.normal(x.mean(), max(x.std(), 1e-6), N), 0.0, 1.0)
        rec_mom.append((float(m.mean()), float(m.std())))
        align = corr(x, m)
        g = np.abs(m - x) < eps
        if arm != "observer":                        # observer: model retrains but
            x = np.where(g, (1 - W) * x + W * m, x)   # its field is never fed back,
            f = np.where(g, (1 - W) * f + W * 1.0, f) # so the population runs free
            for i in range(N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
        has_net = arm in ("model", "frozen", "shuffled", "observer")
        lc0 = float(((net(F).squeeze(1).detach() - y0) ** 2).mean()) if has_net else float("nan")
        traj.append(dict(s=float(f.mean()), contact=float(g.mean()), align=align,
                         pop_std=float(x.std()), lc0=lc0))
    return traj, rec_mom


def model_vs_observer_fig(rows):
    # The counterfactual core: model (full loop) vs observer (retrain kept,
    # feedback cut). The gap between them is the AI's causal effect on each
    # object -- diversity on the left, model forgetting on the right.
    envs = [e for e, _, _ in ENVS]
    arms = ["observer", "model"]
    lab = {"observer": "observer (no feedback)", "model": "full loop"}
    col = {"observer": "#7f7f7f", "model": "#2ca02c"}

    def agg(env, arm, idx):
        v = [r[idx] for r in rows if r[0] == env and r[1] == arm]
        return float(np.mean(v)), float(np.std(v))

    fig, (axd, axl) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(envs)); wbar = 0.36
    for k, arm in enumerate(arms):
        off = (k - 0.5) * wbar
        dm = [agg(e, arm, 6) for e in envs]
        lm = [agg(e, arm, 7) for e in envs]
        axd.bar(x + off, [m for m, _ in dm], wbar, yerr=[s for _, s in dm],
                color=col[arm], label=lab[arm], capsize=3)
        axl.bar(x + off, [m for m, _ in lm], wbar, yerr=[s for _, s in lm],
                color=col[arm], label=lab[arm], capsize=3)
    axd.set(ylabel="diversity kept  (final / initial pop_std)", ylim=(0, 1.1),
            title="Population: the AI contracts diversity")
    axl.set(ylabel="model forgetting  $\\ell_{c0}$  (MSE vs round-0 opinions)",
            title="Model: feedback amplifies forgetting")
    for ax in (axd, axl):
        ax.set_xticks(x); ax.set_xticklabels(envs)
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model vs observer: the gap is the AI's causal effect "
                 "(model retrains in both; feedback cut in observer)", fontsize=12)
    fig.tight_layout()
    fig.savefig("experiments/competition/figs/model_vs_observer.png", dpi=140)
    print("saved experiments/competition/figs/model_vs_observer.png")


def main():
    rows = []   # (env, arm, seed, s, contact, align, divratio, lc0)
    for env, eps, sf in ENVS:
        for seed in SEEDS:
            tm, mom = run("model", seed, eps, sf)
            data = {"model": tm,
                    "observer": run("observer", seed, eps, sf)[0],
                    "frozen": run("frozen", seed, eps, sf)[0],
                    "shuffled": run("shuffled", seed, eps, sf)[0],
                    "rand_model": run("rand_model", seed, eps, sf, moments=mom)[0],
                    "rand_pop": run("rand_pop", seed, eps, sf)[0]}
            for a, tr in data.items():
                divr = tr[-1]["pop_std"] / max(tr[0]["pop_std"], 1e-9)
                align = np.mean([r["align"] for r in tr])
                rows.append((env, a, seed, tr[-1]["s"], tr[-1]["contact"],
                             align, divr, tr[-1]["lc0"]))

    for env, _, _ in ENVS:
        print(f"\n=== env: {env} ===")
        print(f"{'arm':<12}{'s':>7}{'contact':>9}{'align':>8}{'div_ratio':>11}{'l_c0':>7}")
        for a in ARMS:
            er = [r for r in rows if r[0] == env and r[1] == a]
            s, c, al, dr = (np.mean([r[i] for r in er]) for i in (3, 4, 5, 6))
            lc = np.mean([r[7] for r in er])
            lcs = f"{lc:>7.3f}" if not np.isnan(lc) else f"{'--':>7}"
            print(f"{a:<12}{s:>7.3f}{c:>9.3f}{al:>8.3f}{dr:>11.3f}{lcs}")

    col = {"model": "#2ca02c", "observer": "#7f7f7f", "frozen": "#1f77b4",
           "shuffled": "#d62728", "rand_model": "#ff7f0e", "rand_pop": "#9467bd"}
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for a in ARMS:
        er = [r for r in rows if r[1] == a]
        ax.scatter([r[5] for r in er], [r[6] for r in er], c=col[a], label=a, s=45, alpha=0.8)
    ax.set(xlabel="alignment  corr(opinion_i, prediction_i)  (mean over rounds)",
           ylabel="diversity kept  (final pop_std / initial)",
           title="Collapse is geometric: alignment preserves diversity, misalignment contracts it")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("experiments/competition/figs/random_field_alignment.png", dpi=130)
    print("\nsaved experiments/competition/figs/random_field_alignment.png")

    model_vs_observer_fig(rows)


if __name__ == "__main__":
    main()
