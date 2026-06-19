"""Block 25: the control stack (block 24) on PolitiSky24 real polarized stances.

Same 5 arms / alignment metric / 3 environments as block 24, but on the real
Bluesky 2024-election quote graph + stances from block 15 (Harris camp ~0.27,
Trump camp ~0.73, ~76% Harris-skewed, ~6% Trump). Tests whether the mechanism
decomposition holds on real, polarized, imbalanced opinion data:
  contamination structural (s set by gate/absorption, not field values),
  diversity geometric (div_ratio ~ alignment corr(opinion_i, pred_i)),
  forgetting retraining-driven (frozen preserves, continual contracts).
Adds one polarization-specific column: mass_t = the Trump-minority camp's
surviving mass (does the loop erase the minority?).

arms: model / frozen / shuffled / rand_model / rand_pop (see block 24).
Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/25_politisky_controls.py
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

_spec = importlib.util.spec_from_file_location("b15", "experiments/competition/15_politisky.py")
b15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b15)

N = b15.N
C_T = b15.C_T
GAMMA = 1.5
W = 0.3
ROUNDS = 60
TRAIN = 10
P0 = 0.5
SEEDS = [0, 1, 2]
ARMS = ["model", "frozen", "shuffled", "rand_model", "rand_pop"]
ENVS = [("preserve", 0.3, 0.15), ("collapse", 0.3, 1.2), ("unsat-gate", 0.1, 0.15)]


def mlp():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())


def feats(x, rng, sf):
    noise = rng.normal(0, 1.0, (len(x), 2))
    return np.stack([x + rng.normal(0, sf, len(x)), noise[:, 0], noise[:, 1]], 1)


def base_net(rng, seed, sf, p0=P0):
    torch.manual_seed(seed)
    y = np.clip(rng.normal(p0, 0.18, 4000), 0.01, 0.99)
    F = torch.tensor(feats(y, rng, sf), dtype=torch.float32)
    T = torch.tensor(y, dtype=torch.float32)
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        ((net(F).squeeze(1) - T) ** 2).mean().backward()
        opt.step()
    return net


def corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(arm, seed, eps, sf, moments=None):
    rng = np.random.default_rng(seed)
    x0 = b15.X0.copy()
    pop = b15.build_pop(eps, GAMMA, seed, x0)
    F = torch.tensor(feats(x0, rng, sf), dtype=torch.float32)
    y0 = torch.tensor(x0, dtype=torch.float32)
    net = base_net(rng, seed, sf)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    f = np.zeros(N)
    rec_mom, traj, op_seq, pred_seq = [], [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if arm in ("model", "frozen", "shuffled"):
            if arm != "frozen" or t == 0:
                tgt = torch.tensor(x, dtype=torch.float32)
                for _ in range(TRAIN):
                    opt.zero_grad()
                    (((net(F).squeeze(1) - tgt) ** 2).mean()).backward()
                    opt.step()
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
        x = np.where(g, (1 - W) * x + W * m, x)
        f = np.where(g, (1 - W) * f + W * 1.0, f)
        op_seq.append(x.copy())
        pred_seq.append(m.copy())
        b15.write_opinions(pop, x)
        has_net = arm in ("model", "frozen", "shuffled")
        lc0 = float(((net(F).squeeze(1).detach() - y0) ** 2).mean()) if has_net else float("nan")
        traj.append(dict(s=float(f.mean()), contact=float(g.mean()), align=align,
                         pop_std=float(x.std()), lc0=lc0,
                         mass_t=float((np.abs(x - C_T) < 0.15).mean())))
    return traj, rec_mom, np.array(op_seq, dtype=np.float32), np.array(pred_seq, dtype=np.float32)


def _ridge(ax, M, lo, hi, bins=70, step=2, marks=None):
    """Stacked per-round histograms (joyplot), early purple -> late yellow."""
    if marks is None:
        marks = (b15.C_H, C_T)
    T = M.shape[0]
    rounds = list(range(0, T, step))
    if T - 1 not in rounds:
        rounds.append(T - 1)
    dens = []
    for t in rounds:
        h, e = np.histogram(M[t], bins=bins, range=(lo, hi), density=True)
        dens.append((h, 0.5 * (e[:-1] + e[1:])))
    gmax = max((h.max() for h, _ in dens), default=1.0) or 1.0
    scale = 2.2 / gmax
    J = len(dens)
    for j, (h, cx) in enumerate(dens):
        ax.fill_between(cx, j, j + h * scale, color=plt.cm.viridis(j / max(J - 1, 1)),
                        alpha=0.85, lw=0.3, edgecolor="white", zorder=J - j)
    for c in marks:
        ax.axvline(c, color="k", ls=":", lw=0.9, zorder=J + 1)
    ax.set(xlim=(lo, hi))
    ax.set_yticks([])


def _loss_lines(ax, loss):
    """Mean per-agent loss over rounds, split by camp: how well the model can still
    reproduce each camp's original stance (rises = that camp is being forgotten)."""
    T = loss.shape[0]
    r = np.arange(T)
    ax.plot(r, loss[:, b15.CAMP_H].mean(1), color="#1f77b4", lw=1.9, label="Harris (majority)")
    ax.plot(r, loss[:, b15.CAMP_T].mean(1), color="#d62728", lw=1.9, label="Trump (minority)")
    ax.set(xlim=(0, T - 1), ylim=(0, None))
    ax.legend(fontsize=7, frameon=False)


def render_evolution(seed0, env, fname):
    """Opinion + prediction distributions + per-agent-loss map over generations."""
    arms = ["model", "frozen", "shuffled"]
    x0 = b15.X0
    fig, axes = plt.subplots(len(arms), 3, figsize=(14, 3.0 * len(arms)), squeeze=False)
    for i, a in enumerate(arms):
        op, pr = seed0[(env, a)]
        loss = (pr - x0[None, :]) ** 2   # per-agent forgetting: pred vs original opinion
        _ridge(axes[i][0], op, 0.0, 1.0)
        _ridge(axes[i][1], pr, 0.0, 1.0)
        _loss_lines(axes[i][2], loss)
        axes[i][0].set_ylabel(f"{a}\nround ->")
        axes[i][2].set_ylabel("mean loss")
        if i == 0:
            axes[i][0].set_title(f"{env}: opinion distribution")
            axes[i][1].set_title(f"{env}: prediction distribution")
            axes[i][2].set_title(f"{env}: forgetting by camp")
        if i == len(arms) - 1:
            axes[i][0].set_xlabel("value")
            axes[i][1].set_xlabel("value")
            axes[i][2].set_xlabel("round")
    fig.suptitle("PolitiSky24 controls over generations (dotted = Harris/Trump camp centers)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print("saved", fname)


def main():
    smoke = os.environ.get("SMOKE")
    traj_mode = os.environ.get("TRAJ")
    seeds = [0] if (smoke or traj_mode) else SEEDS
    envs = ENVS[:1] if smoke else ENVS
    rows = []
    seed0 = {}
    for env, eps, sf in envs:
        for seed in seeds:
            tr_m, mom, op_m, pr_m = run("model", seed, eps, sf)
            arm_traj = {"model": tr_m}
            if seed == seeds[0]:
                seed0[(env, "model")] = (op_m, pr_m)
            for a in ["frozen", "shuffled", "rand_model", "rand_pop"]:
                mm = mom if a == "rand_model" else None
                tr, _, op, pr = run(a, seed, eps, sf, moments=mm)
                arm_traj[a] = tr
                if seed == seeds[0]:
                    seed0[(env, a)] = (op, pr)
            for a, tr in arm_traj.items():
                divr = tr[-1]["pop_std"] / max(tr[0]["pop_std"], 1e-9)
                align = np.mean([r["align"] for r in tr])
                rows.append((env, a, seed, tr[-1]["s"], tr[-1]["contact"],
                             align, divr, tr[-1]["lc0"], tr[-1]["mass_t"]))

    m0 = float((np.abs(b15.X0 - C_T) < 0.15).mean())
    print(f"\nN={N}  initial Trump-camp mass={m0:.3f}  seeds={seeds}  (mass_t = surviving minority)")
    for env, _, _ in envs:
        print(f"\n=== env: {env} ===")
        print(f"{'arm':<12}{'s':>7}{'contact':>9}{'align':>8}{'div_ratio':>11}{'l_c0':>7}{'mass_t':>8}")
        for a in ARMS:
            er = [r for r in rows if r[0] == env and r[1] == a]
            s, c, al, dr = (np.mean([r[i] for r in er]) for i in (3, 4, 5, 6))
            lc = np.mean([r[7] for r in er])
            mt = np.mean([r[8] for r in er])
            lcs = f"{lc:>7.3f}" if not np.isnan(lc) else f"{'--':>7}"
            print(f"{a:<12}{s:>7.3f}{c:>9.3f}{al:>8.3f}{dr:>11.3f}{lcs}{mt:>8.3f}")

    col = {"model": "#2ca02c", "frozen": "#1f77b4", "shuffled": "#d62728",
           "rand_model": "#ff7f0e", "rand_pop": "#9467bd"}
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for a in ARMS:
        er = [r for r in rows if r[1] == a]
        ax.scatter([r[5] for r in er], [r[6] for r in er], c=col[a], label=a, s=45, alpha=0.8)
    ax.set(xlabel="alignment  corr(opinion_i, prediction_i)  (mean over rounds)",
           ylabel="diversity kept  (final pop_std / initial)",
           title="PolitiSky24 controls: alignment preserves diversity, misalignment contracts it")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("experiments/competition/figs/politisky_controls.png", dpi=130)
    print("\nsaved experiments/competition/figs/politisky_controls.png")

    for env, _, _ in envs:
        render_evolution(seed0, env, f"experiments/competition/figs/politisky_controls_dist_{env}.png")


if __name__ == "__main__":
    main()
