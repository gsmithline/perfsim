"""Counterfactual control stack on REAL ML-Action data (not the synthetic toy).

Ports block 24's arms + observer onto the actual MovieLens-Action loop:
real 723-user LCC, real genre features (cross-fitted ridge R2 ~ 0.76), real
innate opinions (mean Action rating, mean 0.63), the real cosine-kNN graph,
and the same Deffuant + gated blend. Predictor is an MLP standing in for the
LLM. This is the on-dataset version of experiments/competition/24_random_field.py.

Six arms (same gated AB population, same eps/gamma/seed):
  model      MLP retrained each round on the current population (full loop)
  observer   retrained each round BUT predictions never fed back (open loop) <- the baseline
  frozen     trained once at round 0, then static (kills the retraining loop)
  shuffled   predictions permuted across agents each round (kills alignment)
  rand_model random field matched to the model's per-round mean/var
  rand_pop   random field matched to the population's mean/std

Two envs vary the AI gate width (reach): wide (eps_ai 0.4) and narrow
(eps_ai 0.1 = reach-limited). Population Deffuant eps fixed at 0.10 (gamma 0):
the campaign's 0.4 makes the population self-collapse before the AI acts, so
it is narrowed to a bound where the natural population holds its diversity
(retention 0.72) and the AI's marginal effect is visible.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/26_ml_action_controls.py
"""
import copy
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location("rm", "experiments/real_mlp.py")
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)

W = 0.3
# Deffuant confidence bound. At the campaign's 0.4 the population self-collapses
# on its own (natural div_ratio 0.22, eps wide vs innate std 0.137), which hides
# the AI's marginal effect. Narrowed to 0.10 (natural retention 0.72) so the
# population holds its diversity and the AI's contraction becomes visible.
POP_EPS = 0.10
GAMMA = 0.0
ROUNDS = 30
BASE_STEPS = 300
TRAIN = 20
LR = 1e-2
CAP = 4000          # subsample cap on the accumulate training set (matches real_mlp)
SEEDS = [0, 1, 2]
ARMS = ["model", "observer", "frozen", "shuffled", "rand_model", "rand_pop"]
ENVS = [("wide-gate", 0.4), ("narrow-gate", 0.1)]   # (name, gate_eps)


def mlp(d):
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())


def base_net(feat, x0, seed):
    torch.manual_seed(seed)
    net = mlp(feat.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    F = torch.tensor(feat, dtype=torch.float32)
    T = torch.tensor(x0, dtype=torch.float32)
    for _ in range(BASE_STEPS):
        opt.zero_grad(); ((net(F).squeeze(1) - T) ** 2).mean().backward(); opt.step()
    return net


def corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(arm, ds, gate_eps, seed, base, regime="replace", weight="continual", moments=None):
    G, feat, x0 = ds["G"], ds["feat"], ds["x0"]
    N = len(x0)
    pop = rm.build_pop(G, POP_EPS, GAMMA, x0, seed)
    rng = np.random.default_rng(seed)
    F = torch.tensor(feat, dtype=torch.float32)
    y0 = torch.tensor(x0, dtype=torch.float32)
    net = copy.deepcopy(base)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    keep = rng.permutation(N)[: N // 2]
    feat_keep, x0_keep = feat[keep], x0[keep]      # held-out round-0 real data (pristine)
    HXf, Hy = [], []
    f = np.zeros(N)
    traj, rec_mom = [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        if arm in ("model", "frozen", "shuffled", "observer"):
            if regime == "replace":                # data axis: newest round only
                trf, ty = feat, x
            elif regime == "accumulate":           # all rounds so far
                HXf.append(feat); Hy.append(x.copy())
                trf, ty = np.vstack(HXf), np.concatenate(Hy)
            else:                                   # pristine: newest + kept round-0 real
                trf = np.vstack([feat, feat_keep]); ty = np.concatenate([x, x0_keep])
            if len(ty) > CAP:
                sel = rng.permutation(len(ty))[:CAP]
                trf, ty = trf[sel], ty[sel]
            if weight == "fresh":                  # weight axis: reinit from base each round
                net = copy.deepcopy(base)
                opt = torch.optim.Adam(net.parameters(), lr=LR)
            if arm != "frozen" or t == 0:
                Ftr = torch.tensor(trf, dtype=torch.float32)
                Ttr = torch.tensor(ty, dtype=torch.float32)
                for _ in range(TRAIN):
                    opt.zero_grad(); (((net(Ftr).squeeze(1) - Ttr) ** 2).mean()).backward(); opt.step()
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
        g = np.abs(m - x) < gate_eps
        if arm != "observer":                        # observer: model retrains but
            x = np.where(g, (1 - W) * x + W * m, x)   # its field is never fed back,
            f = np.where(g, (1 - W) * f + W * 1.0, f) # so the population runs free
            for i in range(N):
                pop.status[i] = float(x[i])
        has_net = arm in ("model", "frozen", "shuffled", "observer")
        lc0 = float(((net(F).squeeze(1).detach() - y0) ** 2).mean()) if has_net else float("nan")
        traj.append(dict(s=float(f.mean()), contact=float(g.mean()), align=align,
                         pop_std=float(x.std()), lc0=lc0))
    return traj, rec_mom


def model_vs_observer_fig(rows, r2):
    envs = [e for e, _ in ENVS]
    env_lab = {"wide-gate": "wide gate\n$\\epsilon_{\\mathrm{AI}}=0.4$",
               "narrow-gate": "narrow gate\n$\\epsilon_{\\mathrm{AI}}=0.1$"}
    arms = ["observer", "model"]
    lab = {"observer": "observer  (predictions withheld)", "model": "full loop"}
    face = {"observer": "#c2c7cf", "model": "#2f6f9f"}
    edge = {"observer": "#8b929c", "model": "#1d4c70"}

    def agg(env, arm, idx):
        v = [r[idx] for r in rows if r[0] == env and r[1] == arm]
        return float(np.mean(v)), float(np.std(v))

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.9,
        "xtick.labelsize": 10.5, "ytick.labelsize": 9.5,
    })
    fig, (axd, axl) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.085, right=0.985, wspace=0.30)
    x = np.arange(len(envs)); wbar = 0.34

    def draw(ax, idx, fmt, headroom):
        for k, arm in enumerate(arms):
            off = (k - 0.5) * wbar
            vals = [agg(e, arm, idx) for e in envs]
            ax.bar(x + off, [m for m, _ in vals], wbar,
                   yerr=[s for _, s in vals], capsize=3,
                   color=face[arm], edgecolor=edge[arm], linewidth=0.9,
                   error_kw=dict(ecolor="#333", lw=1.0), label=lab[arm], zorder=3)
            for xi, (m, s) in zip(x + off, vals):
                ax.text(xi, m + s + headroom * 0.03, fmt.format(m), ha="center",
                        va="bottom", fontsize=9, color="#222")
        ax.set_xticks(x); ax.set_xticklabels([env_lab[e] for e in envs])
        ax.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    draw(axd, 6, "{:.2f}", 1.0)
    axd.set_ylim(0, 1.05)
    axd.axhline(1.0, color="#c4c4c4", lw=0.9, ls=(0, (4, 3)), zorder=1)
    axd.set_ylabel("population diversity kept", fontsize=11.5)

    draw(axl, 7, "{:.3f}", 0.012)
    axl.set_ylim(0, 0.0145)
    axl.set_ylabel("model forgetting", fontsize=11.5)

    h, l = axd.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10.5,
               handlelength=1.3, columnspacing=2.4, bbox_to_anchor=(0.5, 0.905))
    fig.suptitle("Model-mediated feedback degrades both the population and the model",
                 fontsize=13.5, y=0.99)
    fig.savefig("experiments/competition/figs/ml_action_model_vs_observer.png",
                dpi=200, bbox_inches="tight")
    print("saved experiments/competition/figs/ml_action_model_vs_observer.png")


def regime_fig(reg_rows, r2):
    data_order = ["replace", "accumulate", "pristine"]
    wt_order = ["continual", "fresh"]
    arms = ["observer", "model"]
    lab = {"observer": "observer  (predictions withheld)", "model": "full loop"}
    face = {"observer": "#c2c7cf", "model": "#2f6f9f"}
    edge = {"observer": "#8b929c", "model": "#1d4c70"}

    def agg(data, wt, arm, idx):
        v = [r[idx] for r in reg_rows if r[0] == data and r[1] == wt and r[2] == arm]
        return (float(np.mean(v)), float(np.std(v))) if v else (0.0, 0.0)

    xpos, centers, cur = {}, {}, 0.0
    for wt in wt_order:
        start = cur
        for data in data_order:
            xpos[(wt, data)] = cur; cur += 1.0
        centers[wt] = (start + cur - 1.0) / 2.0
        cur += 0.9
    cells = [(wt, data) for wt in wt_order for data in data_order]
    div_x = (xpos[(wt_order[0], data_order[-1])] + xpos[(wt_order[1], data_order[0])]) / 2.0

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.9,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    })
    fig, (axd, axl) = plt.subplots(1, 2, figsize=(12.2, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.075, right=0.99, wspace=0.22)
    wbar = 0.36

    def draw(ax, idx):
        for k, arm in enumerate(arms):
            off = (k - 0.5) * wbar
            xs = [xpos[c] + off for c in cells]
            vals = [agg(data, wt, arm, idx) for (wt, data) in cells]
            ax.bar(xs, [m for m, _ in vals], wbar, yerr=[s for _, s in vals], capsize=2.5,
                   color=face[arm], edgecolor=edge[arm], linewidth=0.8,
                   error_kw=dict(ecolor="#333", lw=0.9), label=lab[arm], zorder=3)
        ax.set_xticks([xpos[c] for c in cells])
        ax.set_xticklabels([data for (wt, data) in cells])
        ax.grid(axis="y", color="#ececec", lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.axvline(div_x, color="#dddddd", lw=1.0, zorder=0)
        for wt in wt_order:
            ax.text(centers[wt], -0.17, wt, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=11, color="#333")

    draw(axd, 4)
    axd.set_ylim(0, 1.05)
    axd.axhline(1.0, color="#c4c4c4", lw=0.9, ls=(0, (4, 3)), zorder=1)
    axd.set_ylabel("population diversity kept", fontsize=11.5)

    draw(axl, 5)
    axl.set_ylim(bottom=0)
    axl.set_ylabel("model forgetting", fontsize=11.5)

    h, l = axd.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10.5,
               handlelength=1.3, columnspacing=2.4, bbox_to_anchor=(0.5, 0.905))
    fig.suptitle("The AI's effect across the six training regimes (wide gate)",
                 fontsize=13.5, y=0.99)
    fig.savefig("experiments/competition/figs/ml_action_regimes.png",
                dpi=200, bbox_inches="tight")
    print("saved experiments/competition/figs/ml_action_regimes.png")


def main():
    ds = rm.ml_dataset("Action")
    r2 = ds["R2"]
    print(f"ML-Action: N={len(ds['x0'])}  R2={r2:.3f}  innate mean={ds['x0'].mean():.3f}")
    rows = []   # (env, arm, seed, s, contact, align, divratio, lc0)
    for ename, gate_eps in ENVS:
        for seed in SEEDS:
            base = base_net(ds["feat"], ds["x0"], seed)
            tm, mom = run("model", ds, gate_eps, seed, base)
            data = {"model": tm,
                    "observer": run("observer", ds, gate_eps, seed, base)[0],
                    "frozen": run("frozen", ds, gate_eps, seed, base)[0],
                    "shuffled": run("shuffled", ds, gate_eps, seed, base)[0],
                    "rand_model": run("rand_model", ds, gate_eps, seed, base, moments=mom)[0],
                    "rand_pop": run("rand_pop", ds, gate_eps, seed, base)[0]}
            for a, tr in data.items():
                divr = tr[-1]["pop_std"] / max(tr[0]["pop_std"], 1e-9)
                align = np.mean([r["align"] for r in tr])
                rows.append((ename, a, seed, tr[-1]["s"], tr[-1]["contact"],
                             align, divr, tr[-1]["lc0"]))

    for ename, _ in ENVS:
        print(f"\n=== env: {ename} ===")
        print(f"{'arm':<12}{'s':>7}{'contact':>9}{'align':>8}{'div_ratio':>11}{'l_c0':>7}")
        for a in ARMS:
            er = [r for r in rows if r[0] == ename and r[1] == a]
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
           title=f"ML-Action (real, R²={r2:.2f}): alignment preserves diversity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("experiments/competition/figs/ml_action_controls.png", dpi=130)
    print("\nsaved experiments/competition/figs/ml_action_controls.png")

    os.makedirs("experiments/competition/results", exist_ok=True)
    json.dump({"r2": r2, "rows": rows},
              open("experiments/competition/results/ml_action_rows.json", "w"))
    model_vs_observer_fig(rows, r2)

    # regime grid: model vs observer across the 6 data x weight cells, wide gate
    bases = {s: base_net(ds["feat"], ds["x0"], s) for s in SEEDS}
    reg_rows = []
    print("\n=== regime grid (wide gate, div_ratio / l_c0) ===")
    print(f"{'regime':<22}{'observer':>18}{'full loop':>18}")
    for data in ["replace", "accumulate", "pristine"]:
        for wt in ["continual", "fresh"]:
            cell = {}
            for seed in SEEDS:
                for arm in ["observer", "model"]:
                    tr = run(arm, ds, 0.4, seed, bases[seed], regime=data, weight=wt)[0]
                    divr = tr[-1]["pop_std"] / max(tr[0]["pop_std"], 1e-9)
                    reg_rows.append((data, wt, arm, seed, divr, tr[-1]["lc0"]))
                    cell.setdefault(arm, []).append((divr, tr[-1]["lc0"]))
            o = np.mean(cell["observer"], 0); m = np.mean(cell["model"], 0)
            print(f"{data+'/'+wt:<22}{o[0]:>8.2f} /{o[1]:>7.3f}{m[0]:>8.2f} /{m[1]:>7.3f}")
    json.dump(reg_rows, open("experiments/competition/results/ml_action_regime_rows.json", "w"))
    regime_fig(reg_rows, r2)


if __name__ == "__main__":
    main()
