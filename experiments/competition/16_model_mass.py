"""Block 16: model mass. s_t = measured fraction of training data that is
recycled model output, tracked by per-agent provenance f_i in [0,1].

Population is the block-9 AB sweep (Deffuant with bias) reimplemented in
torch, device-agnostic, so f can be carried through every pair interaction:
accepted pair averages f like it averages x; a platform blend sets the blended
share to pure model mass, f <- (1-w)f + w (model output counts as 100% model,
the collapse-literature convention). One sweep = exactly N biased pair
selections, executed as conflict-free batches of disjoint pairs.

Five analyses (gamma=1.5, w=0.3 unless noted): s_t vs eps; exposure clock
E_t = sum s_tau vs the direct self-consuming loop of block 11; defense pricing
at eps=0.3 (replace / mix-with-current "labeled human" / mix-with-round0);
the observable field proxy b_t (drift-vs-gap regression slope) as a
contamination detector; and the two corners (capture eps=0.4, forced
detachment anchor_w=2.0 eps=0.2).

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/16_model_mass.py
"""

import importlib.util
import json
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "ab_mlp_loop", "experiments/competition/09_ab_mlp_loop.py")
m9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m9)

N, P0, ANCHOR_W, TRAIN_STEPS = m9.N, m9.P0, m9.ANCHOR_W, m9.TRAIN_STEPS
GAMMA = 1.5
W = 0.3
ROUNDS = 300
EPS_GRID = [0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 1]
MIN_DIST = 1e-5
BATCH = 75
REG_WIN = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")


def sweep(x, f, eps, gamma):
    """Exactly N biased pair selections; disjoint pairs per batch."""
    ar = torch.arange(BATCH, device=DEVICE)
    done, accepted = 0, 0
    while done < N:
        ini = torch.randperm(N, device=DEVICE)[:BATCH]
        d = (x[ini, None] - x[None, :]).abs().clamp_min(MIN_DIST)
        wts = d.pow(-gamma)
        wts[ar, ini] = 0.0
        par = torch.multinomial(wts, 1).squeeze(1)
        # Luby-style conflict resolution: keep a pair iff it holds the min
        # random priority at both endpoints
        pri = torch.rand(BATCH, device=DEVICE)
        inc = torch.zeros(BATCH, N, dtype=torch.bool, device=DEVICE)
        inc[ar, ini] = True
        inc[ar, par] = True
        best = torch.where(inc, pri[:, None], torch.tensor(2.0, device=DEVICE)).amin(0)
        keep = (pri <= best[ini]) & (pri <= best[par])
        idx = keep.nonzero().squeeze(1)[: N - done]
        i1, i2 = ini[idx], par[idx]
        ok = (x[i1] - x[i2]).abs() < eps
        a1, a2 = i1[ok], i2[ok]
        mid = 0.5 * (x[a1] + x[a2])
        x[a1] = mid
        x[a2] = mid
        fm = 0.5 * (f[a1] + f[a2])
        f[a1] = fm
        f[a2] = fm
        done += len(idx)
        accepted += int(ok.sum())
    return accepted


def init_run(seed):
    torch.manual_seed(seed)
    x = torch.rand(N, device=DEVICE)
    f = torch.zeros(N, device=DEVICE)
    x0 = x.cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    feats = torch.tensor(m9.make_features(x0, rng), dtype=torch.float32)
    net = m9.pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    return x, f, x0, feats, net, opt


def run_loop(eps, seed, w=W, anchor_w=ANCHOR_W, regime="replace"):
    x, f, x0, feats, net, opt = init_run(seed)
    t0 = torch.tensor(x0, dtype=torch.float32)
    traj, pull = [], None
    for t in range(ROUNDS):
        acc = sweep(x, f, eps, GAMMA)
        x_np = x.cpu().numpy().astype(np.float64)
        cur = torch.tensor(x_np, dtype=torch.float32)
        for _ in range(TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            if regime == "replace":
                fit = ((pred - cur) ** 2).mean()
            elif regime == "mix_current":  # same content as replace; only the label differs
                fit = 0.5 * ((pred - cur) ** 2).mean() + 0.5 * ((pred - cur) ** 2).mean()
            else:  # mix_round0
                fit = 0.5 * ((pred - cur) ** 2).mean() + 0.5 * ((pred - t0) ** 2).mean()
            loss = fit + anchor_w * ((pred - P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        mt = torch.tensor(m, dtype=torch.float32, device=DEVICE)
        gate = (mt - x).abs() < eps
        if w > 0:
            x_new = torch.where(gate, (1 - w) * x + w * mt, x)
            if pull is None and bool(gate.any()):
                far = gate & ((mt - x).abs() > 1e-6)
                pull = float(((x_new - x) / (mt - x))[far].mean())
            x = x_new
            f = torch.where(gate, (1 - w) * f + w, f)
        s = float(f.mean())
        xb = x.cpu().numpy()
        traj.append({
            "pop_mean": float(xb.mean()), "pop_std": float(xb.std()),
            "s": s, "contact": float(gate.float().mean()),
            "pred_mean": float(m.mean()), "pred_std": float(m.std()),
            "accepted": acc, "clusters": m9.n_clusters(xb),
            "mix_contam": 0.5 * s if regime == "mix_round0" else s,
        })
    return traj, pull, x0


def run_direct(seed):
    """Block-11 condition A: model trains on its own previous predictions."""
    _, _, x0, feats, net, opt = init_run(seed)
    data = x0
    traj = []
    for _ in range(ROUNDS):
        target = torch.tensor(data, dtype=torch.float32)
        for _ in range(TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() + ANCHOR_W * ((pred - P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        data = m
        traj.append({"pred_mean": float(m.mean()), "pred_std": float(m.std()), "s": 1.0})
    return traj, x0


def run_pop_only(eps, gamma, seed):
    torch.manual_seed(seed)
    x = torch.rand(N, device=DEVICE)
    f = torch.zeros(N, device=DEVICE)
    for _ in range(ROUNDS):
        sweep(x, f, eps, gamma)
    return x.cpu().numpy()


def run_ndlib(eps, gamma, seed):
    pop = m9.build_pop(eps, gamma, seed)
    for _ in range(ROUNDS):
        pop.iteration(node_status=False)
    return np.array([pop.status[i] for i in range(N)])


def seed_avg(rows, field):
    return np.mean([[r[field] for r in t] for t in rows], 0)


def half_time(progress):
    idx = np.where(progress < 0.5)[0]
    return int(idx[0]) if len(idx) else None


def decay_rate(std, t_half, span=60):
    if t_half is None:
        return None
    seg = np.asarray(std[t_half:min(t_half + span, len(std))])
    tt = np.arange(len(seg))
    mask = seg > 1e-6
    if mask.sum() < 5:
        return None
    return float(-np.polyfit(tt[mask], np.log(seg[mask]), 1)[0])


def window_slopes(mean_arr, gap_arr, win=REG_WIN):
    dx = np.diff(mean_arr)
    b = np.full(len(dx), np.nan)
    for t in range(win, len(dx)):
        g, d = gap_arr[t - win:t], dx[t - win:t]
        if g.std() > 1e-6:
            b[t] = np.polyfit(g, d, 1)[0]
    return b


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def smooth(v, win=REG_WIN):
    return np.convolve(v, np.ones(win) / win, "same")


def main():
    print(f"device={DEVICE}, n={N}, gamma={GAMMA}, w={W}, P0={P0}, "
          f"anchor_w={ANCHOR_W}, {ROUNDS} rounds, seeds {SEEDS}")
    out = {"device": str(DEVICE)}

    # --- validation: torch sweep vs NDlib, no platform ---
    print("\nvalidation (w=0, eps=0.3): final pop_std / clusters, torch vs ndlib")
    val = {}
    for gamma in [1.5, 0.0]:
        for impl, fn in [("torch", lambda s, g=gamma: run_pop_only(0.3, g, s)),
                         ("ndlib", lambda s, g=gamma: run_ndlib(0.3, g, s))]:
            xs = [fn(s) for s in SEEDS]
            val[f"g{gamma}|{impl}"] = {
                "pop_std": [float(x.std()) for x in xs],
                "clusters": [m9.n_clusters(x) for x in xs]}
        for impl in ["torch", "ndlib"]:
            v = val[f"g{gamma}|{impl}"]
            print(f"  gamma={gamma:>4} {impl:>5}: std={v['pop_std'][0]:.3f}/"
                  f"{v['pop_std'][1]:.3f} clusters={v['clusters'][0]}/{v['clusters'][1]}",
                  flush=True)
    out["validation"] = val

    # --- runs ---
    med, x0s = {}, {}
    for eps in EPS_GRID:
        rows = []
        for s in SEEDS:
            traj, pull, x0 = run_loop(eps, s)
            rows.append(traj)
            x0s[(eps, s)] = x0
            if eps == EPS_GRID[0] and s == SEEDS[0]:
                out["pull_check"] = pull
        med[eps] = rows
        print(f"mediated eps={eps} done, final s="
              f"{np.mean([t[-1]['s'] for t in rows]):.3f}", flush=True)
    direct = []
    for s in SEEDS:
        traj, x0 = run_direct(s)
        direct.append(traj)
        x0s[("direct", s)] = x0
    defense = {"replace": med[0.3]}
    for regime in ["mix_current", "mix_round0"]:
        defense[regime] = [run_loop(0.3, s, regime=regime)[0] for s in SEEDS]
        print(f"defense {regime} done", flush=True)
    detach = [run_loop(0.2, s, anchor_w=2.0)[0] for s in SEEDS]
    print("detach corner done", flush=True)

    # --- 1: s_t curves ---
    print("\n[1] s_t per eps (final = mean last 5, seed avg)")
    a1 = {}
    for eps in EPS_GRID:
        sct = seed_avg(med[eps], "s")
        cont = seed_avg(med[eps], "contact")
        ds = np.diff(sct, prepend=0.0)
        # mechanistic prediction: ds ~ w * contact * (1 - s), so the raw
        # corr is saturation-confounded; report both
        a1[eps] = {"s_final": float(sct[-5:].mean()),
                   "contact_mean": float(cont.mean()),
                   "corr_ds_contact": corr(ds, cont),
                   "corr_ds_gated_headroom": corr(ds, cont * (1 - sct))}
        print(f"  eps={eps}: s_final={a1[eps]['s_final']:.3f} "
              f"mean_contact={a1[eps]['contact_mean']:.3f} "
              f"corr(ds_t, contact_t)={a1[eps]['corr_ds_contact']:.3f} "
              f"corr(ds_t, contact*(1-s))={a1[eps]['corr_ds_gated_headroom']:.3f}")
    fin_s = np.array([med[e][i][-1]["s"]
                      for e in EPS_GRID for i in range(len(SEEDS))])
    fin_c = np.array([np.mean([r["contact"] for r in med[e][i]])
                      for e in EPS_GRID for i in range(len(SEEDS))])
    a1["corr_finalS_contact_across_eps"] = corr(fin_s, fin_c)
    print(f"  across eps: corr(final s, mean contact)="
          f"{a1['corr_finalS_contact_across_eps']}")
    out["a1"] = a1

    # --- 2: exposure clock ---
    print("\n[2] exposure clock: E at half-collapse")
    a2 = {}
    curves = {}
    for eps in EPS_GRID:
        std0 = np.mean([x0s[(eps, s)].std() for s in SEEDS])
        prog = seed_avg(med[eps], "pop_std") / std0
        E = np.cumsum(seed_avg(med[eps], "s"))
        th = half_time(prog)
        a2[f"mediated_eps{eps}"] = {
            "t_half": th, "E_half": None if th is None else float(E[th])}
        curves[f"eps={eps}"] = (prog, E)
    std0 = np.mean([x0s[("direct", s)].std() for s in SEEDS])
    prog_d = seed_avg(direct, "pred_std") / std0
    E_d = np.arange(1, ROUNDS + 1, dtype=float)
    th = half_time(prog_d)
    a2["direct"] = {"t_half": th, "E_half": None if th is None else float(E_d[th])}
    curves["direct"] = (prog_d, E_d)
    for k, v in a2.items():
        print(f"  {k}: t_half={v['t_half']} E_half={v['E_half']}")
    out["a2"] = a2

    # --- 3: defense pricing at eps=0.3 ---
    print("\n[3] defense pricing eps=0.3 (final = mean last 5, seed avg)")
    a3 = {}
    for regime, rows in defense.items():
        mean_f = float(seed_avg(rows, "pop_mean")[-5:].mean())
        # same seeds -> same torch init across regimes
        init_m = float(np.mean([x0s[(0.3, s)].mean() for s in SEEDS]))
        a3[regime] = {
            "pop_mean_final": mean_f,
            "displacement": mean_f - init_m,
            "pop_std_final": float(seed_avg(rows, "pop_std")[-5:].mean()),
            "mix_contam_final": float(seed_avg(rows, "mix_contam")[-5:].mean()),
            "s_final": float(seed_avg(rows, "s")[-5:].mean())}
        v = a3[regime]
        print(f"  {regime:>11}: mean={v['pop_mean_final']:.3f} "
              f"displ={v['displacement']:+.3f} std={v['pop_std_final']:.3f} "
              f"mix_contam={v['mix_contam_final']:.3f}")
    out["a3"] = a3

    # --- 4: pull detector ---
    print(f"\n[4] per-contact pull check: measured {out['pull_check']:.4f} "
          f"(mechanical w={W})")
    a4 = {"pull_check": out["pull_check"], "cells": {}}
    pooled_b, pooled_s, pooled_ds = [], [], []
    for eps in EPS_GRID:
        pm = seed_avg(med[eps], "pop_mean")
        gap = seed_avg(med[eps], "pred_mean") - pm
        sct = seed_avg(med[eps], "s")
        b = window_slopes(pm, gap)
        ds = smooth(np.diff(sct, prepend=0.0))[:len(b)]
        cb_s = corr(b, sct[:len(b)])
        cb_ds = corr(b, ds)
        a4["cells"][eps] = {"corr_b_s": cb_s, "corr_b_dsdt": cb_ds}
        pooled_b.append(b)
        pooled_s.append(sct[:len(b)])
        pooled_ds.append(ds)
        print(f"  eps={eps}: corr(b_t, s_t)={cb_s} corr(b_t, ds/dt)={cb_ds}")
    a4["pooled_corr_b_s"] = corr(np.concatenate(pooled_b), np.concatenate(pooled_s))
    a4["pooled_corr_b_dsdt"] = corr(np.concatenate(pooled_b), np.concatenate(pooled_ds))
    print(f"  pooled: corr(b_t, s_t)={a4['pooled_corr_b_s']} "
          f"corr(b_t, ds/dt)={a4['pooled_corr_b_dsdt']}")
    out["a4"] = a4

    # --- 5: corner check ---
    print("\n[5] corners")
    std_cap = seed_avg(med[0.4], "pop_std")
    std0_cap = np.mean([x0s[(0.4, s)].std() for s in SEEDS])
    th_cap = half_time(std_cap / std0_cap)
    lam_cap = decay_rate(std_cap, th_cap)
    std_dir = seed_avg(direct, "pred_std")
    th_dir = half_time(prog_d)
    lam_dir = decay_rate(std_dir, th_dir)
    s_cap = seed_avg(med[0.4], "s")
    a5 = {"capture": {"s_late": float(s_cap[-20:].mean()),
                      "lambda": lam_cap, "lambda_direct": lam_dir},
          "detach": {"s_final": float(seed_avg(detach, "s")[-5:].mean()),
                     "s_max": float(seed_avg(detach, "s").max()),
                     "contact_late": float(seed_avg(detach, "contact")[-20:].mean()),
                     "ds_late": float(np.diff(seed_avg(detach, "s"))[-50:].mean())}}
    print(f"  capture eps=0.4: s_late={a5['capture']['s_late']:.3f} "
          f"lambda={lam_cap} vs direct lambda={lam_dir}")
    print(f"  detach anchor_w=2.0 eps=0.2: s_final={a5['detach']['s_final']:.3f} "
          f"s_max={a5['detach']['s_max']:.3f} "
          f"contact_late={a5['detach']['contact_late']:.3f} "
          f"ds_late={a5['detach']['ds_late']:+.5f}")
    out["a5"] = a5

    out["trajectories"] = {
        **{f"mediated_eps{e}|{s}": med[e][i] for e in EPS_GRID
           for i, s in enumerate(SEEDS)},
        **{f"direct|{s}": direct[i] for i, s in enumerate(SEEDS)},
        **{f"defense_{r}|{s}": defense[r][i] for r in ["mix_current", "mix_round0"]
           for i, s in enumerate(SEEDS)},
        **{f"detach|{s}": detach[i] for i, s in enumerate(SEEDS)},
    }
    json.dump(out, open("experiments/competition/figs/model_mass.json", "w"))

    # --- figure ---
    colors = {0.1: "#7f7f7f", 0.2: "#2ca02c", 0.3: "#1f77b4", 0.4: "#9467bd"}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for eps in EPS_GRID:
        ax[0, 0].plot(seed_avg(med[eps], "s"), c=colors[eps], label=f"eps={eps}")
    ax[0, 0].plot(seed_avg(detach, "s"), "k--", label="detach (aw=2, eps=0.2)")
    ax[0, 0].set(xlabel="round", ylabel="s_t = mean model mass",
                 title="model mass vs round")
    ax[0, 0].legend(frameon=False, fontsize=8)
    for name, (prog, E) in curves.items():
        c = "#d62728" if name == "direct" else colors[float(name.split("=")[1])]
        ax[0, 1].plot(prog, c=c, label=name)
        ax[0, 2].plot(E, prog, c=c, label=name)
    for a, xl in [(ax[0, 1], "round"), (ax[0, 2], "cumulative exposure E_t")]:
        a.axhline(0.5, ls=":", c="gray", lw=1)
        a.set(xlabel=xl, ylabel="pop_std_t / std_0 (pred for direct)",
              title=f"collapse progress vs {xl}")
        a.legend(frameon=False, fontsize=8)
    regimes = list(a3.keys())
    xpos = np.arange(len(regimes))
    for k, (field, lab) in enumerate([("displacement", "mean displacement"),
                                      ("pop_std_final", "final pop_std"),
                                      ("mix_contam_final", "train-mix contamination")]):
        ax[1, 0].bar(xpos + 0.25 * k, [a3[r][field] for r in regimes], 0.25, label=lab)
    ax[1, 0].set(xticks=xpos + 0.25, xticklabels=["a replace", "b mix-current",
                                                  "c mix-round0"],
                 title="defense pricing at eps=0.3")
    ax[1, 0].legend(frameon=False, fontsize=8)
    for eps in EPS_GRID:
        pm = seed_avg(med[eps], "pop_mean")
        gap = seed_avg(med[eps], "pred_mean") - pm
        b = window_slopes(pm, gap)
        ax[1, 1].scatter(seed_avg(med[eps], "s")[:len(b)], b, s=4,
                         c=colors[eps], label=f"eps={eps}")
    ax[1, 1].set(xlabel="s_t", ylabel="b_t (drift-vs-gap slope, 20-rd window)",
                 title=f"field proxy vs model mass "
                       f"(pooled r={a4['pooled_corr_b_s']:.2f})")
    ax[1, 1].legend(frameon=False, fontsize=8)
    ax[1, 2].plot(np.log10(np.maximum(std_cap, 1e-8)), c=colors[0.4],
                  label=f"capture eps=0.4 (lambda={lam_cap:.3f})"
                  if lam_cap else "capture eps=0.4")
    ax[1, 2].plot(np.log10(np.maximum(std_dir, 1e-8)), c="#d62728",
                  label=f"direct (lambda={lam_dir:.3f})" if lam_dir else "direct")
    ax[1, 2].set(xlabel="round", ylabel="log10 std",
                 title="corner: late contraction rates")
    ax[1, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"model mass s_t in the AB x MLP loop (device={DEVICE})")
    fig.savefig("experiments/competition/figs/model_mass.png", dpi=130)
    print("saved experiments/competition/figs/model_mass.png")


if __name__ == "__main__":
    main()
