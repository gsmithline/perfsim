"""Block 17: three publication figures on top of the block-16 machinery.

Fig 1 (figs/fig1_master_clock.png): master clock search. Direct loop +
mediated at eps {0.1..0.4}; normalized diversity D_t against five candidate
clocks (round t, exposure E_t, contact-weighted A_t, absorbed-mass B_t, and
s_t itself); pairwise-area collapse score per axis.

Fig 2 (figs/fig2_defense.png): three-line defense at eps=0.3, every batch
70% current + 30% pool; no defense / frozen round-0 pool / lagged crawl
(pool = population snapshot K rounds ago, tags carried). Peel time vs the
pool's own contamination s_pool(t). Bonus K=50 arm.

Fig 3 (figs/fig3_canary.png): canary flow meter. Fixed pattern
zeta_i = +-delta added to predictions before the gated blend; flow meter
c_t = cov(x_t - x_0, zeta)/var(zeta). eps grid x 2 seeds, w=0 control,
invisibility check at eps=0.3.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/17_three_figs.py [fig1|fig2|fig3|all]
"""

import importlib.util
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "model_mass", "experiments/competition/16_model_mass.py")
m16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m16)

N, P0, ANCHOR_W, TRAIN_STEPS = m16.N, m16.P0, m16.ANCHOR_W, m16.TRAIN_STEPS
GAMMA, W, ROUNDS, DEVICE = m16.GAMMA, m16.W, m16.ROUNDS, m16.DEVICE
seed_avg = m16.seed_avg
EPS_GRID = [0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 1, 2]
CANARY_SEEDS = [0, 1]
DELTA = 0.01
LAG_KS = [25, 50]
CACHE = "experiments/competition/figs/cache17"
FIGDIR = "experiments/competition/figs"
COLORS = {0.1: "#7f7f7f", 0.2: "#2ca02c", 0.3: "#1f77b4", 0.4: "#9467bd"}


def cached(key, fn):
    path = f"{CACHE}/{key}.json"
    if os.path.exists(path):
        d = json.load(open(path))
        return d["traj"], np.array(d["x0"])
    traj, x0 = fn()
    json.dump({"traj": traj, "x0": [float(v) for v in x0]}, open(path, "w"))
    print(f"  ran {key}", flush=True)
    return traj, np.array(x0)


def run_loop17(eps, seed, w=W, pool="none", lag_k=25, delta=0.0):
    """Block-16 loop plus a 70/30 defense pool and an additive canary.

    pool: none | frozen0 | lag. delta>0 adds zeta_i = +-delta (fixed signs,
    separate RNG so the sweep stream matches the no-canary run) to the model
    prediction before the gated blend; the MLP itself is untouched.
    """
    x, f, x0, feats, net, opt = m16.init_run(seed)
    t0v = torch.tensor(x0, dtype=torch.float32)
    zeta = None
    if delta > 0:
        g = torch.Generator().manual_seed(10_000 + seed)
        signs = torch.randint(0, 2, (N,), generator=g).float() * 2 - 1
        zeta = (delta * signs).numpy().astype(np.float64)
        zeta_dev = torch.tensor(zeta, dtype=torch.float32, device=DEVICE)
        zc = zeta - zeta.mean()
        vz = float(np.var(zeta))
        # residualizer: remove the x0-bin mean of the displacement so the
        # spurious overlap between mixing dynamics and zeta drops out
        nb = 15
        edges = np.quantile(x0, np.linspace(0, 1, nb + 1))
        bins = np.clip(np.searchsorted(edges, x0, "right") - 1, 0, nb - 1)
    hist_x, hist_f = [], []
    traj = []
    for t in range(ROUNDS):
        m16.sweep(x, f, eps, GAMMA)
        x_np = x.cpu().numpy().astype(np.float64)
        cur = torch.tensor(x_np, dtype=torch.float32)
        if pool == "none":
            pool_t, s_pool = None, None
        elif pool == "frozen0":
            pool_t, s_pool = t0v, 0.0
        else:  # lag
            j = t - lag_k
            pool_t = t0v if j < 0 else hist_x[j]
            s_pool = 0.0 if j < 0 else hist_f[j]
        for _ in range(TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            if pool_t is None:
                fit = ((pred - cur) ** 2).mean()
            else:
                fit = 0.7 * ((pred - cur) ** 2).mean() \
                    + 0.3 * ((pred - pool_t) ** 2).mean()
            loss = fit + ANCHOR_W * ((pred - P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        mt = torch.tensor(m, dtype=torch.float32, device=DEVICE)
        if delta > 0:
            mt = mt + zeta_dev
        gate = (mt - x).abs() < eps
        if w > 0:
            x = torch.where(gate, (1 - w) * x + w * mt, x)
            f = torch.where(gate, (1 - w) * f + w, f)
        xb = x.cpu().numpy().astype(np.float64)
        row = {"pop_mean": float(xb.mean()), "pop_std": float(xb.std()),
               "s": float(f.mean()), "contact": float(gate.float().mean()),
               "pred_mean": float(m.mean()), "pred_std": float(m.std())}
        if pool != "none":
            row["s_pool"] = float(s_pool)
        if delta > 0:
            dx = xb - x0
            row["c"] = float(np.dot(dx - dx.mean(), zc) / (N * vz))
            res = dx.copy()
            for b in range(nb):
                msk = bins == b
                res[msk] -= res[msk].mean()
            row["c_resid"] = float(np.dot(res, zc) / (N * vz))
        traj.append(row)
        if pool == "lag":
            hist_x.append(torch.tensor(xb, dtype=torch.float32))
            hist_f.append(float(f.mean()))
    return traj, x0


def get_mediated(eps, seed):
    return cached(f"med_e{eps}_s{seed}",
                  lambda: m16.run_loop(eps, seed)[::2])


def get_direct(seed):
    return cached(f"direct_s{seed}", lambda: m16.run_direct(seed))


def ramp(v):
    return np.asarray(v, float) + 1e-9 * np.arange(len(v))


def clock_at(curve, axis, level=0.5):
    idx = np.where(curve["D"] < level)[0]
    return None if len(idx) == 0 else float(curve[axis][idx[0]])


def area_score(curves, axis):
    names = list(curves)
    tot, cnt = 0.0, 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            xi, xj = ramp(curves[names[i]][axis]), ramp(curves[names[j]][axis])
            lo, hi = max(xi[0], xj[0]), min(xi[-1], xj[-1])
            if hi <= lo:
                continue
            g = np.linspace(lo, hi, 200)
            di = np.interp(g, xi, curves[names[i]]["D"])
            dj = np.interp(g, xj, curves[names[j]]["D"])
            tot += float(np.abs(di - dj).mean())
            cnt += 1
    return tot / cnt if cnt else None


def fig1():
    print("\n=== FIG 1: master clock search ===", flush=True)
    med, x0s = {}, {}
    for eps in EPS_GRID:
        rows = []
        for s in SEEDS:
            traj, x0 = get_mediated(eps, s)
            rows.append(traj)
            x0s[(eps, s)] = x0
        med[eps] = rows
    direct = []
    for s in SEEDS:
        traj, x0 = get_direct(s)
        direct.append(traj)
        x0s[("direct", s)] = x0

    t_axis = np.arange(1.0, ROUNDS + 1)
    curves = {}
    for eps in EPS_GRID:
        std0 = np.mean([x0s[(eps, s)].std() for s in SEEDS])
        sct = seed_avg(med[eps], "s")
        cont = seed_avg(med[eps], "contact")
        curves[f"eps={eps}"] = {
            "D": seed_avg(med[eps], "pop_std") / std0, "t": t_axis,
            "E": np.cumsum(sct), "A": np.cumsum(cont * sct),
            "B": np.cumsum(W * cont), "s": sct}
    std0 = np.mean([x0s[("direct", s)].std() for s in SEEDS])
    # direct loop: s=1, contact=1, absorbed mass 1/round, so E=A=B=t
    curves["direct"] = {"D": seed_avg(direct, "pred_std") / std0, "t": t_axis,
                        "E": t_axis, "A": t_axis, "B": t_axis,
                        "s": np.ones(ROUNDS)}

    axes_def = [("t", "round t"), ("E", "E_t = sum s"),
                ("A", "A_t = sum contact*s"), ("B", "B_t = sum w*contact")]
    med_only = {n: c for n, c in curves.items() if n != "direct"}
    scores, scores_med, halves = {}, {}, {}
    for ax_key, _ in axes_def:
        scores[ax_key] = area_score(curves, ax_key)
        scores_med[ax_key] = area_score(med_only, ax_key)
        halves[ax_key] = {n: clock_at(curves[n], ax_key) for n in curves}
    # s-axis: direct has s identically 1, no extent, so it is mediated-only
    scores["s"] = None
    scores_med["s"] = area_score(med_only, "s")
    halves["s"] = {n: clock_at(med_only[n], "s") for n in med_only}

    print("collapse scores (mean pairwise |D| area, lower = better):")
    print("  axis | all curves | mediated-only (fair comparison vs s-axis)")
    for k in ["t", "E", "A", "B", "s"]:
        all_s = "  n/a " if scores[k] is None else f"{scores[k]:.4f}"
        print(f"  {k:>4} |   {all_s}   |   {scores_med[k]:.4f}")
    print("clock value at half-collapse (D=0.5), per axis:")
    ratios = {}
    for k in ["t", "E", "A", "B"]:
        hd = halves[k]["direct"]
        line = []
        for n in curves:
            h = halves[k][n]
            line.append(f"{n}={'never' if h is None else f'{h:.1f}'}")
        h02 = halves[k]["eps=0.2"]
        ratios[k] = None if (h02 is None or hd is None) else h02 / hd
        rtxt = "no crossing" if ratios[k] is None else f"{ratios[k]:.1f}x"
        print(f"  {k}: " + " ".join(line) + f" | eps=0.2 vs direct: {rtxt}")

    best = min(scores_med, key=lambda k: scores_med[k])
    within2 = {k: (ratios.get(k) is not None and ratios[k] <= 2.0)
               for k in ["t", "E", "A", "B"]}
    print(f"verdict: best-collapsing axis (mediated-only score) = {best}; "
          f"eps=0.2 within 2x of direct on any axis: {any(within2.values())}")

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9), constrained_layout=True)
    panels = axes_def + [("s", "s_t (state variable)")]
    for k, (ax_key, lab) in enumerate(panels):
        a = ax.flat[k]
        for n, c in curves.items():
            if ax_key == "s" and n == "direct":
                continue
            col = "#d62728" if n == "direct" else COLORS[float(n.split("=")[1])]
            a.plot(c[ax_key], c["D"], c=col, label=n)
        a.axhline(0.5, ls=":", c="gray", lw=1)
        if ax_key in ("E", "A", "B"):
            a.set_xscale("log")
        sc = scores_med[ax_key] if scores[ax_key] is None else scores[ax_key]
        ttl = f"({chr(97 + k)}) D vs {lab}  [score {sc:.3f}]"
        if ax_key == "s":
            ttl += "\n(direct excluded: s = 1 always)"
        a.set(xlabel=lab, ylabel="D_t = std_t / std_0", title=ttl)
        a.legend(frameon=False, fontsize=8)
    a = ax.flat[5]
    keys = ["t", "E", "A", "B", "s"]
    xb = np.arange(len(keys))
    a.bar(xb - 0.18, [scores[k] if scores[k] is not None else 0 for k in keys],
          0.36, label="all curves (incl direct)", color="#1f77b4")
    a.bar(xb + 0.18, [scores_med[k] for k in keys], 0.36,
          label="mediated only", color="#ff7f0e")
    a.set(xticks=xb, xticklabels=keys,
          ylabel="pairwise area score (lower = unified)",
          title="(f) which clock unifies the curves?")
    a.legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 1: searching for a master clock of mediated collapse "
                 f"(n={N}, gamma={GAMMA}, w={W}, {len(SEEDS)} seeds)")
    fig.savefig(f"{FIGDIR}/fig1_master_clock.png", dpi=160)
    print(f"saved {FIGDIR}/fig1_master_clock.png")
    return {"scores": scores, "scores_mediated_only": scores_med,
            "half_clock": halves, "eps02_ratio": ratios,
            "best_axis": best, "any_within_2x": any(within2.values())}


def peel_round(gap, sig, sustain=5):
    above = gap > 2 * np.maximum(sig, 1e-3)
    for t in range(len(gap) - sustain):
        if above[t:t + sustain].all():
            return t
    return None


def fig2():
    print("\n=== FIG 2: three-line defense with lagged crawl ===", flush=True)
    eps = 0.3
    arms = {"no_defense": [get_mediated(eps, s) for s in SEEDS],
            "frozen0": [cached(f"def_frozen_s{s}",
                               lambda s=s: run_loop17(eps, s, pool="frozen0"))
                        for s in SEEDS]}
    for K in LAG_KS:
        arms[f"lag{K}"] = [cached(f"def_lag{K}_s{s}",
                                  lambda s=s, K=K: run_loop17(
                                      eps, s, pool="lag", lag_k=K))
                           for s in SEEDS]

    res = {}
    init_m = float(np.mean([x0.mean() for _, x0 in arms["frozen0"]]))
    for name, runs in arms.items():
        trajs = [t for t, _ in runs]
        res[name] = {
            "std": np.array([[r["pop_std"] for r in t] for t in trajs]),
            "mean": seed_avg(trajs, "pop_mean"),
            "s_pool": seed_avg(trajs, "s_pool") if name.startswith("lag")
            else None,
            "final_std": float(seed_avg(trajs, "pop_std")[-5:].mean()),
            "final_displ": float(seed_avg(trajs, "pop_mean")[-5:].mean()
                                 - init_m)}

    std2 = res["frozen0"]["std"]
    peels, pool_marks = {}, {}
    for K in LAG_KS:
        std3 = res[f"lag{K}"]["std"]
        gap = np.abs(std2.mean(0) - std3.mean(0))
        sig = np.sqrt(0.5 * (std2.std(0) ** 2 + std3.std(0) ** 2))
        pr = peel_round(gap, sig)
        sp = res[f"lag{K}"]["s_pool"]
        first05 = np.where(sp > 0.05)[0]
        first10 = np.where(sp > 0.10)[0]
        peels[K] = pr
        pool_marks[K] = {
            "s_pool_at_peel": None if pr is None else float(sp[pr]),
            "first_s_pool_gt_0.05": None if not len(first05) else int(first05[0]),
            "first_s_pool_gt_0.10": None if not len(first10) else int(first10[0])}
        print(f"K={K}: peel round = {pr}, s_pool at peel = "
              f"{pool_marks[K]['s_pool_at_peel']}, s_pool>0.05 at round "
              f"{pool_marks[K]['first_s_pool_gt_0.05']}, >0.10 at "
              f"{pool_marks[K]['first_s_pool_gt_0.10']}")
    for name in arms:
        print(f"  {name:>10}: final std={res[name]['final_std']:.4f} "
              f"displ={res[name]['final_displ']:+.4f}")
    headline = res["frozen0"]["final_std"] - res["lag25"]["final_std"]
    print(f"headline gap (frozen - lag25) final std: {headline:+.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    style = {"no_defense": ("#d62728", "-", "no defense (100% current)"),
             "frozen0": ("#2ca02c", "-", "frozen round-0 pool"),
             "lag25": ("#1f77b4", "-", "lagged crawl K=25"),
             "lag50": ("#17becf", "--", "lagged crawl K=50")}
    for name, (c, ls, lab) in style.items():
        m = res[name]["std"].mean(0)
        sd = res[name]["std"].std(0)
        ax[0].plot(m, c=c, ls=ls, label=lab)
        ax[0].fill_between(range(ROUNDS), m - sd, m + sd, color=c, alpha=0.15)
    for (K, c), yf in zip([(25, "#1f77b4"), (50, "#17becf")], [0.92, 0.78]):
        if peels[K] is not None:
            ax[0].axvline(peels[K], color=c, ls=":", lw=1.2)
            ax[0].text(peels[K] + 3, ax[0].get_ylim()[1] * yf,
                       f"peel K={K} (t={peels[K]})", fontsize=8, color=c)
    ax[0].set(xlabel="round", ylabel="pop_std",
              title=f"(a) population diversity, eps={eps}, "
                    "batch = 70% current + 30% pool")
    ax[0].legend(frameon=False, fontsize=9)
    for K, c in [(25, "#1f77b4"), (50, "#17becf")]:
        ax[1].plot(res[f"lag{K}"]["s_pool"], c=c, label=f"s_pool, K={K}")
        if peels[K] is not None:
            ax[1].axvline(peels[K], color=c, ls=":", lw=1.2)
    ax[1].set(xlabel="round", ylabel="s_pool(t) = model mass inside the pool",
              title="(b) the crawl pool contaminates itself\n"
                    "(dotted = peel round from panel a)")
    ax[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Fig 2: a lagged crawl defense decays once the pool's own "
                 "model mass arrives")
    fig.savefig(f"{FIGDIR}/fig2_defense.png", dpi=160)
    print(f"saved {FIGDIR}/fig2_defense.png")
    return {"finals": {n: {"std": res[n]["final_std"],
                           "displ": res[n]["final_displ"]} for n in res},
            "peel_rounds": peels, "pool_marks": pool_marks,
            "headline_gap_std": headline}


def fig3():
    print("\n=== FIG 3: canary flow meter ===", flush=True)
    can = {eps: [cached(f"can_e{eps}_s{s}",
                        lambda eps=eps, s=s: run_loop17(eps, s, delta=DELTA))
                 for s in CANARY_SEEDS] for eps in EPS_GRID}
    ctrl = [cached(f"ctrl_w0_e0.3_s{s}",
                   lambda s=s: run_loop17(0.3, s, w=0.0, delta=DELTA))
            for s in CANARY_SEEDS]

    cors, cors_resid, cors_dyn = {}, {}, {}
    pooled_c, pooled_s = [], []
    for eps in EPS_GRID:
        trajs = [t for t, _ in can[eps]]
        ct = seed_avg(trajs, "c")
        cr = seed_avg(trajs, "c_resid")
        st = seed_avg(trajs, "s")
        cors[eps] = m16.corr(ct, st)
        cors_resid[eps] = m16.corr(cr, st)
        dyn = st < 0.9  # pre-saturation phase; plateau noise depresses corr
        cors_dyn[eps] = m16.corr(cr[dyn], st[dyn]) if dyn.sum() >= 3 else None
        pooled_c.append(cr)
        pooled_s.append(st)
        print(f"eps={eps}: corr(c_raw, s)={cors[eps]} "
              f"corr(c_resid, s)={cors_resid[eps]} "
              f"corr_dynamic_phase={cors_dyn[eps]} "
              f"(n_dyn={int(dyn.sum())}) "
              f"c_final={ct[-5:].mean():.3f} c_max={ct.max():.3f} "
              f"s_final={st[-5:].mean():.3f}")
    pooled_r = m16.corr(np.concatenate(pooled_c), np.concatenate(pooled_s))
    print(f"pooled corr(c_resid, s) across cells/rounds: {pooled_r}")

    ctrl_c = np.array([[r["c"] for r in t] for t, _ in ctrl])
    ctrl_cr = np.array([[r["c_resid"] for r in t] for t, _ in ctrl])
    print(f"w=0 control: max|c_raw|={np.abs(ctrl_c).max():.3f} "
          f"max|c_resid|={np.abs(ctrl_cr).max():.3f} (noise floor at "
          f"delta={DELTA}, N={N})")

    # invisibility: same seeds, eps=0.3, with vs without canary
    plain = [get_mediated(0.3, s)[0] for s in CANARY_SEEDS]
    std_plain = np.array([[r["pop_std"] for r in t] for t in plain])
    std_can = np.array([[r["pop_std"] for r in t] for t, _ in can[0.3]])
    dev = np.abs(std_plain.mean(0) - std_can.mean(0)).max()
    seed_scale = std_plain.std(0).max()
    print(f"invisibility eps=0.3: max |pop_std(canary) - pop_std(plain)| = "
          f"{dev:.4f}; across-seed std scale = {seed_scale:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for eps in EPS_GRID:
        trajs = [t for t, _ in can[eps]]
        cr = m16.smooth(seed_avg(trajs, "c_resid"), 5)
        ax[0].plot(cr, c=COLORS[eps], label=f"c_t (resid), eps={eps}")
        ax[0].plot(seed_avg(trajs, "s"), c=COLORS[eps], ls="--", lw=1)
    ax[0].set(xlabel="round", ylabel="c_t (solid) / s_t (dashed)",
              title="(a) canary coefficient vs model mass")
    ax[0].legend(frameon=False, fontsize=8)
    for eps in EPS_GRID:
        trajs = [t for t, _ in can[eps]]
        ax[1].scatter(seed_avg(trajs, "s"), seed_avg(trajs, "c_resid"),
                      s=4, c=COLORS[eps],
                      label=f"eps={eps} (r={cors_resid[eps]:.2f})")
    ax[1].set(xlabel="s_t (hidden truth)", ylabel="c_t (observable)",
              title=f"(b) calibration scatter, pooled r={pooled_r:.2f}")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(ctrl_cr.mean(0), c="#2ca02c", label="c_resid, w=0 control")
    ax[2].plot(ctrl_c.mean(0), c="#9467bd", alpha=0.5, label="c_raw, w=0")
    ax[2].axhline(0, c="gray", lw=1)
    ax[2].set(xlabel="round", ylabel="c_t",
              title=f"(c) w=0 sanity: max|c_resid|={np.abs(ctrl_cr).max():.3f}"
                    f"\ninvisibility: max std deviation {dev:.4f} "
                    f"(seed scale {seed_scale:.4f})")
    ax[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Fig 3: canary flow meter, delta={DELTA} added to "
                 "predictions before the gated blend (MLP untouched)")
    fig.savefig(f"{FIGDIR}/fig3_canary.png", dpi=160)
    print(f"saved {FIGDIR}/fig3_canary.png")
    return {"corr_raw": cors, "corr_resid": cors_resid,
            "corr_dynamic_phase": cors_dyn, "pooled_r": pooled_r,
            "ctrl_max_c_raw": float(np.abs(ctrl_c).max()),
            "ctrl_max_c_resid": float(np.abs(ctrl_cr).max()),
            "invisibility_max_dev": float(dev),
            "seed_scale": float(seed_scale)}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    print(f"device={DEVICE}, n={N}, gamma={GAMMA}, w={W}, {ROUNDS} rounds")
    out = {}
    if which in ("fig1", "all"):
        out["fig1"] = fig1()
    if which in ("fig2", "all"):
        out["fig2"] = fig2()
    if which in ("fig3", "all"):
        out["fig3"] = fig3()
    path = f"{FIGDIR}/three_figs_summary.json"
    if os.path.exists(path):
        prev = json.load(open(path))
        prev.update(out)
        out = prev
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsummary -> {path}")


if __name__ == "__main__":
    main()
