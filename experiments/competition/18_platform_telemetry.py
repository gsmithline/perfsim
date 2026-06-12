"""Block 18: platform-side telemetry. Every logged quantity comes from the
retraining process itself: losses on the incoming batch, gradient norms,
predictions on a fixed probe grid, batch target variance. The provenance
tags f_i still ride through the population but are read only to produce the
ground-truth s_t used for validation scatter and regression targets.

L_init adaptation: for an MLP regressor the own-sample loss is exactly 0
(the direct loop trains on its own deterministic outputs, no sampling
noise), so the ratio L_init / L_init(own) degenerates to division by zero.
We therefore log L_init absolute AND the variance-normalized form
L_init / var(batch targets) -- an unexplained fraction that falls toward 0
as the batch becomes the model's own recycled output.

Note L_final == L_cc (current model on current batch, after training).

Cells: mediated eps {0.1,0.2,0.3,0.4} at w=0.3; w=0 control at eps=0.3;
detachment (anchor_w=2.0, eps=0.2, w=0.3); direct self-consuming loop.
2 seeds, 300 rounds.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/18_platform_telemetry.py
"""

import copy
import importlib.util
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
SEEDS = [0, 1]
N_PROBE = 50
T_REPORT = 150
NORM_CLIP = 10.0
CACHE = "experiments/competition/figs/cache18"
FIGDIR = "experiments/competition/figs"

PROBE = torch.zeros(N_PROBE, 3)
PROBE[:, 0] = torch.linspace(0, 1, N_PROBE)

FEATS = ["l_init_norm", "displacement", "grad_norm",
         "l_cc", "l_c0", "l_0c", "l_00", "batch_var", "dist_init"]

CELL_STYLE = {
    "eps0.1": ("#7f7f7f", "-", "eps=0.1"),
    "eps0.2": ("#2ca02c", "-", "eps=0.2"),
    "eps0.3": ("#1f77b4", "-", "eps=0.3"),
    "eps0.4": ("#9467bd", "-", "eps=0.4"),
    "w0": ("#ff7f0e", "-", "w=0 ctrl (eps=0.3)"),
    "detach": ("k", "--", "detach (aw=2, eps=0.2)"),
    "direct": ("#d62728", "-", "direct loop"),
}
MED_CELLS = [("eps0.1", 0.1), ("eps0.2", 0.2), ("eps0.3", 0.3),
             ("eps0.4", 0.4), ("detach", 0.2)]


class Telemetry:
    """Per-round platform-seat measurements around one retraining call."""

    def __init__(self):
        self.net0 = None
        self.archive0 = None
        self.probe0 = None
        self.prev_probe = None
        self.preds0 = None

    def round(self, net, opt, feats, cur, anchor_w):
        with torch.no_grad():
            l_init = float(((net(feats).squeeze(1) - cur) ** 2).mean())
            if self.prev_probe is None:
                self.prev_probe = net(PROBE).squeeze(1).numpy()
        bvar = float(cur.var(unbiased=False))
        gnorm = 0.0
        for k in range(TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - cur) ** 2).mean() \
                + anchor_w * ((pred - P0) ** 2).mean()
            loss.backward()
            if k == 0:
                # full training-objective gradient: ~0 means retraining no
                # longer moves the model (performative-stability meter)
                gnorm = float(torch.sqrt(sum((p.grad ** 2).sum()
                                             for p in net.parameters())))
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1)
            pp = net(PROBE).squeeze(1).numpy()
        if self.net0 is None:
            self.net0 = copy.deepcopy(net)
            self.archive0 = cur.clone()
            self.probe0 = pp.copy()
            with torch.no_grad():
                self.preds0 = self.net0(feats).squeeze(1)
        row = {
            "l_init": l_init,
            "l_init_norm": l_init / max(bvar, 1e-12),
            "displacement": float(np.abs(pp - self.prev_probe).mean()),
            "grad_norm": gnorm,
            "l_cc": float(((m - cur) ** 2).mean()),
            "l_c0": float(((m - self.archive0) ** 2).mean()),
            "l_0c": float(((self.preds0 - cur) ** 2).mean()),
            "l_00": float(((self.preds0 - self.archive0) ** 2).mean()),
            "batch_var": bvar,
            "dist_init": float(np.abs(pp - self.probe0).mean()),
        }
        self.prev_probe = pp
        return row, m


def run_mediated(eps, seed, w=W, anchor_w=ANCHOR_W):
    x, f, _, feats, net, opt = m16.init_run(seed)
    tel = Telemetry()
    traj = []
    for _ in range(ROUNDS):
        m16.sweep(x, f, eps, GAMMA)
        cur = torch.tensor(x.cpu().numpy().astype(np.float64),
                           dtype=torch.float32)
        row, m = tel.round(net, opt, feats, cur, anchor_w)
        mt = torch.tensor(m.numpy(), dtype=torch.float32, device=DEVICE)
        gate = (mt - x).abs() < eps
        if w > 0:
            x = torch.where(gate, (1 - w) * x + w * mt, x)
            f = torch.where(gate, (1 - w) * f + w, f)
        row["s_true"] = float(f.mean())
        row["contact"] = float(gate.float().mean())
        traj.append(row)
    return traj


def run_direct(seed):
    _, _, x0, feats, net, opt = m16.init_run(seed)
    cur = torch.tensor(x0, dtype=torch.float32)
    tel = Telemetry()
    traj = []
    for _ in range(ROUNDS):
        row, m = tel.round(net, opt, feats, cur, ANCHOR_W)
        cur = m.clone()
        row["s_true"] = 1.0
        traj.append(row)
    return traj


def cached(key, fn):
    path = f"{CACHE}/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))
    traj = fn()
    json.dump(traj, open(path, "w"))
    print(f"  ran {key}", flush=True)
    return traj


def field(runs, cell, key, clip=None):
    v = seed_avg(runs[cell], key)
    return np.minimum(v, clip) if clip is not None else v


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    print(f"device={DEVICE}, n={N}, gamma={GAMMA}, w={W}, P0={P0}, "
          f"{ROUNDS} rounds, seeds {SEEDS}")
    print("L_init adaptation: own-sample loss is exactly 0 for this MLP "
          "regressor, so instead of L_init/L_init(own) we log L_init and "
          "L_init/var(batch) (unexplained fraction).")

    runs = {}
    for eps in EPS_GRID:
        runs[f"eps{eps}"] = [cached(f"med_e{eps}_s{s}",
                                    lambda e=eps, s=s: run_mediated(e, s))
                             for s in SEEDS]
    runs["w0"] = [cached(f"w0_e0.3_s{s}",
                         lambda s=s: run_mediated(0.3, s, w=0.0))
                  for s in SEEDS]
    runs["detach"] = [cached(f"detach_s{s}",
                             lambda s=s: run_mediated(0.2, s, anchor_w=2.0))
                      for s in SEEDS]
    runs["direct"] = [cached(f"direct_s{s}", lambda s=s: run_direct(s))
                      for s in SEEDS]
    out = {}

    # --- 1: L_init ---
    print("\n[1] L_init (early = mean rounds 5-25, late = mean last 20)")
    a1 = {}
    for cell in CELL_STYLE:
        li = field(runs, cell, "l_init")
        ln = field(runs, cell, "l_init_norm")
        a1[cell] = {"l_init_early": float(li[5:25].mean()),
                    "l_init_late": float(li[-20:].mean()),
                    "l_init_norm_early": float(ln[5:25].mean()),
                    "l_init_norm_late": float(ln[-20:].mean())}
        v = a1[cell]
        print(f"  {cell:>7}: L_init {v['l_init_early']:.5f} -> "
              f"{v['l_init_late']:.6f} | L_init/var "
              f"{v['l_init_norm_early']:.3f} -> {v['l_init_norm_late']:.3f}")
    out["a1_l_init"] = a1

    # --- 2: displacement + grad norm ---
    print("\n[2] probe displacement and step-0 grad norm (early/late means)")
    a2 = {}
    for cell in CELL_STYLE:
        dp = field(runs, cell, "displacement")
        gn = field(runs, cell, "grad_norm")
        a2[cell] = {"disp_early": float(dp[5:25].mean()),
                    "disp_late": float(dp[-20:].mean()),
                    "grad_early": float(gn[5:25].mean()),
                    "grad_late": float(gn[-20:].mean())}
        v = a2[cell]
        print(f"  {cell:>7}: disp {v['disp_early']:.5f} -> {v['disp_late']:.6f}"
              f" | grad {v['grad_early']:.4f} -> {v['grad_late']:.4f}")
    out["a2_displacement"] = a2

    # --- 3: the 2x2 loss matrix at t=150 ---
    print(f"\n[3] 2x2 loss matrix at t={T_REPORT} "
          "(rows: current/round-0 model, cols: current batch/round-0 archive)")
    a3 = {}
    for cell in CELL_STYLE:
        a3[cell] = {k: float(field(runs, cell, k)[T_REPORT])
                    for k in ["l_cc", "l_c0", "l_0c", "l_00", "dist_init"]}
        v = a3[cell]
        print(f"  {cell:>7}: [cc={v['l_cc']:.4f} c0={v['l_c0']:.4f} | "
              f"0c={v['l_0c']:.4f} 00={v['l_00']:.4f}] "
              f"probe dist(cur, rd0)={v['dist_init']:.4f}")
    out["a3_matrix_t150"] = a3

    # --- 4: batch variance and the collapse trap ---
    print("\n[4] collapse trap: L_final (=L_cc) falls while var falls faster")
    a4 = {}
    for cell in ["eps0.4", "direct", "w0"]:
        lf = field(runs, cell, "l_cc")
        bv = field(runs, cell, "batch_var")
        a4[cell] = {"l_final_t20": float(lf[20]), "l_final_t280": float(lf[280]),
                    "var_t20": float(bv[20]), "var_t280": float(bv[280]),
                    "fit_frac_t20": float(lf[20] / max(bv[20], 1e-12)),
                    "fit_frac_t280": float(lf[280] / max(bv[280], 1e-12))}
        v = a4[cell]
        print(f"  {cell:>7}: L_final {v['l_final_t20']:.5f} -> "
              f"{v['l_final_t280']:.6f} ({v['l_final_t20'] / max(v['l_final_t280'], 1e-12):.0f}x down), "
              f"var {v['var_t20']:.5f} -> {v['var_t280']:.6f}, "
              f"L/var {v['fit_frac_t20']:.3f} -> {v['fit_frac_t280']:.3f}")
    out["a4_trap"] = a4

    # --- 5: leash length ---
    print("\n[5] dist from round-0 model on probe grid (final mean last 20)")
    a5 = {cell: float(field(runs, cell, "dist_init")[-20:].mean())
          for cell in CELL_STYLE}
    for cell, v in a5.items():
        print(f"  {cell:>7}: {v:.4f}")
    out["a5_dist_init"] = a5

    # --- validation: L_init/var vs true s_t ---
    print("\n[validation] corr(L_init/var, s_t) per mediated cell "
          "(pooled rounds x seeds; w0/direct excluded, s constant)")
    val = {}
    for cell, _ in MED_CELLS:
        ln = np.concatenate([np.minimum([r["l_init_norm"] for r in t],
                                        NORM_CLIP) for t in runs[cell]])
        st = np.concatenate([[r["s_true"] for r in t] for t in runs[cell]])
        val[cell] = m16.corr(ln, st)
        print(f"  {cell:>7}: r={val[cell]}")
    ln_all = np.concatenate([np.minimum([r["l_init_norm"] for r in t], NORM_CLIP)
                             for c, _ in MED_CELLS for t in runs[c]])
    st_all = np.concatenate([[r["s_true"] for r in t]
                             for c, _ in MED_CELLS for t in runs[c]])
    val["pooled"] = m16.corr(ln_all, st_all)
    print(f"   pooled: r={val['pooled']}")
    out["validation_corr"] = val

    # --- regression check ---
    print("\n[regression] telemetry-only features, rounds 10-300, mediated "
          "cells, train seed 0 / test seed 1 (l_init_norm clipped at "
          f"{NORM_CLIP})")

    def design(si, feats):
        X, ys, ye = [], [], []
        for cell, eps in MED_CELLS:
            for r in runs[cell][si][10:]:
                X.append([min(r[k], NORM_CLIP) if k == "l_init_norm" else r[k]
                          for k in feats])
                ys.append(r["s_true"])
                ye.append(eps)
        return np.array(X), np.array(ys), np.array(ye)

    classes = np.array(EPS_GRID)

    def fit_eval(tr, te, feats):
        Xtr, s_tr, e_tr = design(tr, feats)
        Xte, s_te, e_te = design(te, feats)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        res = {"ridge_r2_s": float(
            Ridge(alpha=1.0).fit(Xtr_s, s_tr).score(Xte_s, s_te))}
        ridge_e = Ridge(alpha=1.0).fit(Xtr_s, e_tr)
        res["ridge_r2_eps"] = float(ridge_e.score(Xte_s, e_te))
        pe = ridge_e.predict(Xte_s)
        res["ridge_acc_eps"] = float(
            (classes[np.abs(pe[:, None] - classes).argmin(1)] == e_te).mean())
        rf_s = RandomForestRegressor(200, random_state=0).fit(Xtr, s_tr)
        res["rf_r2_s"] = float(rf_s.score(Xte, s_te))
        rf_e = RandomForestClassifier(200, random_state=0).fit(
            Xtr, np.round(e_tr * 10).astype(int))
        res["rf_acc_eps"] = float(
            (rf_e.predict(Xte) == np.round(e_te * 10).astype(int)).mean())
        res["rf_top_features_s"] = [
            (k, float(v)) for k, v in
            sorted(zip(feats, rf_s.feature_importances_),
                   key=lambda p: -p[1])[:3]]
        return res

    # l_00 is exactly constant within a run (frozen model on frozen archive),
    # i.e. a per-(cell, seed) fingerprint a forest can overfit; eps results
    # are therefore also reported with it dropped and the seed split reversed
    no_l00 = [k for k in FEATS if k != "l_00"]
    reg = {"s0_to_s1": fit_eval(0, 1, FEATS),
           "s1_to_s0": fit_eval(1, 0, FEATS),
           "s0_to_s1_no_l00": fit_eval(0, 1, no_l00),
           "s1_to_s0_no_l00": fit_eval(1, 0, no_l00)}
    for tag, r in reg.items():
        print(f"  {tag:>15}: s_t ridge R^2={r['ridge_r2_s']:.3f} "
              f"RF R^2={r['rf_r2_s']:.3f} | eps ridge R^2="
              f"{r['ridge_r2_eps']:.3f} acc={r['ridge_acc_eps']:.3f} "
              f"RF acc={r['rf_acc_eps']:.3f}")
    print("  RF top features for s_t (s0->s1): "
          + ", ".join(f"{k}={v:.2f}"
                      for k, v in reg["s0_to_s1"]["rf_top_features_s"]))
    out["regression"] = reg

    json.dump(out, open(f"{FIGDIR}/platform_telemetry.json", "w"), indent=1)

    # --- figure ---
    fig, ax = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    panels = [("l_init_norm", "(a) L_init / var(batch)", True, NORM_CLIP),
              ("displacement", "(b) probe displacement", True, None),
              ("grad_norm", "(c) step-0 grad norm", True, None),
              (None, "(d) L_final (solid) vs batch var (dashed)", True, None),
              (None, "(e) fit fraction L_final / var", True, None),
              ("dist_init", "(f) dist from round-0 model (leash)", False, None)]
    for k, (key, title, logy, clip) in enumerate(panels):
        a = ax.flat[k]
        for cell, (c, ls, lab) in CELL_STYLE.items():
            if key is not None:
                a.plot(np.maximum(field(runs, cell, key, clip), 1e-8),
                       c=c, ls=ls, lw=1.2, label=lab)
            elif "L_final" in title:
                a.plot(np.maximum(field(runs, cell, "l_cc"), 1e-8),
                       c=c, ls="-", lw=1.2, label=lab)
                a.plot(np.maximum(field(runs, cell, "batch_var"), 1e-8),
                       c=c, ls="--", lw=0.8)
            else:
                lf = field(runs, cell, "l_cc")
                bv = field(runs, cell, "batch_var")
                a.plot(np.maximum(lf / np.maximum(bv, 1e-12), 1e-8),
                       c=c, ls=ls, lw=1.2, label=lab)
        if logy:
            a.set_yscale("log")
        a.set(xlabel="round", title=title)
        if k == 0:
            a.legend(frameon=False, fontsize=7)

    a = ax.flat[6]
    cells = list(CELL_STYLE)
    xb = np.arange(len(cells))
    for j, mk in enumerate(["l_cc", "l_c0", "l_0c", "l_00"]):
        a.bar(xb + 0.2 * j - 0.3,
              [max(a3[c][mk], 1e-8) for c in cells], 0.2, label=mk)
    a.set_yscale("log")
    a.set(xticks=xb, xticklabels=[CELL_STYLE[c][2] for c in cells],
          title=f"(g) 2x2 loss matrix at t={T_REPORT}")
    a.tick_params(axis="x", labelsize=7, rotation=20)
    a.legend(frameon=False, fontsize=8)

    a = ax.flat[7]
    for cell, _ in MED_CELLS:
        c = CELL_STYLE[cell][0]
        ln = np.concatenate([np.minimum([r["l_init_norm"] for r in t],
                                        NORM_CLIP) for t in runs[cell]])
        st = np.concatenate([[r["s_true"] for r in t] for t in runs[cell]])
        a.scatter(st, ln, s=3, c=c, alpha=0.5,
                  label=f"{CELL_STYLE[cell][2]} (r={val[cell]:.2f})")
    a.set(xlabel="s_t (true tagged model mass)", ylabel="L_init / var",
          title=f"(h) validation: contamination meter vs truth "
                f"(pooled r={val['pooled']:.2f})")
    a.legend(frameon=False, fontsize=7)

    a = ax.flat[8]
    for cell, (c, ls, lab) in CELL_STYLE.items():
        a.plot(np.maximum(field(runs, cell, "l_c0"), 1e-8), c=c, ls="-",
               lw=1.2, label=lab)
        a.plot(np.maximum(field(runs, cell, "l_0c"), 1e-8), c=c, ls=":",
               lw=0.9)
    a.set_yscale("log")
    a.set(xlabel="round",
          title="(i) off-diagonals: L_c0 (solid, population moved)\n"
                "vs L_0c (dotted, model moved)")
    a.legend(frameon=False, fontsize=7)

    accs = [reg[k]["rf_acc_eps"] for k in ("s0_to_s1_no_l00",
                                           "s1_to_s0_no_l00")]
    fig.suptitle("Block 18: telemetry from the platform's seat "
                 f"(tags used only in panel h; ridge R^2 s_t="
                 f"{reg['s0_to_s1']['ridge_r2_s']:.2f}, eps RF acc "
                 f"{min(accs):.2f}-{max(accs):.2f} w/o l_00)")
    fig.savefig(f"{FIGDIR}/platform_telemetry.png", dpi=150)
    print(f"\nsaved {FIGDIR}/platform_telemetry.png and "
          f"{FIGDIR}/platform_telemetry.json")


if __name__ == "__main__":
    main()
