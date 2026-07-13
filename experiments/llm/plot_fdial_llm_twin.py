"""LLM twin of the MLP feature-dial figure (fdial_mlp_dr_vr.png).
Same axes, layout, and population cell (trap cell e040_a040, gamma 0,
W 0.3, 30 rounds, carry): left dr(t) = op_std / innate_std, right
vr(t) = pred_std / op_std, one line per feature strength (realized
cross-fitted ridge R2), GnBu ramp dark = strong. Protocol-matched arm =
fresh weights + replace + beta 0 (the MLP recipe); the continual arm is
saved as a second file for reference. Seed 0 only (the MLP original
averages 3 seeds); bands need the 14-job seed batch.
"""
import os, numpy as np, pandas as pd, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
KNOBS = ["nat", "p100", "p065", "p040", "p033", "p015", "p005"]
BETAS = [("b0", "0"), ("b0p5", "0.5"), ("b1", "1")]

def tag(k, arm, bc="b0"):
    if k == "nat" and arm == "rep":
        return {"b0": "mla2dv2_e040_a040_b0_s0",
                "b0p5": "mla2bv2_e040_a040_b0p5_s0",
                "b1": "mla2bv2_e040_a040_b1_s0"}[bc]
    if k == "nat" and bc == "b0":
        return "mla2dfv2_e040_a040_b0_rep_s0"
    if k == "nat":
        return f"mlat_e040_a040_frep_{bc}_s0"
    return f"mlatF_{k}_e040_a040_{arm}_{bc}_s0"

def ex(t): return os.path.exists(f"{RUNS}/{t}/trajectory.pt")

def load(t): return torch.load(f"{RUNS}/{t}/trajectory.pt", map_location="cpu", weights_only=False)

def r2(d):
    prof = pd.DataFrame(d["profiles"]); y = np.asarray(d["innate"], float); cols = []
    for c in prof.columns:
        v = prof[c]
        cols.append(v.values.astype(float)[:, None] if pd.api.types.is_numeric_dtype(v)
                    else pd.get_dummies(v).values.astype(float))
    X = np.hstack(cols); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); yh = np.zeros_like(y)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        w = np.linalg.solve(X[tr].T @ X[tr] + np.eye(X.shape[1]), X[tr].T @ (y[tr] - y[tr].mean()))
        yh[te] = y[tr].mean() + X[te] @ w
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def curves(t):
    d = load(t)
    op = np.asarray(d["op_raw"], float)
    pred = np.asarray(d["pred_raw"], float)
    innate = np.asarray(d["innate"], float)
    dr = op.std(1) / innate.std()
    pstd = np.array([np.nanstd(np.where(np.isfinite(p), p, np.nan)) for p in pred])
    vr = pstd / np.maximum(op.std(1), 1e-9)
    return r2(d), dr, vr

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

for arm, fname, lab in (("frep", "fdial_llm_dr_vr.png", "fresh weights (MLP protocol)"),
                        ("rep", "fdial_llm_dr_vr_continual.png", "continual weights")):
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.6), constrained_layout=True,
                             sharex=True)
    ts = np.arange(1, 31)
    for ri, (bc, blab) in enumerate(BETAS):
        rows = sorted([curves(tag(k, arm, bc)) for k in KNOBS if ex(tag(k, arm, bc))],
                      key=lambda z: -z[0])
        ramp = plt.cm.GnBu(np.linspace(0.95, 0.35, max(len(rows), 2)))
        axA, axB = axes[ri]
        for (rr, dr, vr), c in zip(rows, ramp):
            axA.plot(ts, dr, color=c, lw=2.2, label=f"{rr:.2f}")
            axB.plot(ts, vr, color=c, lw=2.2)
        axA.set(ylabel=f"$\\beta$={blab}\nDiversity ratio  $d_r$", ylim=(0, 1.05))
        axB.set_ylabel("Platform/population spread  $v_r$")
        axB.axhline(1.0, color="#bbbbbb", lw=1, ls=":")
        if ri == 0:
            axA.legend(title="feature strength $R^2$", frameon=False, fontsize=9,
                       title_fontsize=10, ncols=2)
    for ax in axes[-1]:
        ax.set_xlabel("Timestep")
    fig.suptitle(f"LLM feature dial by anchor strength: Qwen, {lab}, "
                 "replace, trap cell, seed 0", fontsize=13)
    fig.savefig(f"{OUT}/{fname}", dpi=140); plt.close(fig)
    print(f"saved {OUT}/{fname}")
