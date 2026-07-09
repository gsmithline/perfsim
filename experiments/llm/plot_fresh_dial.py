"""Fresh vs continual feature dial (Qwen ML-Action, trap cell, seed 0).
Fig A: 2x3 grid, rows = weight regime (continual / fresh), cols = beta,
       median log10 ppl per round, one line per knob colored by realized R2.
Fig B: final log10 ppl vs R2, continual vs fresh, one panel per beta.
Mirrors loader/r2 from experiments/llm/plot_knob_ppl_lines.py.
"""
import os, numpy as np, pandas as pd, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
KNOBS = ["p100", "p065", "p040", "p033", "p015", "p005", "nat"]
BETAS = [("b0", "0"), ("b0p5", "0.5"), ("b1", "1")]
LEG = {("e040_a040", "b0"): "mla2dv2_e040_a040_b0_s0",
       ("e040_a040", "b0p5"): "mla2bv2_e040_a040_b0p5_s0",
       ("e040_a040", "b1"): "mla2bv2_e040_a040_b1_s0"}

def ex(t): return t and os.path.exists(f"{RUNS}/{t}/trajectory.pt")
def cont_tag(k, b):
    if k == "nat":
        for t in (f"mlat_e040_a040_rep_{b}_s0", LEG.get(("e040_a040", b), "")):
            if ex(t): return t
        return None
    t = f"mlatF_{k}_e040_a040_rep_{b}_s0"; return t if ex(t) else None
def fresh_tag(k, b):
    t = f"mlatF_{k}_e040_a040_frep_{b}_s0"; return t if ex(t) else None
def load(tag): return torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
def medlp(d):
    lp = np.log10(np.clip(np.asarray(d["ppl_raw"], np.float32), 1.0, None)); return np.median(lp, axis=1)
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

# gather series[weight][beta] = list of (R2, curve)
data = {"continual": {}, "fresh": {}}
r2cache = {}
for bc, _ in BETAS:
    for wname, tf in (("continual", cont_tag), ("fresh", fresh_tag)):
        rows = []
        for k in KNOBS:
            t = tf(k, bc)
            if not t: continue
            d = load(t)
            rr = r2cache.setdefault(k, r2(d))
            rows.append((rr, medlp(d)))
        data[wname][bc] = sorted(rows, key=lambda z: -z[0])

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

# ---- Fig A: 2x3 trajectory grid ----
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True, sharey=True,
                         constrained_layout=True)
for ci, (bc, blab) in enumerate(BETAS):
    for ri, wname in enumerate(("continual", "fresh")):
        ax = axes[ri, ci]; rows = data[wname][bc]
        ramp = plt.cm.GnBu(np.linspace(0.95, 0.35, max(len(rows), 2)))
        for (lv, y), c in zip(rows, ramp):
            ax.plot(np.arange(1, len(y) + 1), y, color=c, lw=2.0, label=f"{lv:.2f}")
        ax.set_title(f"{wname},  $\\beta$={blab}", fontsize=12)
        if ci == 0:
            ax.legend(title="$R^2$", fontsize=8, title_fontsize=10, frameon=False, ncols=2, loc="upper left")
for ax in axes[1, :]:
    ax.set_xlabel("round", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("median $\\log_{10}$ ppl", fontsize=12)
fig.suptitle("Fresh vs continual feature dial (Qwen ML-Action, replace, seed 0): "
             "fresh flattens the extremes but detonates the mid-$R^2$ knobs at $\\beta$=0",
             fontsize=13)
fig.savefig(f"{OUT}/fresh_vs_continual_ppl.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/fresh_vs_continual_ppl.png")

# ---- Fig B: final log10 ppl vs R2, continual vs fresh, per beta ----
COL = {"continual": "#2f6f9f", "fresh": "#e08214"}  # blue / orange, CVD-safe pair
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True, constrained_layout=True)
for ci, (bc, blab) in enumerate(BETAS):
    ax = axes[ci]
    for wname in ("continual", "fresh"):
        rows = sorted(data[wname][bc], key=lambda z: z[0])  # by R2 ascending
        xs = [rr for rr, _ in rows]; ys = [y[-1] for _, y in rows]
        ax.plot(xs, ys, "-o", color=COL[wname], lw=2.0, ms=7, label=wname)
    ax.axhline(np.log10(1000), color="#999", lw=1, ls="--")
    ax.text(0.02, np.log10(1000) + 0.05, "ppl 1000", color="#777", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_title(f"$\\beta$={blab}", fontsize=12); ax.set_xlabel("realized probe $R^2$", fontsize=12)
    if ci == 0:
        ax.set_ylabel("final median $\\log_{10}$ ppl", fontsize=12)
        ax.legend(frameon=False, fontsize=11)
fig.suptitle("Final model health vs feature strength: continual is monotone in $R^2$; "
             "fresh is non-monotone (mid-$R^2$ peak) at $\\beta$=0", fontsize=13)
fig.savefig(f"{OUT}/fresh_vs_continual_final.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/fresh_vs_continual_final.png")
