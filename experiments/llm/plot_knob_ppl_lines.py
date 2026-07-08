"""Model-health companion to the knob grids: median log10 perplexity per
round for the gate dial (legend = eps_AI) and feature dial (legend =
realized probe R2), one column per beta. Qwen only (dial runs are Qwen).
"""
import os
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/llm/figs/qwen"
GATE_LEVELS = [("a005", 0.05), ("a010", 0.10), ("a015", 0.15), ("a020", 0.20),
               ("a030", 0.30), ("a040", 0.40), ("a070", 0.70), ("a100", 1.00)]
FEAT_KNOBS = ["p100", "p065", "p040", "p033", "p015", "p005", "nat",
              "q025", "q050", "q100"]
BETAS = [("b0", "0"), ("b0p5", "0.5"), ("b1", "1")]

LEGACY = {}
for c in ["e040_a010", "e040_a020", "e040_a040", "e020_a040"]:
    LEGACY[(c, "b0")] = f"mla2dv2_{c}_b0_s0"
for c in ["e040_a010", "e040_a040", "e020_a040"]:
    for b in ["b0p5", "b1"]:
        LEGACY[(c, b)] = f"mla2bv2_{c}_{b}_s0"


def exists(tag):
    return tag and os.path.exists(f"{RUNS}/{tag}/trajectory.pt")


def gate_tag(ac, beta):
    for t in (f"mlatA_e040_{ac}_rep_{beta}_s0",
              f"mlat_e040_{ac}_rep_{beta}_s0",
              LEGACY.get((f"e040_{ac}", beta), "")):
        if exists(t):
            return t
    return None


def feat_tag(knob, beta):
    if knob == "nat":
        for t in (f"mlat_e040_a040_rep_{beta}_s0", LEGACY.get(("e040_a040", beta), "")):
            if exists(t):
                return t
        return None
    t = f"mlatF_{knob}_e040_a040_rep_{beta}_s0"
    return t if exists(t) else None


def load(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    lp = np.log10(np.clip(np.asarray(d["ppl_raw"], np.float32), 1.0, None))
    return np.median(lp, axis=1), d


def realized_r2(d):
    prof = pd.DataFrame(d["profiles"])
    y = np.asarray(d["innate"], float)
    cols = []
    for c in prof.columns:
        v = prof[c]
        if pd.api.types.is_numeric_dtype(v):
            cols.append(v.values.astype(float)[:, None])
        else:
            cols.append(pd.get_dummies(v).values.astype(float))
    X = np.hstack(cols); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); yh = np.zeros_like(y)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        w = np.linalg.solve(X[tr].T @ X[tr] + np.eye(X.shape[1]),
                            X[tr].T @ (y[tr] - y[tr].mean()))
        yh[te] = y[tr].mean() + X[te] @ w
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, sharey=True,
                         constrained_layout=True)
for ci, (bc, blab) in enumerate(BETAS):
    rows = []
    for ac, aval in GATE_LEVELS:
        tag = gate_tag(ac, bc)
        if tag:
            rows.append((aval, load(tag)[0]))
    ramp = plt.cm.GnBu(np.linspace(0.95, 0.35, max(len(rows), 2)))
    for (lv, y), c in zip(sorted(rows, key=lambda z: -z[0]), ramp):
        axes[0, ci].plot(np.arange(1, len(y) + 1), y, color=c, lw=2.0, label=f"{lv:.2f}")
    axes[0, ci].set_title(f"gate dial, $\\beta$={blab}", fontsize=12)
    if rows:
        axes[0, ci].legend(title="$\\epsilon_{AI}$", fontsize=8, title_fontsize=10,
                           frameon=False, ncols=2)

    rows = []
    for knob in FEAT_KNOBS:
        tag = feat_tag(knob, bc)
        if tag:
            y, d = load(tag)
            rows.append((realized_r2(d), y))
    ramp = plt.cm.GnBu(np.linspace(0.95, 0.35, max(len(rows), 2)))
    for (lv, y), c in zip(sorted(rows, key=lambda z: -z[0]), ramp):
        axes[1, ci].plot(np.arange(1, len(y) + 1), y, color=c, lw=2.0, label=f"{lv:.2f}")
    axes[1, ci].set_title(f"feature dial, $\\beta$={blab}", fontsize=12)
    if rows:
        axes[1, ci].legend(title="$R^2$", fontsize=8, title_fontsize=10,
                           frameon=False, ncols=2)
    axes[1, ci].set_xlabel("round", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("median $\\log_{10}$ ppl", fontsize=12)
fig.suptitle("Qwen ML-Action dials (trap-cell world, rep/continual, seed 0): "
             "model health per round", fontsize=13)
out = f"{FIGS}/knob_ppl_lines.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"saved {out}")
