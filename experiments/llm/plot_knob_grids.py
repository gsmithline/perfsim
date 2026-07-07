"""Knob grids: one 2x2 figure per (model, beta) -- x = timestep everywhere,
left column vr(t), right column dr(t); top row legend = eps_AI (gate dial +
slab gates), bottom row legend = realized probe R2 (feature dial + natural).
Realized R2 is measured from each run's saved (post-shuffle) profiles.
Missing runs are skipped, so this renders partial grids mid-campaign.
"""
import os
import json
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/llm/figs"
MODELS = {"qwen": "mlat", "llama": "mlatL", "gemma": "mlatG"}
BETAS = [("b0", "0"), ("b1", "1")]

GATE_LEVELS = [("a005", 0.05), ("a010", 0.10), ("a015", 0.15), ("a020", 0.20),
               ("a030", 0.30), ("a040", 0.40), ("a070", 0.70), ("a100", 1.00)]
FEAT_KNOBS = ["p100", "p065", "p040", "p033", "p015", "p005", "nat",
              "q025", "q050", "q100"]

LEGACY = {
    ("qwen", "e040_a010", "b0"): "mla2dv2_e040_a010_b0_s0",
    ("qwen", "e040_a020", "b0"): "mla2dv2_e040_a020_b0_s0",
    ("qwen", "e040_a040", "b0"): "mla2dv2_e040_a040_b0_s0",
    ("qwen", "e040_a010", "b1"): "mla2bv2_e040_a010_b1_s0",
    ("qwen", "e040_a040", "b1"): "mla2bv2_e040_a040_b1_s0",
}


def load(tag):
    path = f"{RUNS}/{tag}/trajectory.pt"
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    dr = op.std(1) / (inn.std() + 1e-9)
    vr = np.maximum(pr.std(1) / (op.std(1) + 1e-9), 1e-3)   # log-axis floor
    return dr, vr, d


def realized_r2(d):
    prof = pd.DataFrame(d["profiles"])
    y = np.asarray(d["innate"], float)
    cols = []
    for c in prof.columns:
        v = prof[c]
        if v.dtype == object:
            cols.append(pd.get_dummies(v).values.astype(float))
        else:
            cols.append(v.values.astype(float)[:, None])
    X = np.hstack(cols); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); yh = np.zeros_like(y)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        w = np.linalg.solve(X[tr].T @ X[tr] + np.eye(X.shape[1]),
                            X[tr].T @ (y[tr] - y[tr].mean()))
        yh[te] = y[tr].mean() + X[te] @ w
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def gate_tag(model, ac, beta):
    stem = MODELS[model]
    for t in (f"{stem}A_e040_{ac}_rep_{beta}_s0" if model == "qwen" else "",
              f"{stem}_e040_{ac}_rep_{beta}_s0",
              "mlatA_e040_%s_rep_%s_s0" % (ac, beta) if model == "qwen" else ""):
        if t and os.path.exists(f"{RUNS}/{t}/trajectory.pt"):
            return t
    return LEGACY.get((model, f"e040_{ac}", beta))


def feat_tag(model, knob, beta):
    if knob == "nat":
        t = f"{MODELS[model]}_e040_a040_rep_{beta}_s0"
        if os.path.exists(f"{RUNS}/{t}/trajectory.pt"):
            return t
        return LEGACY.get((model, "e040_a040", beta))
    return f"mlatF_{knob}_e040_a040_rep_{beta}_s0" if model == "qwen" else None


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 12, "ytick.labelsize": 12})

for model in MODELS:
    for beta, blab in BETAS:
        gate_rows, feat_rows = [], []
        for ac, aval in GATE_LEVELS:
            tag = gate_tag(model, ac, beta)
            got = load(tag) if tag else None
            if got:
                gate_rows.append((aval, got[0], got[1]))
        for knob in FEAT_KNOBS:
            tag = feat_tag(model, knob, beta)
            got = load(tag) if tag else None
            if got:
                feat_rows.append((realized_r2(got[2]), got[0], got[1]))
        if not gate_rows and not feat_rows:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), constrained_layout=True)
        for r, (rows, key, title) in enumerate([
                (gate_rows, "$\\epsilon_{AI}$", "gate width"),
                (feat_rows, "$R^2$", "feature strength")]):
            rows = sorted(rows, key=lambda z: -z[0])
            ramp = plt.cm.GnBu(np.linspace(0.95, 0.35, max(len(rows), 2)))
            for (lv, dr, vr), c in zip(rows, ramp):
                t = np.arange(1, len(dr) + 1)
                axes[r, 0].plot(t, vr, color=c, lw=2.0, label=f"{lv:.2f}")
                axes[r, 1].plot(t, dr, color=c, lw=2.0)
            axes[r, 0].set_ylabel(f"$v_r$   ({title})", fontsize=14)
            axes[r, 1].set_ylabel("$d_r$", fontsize=14)
            axes[r, 0].set_yscale("log")
            axes[r, 1].set_ylim(0, 1.5)
            if rows:
                axes[r, 0].legend(title=key, fontsize=9, title_fontsize=11,
                                  frameon=True, framealpha=0.9, edgecolor="0.85", ncols=2)
        for ax in axes[-1]:
            ax.set_xlabel("Timestep", fontsize=14)
        fig.suptitle(f"{model}, $\\beta$={blab} (trap-cell world, rep/continual, carry): "
                     "top = gate dial, bottom = feature dial", fontsize=13)
        os.makedirs(f"{FIGS}/{model}", exist_ok=True)
        out = f"{FIGS}/{model}/knob_grid_{beta}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"saved {out}")
