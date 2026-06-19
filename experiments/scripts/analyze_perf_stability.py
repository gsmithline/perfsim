"""Performative-stability plots for the 4 corners.

Function-space convergence: mean |Δ probe_pred| between consecutive rounds -- the
deployed prediction function's step toward a performatively stable point. Decay to
~0 = the model reproduces itself (stable). Weight-space: the LoRA adapter step
w_step (fresh corners only -- continual ginn predates it). Pure JSON, no LLM.

rows = eps in {0.2, 0.4}, cols = the 4 corners, one line per KL beta.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/scripts/analyze_perf_stability.py
"""

import json
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "runs/pokec_gated_lm"
CORNERS = [("continual+replace", "ginn_%s_l00_%s_s0"),
           ("fresh+replace", "gfr_%s_%s_s0"),
           ("fresh+accumulate", "gfa_%s_%s_s0"),
           ("fresh+pristine", "gfp_%s_%s_s0")]
EPS = [0.2, 0.4]
BETAS = [(0.0, "b0"), (0.1, "b0p1"), (0.5, "b0p5"), (1.0, "b1")]
COL = {0.0: "#1f77b4", 0.1: "#2ca02c", 0.5: "#ff7f0e", 1.0: "#d62728"}


def steps(d):
    tel = [json.loads(l) for l in open(d + "/telemetry.json") if l.strip()]
    P = []
    for r in tel:
        pp = r.get("probe_pred")
        P.append(np.array([np.nan if v is None else float(v) for v in pp]) if pp else None)
    fr, fs = [], []
    for i in range(1, len(P)):
        if P[i] is not None and P[i - 1] is not None:
            fs.append(np.nanmean(np.abs(P[i] - P[i - 1])))
            fr.append(tel[i].get("round", i))
    ws = [(r.get("round"), r.get("w_step")) for r in tel if r.get("w_step") is not None]
    return np.array(fr), np.array(fs), ws


def rolling(a, w=5):
    if len(a) < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def panel(ax, pat, eps, metric):
    e = "e%03d" % int(eps * 100)
    any_data = False
    for beta, btag in BETAS:
        d = os.path.join(ROOT, pat % (e, btag))
        if not os.path.exists(d):
            continue
        fr, fs, ws = steps(d)
        if metric == "f" and len(fs):
            ax.plot(fr, np.clip(rolling(fs), 1e-4, None), color=COL[beta], lw=1.6, label=f"beta={beta}")
            any_data = True
        elif metric == "w" and ws:
            ax.plot([w[0] for w in ws], [w[1] for w in ws], color=COL[beta], lw=1.4, label=f"beta={beta}")
            any_data = True
    return any_data


def figure(metric, fname, ylabel, logy, hline=None):
    fig, axes = plt.subplots(len(EPS), len(CORNERS), figsize=(4 * len(CORNERS), 3 * len(EPS)),
                             squeeze=False, sharex=True)
    for i, eps in enumerate(EPS):
        for j, (name, pat) in enumerate(CORNERS):
            ax = axes[i][j]
            ok = panel(ax, pat, eps, metric)
            if not ok:
                ax.text(0.5, 0.5, "not logged", ha="center", va="center", transform=ax.transAxes)
            if logy:
                ax.set_yscale("log")
            if hline is not None:
                ax.axhline(hline, ls="--", color="gray", lw=0.8)
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"eps={eps}\n{ylabel}", fontsize=9)
            if i == len(EPS) - 1:
                ax.set_xlabel("round")
            if i == 0 and j == len(CORNERS) - 1:
                ax.legend(fontsize=7, frameon=False)
    title = "function-space step |Δ probe_pred|" if metric == "f" else "weight-space step w_step"
    fig.suptitle("Performative stability: " + title, fontsize=12)
    fig.tight_layout()
    out = "experiments/competition/figs/" + fname
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


figure("f", "perf_stability_fspace.png", "|d probe_pred|", True, hline=0.01)
figure("w", "perf_stability_wspace.png", "w_step", False)
