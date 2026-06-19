"""Population opinion-vector L2 across rounds (convergence in opinion space).

Step:  ||op_t - op_{t-1}||_2 over the per-agent opinion vector (op_raw from
       trajectory.pt). Decay to ~0 = the population freezes (a stable point in
       opinion space) -- the population-side analogue of the function-space step.
Drift: ||op_t - op_0||_2 -- cumulative displacement from the initial opinions.

Raw L2 is comparable across these runs (same N). Per-agent RMS = L2 / sqrt(N),
on the [0,1] opinion scale. Needs op_raw, so pull trajectory.pt for these runs.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/scripts/analyze_pop_l2.py
"""

import os
import numpy as np
import torch

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


def op_raw(d):
    tp = os.path.join(d, "trajectory.pt")
    if not os.path.exists(tp):
        return None
    o = torch.load(tp, map_location="cpu", weights_only=False).get("op_raw")
    if o is None or not hasattr(o, "numel") or o.numel() == 0:
        return None
    return np.asarray(o, dtype=np.float64)   # [T, N]


def panel(ax, pat, eps, mode):
    e = "e%03d" % int(eps * 100)
    any_data = False
    for beta, btag in BETAS:
        o = op_raw(os.path.join(ROOT, pat % (e, btag)))
        if o is None:
            continue
        if mode == "step":
            v = np.linalg.norm(o[1:] - o[:-1], axis=1)
            r = np.arange(1, o.shape[0])
        else:
            v = np.linalg.norm(o - o[0], axis=1)
            r = np.arange(o.shape[0])
        ax.plot(r, np.clip(v, 1e-5, None), color=COL[beta], lw=1.5, label=f"beta={beta}")
        any_data = True
    return any_data


def figure(mode, fname, ylabel, logy):
    fig, axes = plt.subplots(len(EPS), len(CORNERS), figsize=(4 * len(CORNERS), 3 * len(EPS)),
                             squeeze=False, sharex=True)
    for i, eps in enumerate(EPS):
        for j, (name, pat) in enumerate(CORNERS):
            ax = axes[i][j]
            if not panel(ax, pat, eps, mode):
                ax.text(0.5, 0.5, "no op_raw\n(pull trajectory.pt)", ha="center",
                        va="center", transform=ax.transAxes)
            if logy:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"eps={eps}\n{ylabel}", fontsize=9)
            if i == len(EPS) - 1:
                ax.set_xlabel("round")
            if i == 0 and j == len(CORNERS) - 1:
                ax.legend(fontsize=7, frameon=False)
    ttl = "step ||op_t - op_{t-1}||_2" if mode == "step" else "drift ||op_t - op_0||_2"
    fig.suptitle("Population opinion L2: " + ttl, fontsize=12)
    fig.tight_layout()
    out = "experiments/competition/figs/" + fname
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


figure("step", "pop_l2_step.png", "L2 step", True)
figure("drift", "pop_l2_drift.png", "L2 drift", False)
