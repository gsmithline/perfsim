"""LLM performance vs usage (eps_AI) on Pokec, from existing gcore runs.
Usage axis = gate eps (adoption breadth). Lines = KL anchor beta.
Three readouts, each the mean of the last 5 rounds:
  A) population diversity  op_std/innate_std   (the real collapse signal)
  B) apparent error   RMSE(pred, current pop)  (the trap: falls with usage)
  C) true error       RMSE(pred, innate)       (accuracy vs real opinion)
Prefer seed 0; fall back to seed 1 where seed 0 is missing/empty.
"""
import os
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/toy/figs"
EPS = [("e010", 0.10), ("e020", 0.20), ("e030", 0.30), ("e040", 0.40)]
BET = [("0", 0.0), ("1", 1.0), ("3", 3.0)]
COL = {"0": "#d62728", "1": "#1f77b4", "3": "#2ca02c"}
TAIL = 5


def sz(p):
    try: return os.path.getsize(p)
    except OSError: return 0


def load(beta, ecode):
    for s in ["s0", "s1"]:
        p = f"{RUNS}/gcore_{ecode}_b{beta}_{s}/trajectory.pt"
        if sz(p) > 1000:
            return torch.load(p, map_location="cpu", weights_only=False), s
    return None, None


def stats(beta, ecode):
    d, s = load(beta, ecode)
    if d is None:
        return None
    op = np.clip(np.asarray(d["op_raw"], dtype=np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], dtype=np.float32), 0, 1)
    inn = np.asarray(d["innate"], dtype=np.float32)
    div = (op.std(1) / (inn.std() + 1e-9))[-TAIL:].mean()
    app = np.sqrt(((pr - op) ** 2).mean(1))[-TAIL:].mean()
    tru = np.sqrt(((pr - inn[None, :]) ** 2).mean(1))[-TAIL:].mean()
    return dict(div=div, app=app, tru=tru, innstd=float(inn.std()))


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
innstd = None
for blab, _ in BET:
    xs, dv, ap, tr = [], [], [], []
    for ec, ev in EPS:
        st = stats(blab, ec)
        if st is None:
            continue
        innstd = st["innstd"]
        xs.append(ev); dv.append(st["div"]); ap.append(st["app"]); tr.append(st["tru"])
    c = COL[blab]
    axes[0].plot(xs, dv, "-o", c=c, label=f"KL $\\beta$={blab}")
    axes[1].plot(xs, ap, "-o", c=c, label=f"KL $\\beta$={blab}")
    axes[2].plot(xs, tr, "-o", c=c, label=f"KL $\\beta$={blab}")

axes[0].axhline(1.0, ls="--", c="gray", lw=1)
axes[0].set(title="A. Population diversity\n(op_std / innate_std)", xlabel="usage  $\\epsilon_{AI}$",
            ylabel="diversity retained")
axes[0].text(0.5, 1.0, " innate", va="bottom", ha="left", c="gray", fontsize=8, transform=axes[0].get_yaxis_transform())
axes[1].set(title="B. Apparent error\n(RMSE vs current population) -- the trap", xlabel="usage  $\\epsilon_{AI}$",
            ylabel="RMSE")
axes[2].axhline(innstd, ls="--", c="gray", lw=1)
axes[2].set(title="C. True error\n(RMSE vs innate real opinion)", xlabel="usage  $\\epsilon_{AI}$", ylabel="RMSE")
axes[2].text(0.02, innstd, " predict-the-mean baseline", va="bottom", ha="left", c="gray", fontsize=8)
for ax in axes:
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.suptitle("LLM (Qwen2.5-7B, KL-SFT) on Pokec: performance vs usage. "
             "Usage collapses the population (A) and makes the model LOOK better (B), "
             "while true accuracy never moves (C).", fontsize=11)
fig.savefig(f"{FIGS}/llm_perf_vs_usage.png", dpi=130)
print(f"saved {FIGS}/llm_perf_vs_usage.png")
