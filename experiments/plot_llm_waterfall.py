"""LLM (Qwen2.5-7B + LoRA, KL-SFT) waterfalls on Pokec, from existing runs.
Grid: rows = KL anchor beta, columns = gate eps (gcore family, seed 0).
Each panel: population opinion distribution over rounds (magma) + platform
prediction band (cyan), with dr = op_std_final/innate_std and vr = pred_std/op_std.
The LLM analog of our gate x anchor equilibria map.
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
BETAS = [("0", 0.0), ("1", 1.0), ("3", 3.0)]      # KL anchor
EPS = [("e010", 0.10), ("e020", 0.20), ("e030", 0.30), ("e040", 0.40)]
bins = np.linspace(0, 1, 41)


def load(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], dtype=np.float32), 0, 1)
    pred = np.clip(np.asarray(d["pred_raw"], dtype=np.float32), 0, 1)
    innate = np.asarray(d["innate"], dtype=np.float32)
    return op, pred, innate


fig, axes = plt.subplots(len(BETAS), len(EPS), figsize=(16, 10), constrained_layout=True)
for i, (blab, _) in enumerate(BETAS):
    for j, (ecode, eval_) in enumerate(EPS):
        ax = axes[i][j]
        tag = f"gcore_{ecode}_b{blab}_s0"
        try:
            op, pred, innate = load(tag)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"missing\n{tag}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([]); continue
        R = op.shape[0]
        H = np.array([np.histogram(op[t], bins=bins, density=True)[0] for t in range(R)])
        ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, R, 0, 1])
        rr = np.arange(R)
        pmean = pred.mean(1); pstd = pred.std(1)
        ax.plot(rr, pmean, c="#00e5ff", lw=1.2)
        ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1), color="#00e5ff", alpha=0.25)
        dr = float(op[-1].std()) / (float(innate.std()) + 1e-9)
        vr = float(pred[-1].std()) / (float(op[-1].std()) + 1e-9)
        ax.text(0.96, 0.05, f"dr={dr:.2f} vr={vr:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                color="white", fontsize=8)
        if i == 0:
            ax.set_title(f"$\\epsilon$={eval_:.2f}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"KL $\\beta$={blab}\nopinion")
        if i == len(BETAS) - 1:
            ax.set_xlabel("round")
fig.suptitle("LLM (Qwen2.5-7B, KL-SFT) waterfalls on Pokec: rows = KL anchor $\\beta$, cols = gate $\\epsilon$ "
             "(gcore, seed 0). Population magma, platform band cyan.", fontsize=12)
fig.savefig(f"{FIGS}/llm_gcore_waterfalls.png", dpi=130)
print(f"saved {FIGS}/llm_gcore_waterfalls.png")
