#!/usr/bin/env python3
"""Post-peer population mean and SD, t = 0..10, for the recursive
update-dose arms (u1/u5/u19 pofdud_ + the reused u181 pofdps_ SFT cell)
against the matched perfect-prediction CPU control.  t = 0 is the innate
population.  NO TITLES; the caption block is printed.  Output stays in
notes/, never paper/."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-ud"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ROOTS = [os.path.join(REPO, "notes/pofd/cluster"),
         os.path.join(REPO, "runs/pokec_gated_lm")]
PP = os.path.join(REPO, "notes/pofd/perfect_prediction",
                  "pp_k1_w1_eaopen_esopen_sw100_s0_r30.pt")
OUT = os.path.join(REPO, "notes/pofd/update_dose")
R = 10
ARMS = [("u1", "pofdud_qwen3_8b_sft_u1_sw100_eaopen_w1_k1_esopen_anch2_s0_r10"),
        ("u5", "pofdud_qwen3_8b_sft_u5_sw100_eaopen_w1_k1_esopen_anch2_s0_r10"),
        ("u19", "pofdud_qwen3_8b_sft_u19_sw100_eaopen_w1_k1_esopen_anch2_s0_r10"),
        ("u181", "pofdps_qwen3_8b_sft_sw100_eaopen_w1_k1_esopen_anch2_s0_r60")]
COLORS = {"u1": "#7fa8d9", "u5": "#4c72b0", "u19": "#2b4f86",
          "u181": "#16294a", "pp": "#c44e52"}


def load(tag):
    for r in ROOTS:
        p = os.path.join(r, tag, "trajectory.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=False)
            return d["op_raw"].float().numpy(), d["innate"].float().numpy()
    raise FileNotFoundError(tag)


def series(op, inn):
    """t=0 (innate) followed by the first R post-peer rounds."""
    m = np.concatenate([[inn.mean()], op[:R].mean(axis=1)])
    s = np.concatenate([[inn.std()], op[:R].std(axis=1)])
    return m, s


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    t = np.arange(R + 1)
    for name, tag in ARMS:
        op, inn = load(tag)
        m, s = series(op, inn)
        lbl = f"SFT, U={name[1:]}" + (" (reused)" if name == "u181" else "")
        axes[0].plot(t, m, marker="o", ms=3.5, lw=1.6,
                     color=COLORS[name], label=lbl)
        axes[1].plot(t, s, marker="o", ms=3.5, lw=1.6, color=COLORS[name])
    d = torch.load(PP, map_location="cpu", weights_only=False)
    m, s = series(d["op_raw"].float().numpy(), d["innate"].float().numpy())
    axes[0].plot(t, m, ls="--", lw=1.6, color=COLORS["pp"],
                 label="perfect prediction")
    axes[1].plot(t, s, ls="--", lw=1.6, color=COLORS["pp"])
    for ax, yl in zip(axes, ("post-peer population mean",
                             "post-peer population SD")):
        ax.set_xlabel("round $t$", fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"update_dose_10r.{ext}"),
                    dpi=180, bbox_inches="tight")
    print(f"[plot_ud] wrote {OUT}/update_dose_10r.pdf/.png")
    print("CAPTION: Qwen3-8B plain SFT on the Section 3 no-memory/full-")
    print("adoption surface (W=1, open gates, S=100, fresh LoRA, seed 0),")
    print("varying ONLY optimizer-step frequency per round via gradient")
    print("accumulation; every arm consumes all 723 live labels once per")
    print("round in the same order. U=181 is ordinary minibatch SFT (the")
    print("archived Section 3 cell); dashed is the matched perfect-")
    print("prediction control. t=0 is the innate population; values are")
    print("POST-PEER.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
