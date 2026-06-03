"""Does a NON-INJECTIVE assistant collapse recoverable signal? (no LLM)

The affine mediator preserves separability (injective), so recoverability never
dropped. Here the assistant snaps each style vector to the nearest of K centroids
(many-to-one), content preserved. As K shrinks, distinct people collapse onto the
same template. We sweep K and measure recoverability of the label:
  - style-borne label  -> should collapse toward chance as K -> 1
  - content label      -> should hold (signal lives in c, untouched)
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfsim.scenarios.ai_mediated import recoverability, style_variance

_spec = importlib.util.spec_from_file_location("bm", "experiments/scripts/ai_mediated/bandit_market_sweep.py")
bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bm)

OUT = Path("runs/two_tickets_analysis")
KS = [1, 2, 4, 8, 16, 32, 64, 128]


def snap(X, z_cols, K):
    Xm = X.copy()
    z = X[:, z_cols]
    km = KMeans(n_clusters=min(K, len(z)), n_init=4, random_state=0).fit(z)
    Xm[:, z_cols] = km.cluster_centers_[km.labels_]
    return Xm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, z_cols, labels = bm.build_features()
    zmask = torch.zeros(X.shape[1], dtype=torch.bool); zmask[z_cols] = True
    base_div = style_variance({"x": torch.tensor(X)}, zmask)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ["style", "content"]:
        y = labels["style" if name == "style" else "experience"]
        rec, div = [], []
        for K in KS:
            Xm = snap(X, z_cols, K)
            rec.append(recoverability({"x": torch.tensor(Xm),
                                       "y": torch.tensor(y, dtype=torch.float32).unsqueeze(-1)}))
            div.append(style_variance({"x": torch.tensor(Xm)}, zmask) / base_div)
        rec0 = recoverability({"x": torch.tensor(X), "y": torch.tensor(y, dtype=torch.float32).unsqueeze(-1)})
        ax[0].plot(KS, rec, marker="o", label=f"{name} label")
        ax[1].plot(KS, div, marker="o", label=f"{name} label")
        print(f"[{name:8s}] recoverability raw={rec0:.3f}  K=1 {rec[0]:.3f} -> K=128 {rec[-1]:.3f}")
    ax[0].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("K templates"); ax[0].set_ylabel("recoverability (AUC)")
    ax[0].set_title("non-injective snap: does merit collapse?"); ax[0].legend()
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("K templates"); ax[1].set_ylabel("style diversity (frac of raw)")
    ax[1].set_title("style diversity vs K"); ax[1].legend()
    fig.suptitle("Non-injective (snap-to-K-templates) assistant, no LLM", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "noninjective_sweep.png", dpi=140)
    print(f"saved {OUT/'noninjective_sweep.png'}")


if __name__ == "__main__":
    main()
