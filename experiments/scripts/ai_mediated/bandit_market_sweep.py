"""Delta(p) sweep for the bandit-market replicator, no LLM.

One assistant vs no-use. At adoption share p the screener theta_p retrains on the
p-mixture (p mediated, 1-p raw); we compute the per-arm payoff under theta_p and
Delta(p) = r_use(p) - r_notuse(p) - kappa. The sign of Delta' over p decides the
replicator flow: Delta' < 0 -> interior pluralism, Delta' > 0 -> tipping to
monoculture. Quasi assistant = pull style z toward the winners' style centroid
(content c preserved). We look at a content label (benign) vs a style-borne label.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib.util
_spec = importlib.util.spec_from_file_location("rl", "experiments/scripts/ai_mediated/run_resume_llm_loop.py")
_rl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rl)

CACHE = Path.home() / ".cache/perfsim/datasets/two_tickets"
OUT = Path("runs/two_tickets_analysis"); OUT.mkdir(parents=True, exist_ok=True)
LAMBDA = 0.6      # mediation strength (pull toward winner style)
KAPPA = 0.02      # cost of using the assistant
PCA_DIM = 64
GRID = np.linspace(0.0, 1.0, 21)


def build_features():
    d = np.load(CACHE / "embeddings.npz")
    df = pd.read_csv(CACHE / "final_paper_resume_outputs_doordash.csv")
    df = df[df["CV"].notna()].reset_index(drop=True)
    raw = d["orig"]
    pca = PCA(PCA_DIM, random_state=0).fit(raw)
    c = StandardScaler().fit_transform(pca.transform(raw))
    z = StandardScaler().fit_transform(_rl.style_features(df["CV"].tolist()))
    n_c = c.shape[1]
    X = np.hstack([c, z]).astype(np.float32)
    z_cols = np.arange(n_c, n_c + z.shape[1])
    exp = pd.to_numeric(df["Experience Years"], errors="coerce").to_numpy()
    y_exp = (exp > np.nanmedian(exp)).astype(int)
    rng = np.random.default_rng(0)
    w_z = rng.standard_normal(len(z_cols))
    s = z @ w_z
    y_sty = (s > np.median(s)).astype(int)
    return X, z_cols, {"experience": y_exp, "style": y_sty}


def mediate(X, z_cols, y):
    target = X[y == 1][:, z_cols].mean(0)          # winners' style centroid
    Xm = X.copy()
    Xm[:, z_cols] = (1 - LAMBDA) * X[:, z_cols] + LAMBDA * target
    return Xm


def delta_curve(X, z_cols, y, seed=0):
    Xm = mediate(X, z_cols, y)
    rng = np.random.default_rng(seed)
    out = []
    for p in GRID:
        use = rng.random(len(X)) < p
        obs = np.where(use[:, None], Xm, X)
        sc = StandardScaler().fit(obs)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(obs), y)
        r_use = clf.predict_proba(sc.transform(Xm))[:, 1].mean()
        r_not = clf.predict_proba(sc.transform(X))[:, 1].mean()
        out.append((r_use - r_not) - KAPPA)
    return np.array(out)


def replicator(delta_fn, p0, steps=4000, dt=0.01):
    p = p0
    for _ in range(steps):
        p = min(1.0, max(0.0, p + dt * p * (1 - p) * delta_fn(p)))
    return p


def main():
    X, z_cols, labels = build_features()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, name in enumerate(["experience", "style"]):
        y = labels[name]
        D = delta_curve(X, z_cols, y)
        interp = lambda p: float(np.interp(p, GRID, D))
        ax[0].plot(GRID, D, marker="o", ms=3, label=name)
        slope = np.polyfit(GRID, D, 1)[0]
        ends = {p0: replicator(interp, p0) for p0 in (0.1, 0.5, 0.9)}
        roots = GRID[:-1][(D[:-1] > 0) != (D[1:] > 0)]
        print(f"[{name}] Delta(0)={D[0]:+.3f} Delta(1)={D[-1]:+.3f} slope={slope:+.3f} "
              f"zero-crossing@p={roots.round(2).tolist()} replicator_ends={ {k: round(v,2) for k,v in ends.items()} }")
        for p0, pe in ends.items():
            ax[1].plot([0, 1], [p0, pe], alpha=0.5)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xlabel("adoption share p")
    ax[0].set_ylabel("Delta(p) = r_use - r_notuse - kappa"); ax[0].set_title("fitness gap vs share")
    ax[0].legend()
    ax[1].set_xlabel("(start, end)"); ax[1].set_ylabel("p"); ax[1].set_title("replicator endpoints")
    ax[1].set_ylim(-0.05, 1.05)
    fig.suptitle(f"Bandit-market Delta(p) sweep (lambda={LAMBDA}, kappa={KAPPA}, no LLM)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "bandit_market_sweep.png", dpi=140)
    print(f"saved {OUT/'bandit_market_sweep.png'}")


if __name__ == "__main__":
    main()
