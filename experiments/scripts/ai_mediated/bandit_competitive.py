"""Competitive (top-k) reward vs absolute reward, no LLM.

Hiring is zero-sum: a fixed slot fraction gets hired by score under theta_p. So at
full adoption the hire RATE is unchanged but everyone has paid the cost, and we
check whether the screener also gets worse at picking the truly-qualified (R_H).
Compare Delta_competitive(p) to the absolute Delta(p) from the earlier sweep.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location("bm", "experiments/scripts/ai_mediated/bandit_market_sweep.py")
bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bm)

OUT = Path("runs/two_tickets_analysis")
SLOT = 0.30 


def curves(X, z_cols, y, seed=0):
    Xm = bm.mediate(X, z_cols, y)
    rng = np.random.default_rng(seed)
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=seed, stratify=y)
    d_abs, d_comp, r_h, hire_rate = [], [], [], []
    for p in bm.GRID:
        use = rng.random(len(X)) < p
        obs = np.where(use[:, None], Xm, X)
        sc = StandardScaler().fit(obs[idx_tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(obs[idx_tr]), y[idx_tr])
        s_m = clf.predict_proba(sc.transform(Xm))[:, 1]
        s_r = clf.predict_proba(sc.transform(X))[:, 1]
        s_obs = np.where(use, s_m, s_r)
        d_abs.append(s_m.mean() - s_r.mean())
        thr = np.quantile(s_obs, 1 - SLOT)
        d_comp.append((s_m > thr).mean() - (s_r > thr).mean())
        r_h.append(roc_auc_score(y[idx_te], s_r[idx_te]))   # screener on raw held-out
        hire_rate.append((s_obs > thr).mean())
    return np.array(d_abs), np.array(d_comp), np.array(r_h), np.array(hire_rate)


def replicator_eq(interp, kappa, p0=0.5, steps=6000, dt=0.01):
    p = p0
    for _ in range(steps):
        p = min(1.0, max(0.0, p + dt * p * (1 - p) * (interp(p) - kappa)))
    return p


def main():
    X, z_cols, labels = bm.build_features()
    y = labels["style"]
    d_abs, d_comp, r_h, hire = curves(X, z_cols, y)
    gi_c = lambda p: float(np.interp(p, bm.GRID, d_comp))

    print(f"absolute     Delta(0)={d_abs[0]:+.3f} Delta(1)={d_abs[-1]:+.3f}")
    print(f"competitive  Delta(0)={d_comp[0]:+.3f} Delta(1)={d_comp[-1]:+.3f}  (->0 = advantage fully competed away)")
    print(f"hire rate    p=0 {hire[0]:.2f} -> p=1 {hire[-1]:.2f}  (flat = zero-sum, AI use does not add hires)")
    print(f"screener R_H p=0 {r_h[0]:.3f} -> p=1 {r_h[-1]:.3f}  (drop = worse at picking truly-qualified)")
    for k in (0.02, 0.05, 0.10):
        print(f"  competitive replicator, kappa={k:.2f} -> p*={replicator_eq(gi_c, k):.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(bm.GRID, d_abs, marker="o", ms=3, label="absolute reward")
    ax[0].plot(bm.GRID, d_comp, marker="o", ms=3, label="competitive (top-k)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("adoption share p"); ax[0].set_ylabel("Delta_gross(p)")
    ax[0].set_title("advantage of using: absolute vs competitive"); ax[0].legend()
    ax[1].plot(bm.GRID, r_h, marker="o", ms=3, color="purple", label="screener R_H (raw)")
    ax[1].plot(bm.GRID, hire, marker="s", ms=3, color="gray", label="hire rate")
    ax[1].set_xlabel("adoption share p"); ax[1].set_ylim(0, 1.0)
    ax[1].set_title("hires unchanged, screener degrades"); ax[1].legend()
    fig.suptitle("Competitive reward: the red-queen trap (no LLM)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "bandit_competitive.png", dpi=140)
    print(f"saved {OUT/'bandit_competitive.png'}")


if __name__ == "__main__":
    main()
