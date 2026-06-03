"""Phase diagram: equilibrium adoption vs cost-of-use kappa (no LLM).

Compute the cost-free advantage Delta_gross(p) once for the style label, then for
each kappa run the replicator to its equilibrium. Delta_gross is decreasing, so:
  kappa < Delta_gross(1)        -> Delta>0 everywhere -> monoculture (p*=1)
  Delta_gross(1)<kappa<Delta_gross(0) -> stable interior p* -> pluralism
  kappa > Delta_gross(0)        -> Delta<0 everywhere -> no adoption (p*=0)
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location("bm", "experiments/scripts/ai_mediated/bandit_market_sweep.py")
bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bm)

OUT = Path("runs/two_tickets_analysis")
KAPPAS = np.linspace(0.0, 0.40, 41)


def replicator_eq(gross_interp, kappa, p0, steps=6000, dt=0.01):
    p = p0
    for _ in range(steps):
        p = min(1.0, max(0.0, p + dt * p * (1 - p) * (gross_interp(p) - kappa)))
    return p


def main():
    X, z_cols, labels = bm.build_features()
    y = labels["style"]
    gross = bm.delta_curve(X, z_cols, y) + bm.KAPPA  # add back the cost to get Delta_gross
    gi = lambda p: float(np.interp(p, bm.GRID, gross))
    g0, g1 = gross[0], gross[-1]

    eq = []
    for k in KAPPAS:
        ends = [replicator_eq(gi, k, p0) for p0 in (0.1, 0.5, 0.9)]
        eq.append(np.mean(ends))
    eq = np.array(eq)

    print(f"Delta_gross(0)={g0:.3f}  Delta_gross(1)={g1:.3f}")
    print(f"predicted thresholds: monoculture below kappa={g1:.3f}, no-adoption above kappa={g0:.3f}")
    for k in (0.0, 0.10, 0.18, 0.25, 0.33, 0.40):
        print(f"  kappa={k:.2f} -> equilibrium adoption p*={replicator_eq(gi, k, 0.5):.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(KAPPAS, eq, marker="o", ms=3)
    ax.axvline(g1, ls="--", c="green", label=f"Delta_gross(1)={g1:.2f} (monoculture below)")
    ax.axvline(g0, ls="--", c="red", label=f"Delta_gross(0)={g0:.2f} (no adoption above)")
    ax.set_xlabel("cost of using the assistant  kappa")
    ax.set_ylabel("equilibrium adoption  p*")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Bandit-market phase diagram: adoption vs cost (style label, no LLM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bandit_kappa_phase.png", dpi=140)
    print(f"saved {OUT/'bandit_kappa_phase.png'}")


if __name__ == "__main__":
    main()
