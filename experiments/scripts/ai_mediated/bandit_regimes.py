"""Three regimes of the adoption x collapse system (no LLM, no data).

State: p = adoption share, h = style homogenization of what theta sees.
  p_dot = p(1-p) [ A(1-h) - kappa ]            # replicator: gaming gain A(1-h), cost kappa
  h_dot = alpha (mu p - h)        if mu p >= h  # homogenization rises toward mu*p
        = rho alpha (mu p - h)    if mu p <  h  # relaxes down at rate rho*alpha
  merit q = 1 - h                              # screener's merit-power; q->0 = hiring at chance

mu  = mediation strength (homogenization at full adoption)
rho = reversibility of the population (1 = D^H recovers, 0 = internalized ratchet)

Predicted equilibria:
  A(1-mu) > kappa            -> monoculture p*=1,  q*=1-mu              (weak mediation)
  A(1-mu) < kappa, rho>0     -> interior p*=(1-kappa/A)/mu, q*=1-mu p*  (strong + reversible)
  A(1-mu) < kappa, rho~0     -> ratchet: p collapses, h stuck, q stuck low (irreversible)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("runs/two_tickets_analysis")
A = 0.35
R = 3.0          # replicator rate (fast adoption relative to theta recovery)
ALPHA = 0.03     # homogenization / theta-recovery rate (slow -> adoption overshoots)
DT = 0.05
T = 4000


def simulate(mu, rho, kappa, p0=0.05, h0=0.0):
    p, h = p0, h0
    P, H = [], []
    for _ in range(T):
        delta = A * (1 - h) - kappa
        dp = R * p * (1 - p) * delta
        target = mu * p
        dh = ALPHA * (target - h) if target >= h else rho * ALPHA * (target - h)
        p = min(1.0, max(0.0, p + DT * dp))
        h = min(1.0, max(0.0, h + DT * dh))
        P.append(p); H.append(h)
    return np.array(P), np.array(H)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    regimes = [
        ("weak mediation",        dict(mu=0.30, rho=1.0, kappa=0.05)),
        ("strong + reversible",   dict(mu=0.90, rho=1.0, kappa=0.15)),
        ("strong + irreversible", dict(mu=0.90, rho=0.0, kappa=0.15)),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for j, (name, par) in enumerate(regimes):
        P, H = simulate(**par)
        q = 1 - H
        ax[j].plot(P, label="adoption p", color="#3b7dd8")
        ax[j].plot(q, label="merit q = 1-h", color="purple")
        ax[j].set_title(name); ax[j].set_xlabel("time"); ax[j].set_ylim(-0.05, 1.05)
        ax[j].legend()
        astar = A * (1 - par["mu"]) - par["kappa"]
        pstar = (1 - par["kappa"] / A) / par["mu"]
        print(f"[{name:22s}] A(1-mu)-kappa={astar:+.3f}  predicted interior p*={pstar:.2f}  "
              f"-> final p={P[-1]:.2f} h={H[-1]:.2f} q={q[-1]:.2f}")
    fig.suptitle("Adoption x collapse: three regimes (replicator + theta recovery, no LLM)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "bandit_regimes.png", dpi=140)

    # regime map over (mu, rho) at fixed kappa
    mus = np.linspace(0.1, 1.0, 28)
    rhos = np.linspace(0.0, 1.0, 28)
    Z = np.zeros((len(rhos), len(mus)))
    for i, rho in enumerate(rhos):
        for k, mu in enumerate(mus):
            P, H = simulate(mu, rho, kappa=0.15)
            if P[-1] > 0.9:
                Z[i, k] = 0          # monoculture
            elif P[-1] > 0.05:
                Z[i, k] = 1          # interior
            else:
                Z[i, k] = 2 if H[-1] > 0.2 else 3   # ratchet (stuck) vs clean exit
    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    im = ax2.imshow(Z, origin="lower", extent=[mus[0], mus[-1], rhos[0], rhos[-1]],
                    aspect="auto", cmap="viridis", vmin=0, vmax=3)
    ax2.set_xlabel("mediation strength mu"); ax2.set_ylabel("reversibility rho")
    ax2.set_title("regime map (kappa=0.15): 0=monoculture 1=interior 2=ratchet 3=exit")
    fig2.colorbar(im, ticks=[0, 1, 2, 3])
    fig2.tight_layout()
    fig2.savefig(OUT / "bandit_regime_map.png", dpi=140)
    print(f"saved {OUT/'bandit_regimes.png'} and {OUT/'bandit_regime_map.png'}")


if __name__ == "__main__":
    main()
