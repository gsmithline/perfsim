"""Single-platform distortion phase diagram. Panel A: many (alpha,beta) combos
collapse onto one curve in alpha*beta (proving the product identity). Panel B:
3D surface of faithfulness over (alpha*beta, eta).
Run: python experiments/recommender/run_phase_distortion.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import build_recommender_env, engagement_faithfulness
from perfsim.simulator import Simulator

OUT = Path("runs/recommender")
SEEDS = tuple(range(10))


def final_faithfulness(alpha: float, beta: float, eta: float, seed: int,
                       n_rounds: int = 20, epoch_size: int = 8) -> float:
    env = build_recommender_env(n_items=30, n_users=200, dim=8,
                                alpha=alpha, beta=beta, eta=eta, seed=seed)
    model = LinearModel(8, 1)
    sim = Simulator(env=env, learner=ERMLearner(model, MSELoss()), loss=MSELoss())
    last = {"v": float("nan")}
    sim.run(n_rounds=n_rounds, epoch_size=epoch_size, seed=seed,
            on_round=lambda t, r: last.update(v=engagement_faithfulness(env)))
    return last["v"]


def mean_seed(alpha, beta, eta):
    vals = [final_faithfulness(alpha, beta, eta, s) for s in SEEDS]
    return sum(vals) / len(vals), min(vals), max(vals)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 5.5))

    # Panel A: product collapse (eta=0)
    axA = fig.add_subplot(1, 2, 1)
    combos = [(a, b) for b in (2.0, 4.0, 8.0) for a in (0.0, 1.0, 2.0, 4.0, 8.0)]
    xs, ys, los, his = [], [], [], []
    for a, b in combos:
        m, lo, hi = mean_seed(a, b, 0.0)
        xs.append(a * b); ys.append(m); los.append(lo); his.append(hi)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
    los = [los[i] for i in order]; his = [his[i] for i in order]
    axA.fill_between(xs, los, his, color="#3b6fd8", alpha=0.15)
    axA.plot(xs, ys, "o-", color="#3b6fd8")
    axA.set_xscale("symlog")
    axA.set_xlabel("alpha * beta (manipulation strength)")
    axA.set_ylabel("final corr(engagement, true relevance)")
    axA.set_title("(a) all (alpha,beta) combos collapse onto one curve  (eta=0)")
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # Panel B: faithfulness vs product, one line per eta
    axB = fig.add_subplot(1, 2, 2)
    products = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    etas = (0.0, 0.05, 0.15, 0.3)
    cmap = plt.cm.viridis
    xs_b = [max(p, 0.5) for p in products]
    for i, eta in enumerate(etas):
        ys = [mean_seed(prod / 4.0, 4.0, eta)[0] for prod in products]
        axB.plot(xs_b, ys, "o-", color=cmap(i / (len(etas) - 1)),
                 label=f"eta={eta:g}")
    axB.set_xscale("log")
    axB.set_xlabel("alpha * beta (manipulation strength)")
    axB.set_ylabel("final faithfulness")
    axB.set_title("(b) drift lowers the whole curve; manipulation has a threshold")
    axB.legend(fontsize=8, title="drift")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.suptitle("Single-platform distortion: faithfulness vs manipulation strength and drift "
                 f"(mean of seeds {SEEDS})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "phase_distortion.png", dpi=150)

    print(f"{'a*b':>6s} {'eta':>5s} {'faithfulness':>13s}")
    for eta in etas:
        for prod in products:
            m, lo, hi = mean_seed(prod / 4.0, 4.0, eta)
            print(f"{prod:6.0f} {eta:5.2f} {m:13.3f}  [{lo:.3f},{hi:.3f}]")
    print(f"[recommender] figure -> {OUT / 'phase_distortion.png'}")


if __name__ == "__main__":
    main()
