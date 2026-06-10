"""Per-round trajectories for the prior-pull endpoint result (block 3, Exp 3).

Endpoints showed shared prior = no protection, distinct priors = full rescue.
Endpoints can't say whether a shared prior *accelerates* the collapse relative
to no prior, so here we record sep and pop circ-var every round at the
collapsing malleability (0.6) and compare collapse speed.

Run: python experiments/competition/fig_prior_traj.py
"""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_s2 = importlib.util.spec_from_file_location("b2", "experiments/competition/02_phase_diagram.py")
b2 = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(b2)
_s3 = importlib.util.spec_from_file_location("b3", "experiments/competition/03_prior_pull.py")
b3 = importlib.util.module_from_spec(_s3)
_s3.loader.exec_module(b3)

OUT = Path("experiments/competition/figs")

K, TAU, MALL = 3, 0.2, 0.6
SEG = torch.tensor([(i + 0.5) / K for i in range(K)])
SHARED = torch.full((K,), 0.5)
SEEDS = [0, 1]
ROUNDS = 25

# (tag, label, prior points, prior_w, color)
CONDS = [
    ("none", "no prior", SHARED, 0.0, "#777777"),
    ("shared01", "shared w=0.1", SHARED, 0.1, "#d8633b"),
    ("shared03", "shared w=0.3", SHARED, 0.3, "#a03b1e"),
    ("distinct01", "distinct w=0.1", SEG, 0.1, "#3b6fd8"),
]


def run_traj(prior_pts, prior_w, seed, n=3000, degree=10, peer_share=0.5):
    torch.manual_seed(seed)
    w_graph = b2.random_graph(n, degree, seed)
    innate = torch.rand(n)
    x = innate.clone()
    pos = SEG.clone()
    sep, cv = [], []
    for _ in range(ROUNDS):
        pos = b3.compete_prior(pos, x, TAU, prior_pts, prior_w)
        x = b2.fj_step(x, innate, w_graph, pos, TAU, MALL, peer_share)
        sep.append(b2.plat_min_gap(pos) * K)
        cv.append(b2.circ_var(x))
    return sep, cv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traj = {}
    print(f"Prior-pull trajectories: mall={MALL}, K={K}, tau={TAU}, {ROUNDS} rounds.")
    for tag, label, pts, w, _ in CONDS:
        runs = [run_traj(pts, w, s) for s in SEEDS]
        sep = [sum(r[0][t] for r in runs) / len(runs) for t in range(ROUNDS)]
        cv = [sum(r[1][t] for r in runs) / len(runs) for t in range(ROUNDS)]
        traj[tag] = {"sep": sep, "cv": cv}
        half = next((t for t, v in enumerate(sep) if v < 0.5), None)
        print(f"{label:>15}: final sep={sep[-1]:.3f} cv={cv[-1]:.3f} "
              f"rounds-to-sep<0.5={'never' if half is None else half}", flush=True)

    json.dump(traj, open(OUT / "prior_traj.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for tag, label, _, _, color in CONDS:
        ax[0].plot(traj[tag]["sep"], color=color, label=label)
        ax[1].plot(traj[tag]["cv"], color=color, label=label)
    ax[0].set(xlabel="round", ylabel="platform sep", title=f"segmentation, mall={MALL}")
    ax[1].set(xlabel="round", ylabel="pop circ-var", title="population diversity")
    ax[0].legend(frameon=False)
    fig.savefig(OUT / "prior_traj.png", dpi=150)
    print(f"saved {OUT / 'prior_traj.png'}")


if __name__ == "__main__":
    main()
