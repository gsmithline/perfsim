"""End-to-end PP loop with an MLP predictor and gradient-strategic agents.

Trains an MLPModel on a synthetic 2-D binary classification problem where
agents best-respond to the current predictor via the input gradient of its
output. Each round:

  1. Population computes ∂f(x_0; θ_t) / ∂x at every agent's x_0.
  2. Agents shift: x_t = x_0 + ε · ∂f/∂x (Perdomo sign: ε = -μ).
  3. Predictor sees (x_t, y) and runs k SGD steps on BCE-with-logits loss.

Saves a 2x2 panel: (top-left) initial vs final decision boundary on the
fixed grid; (top-right) per-round PR; (bottom-left) per-round ||θ_t -
θ_{t-1}||; (bottom-right) ||shift||_2 per round (how much agents move).

Run:
    python examples/nonlinear_strategic.py
    python examples/nonlinear_strategic.py --mu 1.0 --n-rounds 30
    python examples/nonlinear_strategic.py --hidden 32 32 --rgd-lr 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from perfsim.history import History
from perfsim.learners import GradientLearner
from perfsim.losses import BCEWithLogitsLoss, L2RegularizedLoss
from perfsim.metrics import performative_risk
from perfsim.models import MLPModel
from perfsim.simulator import Simulator
from perfsim.worlds.strategic_gradient import StrategicGradientWorld


def _make_population(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x_pos = torch.randn(n // 2, 2, generator=g) + torch.tensor([1.5, 0.0])
    x_neg = torch.randn(n // 2, 2, generator=g) + torch.tensor([-1.5, 0.0])
    x0 = torch.cat([x_pos, x_neg], dim=0)
    y = torch.cat([torch.ones(n // 2, 1), torch.zeros(n // 2, 1)], dim=0)
    perm = torch.randperm(n, generator=g)
    return x0[perm], y[perm]


def _decision_grid(model: MLPModel, lim: float = 4.0, n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-lim, lim, n)
    ys = torch.linspace(-lim, lim, n)
    XX, YY = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=-1)
    with torch.no_grad():
        logits = model(grid).reshape(n, n)
        probs = torch.sigmoid(logits)
    return XX.numpy(), YY.numpy(), probs.numpy()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mu", type=float, default=0.5, help="strategic strength (epsilon = -mu)")
    p.add_argument("--n-agents", type=int, default=400)
    p.add_argument("--n-rounds", type=int, default=20)
    p.add_argument("--hidden", type=int, nargs="+", default=[16, 16])
    p.add_argument("--rgd-lr", type=float, default=0.05)
    p.add_argument("--rgd-steps", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "nonlinear_strategic.png",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    torch.manual_seed(args.seed)

    x0, y = _make_population(args.n_agents, args.seed)
    world = StrategicGradientWorld(x0=x0, y=y, epsilon=-args.mu)

    model = MLPModel(in_features=2, hidden_dims=args.hidden, init_seed=args.seed)
    base_loss = BCEWithLogitsLoss()
    train_loss = L2RegularizedLoss(base_loss, weight_decay=args.weight_decay, decay_bias=False)
    learner = GradientLearner(
        model, train_loss, lr=args.rgd_lr, steps_per_round=args.rgd_steps, optimizer="adam"
    )

    initial_XX, initial_YY, initial_prob = _decision_grid(model)

    def pr_metric(sim: Simulator) -> torch.Tensor:
        return performative_risk(sim.world, sim.learner.model, base_loss)

    def shift_norm(sim: Simulator) -> torch.Tensor:
        data = sim.world.sample(sim.learner.model)
        return (data["x"] - x0).norm(dim=-1).mean()

    sim = Simulator(
        world=world,
        learner=learner,
        loss=train_loss,
        metrics={"PR": pr_metric, "shift_norm": shift_norm},
    )
    history: History = sim.run(n_rounds=args.n_rounds, seed=args.seed)

    final_XX, final_YY, final_prob = _decision_grid(model)

    rounds = [int(r["round"]) for r in history.records]
    prs = [float(r["PR"].item()) for r in history.records]
    gaps = [
        float(r["stability_gap"].item()) if "stability_gap" in r else None
        for r in history.records
    ]
    shifts = [float(r["shift_norm"].item()) for r in history.records]

    print(f"# initial PR = {prs[0]:.6f}")
    print(f"# final   PR = {prs[-1]:.6f}")
    print(f"# initial mean shift = {shifts[0]:.4f}")
    print(f"# final   mean shift = {shifts[-1]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_init, ax_pr = axes[0]
    ax_gap, ax_shift = axes[1]

    levels = np.linspace(0.0, 1.0, 11)
    ax_init.contourf(initial_XX, initial_YY, initial_prob, levels=levels, cmap="RdBu_r", alpha=0.6)
    ax_init.contour(initial_XX, initial_YY, initial_prob, levels=[0.5], colors="black", linewidths=0.8, linestyles="--")
    cs = ax_init.contour(final_XX, final_YY, final_prob, levels=[0.5], colors="black", linewidths=1.5)
    ax_init.scatter(x0[y.squeeze() == 1, 0], x0[y.squeeze() == 1, 1], s=8, c="red", alpha=0.6, label="y=1 (defaulters)")
    ax_init.scatter(x0[y.squeeze() == 0, 0], x0[y.squeeze() == 0, 1], s=8, c="blue", alpha=0.6, label="y=0")
    ax_init.set_title("Decision boundary: dashed = initial, solid = final")
    ax_init.set_xlabel("x_1"); ax_init.set_ylabel("x_2")
    ax_init.legend(loc="lower left", fontsize=8)
    ax_init.set_xlim(-4, 4); ax_init.set_ylim(-4, 4)

    ax_pr.plot(rounds, prs, marker="o", markersize=3)
    ax_pr.set_xlabel("round t"); ax_pr.set_ylabel("PR (BCE)"); ax_pr.set_title("Performative risk per round")
    ax_pr.grid(True, alpha=0.3)

    gap_x = [r for r, g in zip(rounds, gaps) if g is not None]
    gap_y = [g for g in gaps if g is not None]
    if gap_y:
        ax_gap.plot(gap_x, gap_y, marker="o", markersize=3)
        ax_gap.set_yscale("log")
    ax_gap.set_xlabel("round t"); ax_gap.set_ylabel(r"$\|\theta_t - \theta_{t-1}\|_2$"); ax_gap.set_title("Stability gap (log y)")
    ax_gap.grid(True, which="both", alpha=0.3)

    ax_shift.plot(rounds, shifts, marker="o", markersize=3, color="darkorange")
    ax_shift.set_xlabel("round t"); ax_shift.set_ylabel(r"mean$_i\,\|x_t^i - x_0^i\|_2$")
    ax_shift.set_title("Mean per-agent strategic shift")
    ax_shift.grid(True, alpha=0.3)

    fig.suptitle(f"Nonlinear strategic PP: MLP{args.hidden}, μ={args.mu}, k={args.rgd_steps}, lr={args.rgd_lr}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"# saved figure -> {args.out}")


if __name__ == "__main__":
    main()
