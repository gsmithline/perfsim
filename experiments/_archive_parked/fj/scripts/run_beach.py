"""Beach operators under ERM predictive loops on an FJ population: the model
class (bias-only / linear / community / mlp) selects what the opinion
distribution does.
Run: python experiments/fj/run_beach.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.environments.dynamics.fj import FJWorld, normalize_adjacency
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel, MLPModel
from perfsim.simulator import Simulator

OUT = Path("runs/fj_beach")
SEEDS = (0, 1, 2)
N = 200
ROUNDS = 30
K_INNER = 10
BETA_PANEL = 0.6


class BiasOnly(Model):
    """One shared scalar prediction: the broadcast-mean platform."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.bias.expand(x.shape[0], 1)


def make_population(seed: int) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    community = (torch.arange(N) >= N // 2).long()
    innate = torch.where(community == 0, -1.0, 1.0) + 0.3 * torch.randn(N, generator=gen)
    p = torch.where(community.unsqueeze(0) == community.unsqueeze(1), 0.10, 0.01)
    adj = (torch.rand(N, N, generator=gen) < p).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()
    return innate, normalize_adjacency(adj), community


def make_model(kind: str, community: Tensor) -> tuple[Model, Tensor | None]:
    """Returns (model, features); features=None keeps the world default (innate)."""
    if kind == "bias only":
        return BiasOnly(), None
    if kind == "linear":
        return LinearModel(1, 1), None
    if kind == "community":
        onehot = torch.nn.functional.one_hot(community, 2).float()
        return LinearModel(2, 1, bias=False), onehot
    if kind == "mlp":
        return MLPModel(1, (32, 32), init_seed=0), None
    raise ValueError(kind)


MODELS = ("bias only", "linear", "community", "mlp")
COLORS = {
    "no platform": "#888888", "bias only": "#d8633b", "linear": "#3b6fd8",
    "community": "#3bb0a0", "mlp": "#8d3bd8",
}


def trajectory(kind: str, beta: float, seed: int) -> dict[str, list[float]]:
    innate, graph, community = make_population(seed)
    if kind == "no platform":
        beta, kind_eff = 0.0, "bias only"
    else:
        kind_eff = kind
    model, features = make_model(kind_eff, community)
    world = FJWorld(
        innate, graph, torch.full((N,), 0.5), platform_sus=beta, features=features,
    )
    sim = Simulator(env=world, learner=ERMLearner(model, MSELoss()), loss=MSELoss())
    out: dict[str, list[float]] = {"mean": [], "std": [], "sep": []}
    out["innate_std"] = [float(innate.std())]
    out["innate_sep"] = [
        float((innate[community == 1].mean() - innate[community == 0].mean()).abs())
    ]

    def on_round(t: int, record: dict) -> None:
        x = world.current_opinion
        out["mean"].append(float(x.mean()))
        out["std"].append(float(x.std()))
        out["sep"].append(float((x[community == 1].mean() - x[community == 0].mean()).abs()))

    x_feat = world.features
    initial = {
        "x": x_feat.unsqueeze(-1) if x_feat.ndim == 1 else x_feat,
        "y": innate.unsqueeze(-1),
        "agent_idx": torch.arange(N),
    }
    sim.run(n_rounds=ROUNDS, epoch_size=K_INNER, seed=seed,
            initial_data=initial, on_round=on_round)
    return out


def _band(ax, runs, key, color, label):
    curves = torch.tensor([r[key] for r in runs])
    mean = curves.mean(dim=0)
    ax.plot(range(len(mean)), mean, color=color, label=label)
    ax.fill_between(range(len(mean)), curves.min(dim=0).values,
                    curves.max(dim=0).values, color=color, alpha=0.12)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = ("no platform",) + MODELS
    runs_at_beta = {name: [trajectory(name, BETA_PANEL, s) for s in SEEDS] for name in names}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = (("mean", "beach mean (translation)"),
              ("std", "beach std (contraction)"),
              ("sep", "community separation (split survival)"))
    for ax, (key, title) in zip(axes.flat, panels):
        for name, runs in runs_at_beta.items():
            _band(ax, runs, key, COLORS[name], name)
        ax.set_title(f"{title}  (platform_sus={BETA_PANEL:g})")
        ax.set_xlabel("round")
    axes.flat[0].legend(fontsize=8)

    ax = axes.flat[3]
    betas = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
    for name in MODELS:
        finals = []
        for b in betas:
            vals = [trajectory(name, b, s)["std"][-1] for s in SEEDS]
            finals.append(sum(vals) / len(vals))
        ax.plot(betas, finals, "o-", color=COLORS[name], label=name)
    ax.set_xlabel("platform_sus (paper's beta)")
    ax.set_ylabel("equilibrium beach std")
    ax.set_title("homogenization gate: std falls with platform trust")
    ax.legend(fontsize=8)

    fig.suptitle("The beach under ERM predictive loops: model capacity selects the operator",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "beach.png", dpi=140)

    print(f"{'model':>12s} | {'displacement':>12s} {'std vs innate':>13s} {'sep vs innate':>13s}")
    for name, runs in runs_at_beta.items():
        dm = sum(r["mean"][-1] for r in runs) / len(runs)
        sr = sum(r["std"][-1] / r["innate_std"][0] for r in runs) / len(runs)
        pr = sum(r["sep"][-1] / r["innate_sep"][0] for r in runs) / len(runs)
        print(f"{name:>12s} | {dm:+12.3f} {sr:13.3f} {pr:13.3f}")
    print(f"[fj] figure -> {OUT / 'beach.png'}  (seeds {SEEDS})")


if __name__ == "__main__":
    main()
