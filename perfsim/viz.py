"""Training-trajectory visualization: risk landscapes, convergence, gradient diagnostics.

Three views of a PP run, following the standard decoupled-risk picture
(the ICLR 2026 performative-prediction blogpost):

Requires matplotlib (`pip install perfsim[viz]`). 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from torch import Tensor

from perfsim.core.environment import Environment
from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.history import History

Trajectories = Mapping[str, "History | Tensor"]


def theta_path(history: History | Tensor, *, key: str = "theta") -> Tensor:
    """(T, d) stack of recorded parameters from a History (or pass-through tensor)."""
    if isinstance(history, Tensor):
        return history.reshape(len(history), -1)
    rows = []
    for r in history.records:
        v = r[key]
        rows.append(v if isinstance(v, Tensor) else torch.tensor(v))
    return torch.stack([v.reshape(-1).float() for v in rows])


@dataclass
class RiskSurface:
    """DPR evaluated on a 1-D parameter slice theta(alpha) = base + alpha * direction.

    dpr[i, j] = E_{z ~ D(theta(grid[i]))} loss(theta(grid[j]), z): row = deployed
    (distribution-generating) parameter, column = evaluated parameter. `pr` is the
    diagonal, `best_response[i]` the alpha minimizing row i.
    """

    grid: Tensor
    dpr: Tensor
    pr: Tensor
    best_response: Tensor
    alpha_stable: float
    alpha_opt: float
    base: Tensor
    direction: Tensor

    @property
    def theta_stable(self) -> Tensor:
        return self.base + self.alpha_stable * self.direction

    @property
    def theta_opt(self) -> Tensor:
        return self.base + self.alpha_opt * self.direction

    def project(self, thetas: Tensor) -> Tensor:
        """Orthogonal projection of (T, d) thetas onto the slice coordinate alpha."""
        flat = thetas.reshape(len(thetas), -1) - self.base
        return (flat @ self.direction) / (self.direction @ self.direction)


def risk_surface(
    env: Environment,
    model: Model,
    loss: Loss,
    grid: Tensor,
    *,
    base: Tensor | None = None,
    direction: Tensor | None = None,
    n_samples: int = 4,
    seed: int = 0,
) -> RiskSurface:
    """Estimate DPR on grid x grid along a 1-D parameter slice.

    For scalar-parameter models the slice is parameter space itself. For d > 1
    pass `direction` (and optionally `base`) to choose the line theta(alpha) =
    base + alpha * direction. Each deploy point samples `n_samples` batches
    (env reset to seed..seed+n_samples-1) and every eval point is scored on the
    same batches, so rows share common random numbers.
    """
    d = model.num_params
    if direction is None:
        if d != 1:
            raise ValueError(
                f"model has {d} params; pass `direction` (and optionally `base`) "
                "to choose a 1-D slice"
            )
        direction = torch.ones(1)
    direction = direction.reshape(-1).float()
    base = torch.zeros(d) if base is None else base.reshape(-1).float()
    grid = grid.reshape(-1).float()
    thetas = base.unsqueeze(0) + grid.unsqueeze(1) * direction.unsqueeze(0)

    deploy, evaluate = model.clone(), model.clone()
    g = len(grid)
    dpr = torch.zeros(g, g)
    with torch.no_grad():
        for i in range(g):
            deploy.set_params(thetas[i])
            for s in range(n_samples):
                env.reset(seed=seed + s)
                data = env.sample(deploy)
                for j in range(g):
                    evaluate.set_params(thetas[j])
                    dpr[i, j] += loss(evaluate, data, reduction="mean") / n_samples

    pr = dpr.diagonal().clone()
    br_idx = dpr.argmin(dim=1)
    best_response = grid[br_idx]
    alpha_opt = float(grid[pr.argmin()])
    alpha_stable = float(grid[(best_response - grid).abs().argmin()])
    return RiskSurface(
        grid=grid,
        dpr=dpr,
        pr=pr,
        best_response=best_response,
        alpha_stable=alpha_stable,
        alpha_opt=alpha_opt,
        base=base,
        direction=direction,
    )


def _resolve_ax(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    return ax


def plot_landscape(
    surface: RiskSurface,
    *,
    trajectories: Trajectories | None = None,
    show_best_response: bool = True,
    cmap: str = "viridis",
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of the decoupled risk with stable/optimal points and cobweb trajectories.

    x = evaluated theta, y = deployed theta; the dashed diagonal is the
    performative risk. Each trajectory is drawn as retraining steps
    (theta_t, theta_t) -> (theta_{t+1}, theta_t) -> (theta_{t+1}, theta_{t+1}).
    """
    ax = _resolve_ax(ax)
    g = surface.grid
    ax.pcolormesh(g, g, surface.dpr, cmap=cmap, shading="nearest")
    ax.plot(g, g, ls="--", c="white", lw=1, label="deployed = evaluated")
    if show_best_response:
        ax.plot(surface.best_response, g, ls=":", c="white", lw=1.2, label="best response")
    ax.plot(surface.alpha_stable, surface.alpha_stable, "o", c="#ffd166", ms=9,
            mec="black", label="stable")
    ax.plot(surface.alpha_opt, surface.alpha_opt, "*", c="#ef476f", ms=14,
            mec="black", label="optimal")

    if trajectories:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for c, (name, traj) in zip(colors, trajectories.items()):
            a = surface.project(theta_path(traj)).tolist()
            xs, ys = [a[0]], [a[0]]
            for t in range(1, len(a)):
                xs += [a[t], a[t]]
                ys += [a[t - 1], a[t]]
            ax.plot(xs, ys, c=c, lw=1.5, alpha=0.9, label=name)
            ax.plot(a[0], a[0], "s", c=c, ms=5, mec="black")
            ax.plot(a[-1], a[-1], "D", c=c, ms=6, mec="black")

    ax.set(xlabel="evaluated theta", ylabel="deployed theta", title="decoupled risk")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return ax


def _surface_z(surface: RiskSurface, alpha_eval: Tensor, alpha_deploy: Tensor) -> Tensor:
    """Bilinear interpolation of dpr at (eval, deploy) slice coordinates."""
    g = surface.grid
    span = g[-1] - g[0]
    fx = ((alpha_eval - g[0]) / span * (len(g) - 1)).clamp(0, len(g) - 1 - 1e-6)
    fy = ((alpha_deploy - g[0]) / span * (len(g) - 1)).clamp(0, len(g) - 1 - 1e-6)
    x0, y0 = fx.long(), fy.long()
    wx, wy = fx - x0, fy - y0
    d = surface.dpr
    return ((1 - wy) * ((1 - wx) * d[y0, x0] + wx * d[y0, x0 + 1])
            + wy * ((1 - wx) * d[y0 + 1, x0] + wx * d[y0 + 1, x0 + 1]))


def plot_landscape_3d(
    surface: RiskSurface,
    *,
    trajectories: Trajectories | None = None,
    cross_sections: Sequence[float] = (),
    cmap: str = "viridis",
    elev: float = 22.0,
    azim: float = -125.0,
    ax: Axes | None = None,
) -> Axes:
    """3-D decoupled-risk surface in the blogpost style.

    z = DPR(deployed theta, evaluated theta), with the performative risk
    DPR(theta, theta) drawn along the diagonal, optional fixed-distribution
    cross-sections (deployed theta frozen), stable/optimal markers, and
    trajectories as retraining cobwebs draped on the surface.
    """
    if ax is None:
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(projection="3d")
    g = surface.grid
    xx, yy = torch.meshgrid(g, g, indexing="xy")
    ax.plot_surface(xx.numpy(), yy.numpy(), surface.dpr.numpy(), cmap=cmap,
                    alpha=0.65, linewidth=0, antialiased=True)
    lift = 0.01 * float(surface.dpr.max() - surface.dpr.min())

    ax.plot(g, g, surface.pr + lift, c="black", lw=2.5, label="performative risk")
    for a in cross_sections:
        i = int((g - a).abs().argmin())
        ax.plot(g, [float(g[i])] * len(g), surface.dpr[i] + lift, c="gray", lw=1.5)
        j = int(surface.dpr[i].argmin())
        ax.plot([float(g[j])], [float(g[i])], [float(surface.dpr[i, j]) + lift],
                "v", c="gray", ms=6)

    for alpha, marker, color, label in (
        (surface.alpha_stable, "o", "#ffd166", "stable"),
        (surface.alpha_opt, "*", "#ef476f", "optimal"),
    ):
        z = float(_surface_z(surface, torch.tensor([alpha]), torch.tensor([alpha]))) + lift
        ax.plot([alpha], [alpha], [z], marker, c=color, ms=11 if marker == "*" else 8,
                mec="black", label=label)

    if trajectories:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for c, (name, traj) in zip(colors, trajectories.items()):
            a = surface.project(theta_path(traj))
            xs, ys = [a[:1]], [a[:1]]
            steps = torch.linspace(0, 1, 16)
            for t in range(1, len(a)):
                xs += [a[t - 1] + steps * (a[t] - a[t - 1]), a[t].repeat(16)]
                ys += [a[t - 1].repeat(16), a[t - 1] + steps * (a[t] - a[t - 1])]
            x, y = torch.cat(xs), torch.cat(ys)
            z = _surface_z(surface, x, y) + 2 * lift
            ax.plot(x, y, z, c=c, lw=2, label=name)

    ax.view_init(elev=elev, azim=azim)
    ax.set(xlabel="evaluated theta", ylabel="deployed theta", zlabel="risk")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return ax


def plot_risk_curve(
    surface: RiskSurface,
    *,
    cross_sections: Sequence[float] = (),
    ax: Axes | None = None,
) -> Axes:
    """Performative risk along the diagonal, with optional fixed-distribution slices.

    Each alpha in `cross_sections` adds the curve DPR(alpha, .) -- the risk
    landscape if the distribution were frozen at D(theta(alpha)). At a stable
    point that slice is minimized on the diagonal.
    """
    ax = _resolve_ax(ax)
    g = surface.grid
    for a in cross_sections:
        i = int((g - a).abs().argmin())
        ax.plot(g, surface.dpr[i], c="gray", lw=1, alpha=0.6)
        ax.annotate(f"D fixed at {float(g[i]):.2g}", (float(g[-1]), float(surface.dpr[i, -1])),
                    fontsize=7, color="gray")
    ax.plot(g, surface.pr, c="black", lw=2, label="performative risk")
    ax.axvline(surface.alpha_stable, ls="--", c="#ffd166", label="stable")
    ax.axvline(surface.alpha_opt, ls="--", c="#ef476f", label="optimal")
    ax.set(xlabel="theta", ylabel="risk")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_convergence(
    trajectories: Trajectories,
    *,
    theta_star: Tensor | None = None,
    logy: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Per-round ||theta_t - theta_star|| (or stability gap if theta_star is None)."""
    ax = _resolve_ax(ax)
    for name, traj in trajectories.items():
        thetas = theta_path(traj)
        if theta_star is not None:
            y = (thetas - theta_star.reshape(-1)).norm(dim=1)
            rounds = range(len(thetas))
        else:
            y = (thetas[1:] - thetas[:-1]).norm(dim=1)
            rounds = range(1, len(thetas))
        ax.plot(rounds, y, marker="o", ms=3, label=name)
    if logy:
        ax.set_yscale("log")
    ylabel = "||theta_t - theta*||" if theta_star is not None else "||theta_t - theta_{t-1}||"
    ax.set(xlabel="round", ylabel=ylabel)
    ax.legend(frameon=False, fontsize=8)
    return ax


def gradient_norms(
    env: Environment,
    model: Model,
    loss: Loss,
    thetas: Tensor,
    *,
    eps: float = 1e-2,
    n_samples: int = 4,
    seed: int = 0,
) -> dict[str, Tensor]:
    """Gradient decomposition norms along a (T, d) parameter trajectory.

    Returns per-round norms of: the model term grad_theta DPR(theta_t, .) with
    the distribution frozen (autograd; zero at stable points), the total
    performative-risk gradient (central finite differences with common random
    numbers; zero at optimal points), and the distribution term (their
    difference). Cost is O(2d) risk evaluations per round.
    """
    work = model.clone()
    thetas = thetas.reshape(len(thetas), -1)
    d = thetas.shape[1]

    def avg_pr(theta: Tensor) -> float:
        work.set_params(theta)
        total = 0.0
        with torch.no_grad():
            for s in range(n_samples):
                env.reset(seed=seed + s)
                total += float(loss(work, env.sample(work), reduction="mean"))
        return total / n_samples

    out = {"model": [], "dist": [], "total": []}
    for theta in thetas:
        work.set_params(theta)
        gm = torch.zeros(d)
        for s in range(n_samples):
            env.reset(seed=seed + s)
            data = env.sample(work)
            value = loss(work, data, reduction="mean")
            grads = torch.autograd.grad(value, list(work.parameters()))
            gm += torch.cat([gr.reshape(-1) for gr in grads]) / n_samples
        gt = torch.zeros(d)
        for k in range(d):
            e = torch.zeros(d)
            e[k] = eps
            gt[k] = (avg_pr(theta + e) - avg_pr(theta - e)) / (2 * eps)
        out["model"].append(gm.norm())
        out["total"].append(gt.norm())
        out["dist"].append((gt - gm).norm())
    return {k: torch.stack(v) for k, v in out.items()}


def plot_gradient_norms(
    env: Environment,
    model: Model,
    loss: Loss,
    history: History | Tensor,
    *,
    eps: float = 1e-2,
    n_samples: int = 4,
    seed: int = 0,
    logy: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plot the gradient decomposition along a recorded trajectory.

    ||grad_model DPR|| -> 0 identifies convergence to a stable point;
    ||grad PR|| -> 0 identifies convergence to an optimal point.
    """
    ax = _resolve_ax(ax)
    norms = gradient_norms(
        env, model, loss, theta_path(history), eps=eps, n_samples=n_samples, seed=seed
    )
    labels = {
        "model": "||grad_model DPR|| (0 at stable)",
        "dist": "||grad_dist DPR||",
        "total": "||grad PR|| (0 at optimal)",
    }
    for key, lab in labels.items():
        ax.plot(range(len(norms[key])), norms[key], marker="o", ms=3, label=lab)
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel="round", ylabel="gradient norm")
    ax.legend(frameon=False, fontsize=8)
    return ax
