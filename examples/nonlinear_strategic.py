"""End-to-end PP loop with an MLP predictor and gradient-strategic agents.

Trains an MLPModel on a synthetic 2-D binary classification problem where
agents best-respond to the current predictor via the input gradient of its
output. Each round:

  1. Population computes d f(x_0; theta_t) / dx at every agent's x_0.
  2. Agents shift: x_t = x_0 + eps * d f/dx (Perdomo sign: eps = -mu).
  3. Predictor sees (x_t, y) and runs k SGD steps on BCE-with-logits loss.
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from perfsim.history import History
    from perfsim.learners import GradientLearner
    from perfsim.losses import BCEWithLogitsLoss, L2RegularizedLoss
    from perfsim.metrics import performative_risk
    from perfsim.models import MLPModel
    from perfsim.simulator import Simulator
    from perfsim.environments.dynamics.strategic_gradient import StrategicGradientWorld
    return (
        BCEWithLogitsLoss,
        GradientLearner,
        History,
        L2RegularizedLoss,
        MLPModel,
        Simulator,
        StrategicGradientWorld,
        np,
        performative_risk,
        plt,
        torch,
    )


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # Nonlinear strategic PP

        2-D binary classification with an MLP. Each round agents shift along
        the input gradient of the deployed predictor's output. The predictor
        retrains. Visualizes the moving decision boundary, performative risk,
        stability gap, and per-agent shift magnitude.
        """
    )
    return (mo,)


@app.cell
def config():
    MU = 0.5
    N_AGENTS = 400
    N_ROUNDS = 20
    HIDDEN = (16, 16)
    RGD_LR = 0.05
    RGD_STEPS = 10
    WEIGHT_DECAY = 1e-4
    SEED = 0
    return HIDDEN, MU, N_AGENTS, N_ROUNDS, RGD_LR, RGD_STEPS, SEED, WEIGHT_DECAY


@app.cell
def make_population(N_AGENTS, SEED, torch):
    def _make_population(n, seed):
        g = torch.Generator().manual_seed(seed)
        x_pos = torch.randn(n // 2, 2, generator=g) + torch.tensor([1.5, 0.0])
        x_neg = torch.randn(n // 2, 2, generator=g) + torch.tensor([-1.5, 0.0])
        x0 = torch.cat([x_pos, x_neg], dim=0)
        y = torch.cat([torch.ones(n // 2, 1), torch.zeros(n // 2, 1)], dim=0)
        perm = torch.randperm(n, generator=g)
        return x0[perm], y[perm]

    x0, y = _make_population(N_AGENTS, SEED)
    return x0, y


@app.cell
def build_world_and_learner(
    BCEWithLogitsLoss,
    GradientLearner,
    HIDDEN,
    L2RegularizedLoss,
    MLPModel,
    MU,
    RGD_LR,
    RGD_STEPS,
    SEED,
    StrategicGradientWorld,
    WEIGHT_DECAY,
    torch,
    x0,
    y,
):
    torch.manual_seed(SEED)
    world = StrategicGradientWorld(x0=x0, y=y, epsilon=-MU)
    model = MLPModel(in_features=2, hidden_dims=list(HIDDEN), init_seed=SEED)
    base_loss = BCEWithLogitsLoss()
    train_loss = L2RegularizedLoss(base_loss, weight_decay=WEIGHT_DECAY, decay_bias=False)
    learner = GradientLearner(
        model, train_loss, lr=RGD_LR, steps_per_round=RGD_STEPS, optimizer="adam"
    )
    return base_loss, learner, model, train_loss, world


@app.cell
def decision_grid_fn(torch):
    def decision_grid(model, lim=4.0, n=80):
        xs = torch.linspace(-lim, lim, n)
        ys = torch.linspace(-lim, lim, n)
        XX, YY = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=-1)
        with torch.no_grad():
            logits = model(grid).reshape(n, n)
            probs = torch.sigmoid(logits)
        return XX.numpy(), YY.numpy(), probs.numpy()
    return (decision_grid,)


@app.cell
def snapshot_initial(decision_grid, model):
    initial_XX, initial_YY, initial_prob = decision_grid(model)
    return initial_XX, initial_YY, initial_prob


@app.cell
def run_sim(
    History,
    N_ROUNDS,
    SEED,
    Simulator,
    base_loss,
    learner,
    performative_risk,
    train_loss,
    world,
    x0,
):
    def _pr_metric(sim):
        return performative_risk(sim.world, sim.learner.model, base_loss)

    def _shift_norm(sim):
        data = sim.world.sample(sim.learner.model)
        return (data["x"] - x0).norm(dim=-1).mean()

    sim = Simulator(
        world=world,
        learner=learner,
        loss=train_loss,
        metrics={"PR": _pr_metric, "shift_norm": _shift_norm},
    )
    history: History = sim.run(n_rounds=N_ROUNDS, seed=SEED)
    return history, sim


@app.cell
def final_grid(decision_grid, model):
    final_XX, final_YY, final_prob = decision_grid(model)
    return final_XX, final_YY, final_prob


@app.cell
def summarize(history):
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
    return gaps, prs, rounds, shifts


@app.cell
def plot(
    HIDDEN,
    MU,
    RGD_LR,
    RGD_STEPS,
    final_XX,
    final_YY,
    final_prob,
    gaps,
    initial_XX,
    initial_YY,
    initial_prob,
    np,
    plt,
    prs,
    rounds,
    shifts,
    x0,
    y,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_init, ax_pr = axes[0]
    ax_gap, ax_shift = axes[1]

    _levels = np.linspace(0.0, 1.0, 11)
    ax_init.contourf(initial_XX, initial_YY, initial_prob, levels=_levels, cmap="RdBu_r", alpha=0.6)
    ax_init.contour(initial_XX, initial_YY, initial_prob, levels=[0.5], colors="black", linewidths=0.8, linestyles="--")
    ax_init.contour(final_XX, final_YY, final_prob, levels=[0.5], colors="black", linewidths=1.5)
    ax_init.scatter(x0[y.squeeze() == 1, 0], x0[y.squeeze() == 1, 1], s=8, c="red", alpha=0.6, label="y=1")
    ax_init.scatter(x0[y.squeeze() == 0, 0], x0[y.squeeze() == 0, 1], s=8, c="blue", alpha=0.6, label="y=0")
    ax_init.set_title("Decision boundary: dashed = initial, solid = final")
    ax_init.set_xlabel("x_1"); ax_init.set_ylabel("x_2")
    ax_init.legend(loc="lower left", fontsize=8)
    ax_init.set_xlim(-4, 4); ax_init.set_ylim(-4, 4)

    ax_pr.plot(rounds, prs, marker="o", markersize=3)
    ax_pr.set_xlabel("round"); ax_pr.set_ylabel("PR (BCE)"); ax_pr.set_title("Performative risk")
    ax_pr.grid(True, alpha=0.3)

    _gap_x = [r for r, g in zip(rounds, gaps) if g is not None]
    _gap_y = [g for g in gaps if g is not None]
    if _gap_y:
        ax_gap.plot(_gap_x, _gap_y, marker="o", markersize=3)
        ax_gap.set_yscale("log")
    ax_gap.set_xlabel("round"); ax_gap.set_ylabel("||theta_t - theta_{t-1}||"); ax_gap.set_title("Stability gap (log y)")
    ax_gap.grid(True, which="both", alpha=0.3)

    ax_shift.plot(rounds, shifts, marker="o", markersize=3, color="darkorange")
    ax_shift.set_xlabel("round"); ax_shift.set_ylabel("mean ||x_t - x_0||")
    ax_shift.set_title("Mean per-agent shift")
    ax_shift.grid(True, alpha=0.3)

    fig.suptitle(f"Nonlinear strategic PP: MLP{list(HIDDEN)}, mu={MU}, k={RGD_STEPS}, lr={RGD_LR}")
    fig.tight_layout()
    fig
    return (axes, fig)


if __name__ == "__main__":
    app.run()
