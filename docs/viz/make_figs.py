"""Regenerate the README viz figures: python docs/viz/make_figs.py"""

import matplotlib
import torch

matplotlib.use("Agg")

from perfsim import viz
from perfsim.environments.dynamics import GaussianShiftWorld
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.simulator import Simulator

OUT = "docs/viz"


def world():
    w = GaussianShiftWorld(A=torch.tensor([[0.5]]), b=torch.tensor([1.0]),
                           sigma_noise=0.05, batch_size=512)
    w.reset(seed=0)
    return w


def model():
    m = LinearModel(in_features=1, out_features=1, bias=False)
    m.set_params(torch.tensor([0.0]))
    return m


def main():
    loss = MSELoss()
    hist = {
        "RRM": Simulator(env=world(), learner=ERMLearner(model(), loss),
                         loss=loss).run(8, seed=0),
        "RGD": Simulator(env=world(), learner=GradientLearner(model(), loss, lr=0.1,
                         steps_per_round=5), loss=loss).run(8, seed=0),
    }
    surf = viz.risk_surface(world(), model(), loss, torch.linspace(0, 3, 41), n_samples=4)

    ax = viz.plot_landscape(surf, trajectories=hist)
    ax.figure.savefig(f"{OUT}/landscape.png", dpi=150)
    ax = viz.plot_landscape_3d(surf, trajectories=hist,
                               cross_sections=[0.5, surf.alpha_stable])
    ax.figure.savefig(f"{OUT}/landscape_3d.png", dpi=150)
    ax = viz.plot_risk_curve(surf, cross_sections=[0.5, surf.alpha_stable])
    ax.figure.savefig(f"{OUT}/risk_curve.png", dpi=150)
    ax = viz.plot_convergence(hist, theta_star=torch.tensor([2.0]))
    ax.figure.savefig(f"{OUT}/convergence.png", dpi=150)
    ax = viz.plot_gradient_norms(world(), model(), loss, hist["RGD"], n_samples=4)
    ax.figure.savefig(f"{OUT}/gradient_norms.png", dpi=150)


if __name__ == "__main__":
    main()
