"""Initial interpretability-diagnostic test on controlled maps where ground
truth is known. (A) harness check: RRM on GaussianShiftMap should hit the
closed-form stable point. (B) gameable-feature downweighting: on
StrategicLinearMap the strat_features are the gameable ones, does a
performative method downweight them relative to a performance-blind fit?
Run: python experiments/diagnostics/run_diagnostics.py
"""

from __future__ import annotations

import torch

from perfsim.environments.map_env import MapEnvironment
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import MSELoss
from perfsim.maps.gaussian_shift import GaussianShiftMap
from perfsim.maps.strategic import StrategicLinearMap
from perfsim.models import LinearModel
from perfsim.simulator import Simulator


def gaussian_harness_check() -> None:
    d = 3
    gmap = GaussianShiftMap(A=0.5 * torch.eye(d), b=torch.ones(d), sigma_noise=0.01)
    model = LinearModel(d, 1, bias=False)
    sim = Simulator(env=MapEnvironment(gmap, batch_size=512),
                    learner=ERMLearner(model, MSELoss()), loss=MSELoss())
    sim.run(n_rounds=30, epoch_size=1, seed=0)
    got = model.get_params()[:d].detach()
    fp = gmap.closed_form_fp()
    print("[A] GaussianShift RRM vs closed-form theta_PS:")
    print(f"    got  {[round(float(v),3) for v in got]}")
    print(f"    PS   {[round(float(v),3) for v in fp]}")
    print(f"    ||err|| = {float((got - fp).norm()):.4f}  (small = harness OK)\n")


def downweighting_test() -> None:
    D, N = 4, 4000
    strat, nonstrat = [0, 1], [2, 3]
    gen = torch.Generator(); gen.manual_seed(1)
    x0 = torch.randn(N, D, generator=gen)
    w_true = torch.ones(D)
    y = (x0 @ w_true).unsqueeze(-1) + 0.1 * torch.randn(N, 1, generator=gen)
    base = {"x": x0, "y": y, "agent_idx": torch.arange(N)}
    smap = StrategicLinearMap(x0, y, epsilon=1.0, strat_features=strat)

    def wvec(m):
        return m.get_params()[:D].detach()

    def ratio(w):
        return float(w[strat].abs().mean() / w[nonstrat].abs().mean())

    # performance-blind: ERM on the clean base, no deployment feedback
    mn = LinearModel(D, 1, bias=False)
    ERMLearner(mn, MSELoss()).train(base)
    wn = wvec(mn)

    # RRM: retrain to the performative fixed point on the gamed distribution
    mr = LinearModel(D, 1, bias=False)
    Simulator(env=MapEnvironment(smap, batch_size=512),
              learner=ERMLearner(mr, MSELoss()), loss=MSELoss()).run(
        n_rounds=40, epoch_size=1, seed=0)
    wr = wvec(mr)

    # RGD: k gradient steps per round on the gamed distribution
    mg = LinearModel(D, 1, bias=False)
    Simulator(env=MapEnvironment(smap, batch_size=512),
              learner=GradientLearner(mg, MSELoss(), lr=0.05, steps_per_round=5),
              loss=MSELoss()).run(n_rounds=300, epoch_size=1, seed=0)
    wg = wvec(mg)

    print("[B] Gameable-feature downweighting (strat=[0,1] gameable, [2,3] not):")
    print(f"    {'method':>14s} | {'weights':>32s} | strat:nonstrat |w|")
    for name, w in (("naive ERM(base)", wn), ("RRM", wr), ("RGD", wg)):
        ws = [round(float(v), 3) for v in w]
        print(f"    {name:>14s} | {str(ws):>32s} | {ratio(w):.3f}")
    print("    (ratio ~1 = no downweighting; <1 = gameable features downweighted)")


def main() -> None:
    gaussian_harness_check()
    downweighting_test()


if __name__ == "__main__":
    main()
