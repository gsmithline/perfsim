"""Smoke test for the AI-mediated human data harness.

Builds a synthetic population with an explicit z/c/y split:
  - c (fundamentals, preserved by the mediator) -- columns 0..3
  - z (style, rewritten by the mediator)        -- columns 4..7
  - label y = f(z): the signal lives entirely in the style columns.

It demonstrates faithfully (mechanics check on synthetic data, NOT a scientific
result) what the harness actually shows, which is the SC-notebook null in
miniature: recoverable information is surprisingly hard to destroy.

  Part 1  the metric has power, AND variance shrinking is not collapse. A linear
          pull-toward-centroid (lambda<1) is an affine map on z, so it crushes
          style variance to ~0 but PRESERVES linear separability -> recoverability
          holds at ~1.0. Only lambda=1 (exact, non-injective collapse: every
          style vector mapped to the same point) actually drops it to chance.
  Part 2  additive noise does not remove a linear signal either. Even large
          jitter (variance going UP) leaves recoverability ~1.0, because the
          linear component survives underneath and a probe averages the noise
          out. Diversity dropping and diversity rising are both red herrings for
          information.
  Part 3  the clean-anchor vs mediated-anchor contrast (all four regimes run).
          The ORDERING is right (clean anchor / accumulate hold; replace and
          mediated anchor dip), but on this perfectly-separable synthetic label
          the gap is tiny -- a real effect needs a task where the clean signal
          is not trivially recoverable.

Run:  python examples/ai_mediated_smoke.py
"""

from __future__ import annotations

import torch

from perfsim.core.predictor import Predictor
from perfsim.environments.mediation import ContractionMediator, MediationWorld
from perfsim.learners import ERMLearner
from perfsim.losses import BCELoss
from perfsim.models import LogisticModel
from perfsim.scenarios.ai_mediated import (
    model_auc,
    recoverability,
    run_mediated,
    style_variance,
)

N = 1500
N_C = 4
N_Z = 4
D = N_C + N_Z
STYLE_FEATURES = list(range(N_C, D))  # the z columns


def build_fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(N, N_C, generator=g)
    z = torch.randn(N, N_Z, generator=g)
    x = torch.cat([c, z], dim=1)
    w_z = torch.randn(N_Z, generator=g)
    y_z = (torch.sigmoid(z @ w_z) > 0.5).float().unsqueeze(-1)  # label lives in style
    return x, y_z


def fresh_predictor() -> Predictor:
    model = LogisticModel(in_features=D)
    learner = ERMLearner(model, BCELoss(), max_iter=200)
    return Predictor(model=model, loss=BCELoss(), learner=learner)


def one_shot_mediated(x, y, mediator) -> dict:
    world = MediationWorld(x, y, mediator=mediator, style_features=STYLE_FEATURES)
    recs = run_mediated(world, fresh_predictor(), n_rounds=1, regime="replace",
                        probes={"rec": recoverability,
                                "sv": lambda d: style_variance(d, world.style_mask)})
    return recs[0]


def main() -> None:
    x, y = build_fixture()
    raw = MediationWorld(x, y, mediator=ContractionMediator(0.0),
                         style_features=STYLE_FEATURES).raw_data
    print(f"raw data: recoverability={recoverability(raw):.3f}  "
          f"style_var={style_variance(raw, torch.tensor([c in STYLE_FEATURES for c in range(D)])):.2f}")

    print("\n=== Part 1: metric power + variance is NOT collapse (affine, no noise) ===")
    print("  label=f(z); pure lambda-contraction toward the style centroid")
    for lam in (0.5, 0.9, 0.99, 1.0):
        r = one_shot_mediated(x, y, ContractionMediator(lam, target_mode="centroid"))
        print(f"  lambda={lam:<5}  style_var={r['sv']:7.3f}  recoverability={r['rec']:.3f}")
    print("  -> variance collapses, recoverability holds (affine-invariant); only")
    print("     lambda=1 (non-injective: all z -> one point) drops it to chance")

    print("\n=== Part 2: additive noise does NOT remove a linear signal (lambda=0.8) ===")
    for jit in (0.0, 0.5, 1.0, 2.0):
        r = one_shot_mediated(x, y, ContractionMediator(0.8, target_mode="centroid", jitter=jit))
        print(f"  jitter={jit:<5} style_var={r['sv']:7.3f}  recoverability={r['rec']:.3f}")
    print("  -> variance now RISES with jitter yet recoverability stays ~1.0:")
    print("     the linear signal survives the noise; diversity is not information")

    print("\n=== Part 3: clean-anchor vs mediated-anchor (degrading channel) ===")
    print("  mediator = lambda 0.8 + jitter 1.5; platform raw-AUC on the clean population each round")
    for regime in ("replace", "accumulate", "clean_anchor", "mediated_anchor"):
        world = MediationWorld(
            x, y,
            mediator=ContractionMediator(0.8, target_mode="centroid", jitter=1.5),
            style_features=STYLE_FEATURES,
        )
        pred = fresh_predictor()
        raw_auc_traj: list[float] = []
        run_mediated(
            world, pred, n_rounds=6, regime=regime, seed=0, alpha=0.5,
            probes={"rec": recoverability},
            on_round=lambda t, rec: raw_auc_traj.append(model_auc(pred.model, world.raw_data)),
        )
        traj = " ".join(f"{a:.3f}" for a in raw_auc_traj)
        print(f"  {regime:16s}  platform raw-AUC/round: {traj}")
    print("  -> ordering is right (clean_anchor/accumulate hold, replace/mediated dip)")
    print("     but the gap is tiny here: this synthetic label is trivially separable")


if __name__ == "__main__":
    main()
