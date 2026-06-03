"""Social-dilemma performative-prediction scenario.

A population plays a public-goods game; the platform model theta predicts each
agent's equilibrium cooperation from fixed features x, and its prediction
re-enters the dynamics as a behavioral mediator (strength gamma). The central
question is whether the performatively stable predictor is accurate only by
inducing a degraded (less cooperative, homogenized) population.

Two population engines, same payoff + gamma-shaping semantics:
- PublicGoodsDilemma (dilemma.py): cheap surrogate -- payoff-weighted bootstrap
  imitation with a type anchor and a theta anchor. Fast, no RL.
- PublicGoodsMARL (ppo_dilemma.py): the real thing -- a parameter-shared PPO
  population (torchrl IPPO) acting in PublicGoodsSubstrate, with theta entering
  via reward shaping.
"""

from perfsim.scenarios.social_dilemma.dilemma import PublicGoodsDilemma
from perfsim.scenarios.social_dilemma.probes import (
    calibration_error,
    cooperation_variance,
    mean_cooperation,
    type_r2,
)

__all__ = [
    "PublicGoodsDilemma",
    "calibration_error",
    "cooperation_variance",
    "mean_cooperation",
    "type_r2",
]

# PublicGoodsMARL pulls in torchrl, an optional dependency; import lazily so the
# cheap surrogate path works without it.
try:
    from perfsim.scenarios.social_dilemma.ppo_dilemma import PublicGoodsMARL  # noqa: F401
    from perfsim.scenarios.social_dilemma.substrate import PublicGoodsSubstrate  # noqa: F401

    __all__ += ["PublicGoodsMARL", "PublicGoodsSubstrate"]
except ImportError:
    pass
