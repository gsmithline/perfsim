"""Library API for the Perdomo loan scenario.

The CLI lives in `__main__.py`; invoke with:
    python -m perfsim.scenarios.perdomo_loan ...

Use `run(config)` from code.

Loss handling matches Perdomo:
- Training loss is `L2RegularizedLoss(BCEWithLogits, weight_decay,
  decay_bias=False)`. The bias term is NOT regularized.
- Reported PR uses the unregularized base BCE loss (Perdomo's convention
  for reporting performative risk).

Strategic features: per `config.strat_features` (default Perdomo's
[0, 5, 7]); only these columns are shifted each round.
"""

from __future__ import annotations

from perfsim.history import History
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import BCEWithLogitsLoss, L2RegularizedLoss
from perfsim.metrics import performative_risk
from perfsim.models import LinearModel
from perfsim.scenarios.perdomo_loan.config import PerdomoLoanConfig, build_dataset
from perfsim.scenarios.perdomo_loan.world import build_world
from perfsim.simulator import Simulator


def run(config: PerdomoLoanConfig) -> History:
    """Run the Perdomo loan scenario for one config and return the History."""
    dataset = build_dataset(config)
    world = build_world(
        dataset,
        mu=config.mu,
        standardize=config.standardize,
        robust=config.robust,
        clip=config.clip,
        strat_features=config.strat_features,
    )
    model = LinearModel(in_features=world.dim, out_features=1, bias=True)

    # Juan's exact lam = 1/n (bias excluded); weight_decay=None resolves to that
    # from the training set size.
    n_train = world.n_agents
    weight_decay = 1.0 / n_train if config.weight_decay is None else config.weight_decay

    base_loss = BCEWithLogitsLoss()
    if weight_decay > 0.0:
        train_loss = L2RegularizedLoss(
            base_loss,
            weight_decay=weight_decay,
            decay_bias=config.decay_bias,
        )
    else:
        train_loss = base_loss

    if config.learner == "erm":
        learner = ERMLearner(model, train_loss, max_iter=200)
    elif config.learner == "gradient":
        learner = GradientLearner(
            model,
            train_loss,
            lr=config.learner_lr,
            steps_per_round=config.learner_steps,
            optimizer="sgd",
        )
    else:
        raise ValueError(
            f"unknown learner {config.learner!r}; expected 'erm' or 'gradient'"
        )

    def pr_metric(sim: Simulator):
        # Reported PR uses unregularized base loss (Perdomo convention).
        return performative_risk(sim.world, sim.learner.model, base_loss)

    sim = Simulator(
        world=world,
        learner=learner,
        loss=train_loss,
        metrics={"PR": pr_metric},
        dataset=dataset,
    )
    return sim.run(n_rounds=config.n_rounds, seed=config.seed)
