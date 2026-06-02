"""AI-mediated human data scenario: the mediation channel as D(theta).

Plug any Dataset into a MediationWorld (the env), pick a Mediator (psi), pick
where the label lives (style vs fundamentals), pick a retrain regime, run the
loop, and measure recoverability of the label across rounds.

See perfsim.environments.mediation for the env + mediators, and `run_mediated`
here for the retrain-regime driver.
"""

from perfsim.environments.mediation import (
    ContractionMediator,
    IdentityMediator,
    LinearSurrogateMediator,
    MediationWorld,
    Mediator,
)
from perfsim.scenarios.ai_mediated.driver import REGIMES, run_mediated
from perfsim.scenarios.ai_mediated.probes import (
    model_auc,
    recoverability,
    style_variance,
)

__all__ = [
    "ContractionMediator",
    "IdentityMediator",
    "LinearSurrogateMediator",
    "MediationWorld",
    "Mediator",
    "REGIMES",
    "run_mediated",
    "recoverability",
    "style_variance",
    "model_auc",
]
