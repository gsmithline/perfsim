"""Covid scenario driving the bundled agent_torch covid model through perfsim.

Public API:
    make_covid_env(...) -> AgentTorchEnvironment
    build_covid_runner(seed) -> agent_torch.Runner
    PerfsimIsolationDecision  (SubstepAction)
    default_feature_provider, default_signal_writer, default_state_extractor

Requires the `agenttorch` extra: `pip install 'perfsim[agenttorch]'`.

See README.md in this directory for caveats (langchain shim, bundled
astoria path, OmegaConf resolver workaround).
"""

from perfsim.scenarios.at_covid.action import PerfsimIsolationDecision
from perfsim.scenarios.at_covid.env import (
    build_covid_runner,
    default_feature_provider,
    default_signal_writer,
    default_signal_writer_grad,
    default_state_extractor,
    make_covid_env,
    seed_initial_infections,
)

__all__ = [
    "PerfsimIsolationDecision",
    "build_covid_runner",
    "default_feature_provider",
    "default_signal_writer",
    "default_signal_writer_grad",
    "default_state_extractor",
    "make_covid_env",
    "seed_initial_infections",
]
