"""perfsim scenario: AgentTorch macro_economics (NYC labor + consumption ABM).

Mirrors the at_covid scenario for the bundled macro_economics model. The LM
acts as a "financial recommender" — per agent, it outputs a consumption
propensity in [0, 1] which gates the agent's spending decision. Aggregate
state (prices, inflation, unemployment) evolves under those decisions and
becomes input to the next round's SFT target. Direct macro analog of the
public-health-recommender pattern in at_covid.

Public entry:  make_macro_env(...) -> AgentTorchEnvironment
"""

from perfsim.scenarios.at_macro.env import (
    build_macro_runner,
    default_feature_provider,
    default_signal_writer,
    default_signal_writer_grad,
    default_state_extractor,
    make_macro_env,
)

__all__ = [
    "build_macro_runner",
    "default_feature_provider",
    "default_signal_writer",
    "default_signal_writer_grad",
    "default_state_extractor",
    "make_macro_env",
]
