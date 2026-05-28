"""AT-driven Schelling segregation scenario for perfsim."""

from perfsim.scenarios.at_schelling.data import (
    NYC_ACS_PROPORTIONS_4TYPE,
    TYPE_NAMES,
    initial_placement,
    load_nyc_demographics,
)
from perfsim.scenarios.at_schelling.default_callables import (
    default_feature_provider,
    default_signal_writer,
    default_signal_writer_grad,
    default_state_extractor,
)
from perfsim.scenarios.at_schelling.env import (
    build_schelling_runner,
    make_schelling_env,
)


def __getattr__(name: str):
    # model_scoring pulls in transformers, which is slow / can deadlock on
    # some installs (TF init). Keep it out of the eager-import path.
    if name == "BinaryLMScorer":
        from perfsim.scenarios.at_schelling.model_scoring import BinaryLMScorer
        return BinaryLMScorer
    raise AttributeError(name)


__all__ = [
    "make_schelling_env",
    "build_schelling_runner",
    "BinaryLMScorer",
    "default_feature_provider",
    "default_signal_writer",
    "default_signal_writer_grad",
    "default_state_extractor",
    "load_nyc_demographics",
    "initial_placement",
    "NYC_ACS_PROPORTIONS_4TYPE",
    "TYPE_NAMES",
]
