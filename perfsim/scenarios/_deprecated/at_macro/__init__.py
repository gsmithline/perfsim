"""perfsim scenario: AgentTorch macro_economics (NYC labor + consumption ABM).

PARKED / NOT RECOMMENDED FOR ACTIVE USE.

The bundled AgentTorch macro_economics simulator has multiple structural bugs.
After patching all of them (see _PatchedAssetsGoods, _PatchedUpdateAssets,
_PatchedMacroRates, _PatchedFinancialMarket in env.py) and adding
employment-responds-to-imbalance dynamics, the simulator does respond to
consumption decisions -- but LM beta sweeps don't produce meaningful
differentiation in the headline economic outcomes (inflation saturates at
the rate cap, unemployment is only weakly linked to consumption, and the
per-bucket consumption-aware target collapses the LM to ~0.65 for all
demographics). The hand-crafted SFT target is the binding limitation: it
isn't genuinely produced by ABM dynamics in the FJ sense, so the LM's loss
has no real signal tied to what the population would actually do.

The patches and code are preserved here in case someone returns to this
work. The covid scenario at_covid is the active ABM. Mastodon-sim is the
intended richer replacement (see perfsim README).

Public entry:  make_macro_env(...) -> AgentTorchEnvironment
"""

from perfsim.scenarios._deprecated.at_macro.env import (
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
