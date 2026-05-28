"""Substep functions for the Schelling scenario.

Each substep takes (state, config) and mutates state in place. The order
in which they appear in `env.build_schelling_substeps` is the order they
run within one Schelling round:

    compute_neighborhood -> happiness_predict_action -> move_decision ->
    execute_moves -> compute_realized_happiness

This mirrors the AT covid substep split (observation/policy/transition)
but flattened: we don't separate observation/policy from transition
because each substep is a pure function of state + config. The
`happiness_predict_action` substep is effectively a no-op transformer
that reads the LM's `platform_signal` and exposes it as `p_pred` for the
move-decision substep to consume; the LM call itself happens outside
runner.step() (perfsim.adapters.agenttorch.AgentTorchEnvironment.run).
"""

from perfsim.scenarios.at_schelling.substeps.compute_neighborhood import (
    compute_neighborhood,
)
from perfsim.scenarios.at_schelling.substeps.happiness_predict_action import (
    happiness_predict_action,
)
from perfsim.scenarios.at_schelling.substeps.move_decision import move_decision
from perfsim.scenarios.at_schelling.substeps.execute_moves import execute_moves
from perfsim.scenarios.at_schelling.substeps.compute_realized_happiness import (
    compute_realized_happiness,
)

__all__ = [
    "compute_neighborhood",
    "happiness_predict_action",
    "move_decision",
    "execute_moves",
    "compute_realized_happiness",
]
