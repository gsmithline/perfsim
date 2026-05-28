"""Schelling segregation scenario for perfsim.

Public API:
    make_schelling_env(...) -> AgentTorchEnvironment
    build_schelling_runner(seed, ...) -> SchellingRunner
    BinaryLMScorer(lm, yes_token, no_token)
    default_feature_provider, default_signal_writer, default_state_extractor
    load_nyc_demographics, NYC_ACS_PROPORTIONS_4TYPE, TYPE_NAMES

The Schelling LM-FJ-style PP loop:
    1. perfsim queries the BinaryLMScorer once per round; the scorer
       returns per-agent P(HAPPY | x_i) using next-token logits over
       the HAPPY / UNHAPPY token ids.
    2. signal_writer deposits p_i at
       runner.state["agents"]["residents"]["platform_signal"].
    3. runner.step(num_steps=K) runs K Schelling rounds. Within each
       round:
         a. compute_neighborhood -> same_frac, opp_frac, empty_frac.
         b. happiness_predict_action -> lift platform_signal to p_pred.
         c. move_decision -> H_i = clip(H_0 - lambda*(p_i - 0.5)),
            move_i = 1[s_i < H_i] via straight-through Bernoulli.
         d. execute_moves -> relocate movers to random empty cells.
         e. compute_realized_happiness -> y_i = 1[s_i_new >= H_0]
            (baseline threshold, NOT the LM-modulated one).
    4. state_extractor returns (x, y, agent_idx) to perfsim.
    5. SFTLearner (or KLSFTLearner) trains the LM on (prompt_i, target_i)
       where target_i is "HAPPY" or "UNHAPPY" based on y_i.

See README.md for usage examples.
"""

from perfsim.scenarios.at_schelling.data import (
    NYC_ACS_PROPORTIONS_4TYPE,
    TYPE_NAMES,
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
    build_schelling_substeps,
    make_schelling_env,
)
from perfsim.scenarios.at_schelling.model_scoring import BinaryLMScorer
from perfsim.scenarios.at_schelling.runner import SchellingRunner

__all__ = [
    "make_schelling_env",
    "build_schelling_runner",
    "build_schelling_substeps",
    "BinaryLMScorer",
    "SchellingRunner",
    "default_feature_provider",
    "default_signal_writer",
    "default_signal_writer_grad",
    "default_state_extractor",
    "load_nyc_demographics",
    "NYC_ACS_PROPORTIONS_4TYPE",
    "TYPE_NAMES",
]
