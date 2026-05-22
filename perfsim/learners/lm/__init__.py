"""LM-backed Learners: TRL-wrapped fine-tuning of an HFCausalLMModel.

v0 ships SFT and KL-anchored SFT. RL trainers (PG, PPO, GRPO, DPO) land
in v2 alongside the trajectory data schema (DESIGN.md §14).

Both classes import TRL / transformers lazily (inside `train`), so the
package import works without the `[lm]` extra installed.
"""

from perfsim.learners.lm.kl_sft import KLSFTLearner
from perfsim.learners.lm.sft import SFTLearner

__all__ = ["KLSFTLearner", "SFTLearner"]
