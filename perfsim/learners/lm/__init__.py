"""LM-backed Learners: TRL-wrapped fine-tuning of an HFCausalLMModel.

Ships SFT and KL-anchored SFT. Both import TRL / transformers lazily (inside
`train`), so the package import works without the `[lm]` extra installed.
"""

from perfsim.learners.lm.kl_sft import KLSFTLearner
from perfsim.learners.lm.sft import SFTLearner

__all__ = ["KLSFTLearner", "SFTLearner"]
