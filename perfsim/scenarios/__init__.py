"""Scenario bundles (World + Model + Loss + Learner + Dataset + reproduction).

Currently shipped:
    perfsim.scenarios.perdomo_loan : Perdomo et al. ICML 2020 strategic loan
        example, with GiveMeSomeCredit via Kaggle (real-data) and a
        synthetic fallback for CI.
"""

from perfsim.scenarios import perdomo_loan

__all__ = ["perdomo_loan"]
