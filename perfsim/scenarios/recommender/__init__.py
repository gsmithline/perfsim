"""Performative recommender ecosystem scenario.

A platform ranker theta scores items; exposure is allocated by softmax over
scores; a batched user population (RecSim NG-style, community mixture) chooses
items via a differentiable multinomial-logit biased by exposure; logged
engagement becomes theta's next training data; and user interests drift toward
what was surfaced. Both labels (exposure bias) and population (taste drift) are
performative.

ProducerEcosystemWorld adds a content-producer population whose availability
share evolves by replicator dynamics driven by engagement -- giving rich-get-
richer, topic-diversity collapse, and an optional ecosystem-aware exposure boost.

Design follows the entity / behavior / story decomposition of RecSim NG
(Mladenov et al. 2021, arXiv:2103.08057), reimplemented in torch and wrapped to
perfsim's Environment / FullyDifferentiable contract. The producer dynamics are
replicator-based (perfsim's ReplicatorWorld math), inspired by -- not a port of
-- RecSim NG's provider model.
"""

from perfsim.scenarios.recommender.entities import (
    Corpus,
    UserPopulation,
    sample_corpus,
    sample_user_population,
)
from perfsim.scenarios.recommender.env import RecommenderEcosystemWorld
from perfsim.scenarios.recommender.env_producers import ProducerEcosystemWorld
from perfsim.scenarios.recommender.probes import (
    alignment_induced,
    alignment_truth,
    engagement_entropy,
    engagement_faithfulness,
    exposure_faithfulness,
    exposure_gini,
    induced_calibration,
    interest_drift,
    producer_diversity,
    surviving_producers,
    user_welfare,
)
from perfsim.scenarios.recommender.story import build_producer_env, build_recommender_env

__all__ = [
    "Corpus",
    "UserPopulation",
    "sample_corpus",
    "sample_user_population",
    "RecommenderEcosystemWorld",
    "ProducerEcosystemWorld",
    "build_recommender_env",
    "build_producer_env",
    "alignment_induced",
    "alignment_truth",
    "engagement_faithfulness",
    "exposure_faithfulness",
    "induced_calibration",
    "exposure_gini",
    "engagement_entropy",
    "interest_drift",
    "producer_diversity",
    "surviving_producers",
    "user_welfare",
]
