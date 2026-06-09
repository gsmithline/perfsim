# `perfsim.scenarios.recommender`

A recommender that feeds on its own choices. The model is the ranker: it scores items, those scores decide what gets shown, users mostly engage with what they're shown, and that engagement becomes the model's next training set. At the same time users' tastes drift toward whatever they consumed. So both the labels (engagement is inflated by exposure) and the population (tastes move) react to the model.

Inspired by RecSim NG (Mladenov et al. 2021), reimplemented in torch. The population is one batched `(N, D)` interest tensor, not one object per user.

## What one round does (demand side, `RecommenderEcosystemWorld`)

1. the model scores items
2. exposure = softmax over scores (sharper `beta` = more winner-take-all)
3. each user picks items, pulled both by their true interest and by what was shown them (`alpha`)
4. engagement is logged as the label
5. interests drift toward what was consumed (`eta`)
6. the model retrains on the logged engagement

`epoch_size > 1` lets tastes evolve a few steps before retraining; `epoch_size = 1` is the lockstep loop.

### Knobs

| Param | Meaning |
|---|---|
| `alpha` | how much being shown an item inflates engagement (the label channel) |
| `beta` | ranker sharpness; high `beta` concentrates attention on a few items |
| `eta` | how fast tastes drift toward what was consumed (the population channel) |
| `temp` | choice randomness |
| `n_communities`, `community_spread` | how varied the user population is |

You can isolate the two effects: sweep `alpha` with `eta=0` for the label channel, sweep `eta` with `alpha=0` for the taste-drift channel.

### Measuring what happens (`probes.py`)

The initial population's true tastes (`interest0`) are never trained on, so `world.hidden_quality()` gives you a ground truth to compare against. The headline probe is **`engagement_faithfulness`**: how well the logged engagement still tracks what users truly like. It stays flat with no exposure bias (`alpha=0`) and settles lower once the loop is on (about 0.33 to 0.22 at `alpha>=4`), the self-corruption. Others: `exposure_faithfulness` (does attention reach genuinely relevant items), `exposure_gini` (feed concentration), `interest_drift` (how far tastes moved).

## Producers (`ProducerEcosystemWorld`)

Adds content creators. Each item is a producer with a surviving share that grows or shrinks by how much engagement it gets (a replicator). So the ranker's exposure decides which producers survive, which decides next round's content. A myopic ranker drives rich-get-richer (diversity collapses to one producer); an ecosystem-aware `boost` to smaller producers counteracts it. Extra knobs: `gamma_supply` (only performative when > 0), `repl_rate`, `kappa`, `boost`. Extra probes: `producer_diversity`, `surviving_producers`, `user_welfare`.

## Competing platforms (`CompetingRecommendersWorld`)

Several rankers over one shared, drifting population. Each user splits attention across the platforms, and each platform trains only on the engagement its own share produced; `eta_mob` lets attention move between platforms over time. Run it with `run_competition`. Use it to ask whether competing platforms differentiate or converge.

## Quickstart

```python
from perfsim.scenarios.recommender import build_recommender_env, build_producer_env
from perfsim.models import LinearModel
from perfsim.losses import MSELoss
from perfsim.learners import ERMLearner
from perfsim.simulator import Simulator

env = build_recommender_env(n_items=30, n_users=200, dim=8, alpha=4.0, beta=4.0, eta=0.15)
# or: build_producer_env(n_items=30, n_users=200, dim=8, gamma_supply=1.0, boost=0.0)

sim = Simulator(env=env, learner=ERMLearner(LinearModel(8, 1), MSELoss()), loss=MSELoss())
hist = sim.run(n_rounds=40, epoch_size=1, seed=0)
```

Everything is differentiable, so these worlds also work with `PerfGDLearner` (gradient straight through exposure and choice to engagement).

## Files

- `entities.py` — items and the user population
- `choice.py` — the differentiable user choice
- `behaviors.py` — exposure, engagement, drift, producer dynamics
- `env.py`, `env_producers.py`, `env_competition.py` — the three worlds
- `probes.py` — the metrics
- `story.py` — the `build_*_env` helpers
